"""
vision_features/gallery/pose_conditioned.py
─────────────────────────────────────────────
Pose-conditioned ReID gallery: 8 slots per cow per modality.

Slots = {standing, lying} × {left, right, toward, away}
Gallery files:
    gallery_pose_day.npy   → dict { real_id (int) : np.ndarray (8, 128) }
    gallery_pose_night.npy → same

A slot is "populated" if gallery_n[slot] > 0.
An unpopulated slot returns None on query; callers must handle the fallback chain.

The flat (non-slot) galleries in reconcile.py (gallery_day.npy / gallery_night.npy)
are untouched — this module adds the 8-slot galleries alongside them. reconcile.py
step B still updates the flat gallery; step C (cosine resolver) can optionally use
the slot galleries for finer discrimination once they are populated.

Fallback query chain (slot-aware cosine resolver):
    1. Exact slot (posture + facing match)
    2. Same posture, any facing  (4 slots aggregated by mean)
    3. Any populated slot        (all 8, mean — equivalent to flat gallery)
    4. None                      (caller falls back to kinetics-only)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ..schema import (
    N_SLOTS, GALLERY_SLOTS, SLOT_NAMES,
    Posture, Facing,
    slot_index, slot_name,
)

# ─────────────────────────────────────────────────────────────────────────────
# Gallery data container
# ─────────────────────────────────────────────────────────────────────────────

class PoseGallery:
    """
    In-memory representation of the 8-slot pose-conditioned gallery.

    Attributes
    ----------
    embeds : dict { real_id -> np.ndarray (8, 128) }
             NaN rows indicate empty slots.
    counts : dict { real_id -> np.ndarray (8,) int }
             Number of sessions that have contributed to each slot.
    """

    def __init__(self) -> None:
        self.embeds: dict[int, np.ndarray] = {}   # (8, 128) per cow
        self.counts: dict[int, np.ndarray] = {}   # (8,)     per cow

    # ── I/O ──────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, gallery_dir: str, modality: str) -> "PoseGallery":
        """
        Load from gallery_pose_{modality}.npy.
        Returns an empty gallery if the file doesn't exist.
        modality: 'day' | 'night'
        """
        path = Path(gallery_dir) / f"gallery_pose_{modality}.npy"
        g    = cls()
        if not path.exists():
            return g
        data = np.load(str(path), allow_pickle=True).item()
        g.embeds = {int(k): v["embeds"].astype(np.float32) for k, v in data.items()}
        g.counts = {int(k): v["counts"].astype(np.int32)   for k, v in data.items()}
        return g

    def save(self, gallery_dir: str, modality: str) -> None:
        """Save to gallery_pose_{modality}.npy. Creates directory if needed."""
        Path(gallery_dir).mkdir(parents=True, exist_ok=True)
        path = Path(gallery_dir) / f"gallery_pose_{modality}.npy"
        data = {
            k: {"embeds": self.embeds[k], "counts": self.counts[k]}
            for k in self.embeds
        }
        np.save(str(path), data)

    # ── Update (EMA per slot) ─────────────────────────────────────────────────

    def update(
        self,
        real_id: int,
        slot_embeds: dict[int, np.ndarray],  # {slot_idx -> mean_embed (128,)}
        alpha: float = 0.15,
    ) -> dict[int, float]:
        """
        EMA-update each populated slot for a single cow.

        Parameters
        ----------
        real_id     : cow identity
        slot_embeds : {slot_idx -> session mean embed (128,)} for confirmed frames.
                      Only slots with entries are updated; empty slots untouched.
        alpha       : EMA learning rate (full α for kinetic-confirmed,
                      α/2 for cosine-only — callers divide before passing in).

        Returns
        -------
        dict {slot_idx -> cosine(old, new)} for updated slots (for logging).
        """
        if real_id not in self.embeds:
            self.embeds[real_id] = np.full((N_SLOTS, 128), np.nan, dtype=np.float32)
            self.counts[real_id] = np.zeros(N_SLOTS, dtype=np.int32)

        cosines = {}
        for slot_idx, new_vec in slot_embeds.items():
            old_vec = self.embeds[real_id][slot_idx]

            if np.all(np.isnan(old_vec)):
                # First time this slot is populated
                updated = _l2_norm(new_vec)
                cosines[slot_idx] = float("nan")
            else:
                updated  = _l2_norm(alpha * new_vec + (1 - alpha) * old_vec)
                cosines[slot_idx] = float(np.dot(old_vec, new_vec))

            self.embeds[real_id][slot_idx] = updated
            self.counts[real_id][slot_idx] += 1

        return cosines

    # ── Query (slot-aware cosine similarity) ─────────────────────────────────

    def query(
        self,
        query_embed: np.ndarray,      # (128,) L2-normalised
        posture: Posture | None,
        facing:  Facing  | None,
        threshold: float = 0.75,
    ) -> tuple[int | None, float, int | None]:
        """
        Find the best-matching cow using the fallback chain.

        Fallback:
          1. Exact slot                (posture + facing)
          2. Same posture, any facing  (4 slots)
          3. All populated slots       (8 slots)
          4. None

        Parameters
        ----------
        query_embed : (128,) L2-normalised embedding to match.
        posture     : Posture of the query frame (or None if uncertain).
        facing      : Facing  of the query frame (or None if uncertain).
        threshold   : minimum cosine similarity to accept a match.

        Returns
        -------
        (real_id or None, cosine_similarity, slot_idx_used or None)
        """
        if not self.embeds:
            return None, 0.0, None

        # ── determine which slots to use (fallback cascade) ──────────────────
        exact_slot = None
        if (posture is not None and facing is not None
                and posture != Posture.UNCERTAIN and facing != Facing.UNCERTAIN):
            exact_slot = slot_index(posture, facing)

        # slots grouped by posture for fallback level 2
        posture_slots: list[int] | None = None
        if posture is not None and posture != Posture.UNCERTAIN:
            posture_slots = [
                idx for idx, (p, _) in enumerate(GALLERY_SLOTS) if p == posture
            ]

        def _best_from_slots(slots: list[int]) -> tuple[int | None, float, int | None]:
            """Score all cows on the given slot indices, return best match."""
            best_id, best_cos, best_slot = None, -1.0, None
            for real_id, embeds_mat in self.embeds.items():
                for s in slots:
                    vec = embeds_mat[s]
                    if np.all(np.isnan(vec)):
                        continue
                    cos = float(np.dot(query_embed, vec))
                    if cos > best_cos:
                        best_cos, best_id, best_slot = cos, real_id, s
            return best_id, best_cos, best_slot

        # Level 1: exact slot
        if exact_slot is not None:
            rid, cos, s = _best_from_slots([exact_slot])
            if rid is not None and cos >= threshold:
                return rid, cos, s

        # Level 2: same posture, any facing
        if posture_slots is not None:
            rid, cos, s = _best_from_slots(posture_slots)
            if rid is not None and cos >= threshold:
                return rid, cos, s

        # Level 3: all populated slots
        all_slots = list(range(N_SLOTS))
        rid, cos, s = _best_from_slots(all_slots)
        if rid is not None and cos >= threshold:
            return rid, cos, s

        return None, max(0.0, cos if 'cos' in dir() else 0.0), None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def populated_slots(self, real_id: int) -> list[str]:
        """Return human-readable slot names that have at least one observation."""
        if real_id not in self.counts:
            return []
        return [SLOT_NAMES[i] for i, n in enumerate(self.counts[real_id]) if n > 0]

    def summary(self) -> str:
        lines = [f"PoseGallery: {len(self.embeds)} cows"]
        for rid in sorted(self.embeds):
            slots = self.populated_slots(rid)
            lines.append(f"  real_id {rid:>6}: {len(slots)}/8 slots  [{', '.join(slots)}]")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Slot embedding builder
# ─────────────────────────────────────────────────────────────────────────────

def build_slot_embeds(
    embed_df,             # pd.DataFrame with columns [temp_id, embed (np.ndarray 128)]
    per_frame_labels,     # dict with "posture" (N,) int8 and "facing" (N,) int8 arrays
    frame_temp_ids,       # np.ndarray (N,) int — temp_id per embed row (aligned with labels)
    min_conf_posture,     # np.ndarray (N,) float32 — posture confidence
    min_conf_facing,      # np.ndarray (N,) float32 — facing confidence
    min_conf: float = 0.30,
) -> dict[int, dict[int, np.ndarray]]:
    """
    Group embed rows by slot, mean-pool each slot per temp_id.

    Parameters
    ----------
    embed_df           : DataFrame of embed rows for this session.
    per_frame_labels   : {"posture": (N,), "facing": (N,)} arrays aligned to embed_df rows.
    frame_temp_ids     : (N,) temp_id per frame — must align with per_frame_labels arrays.
    min_conf_posture   : (N,) confidence of posture label.
    min_conf_facing    : (N,) confidence of facing label.
    min_conf           : both posture_conf and facing_conf must exceed this to assign a slot.

    Returns
    -------
    dict { temp_id -> { slot_idx -> mean_embed (128,) } }
    For each temp_id, only slots with ≥1 qualifying frame are included.
    """
    import pandas as pd

    posture_arr = per_frame_labels["posture"]
    facing_arr  = per_frame_labels["facing"]

    # Filter to confident, labelled frames
    confident = (
        (min_conf_posture >= min_conf)
        & (min_conf_facing >= min_conf)
        & (posture_arr != int(Posture.UNCERTAIN))
        & (facing_arr  != int(Facing.UNCERTAIN))
    )

    result: dict[int, dict[int, np.ndarray]] = {}
    embeds_array = np.stack(embed_df["embed"].values)   # (N, 128)

    for i, is_good in enumerate(confident):
        if not is_good:
            continue
        p   = Posture(posture_arr[i])
        f   = Facing(facing_arr[i])
        s   = slot_index(p, f)
        if s is None:
            continue
        tid = int(frame_temp_ids[i])
        vec = embeds_array[i]

        result.setdefault(tid, {}).setdefault(s, []).append(vec)

    # Mean-pool and L2-normalise each slot
    pooled: dict[int, dict[int, np.ndarray]] = {}
    for tid, slots in result.items():
        pooled[tid] = {}
        for slot_idx, vecs in slots.items():
            stack = np.stack(vecs)
            mean  = stack.mean(axis=0)
            pooled[tid][slot_idx] = _l2_norm(mean)

    return pooled


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _l2_norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32) if norm > 1e-8 else v.astype(np.float32)
