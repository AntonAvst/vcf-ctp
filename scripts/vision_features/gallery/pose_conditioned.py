"""
vision_features/gallery/pose_conditioned.py
─────────────────────────────────────────────
Two pose-conditioned ReID galleries:

  PoseGallery      — keyed by real_id (int, positive)
                     Confirmed identities. Persists across sessions.
                     Files: gallery_pose_{day|night}.npy

  TempPoseGallery  — keyed by (camera_id, session_id, temp_id)
                     Unresolved identities. Cross-session within a camera.
                     Files: temp_gallery_pose_{day|night}.npy
                     Entries are deleted when the temp_id gets resolved.

Both use 8 slots = {standing, lying} × {left, right, toward, away}.

Synthetic IDs
─────────────
When two unresolved temp_id entries match above threshold but neither has
a real_id, a synthetic id is minted (negative integer, globally unique).
Written to reid_registry like any real_id, but with match_method='synthetic'.

Counter file: reid_gallery/synthetic_id_counter.npy  (single int, global)
This file is shared across all cameras / databases so ids are never reused.

When a synthetic id is later confirmed (kinetic, manual, or cosine against
a real_id gallery entry), backpropagate_resolution() replaces every
occurrence of the synthetic id with the confirmed real_id across the DB,
folds the gallery vectors, then deletes the synthetic entry.

Fallback query chain (same for both galleries):
    1. Exact slot         (posture + facing match)
    2. Same posture       (any facing, 4 slots)
    3. All populated slots (8 slots)
    4. None
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..schema import (
    N_SLOTS, GALLERY_SLOTS, SLOT_NAMES,
    Posture, Facing,
    slot_index, slot_name,
)


# ─────────────────────────────────────────────────────────────────────────────
# Temp gallery key
# ─────────────────────────────────────────────────────────────────────────────

class TempKey(NamedTuple):
    camera_id:  str
    session_id: str
    temp_id:    int


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic ID counter  (global .npy file)
# ─────────────────────────────────────────────────────────────────────────────

def mint_synthetic_id(gallery_dir: str) -> int:
    """
    Atomically decrement and return the next synthetic id (negative integer).
    Counter stored in reid_gallery/synthetic_id_counter.npy.
    Thread-safe within a single process; safe across processes because
    reconcile.py runs serially (one session at a time).

    Returns: e.g. -1, -2, -3, ...
    """
    path = Path(gallery_dir) / "synthetic_id_counter.npy"
    if path.exists():
        current = int(np.load(str(path)))
    else:
        current = 0   # first mint will return -1
    new_val = current - 1
    np.save(str(path), np.array(new_val, dtype=np.int64))
    return new_val


def peek_synthetic_counter(gallery_dir: str) -> int:
    """Return the current counter value without decrementing."""
    path = Path(gallery_dir) / "synthetic_id_counter.npy"
    if not path.exists():
        return 0
    return int(np.load(str(path)))


def is_synthetic(real_id: int) -> bool:
    return real_id < 0


# ─────────────────────────────────────────────────────────────────────────────
# Shared slot logic (used by both gallery classes)
# ─────────────────────────────────────────────────────────────────────────────

def _best_from_slots(
    embeds_dict: dict,          # { key -> np.ndarray (8, 128) }
    slots: list[int],
    query_embed: np.ndarray,
) -> tuple[object, float, int | None]:
    """Score all entries on the given slot indices, return (key, cosine, slot)."""
    best_key, best_cos, best_slot = None, -1.0, None
    for key, embeds_mat in embeds_dict.items():
        for s in slots:
            vec = embeds_mat[s]
            if np.all(np.isnan(vec)):
                continue
            cos = float(np.dot(query_embed, vec))
            if cos > best_cos:
                best_cos, best_key, best_slot = cos, key, s
    return best_key, best_cos, best_slot


def _slot_query(
    embeds_dict: dict,
    query_embed: np.ndarray,
    posture: Posture | None,
    facing:  Facing  | None,
    threshold: float,
) -> tuple[object, float, int | None]:
    """
    Shared fallback query logic for both PoseGallery and TempPoseGallery.
    Returns (key, cosine, slot_idx) or (None, best_cos, None).
    """
    if not embeds_dict:
        return None, 0.0, None

    exact_slot = None
    if (posture is not None and facing is not None
            and posture != Posture.UNCERTAIN and facing != Facing.UNCERTAIN):
        exact_slot = slot_index(posture, facing)

    posture_slots = None
    if posture is not None and posture != Posture.UNCERTAIN:
        posture_slots = [i for i, (p, _) in enumerate(GALLERY_SLOTS) if p == posture]

    best_cos = -1.0

    # Level 1: exact slot
    if exact_slot is not None:
        key, cos, s = _best_from_slots(embeds_dict, [exact_slot], query_embed)
        best_cos = max(best_cos, cos)
        if key is not None and cos >= threshold:
            return key, cos, s

    # Level 2: same posture, any facing
    if posture_slots is not None:
        key, cos, s = _best_from_slots(embeds_dict, posture_slots, query_embed)
        best_cos = max(best_cos, cos)
        if key is not None and cos >= threshold:
            return key, cos, s

    # Level 3: all populated slots
    key, cos, s = _best_from_slots(embeds_dict, list(range(N_SLOTS)), query_embed)
    best_cos = max(best_cos, cos)
    if key is not None and cos >= threshold:
        return key, cos, s

    return None, max(0.0, best_cos), None


# ─────────────────────────────────────────────────────────────────────────────
# PoseGallery  (real_id keyed)
# ─────────────────────────────────────────────────────────────────────────────

class PoseGallery:
    """
    8-slot pose-conditioned gallery keyed by real_id (positive int).
    Includes synthetic ids (negative int) until they are confirmed.
    """

    def __init__(self) -> None:
        self.embeds: dict[int, np.ndarray] = {}   # (8, 128) per cow
        self.counts: dict[int, np.ndarray] = {}   # (8,)     per cow

    @classmethod
    def load(cls, gallery_dir: str, modality: str) -> "PoseGallery":
        path = Path(gallery_dir) / f"gallery_pose_{modality}.npy"
        g = cls()
        if not path.exists():
            return g
        data = np.load(str(path), allow_pickle=True).item()
        g.embeds = {int(k): v["embeds"].astype(np.float32) for k, v in data.items()}
        g.counts = {int(k): v["counts"].astype(np.int32)   for k, v in data.items()}
        return g

    def save(self, gallery_dir: str, modality: str) -> None:
        Path(gallery_dir).mkdir(parents=True, exist_ok=True)
        path = Path(gallery_dir) / f"gallery_pose_{modality}.npy"
        data = {k: {"embeds": self.embeds[k], "counts": self.counts[k]}
                for k in self.embeds}
        np.save(str(path), data)

    def update(
        self,
        real_id: int,
        slot_embeds: dict[int, np.ndarray],
        alpha: float = 0.15,
    ) -> dict[int, float]:
        if real_id not in self.embeds:
            self.embeds[real_id] = np.full((N_SLOTS, 128), np.nan, dtype=np.float32)
            self.counts[real_id] = np.zeros(N_SLOTS, dtype=np.int32)

        cosines = {}
        for slot_idx, new_vec in slot_embeds.items():
            old_vec = self.embeds[real_id][slot_idx]
            if np.all(np.isnan(old_vec)):
                updated = _l2_norm(new_vec)
                cosines[slot_idx] = float("nan")
            else:
                updated = _l2_norm(alpha * new_vec + (1 - alpha) * old_vec)
                cosines[slot_idx] = float(np.dot(old_vec, new_vec))
            self.embeds[real_id][slot_idx] = updated
            self.counts[real_id][slot_idx] += 1

        return cosines

    def merge_and_delete(self, src_id: int, dst_id: int, alpha: float = 0.5) -> None:
        """
        Fold src_id gallery slots into dst_id (used when a synthetic id is
        confirmed as dst_id), then remove src_id.
        alpha controls how much weight the src history gets.
        """
        if src_id not in self.embeds:
            return
        if dst_id not in self.embeds:
            self.embeds[dst_id] = np.full((N_SLOTS, 128), np.nan, dtype=np.float32)
            self.counts[dst_id] = np.zeros(N_SLOTS, dtype=np.int32)

        for s in range(N_SLOTS):
            src_vec = self.embeds[src_id][s]
            dst_vec = self.embeds[dst_id][s]
            if np.all(np.isnan(src_vec)):
                continue
            if np.all(np.isnan(dst_vec)):
                self.embeds[dst_id][s] = src_vec.copy()
            else:
                merged = _l2_norm(alpha * src_vec + (1 - alpha) * dst_vec)
                self.embeds[dst_id][s] = merged
            self.counts[dst_id][s] += self.counts[src_id][s]

        del self.embeds[src_id]
        del self.counts[src_id]

    def query(
        self,
        query_embed: np.ndarray,
        posture: Posture | None,
        facing:  Facing  | None,
        threshold: float = 0.75,
    ) -> tuple[int | None, float, int | None]:
        key, cos, s = _slot_query(self.embeds, query_embed, posture, facing, threshold)
        return key, cos, s

    def populated_slots(self, real_id: int) -> list[str]:
        if real_id not in self.counts:
            return []
        return [SLOT_NAMES[i] for i, n in enumerate(self.counts[real_id]) if n > 0]

    def summary(self) -> str:
        lines = [f"PoseGallery: {len(self.embeds)} entries"]
        for rid in sorted(self.embeds):
            tag = " [SYNTHETIC]" if is_synthetic(rid) else ""
            slots = self.populated_slots(rid)
            lines.append(f"  id {rid:>8}{tag}: {len(slots)}/8 slots  [{', '.join(slots)}]")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TempPoseGallery  (TempKey keyed, camera-scoped)
# ─────────────────────────────────────────────────────────────────────────────

class TempPoseGallery:
    """
    8-slot pose-conditioned gallery for unresolved temp_ids.

    Keyed by TempKey(camera_id, session_id, temp_id).
    Scoped by camera_id so cows are never matched across pens.
    Persists cross-session; entries deleted when temp_id is resolved.

    Files: temp_gallery_pose_{day|night}.npy
    """

    def __init__(self) -> None:
        # embeds: { TempKey -> np.ndarray (8, 128) }
        self.embeds: dict[TempKey, np.ndarray] = {}
        self.counts: dict[TempKey, np.ndarray] = {}

    @classmethod
    def load(cls, gallery_dir: str, modality: str) -> "TempPoseGallery":
        path = Path(gallery_dir) / f"temp_gallery_pose_{modality}.npy"
        g = cls()
        if not path.exists():
            return g
        data = np.load(str(path), allow_pickle=True).item()
        for k, v in data.items():
            key = TempKey(*k)
            g.embeds[key] = v["embeds"].astype(np.float32)
            g.counts[key] = v["counts"].astype(np.int32)
        return g

    def save(self, gallery_dir: str, modality: str) -> None:
        Path(gallery_dir).mkdir(parents=True, exist_ok=True)
        path = Path(gallery_dir) / f"temp_gallery_pose_{modality}.npy"
        # Convert TempKey (NamedTuple) to plain tuple for numpy serialisation
        data = {tuple(k): {"embeds": self.embeds[k], "counts": self.counts[k]}
                for k in self.embeds}
        np.save(str(path), data)

    def update(
        self,
        key: TempKey,
        slot_embeds: dict[int, np.ndarray],
        alpha: float = 0.15,
    ) -> dict[int, float]:
        if key not in self.embeds:
            self.embeds[key] = np.full((N_SLOTS, 128), np.nan, dtype=np.float32)
            self.counts[key] = np.zeros(N_SLOTS, dtype=np.int32)

        cosines = {}
        for slot_idx, new_vec in slot_embeds.items():
            old_vec = self.embeds[key][slot_idx]
            if np.all(np.isnan(old_vec)):
                updated = _l2_norm(new_vec)
                cosines[slot_idx] = float("nan")
            else:
                updated = _l2_norm(alpha * new_vec + (1 - alpha) * old_vec)
                cosines[slot_idx] = float(np.dot(old_vec, new_vec))
            self.embeds[key][slot_idx] = updated
            self.counts[key][slot_idx] += 1

        return cosines

    def delete_resolved(
        self,
        session_id: str,
        resolved_temp_ids: set[int],
    ) -> list[TempKey]:
        """
        Remove all TempKey entries where session_id matches and temp_id is
        in resolved_temp_ids. Returns list of deleted keys for logging.
        """
        to_delete = [
            k for k in self.embeds
            if k.session_id == session_id and k.temp_id in resolved_temp_ids
        ]
        for k in to_delete:
            del self.embeds[k]
            del self.counts[k]
        return to_delete

    def query_for_camera(
        self,
        camera_id: str,
        query_embed: np.ndarray,
        posture: Posture | None,
        facing:  Facing  | None,
        threshold: float = 0.75,
        exclude_session: str | None = None,   # don't match against current session
    ) -> tuple[TempKey | None, float, int | None]:
        """
        Query only entries belonging to camera_id.
        Optionally exclude the current session (to avoid self-match).
        Returns (TempKey or None, cosine, slot_idx).
        """
        scoped = {
            k: v for k, v in self.embeds.items()
            if k.camera_id == camera_id
            and (exclude_session is None or k.session_id != exclude_session)
        }
        key, cos, s = _slot_query(scoped, query_embed, posture, facing, threshold)
        return key, cos, s

    def get_mean_embed(self, key: TempKey) -> np.ndarray | None:
        """Return mean of all populated slots for a key (for flat cosine fallback)."""
        if key not in self.embeds:
            return None
        vecs = [self.embeds[key][s] for s in range(N_SLOTS)
                if not np.all(np.isnan(self.embeds[key][s]))]
        if not vecs:
            return None
        mean = np.stack(vecs).mean(axis=0)
        return _l2_norm(mean)

    def entries_for_camera(self, camera_id: str) -> list[TempKey]:
        return [k for k in self.embeds if k.camera_id == camera_id]

    def summary(self, camera_id: str | None = None) -> str:
        keys = self.entries_for_camera(camera_id) if camera_id else list(self.embeds)
        lines = [f"TempPoseGallery: {len(keys)} entries"
                 + (f" (camera={camera_id})" if camera_id else "")]
        for k in sorted(keys):
            populated = [SLOT_NAMES[i] for i, n in enumerate(self.counts[k]) if n > 0]
            lines.append(f"  {k.camera_id}/{k.session_id}/t{k.temp_id}: "
                         f"{len(populated)}/8 slots  [{', '.join(populated)}]")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Backpropagation  (synthetic id → confirmed real_id)
# ─────────────────────────────────────────────────────────────────────────────

def backpropagate_resolution(
    synthetic_id: int,
    real_id: int,
    conn: sqlite3.Connection,
    pose_gallery: PoseGallery,
    gallery_dir: str,
    modality: str,
    dry_run: bool = False,
) -> int:
    """
    Replace every occurrence of synthetic_id with real_id across the DB,
    fold the gallery vectors, and delete the synthetic entry.

    Tables updated:
        resolved_cow_timeline   (real_id column)
        reid_registry           (real_id PK — delete synthetic, upsert real)
        temp_id_merges          (real_id column)
        manual_assignments      (real_id column)

    Returns number of timeline rows updated.
    """
    assert is_synthetic(synthetic_id), "synthetic_id must be negative"
    assert real_id > 0, "real_id must be positive"

    if dry_run:
        rows = conn.execute(
            "SELECT COUNT(*) FROM resolved_cow_timeline WHERE real_id = ?",
            (synthetic_id,)
        ).fetchone()[0]
        print(f"[backprop dry_run] Would update {rows} timeline rows "
              f"{synthetic_id} → {real_id}")
        return rows

    # ── Update all tables ────────────────────────────────────────────────────
    conn.execute(
        "UPDATE resolved_cow_timeline SET real_id = ? WHERE real_id = ?",
        (real_id, synthetic_id)
    )
    rows_updated = conn.execute(
        "SELECT changes()"
    ).fetchone()[0]

    conn.execute(
        "UPDATE temp_id_merges SET real_id = ? WHERE real_id = ?",
        (real_id, synthetic_id)
    )
    conn.execute(
        "UPDATE manual_assignments SET real_id = ? WHERE real_id = ?",
        (real_id, synthetic_id)
    )

    # ── Delete synthetic reid_registry row ───────────────────────────────────
    conn.execute(
        "DELETE FROM reid_registry WHERE real_id = ?",
        (synthetic_id,)
    )
    conn.commit()

    # ── Fold gallery vectors ─────────────────────────────────────────────────
    pose_gallery.merge_and_delete(synthetic_id, real_id, alpha=0.5)
    pose_gallery.save(gallery_dir, modality)

    return rows_updated


# ─────────────────────────────────────────────────────────────────────────────
# Slot embedding builder  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_slot_embeds(
    embed_df,
    per_frame_labels,
    frame_temp_ids,
    min_conf_posture,
    min_conf_facing,
    min_conf: float = 0.30,
) -> dict[int, dict[int, np.ndarray]]:
    """
    Group embed rows by slot, mean-pool each slot per temp_id.

    Returns dict { temp_id -> { slot_idx -> mean_embed (128,) } }
    """
    posture_arr = per_frame_labels["posture"]
    facing_arr  = per_frame_labels["facing"]

    confident = (
        (min_conf_posture >= min_conf)
        & (min_conf_facing >= min_conf)
        & (posture_arr != int(Posture.UNCERTAIN))
        & (facing_arr  != int(Facing.UNCERTAIN))
    )

    result: dict[int, dict[int, list]] = {}
    embeds_array = np.stack(embed_df["embed"].values)

    for i, is_good in enumerate(confident):
        if not is_good:
            continue
        p  = Posture(posture_arr[i])
        f  = Facing(facing_arr[i])
        s  = slot_index(p, f)
        if s is None:
            continue
        tid = int(frame_temp_ids[i])
        result.setdefault(tid, {}).setdefault(s, []).append(embeds_array[i])

    pooled: dict[int, dict[int, np.ndarray]] = {}
    for tid, slots in result.items():
        pooled[tid] = {}
        for slot_idx, vecs in slots.items():
            mean = np.stack(vecs).mean(axis=0)
            pooled[tid][slot_idx] = _l2_norm(mean)

    return pooled


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _l2_norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32) if norm > 1e-8 else v.astype(np.float32)