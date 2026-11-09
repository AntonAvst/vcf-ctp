"""
gallery_ledger.py — per-session contribution ledger for the flat and
pose-conditioned galleries (ToDo item 5), plus the ID-switch review queue
fed by appearance_quality.mean_embed_report()'s split check (item 10).

Why a ledger instead of a running EMA
──────────────────────────────────────
The galleries used to be updated by an irreversible EMA:
    gallery[aid] = alpha * session_mean + (1 - alpha) * gallery[aid]
Re-running a session applied its update a second time, and a wrong match
could never be taken back out.

Now every session's contribution to one (real_id, modality, gallery, slot)
key is its own row: the session's robust, quality-weighted mean embedding,
the source that resolved the id, its effective frame count and a timestamp.
The gallery vector is RECOMPUTED from the rows on every change:

    weight(row) = source_weight * 0.5 ** (age_days / half_life_days)
                  * n_effective_frames ** frames_weight_power
    vector      = L2( sum(weight * row.vector) / sum(weight) )

Exponential decay keeps the RATIO between two rows' weights fixed over time,
so a cow's stored vector stays correct between commits; decay only decides
how much old rows count against NEW ones ("old contributions fade").

Re-runs (SessionCommit)
───────────────────────
A run opens one SessionCommit per gallery ('flat' in reconcile.py,
'pose' in vision_features/extractor.py). Opening it removes EVERY row this
session wrote to that gallery on earlier runs — not only the keys it will
write again — so a contribution the new run no longer makes (the track was
re-assigned to another cow, fell under min_embeds, lost a slot, changed
camera mode) disappears too. finish() returns the rebuilt vector for each
such orphaned key so the caller can update the .npy gallery. Under --dry_run
nothing is deleted; every rebuild just ignores this session's stored rows,
so a dry run previews exactly what a real run would produce.

A human decision on one of this session's rows (approve / reject, i.e. a
row with reviewed_dt set) is carried over to the re-run's row for the same
key: a rejected contribution stays rejected and an approved one skips the
drift gate. Deleting a row ('remove') is NOT sticky — the next re-run of
that session writes it again. To take a wrong match out for good, use
'reject' (or fix the assignment and re-run).

Drift gate
──────────
Before a contribution is folded in, its cosine with the cow's current
rebuilt vector (excluding this session) is computed and stored as `drift`.
Below gallery.contribution.drift_gate_cosine the row is stored as
'held_for_review' and does not count until approved. Note the gate is below
cosine_resolver.threshold, so a cosine-sourced contribution (which already
matched this gallery) practically never trips it: the gate guards against
kinetic / manual / boundary ids that disagree with what the cow looks like.

Existing galleries (legacy seed)
────────────────────────────────
Galleries built before the ledger have vectors but no rows. The first time
a key with a stored vector and no rows (of any status) is committed to, the
stored vector is inserted as one 'legacy' row (session_id '__legacy__',
n_effective_frames = gallery.contribution.legacy_seed_frames) before the new
contribution is gated and added — so deploying the ledger does not replace
every cow's history with a single session. The seed is an ordinary row: it
fades with age and can be removed or rejected like any other.

Scope
─────
Real (positive) ids in gallery_{day|night}.npy and gallery_pose_{day|night}
.npy. Synthetic ids and TempPoseGallery keep their plain EMA updates — they
are scratch space that is folded or deleted once an id is resolved.

This module never loads or saves .npy files during a reconcile run; callers
apply the vectors it returns to the gallery objects they already hold.

CLI (review / maintenance; pulls DB + galleries from Drive, writes back):

    python3 gallery_ledger.py list-pending [--real_id ID]
    python3 gallery_ledger.py approve ID | --session S --real_id R
    python3 gallery_ledger.py reject  ID | --session S --real_id R
    python3 gallery_ledger.py remove  ID | --session S --real_id R
    python3 gallery_ledger.py show REAL_ID --modality day --gallery flat [--slot N]
    python3 gallery_ledger.py list-id-switch-flags [--all]
    python3 gallery_ledger.py review-id-switch-flag ID
  common flags: --bypass_upload_check, --params

The --session/--real_id form acts on every row (flat and pose, both
modalities) that one session contributed to one cow — the usual unit of a
wrong match.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config

FLAT_SLOT = -1            # slot_idx used for the flat (non pose-conditioned) gallery
LEGACY_SESSION = "__legacy__"
LEGACY_SOURCE = "legacy"

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gallery_contributions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT    NOT NULL,
    real_id             INTEGER NOT NULL,
    modality            TEXT    NOT NULL,   -- 'day' | 'night'
    gallery             TEXT    NOT NULL,   -- 'flat' | 'pose'
    slot_idx            INTEGER NOT NULL,   -- -1 (FLAT_SLOT) for 'flat'; 0-7 for 'pose'
    source              TEXT    NOT NULL,   -- 'manual' | 'kinetic' | 'boundary_rendezvous' | 'cosine'
                                             -- | 'legacy' (pre-ledger vector) | 'synthetic' (a
                                             -- resolved synthetic id's pose history)
    source_weight       REAL    NOT NULL,   -- gallery.source_weights[source] snapshot at commit time
    n_effective_frames  REAL    NOT NULL,
    vector              TEXT    NOT NULL,   -- JSON list[128], L2-normalised
    drift               REAL,               -- cosine(vector of the other active rows, this vector);
                                             -- NULL = nothing to compare against (gate bypassed)
    status              TEXT    NOT NULL DEFAULT 'active',  -- 'active' | 'held_for_review' | 'rejected'
    created_dt          TEXT    NOT NULL,
    reviewed_dt          TEXT,
    UNIQUE(session_id, real_id, modality, gallery, slot_idx)
);
CREATE INDEX IF NOT EXISTS idx_gallery_contrib_key
    ON gallery_contributions(real_id, modality, gallery, slot_idx, status);

CREATE TABLE IF NOT EXISTS id_switch_flags (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT    NOT NULL,
    temp_id             INTEGER NOT NULL,
    real_id             INTEGER,            -- NULL = unresolved track; negative = synthetic id
    modality            TEXT    NOT NULL,
    n_effective_frames  REAL    NOT NULL,
    trim_frac           REAL    NOT NULL,
    split_gap           REAL    NOT NULL,
    split_cluster_frac  REAL    NOT NULL,
    interleave_frac     REAL,               -- frame weight on the wrong side of the switch point
    switch_frame        INTEGER,            -- frame_index where the second animal takes over
    created_dt          TEXT    NOT NULL,
    reviewed             INTEGER NOT NULL DEFAULT 0,
    UNIQUE(session_id, temp_id, modality)
);
"""


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Safe to call every run (CREATE TABLE/INDEX IF NOT EXISTS)."""
    conn.executescript(SCHEMA_SQL)
    for sql in ("ALTER TABLE id_switch_flags ADD COLUMN interleave_frac REAL",
                "ALTER TABLE id_switch_flags ADD COLUMN switch_frame INTEGER"):
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass   # column already present
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Weighting
# ─────────────────────────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32) if norm > 1e-8 else v.astype(np.float32)


def _decay(age_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return float(0.5 ** (max(age_days, 0.0) / half_life_days))


def _weight(source_weight: float, n_effective_frames: float, age_days: float, cp) -> float:
    return (source_weight
            * _decay(age_days, cp.half_life_days)
            * (max(n_effective_frames, 1e-6) ** cp.frames_weight_power))


def _age_days(created_dt: str, as_of: pd.Timestamp) -> float:
    return float((as_of - pd.Timestamp(created_dt)).total_seconds() / 86400.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_vector(
    conn: sqlite3.Connection,
    real_id: int,
    modality: str,
    gallery: str,
    slot_idx: int = FLAT_SLOT,
    as_of: pd.Timestamp | None = None,
    exclude_session: str | None = None,
    extras: list | None = None,
) -> tuple[np.ndarray | None, int, float]:
    """
    Weighted mean over every ACTIVE row for this key, decayed to `as_of`
    (default now). Returns (vector or None, rows used, total weight); rows
    whose weight is below gallery.contribution.min_weight don't count.

    exclude_session: ignore this session's stored rows (dry runs, and the
    prior vector a re-run is gated against).
    extras: not-yet-stored rows to include, each a tuple
    (vector, source_weight, n_effective_frames, age_days) — dry-run previews.
    """
    cp = config.params().gallery.contribution
    as_of = as_of or pd.Timestamp.now()
    q = ("SELECT source_weight, n_effective_frames, vector, created_dt "
         "FROM gallery_contributions WHERE real_id = ? AND modality = ? "
         "AND gallery = ? AND slot_idx = ? AND status = 'active'")
    params: list = [real_id, modality, gallery, slot_idx]
    if exclude_session is not None:
        q += " AND session_id != ?"
        params.append(exclude_session)

    items = [(np.asarray(json.loads(r["vector"]), dtype=np.float32),
              r["source_weight"], r["n_effective_frames"],
              _age_days(r["created_dt"], as_of))
             for r in conn.execute(q, params).fetchall()]
    items += list(extras or [])

    acc, total_w, n_used = None, 0.0, 0
    for vec, sw, n_eff, age in items:
        w = _weight(sw, n_eff, age, cp)
        if w < cp.min_weight:
            continue
        acc = vec * w if acc is None else acc + vec * w
        total_w += w
        n_used += 1
    if acc is None or total_w <= 0:
        return None, 0, 0.0
    return _l2(acc / total_w), n_used, total_w


def _has_any_rows(conn, real_id, modality, gallery, slot_idx, exclude_session) -> bool:
    """Any row for this key, of ANY status, not written by exclude_session.
    A rejected-only key counts: it must never be re-seeded from the gallery."""
    return conn.execute(
        "SELECT 1 FROM gallery_contributions WHERE real_id = ? AND modality = ? "
        "AND gallery = ? AND slot_idx = ? AND session_id != ? LIMIT 1",
        (real_id, modality, gallery, slot_idx, exclude_session),
    ).fetchone() is not None


# ─────────────────────────────────────────────────────────────────────────────
# One run's contributions to one gallery
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContributionResult:
    status: str                       # 'active' | 'held_for_review' | 'rejected'
    drift: float | None               # None when nothing was there to compare against
    vector_after: np.ndarray | None   # vector the caller should now hold for this key
    n_contributions: int              # active rows behind vector_after
    seeded_legacy: bool = False       # the stored pre-ledger vector was seeded as a row
    carried_decision: str | None = None   # human decision carried over from an earlier run
    note: str = ""


class SessionCommit:
    """All of one run's contributions from one session to one gallery
    ('flat' or 'pose'). Construct → record() per key → finish().
    See the module docstring, "Re-runs"."""

    def __init__(self, conn: sqlite3.Connection, session_id: str, gallery: str,
                 dry_run: bool = False, now: pd.Timestamp | None = None):
        assert gallery in ("flat", "pose")
        if session_id == LEGACY_SESSION:
            raise ValueError(f"session_id {LEGACY_SESSION!r} is reserved")
        self.conn, self.session_id, self.gallery = conn, session_id, gallery
        self.dry_run = dry_run
        self.now = now or pd.Timestamp.now()
        self.recorded: set = set()
        self.finished = False
        # per key, this run's not-excluded contributions as rebuild extras:
        # its own active row, plus (dry run) a legacy seed it didn't store
        self._run_extras: dict = {}

        rows = conn.execute(
            "SELECT real_id, modality, slot_idx, status, reviewed_dt "
            "FROM gallery_contributions WHERE session_id = ? AND gallery = ?",
            (session_id, gallery),
        ).fetchall()
        self.old_keys = {(r["real_id"], r["modality"], r["slot_idx"]) for r in rows}
        self.decisions = {(r["real_id"], r["modality"], r["slot_idx"]): (r["status"], r["reviewed_dt"])
                          for r in rows if r["reviewed_dt"] is not None}
        if rows and not dry_run:
            conn.execute("DELETE FROM gallery_contributions WHERE session_id = ? AND gallery = ?",
                         (session_id, gallery))
            conn.commit()

    def record(
        self,
        real_id: int,
        modality: str,
        source: str,
        source_weight: float,
        n_effective_frames: float,
        vector: np.ndarray,
        slot_idx: int = FLAT_SLOT,
        current_vector: np.ndarray | None = None,   # what the gallery holds now (legacy seed)
        current_dt: str | None = None,              # when that vector was last updated, if known
    ) -> ContributionResult:
        cp = config.params().gallery.contribution
        conn, sid, g = self.conn, self.session_id, self.gallery
        key = (int(real_id), modality, int(slot_idx))
        vector = _l2(np.asarray(vector, dtype=np.float32))
        note = ""
        if key in self.recorded:
            # Two temp_ids of one session gave the same cow the same slot —
            # the Duplicate Resolver normally prevents this. Last one wins.
            note = "key recorded twice in this run; the earlier row was replaced"
            if not self.dry_run:
                conn.execute(
                    "DELETE FROM gallery_contributions WHERE session_id = ? AND real_id = ? "
                    "AND modality = ? AND gallery = ? AND slot_idx = ?", (sid, *key[:2], g, key[2]))

        # ── legacy seed: a pre-ledger vector with no rows behind it ───────────
        extras, seeded = [], False
        # (this session's own earlier rows count as rows: the vector the gallery
        # holds may have been built from them, so it is not a pre-ledger vector)
        if (current_vector is not None and key not in self.old_keys
                and not _has_any_rows(conn, *key[:2], g, key[2], sid)):
            seed_vec = _l2(np.asarray(current_vector, dtype=np.float32))
            seed_dt = pd.Timestamp(current_dt) if current_dt else self.now
            seeded = True
            if self.dry_run:
                extras.append((seed_vec, 1.0, cp.legacy_seed_frames,
                               _age_days(seed_dt.isoformat(), self.now)))
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO gallery_contributions (session_id, real_id, modality, "
                    "gallery, slot_idx, source, source_weight, n_effective_frames, vector, drift, "
                    "status, created_dt) VALUES (?,?,?,?,?,?,?,?,?,NULL,'active',?)",
                    (LEGACY_SESSION, key[0], key[1], g, key[2], LEGACY_SOURCE, 1.0,
                     float(cp.legacy_seed_frames), json.dumps(seed_vec.tolist()),
                     seed_dt.isoformat()))

        # ── gate (or a human decision carried over from an earlier run) ───────
        prior, prior_n, _ = rebuild_vector(conn, *key[:2], g, key[2], as_of=self.now,
                                           exclude_session=sid, extras=extras)
        drift = float(np.dot(prior, vector)) if prior is not None else None
        carried = self.decisions.get(key)
        if carried is not None:
            status, reviewed_dt = carried
        else:
            status = ("held_for_review" if drift is not None and drift < cp.drift_gate_cosine
                      else "active")
            reviewed_dt = None

        if not self.dry_run:
            conn.execute(
                "INSERT INTO gallery_contributions (session_id, real_id, modality, gallery, "
                "slot_idx, source, source_weight, n_effective_frames, vector, drift, status, "
                "created_dt, reviewed_dt) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, key[0], key[1], g, key[2], source, float(source_weight),
                 float(n_effective_frames), json.dumps(vector.tolist()), drift, status,
                 self.now.isoformat(), reviewed_dt))
            conn.commit()
        self.recorded.add(key)

        run_extras = list(extras)
        if status != "active":
            vec_after, n_after = prior, prior_n
        else:
            run_extras.append((vector, float(source_weight), float(n_effective_frames), 0.0))
            vec_after, n_after, _ = rebuild_vector(
                conn, *key[:2], g, key[2], as_of=self.now, exclude_session=sid,
                extras=run_extras)
        self._run_extras[key] = run_extras
        return ContributionResult(status=status, drift=drift, vector_after=vec_after,
                                  n_contributions=n_after, seeded_legacy=seeded,
                                  carried_decision=carried[0] if carried else None, note=note)

    def reference(self, real_id: int, modality: str, slot_idx: int = FLAT_SLOT,
                  fallback: np.ndarray | None = None) -> np.ndarray | None:
        """The cow's vector as it stands for THIS run: every other session's
        active rows plus whatever this run has recorded so far, never this
        session's rows from earlier runs (a re-run must not be judged against
        a vector built from itself). `fallback` — the vector the gallery file
        holds — is used only for a pre-ledger key with no rows at all."""
        key = (int(real_id), modality, int(slot_idx))
        vec, _, _ = rebuild_vector(self.conn, *key[:2], self.gallery, key[2], as_of=self.now,
                                   exclude_session=self.session_id,
                                   extras=self._run_extras.get(key))
        if vec is None and fallback is not None and key not in self.old_keys \
                and key not in self._run_extras \
                and not _has_any_rows(self.conn, *key[:2], self.gallery, key[2], self.session_id):
            return _l2(np.asarray(fallback, dtype=np.float32))
        return vec

    def known_ids(self, modality: str) -> set:
        """Real ids with any row for this gallery + modality (other sessions)
        or recorded in this run."""
        ids = {r[0] for r in self.conn.execute(
            "SELECT DISTINCT real_id FROM gallery_contributions WHERE gallery = ? "
            "AND modality = ? AND session_id != ?", (self.gallery, modality, self.session_id))}
        return ids | {k[0] for k in self._run_extras if k[1] == modality}

    def finish(self) -> dict:
        """{(real_id, modality, slot_idx) -> (vector or None, n_contributions)}
        for every key this session contributed to on an earlier run but not
        on this one. The caller must write these into its gallery objects
        (None = the key has no active rows left: clear it)."""
        self.finished = True
        out = {}
        for key in sorted(self.old_keys - self.recorded):
            vec, n, _ = rebuild_vector(self.conn, *key[:2], self.gallery, key[2],
                                       as_of=self.now, exclude_session=self.session_id)
            out[key] = (vec, n)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Review / maintenance — exact, one row (or one session × cow) at a time
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LedgerEditResult:
    rows: list                                   # the rows acted on (as they were before)
    rebuilt: dict = field(default_factory=dict)  # {(real_id, modality, gallery, slot) -> (vec|None, n)}


def _select_rows(conn, contribution_id: int | None, session_id: str | None,
                 real_id: int | None) -> list[dict]:
    if contribution_id is not None:
        rows = conn.execute("SELECT * FROM gallery_contributions WHERE id = ?",
                            (contribution_id,)).fetchall()
        if not rows:
            raise ValueError(f"no gallery_contributions row with id={contribution_id}")
    else:
        if session_id is None or real_id is None:
            raise ValueError("give a row id, or both session_id and real_id")
        rows = conn.execute("SELECT * FROM gallery_contributions WHERE session_id = ? "
                            "AND real_id = ?", (session_id, real_id)).fetchall()
        if not rows:
            raise ValueError(f"no gallery_contributions rows for session {session_id!r}, "
                             f"real_id {real_id}")
    return [dict(r) for r in rows]


def _rebuild_keys(conn, rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        k = (r["real_id"], r["modality"], r["gallery"], r["slot_idx"])
        if k not in out:
            vec, n, _ = rebuild_vector(conn, r["real_id"], r["modality"], r["gallery"], r["slot_idx"])
            out[k] = (vec, n)
    return out


def edit_contributions(conn: sqlite3.Connection, action: str,
                       contribution_id: int | None = None,
                       session_id: str | None = None,
                       real_id: int | None = None) -> LedgerEditResult:
    """action: 'approve' | 'reject' | 'remove'. Commits, then returns the
    rebuilt vector of every affected key for the caller to apply."""
    assert action in ("approve", "reject", "remove")
    rows = _select_rows(conn, contribution_id, session_id, real_id)
    ids = [r["id"] for r in rows]
    ph = ",".join("?" * len(ids))
    if action == "remove":
        conn.execute(f"DELETE FROM gallery_contributions WHERE id IN ({ph})", ids)
    else:
        status = "active" if action == "approve" else "rejected"
        conn.execute(f"UPDATE gallery_contributions SET status = ?, reviewed_dt = ? "
                     f"WHERE id IN ({ph})", [status, pd.Timestamp.now().isoformat(), *ids])
    conn.commit()
    return LedgerEditResult(rows=rows, rebuilt=_rebuild_keys(conn, rows))


def list_pending(conn: sqlite3.Connection, real_id: int | None = None) -> list[dict]:
    q = "SELECT * FROM gallery_contributions WHERE status = 'held_for_review'"
    params: list = []
    if real_id is not None:
        q += " AND real_id = ?"
        params.append(real_id)
    return [dict(r) for r in conn.execute(q + " ORDER BY created_dt", params).fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# ID-switch flags (item 10) — a review queue only; nothing here changes an
# assignment or a gallery. See appearance_quality.mean_embed_report().
# ─────────────────────────────────────────────────────────────────────────────

def clear_id_switch_flags(conn: sqlite3.Connection, session_id: str, dry_run: bool = False) -> None:
    """A re-run replaces the session's flags (temp_ids can be renumbered by
    stitching, or no longer split)."""
    if dry_run:
        return
    conn.execute("DELETE FROM id_switch_flags WHERE session_id = ?", (session_id,))
    conn.commit()


def flag_id_switch(conn: sqlite3.Connection, session_id: str, temp_id: int,
                   real_id: int | None, modality: str, report, dry_run: bool = False) -> None:
    if dry_run:
        return
    conn.execute(
        "INSERT OR REPLACE INTO id_switch_flags (session_id, temp_id, real_id, modality, "
        "n_effective_frames, trim_frac, split_gap, split_cluster_frac, interleave_frac, "
        "switch_frame, created_dt) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (session_id, int(temp_id), None if real_id is None else int(real_id), modality,
         report.n_total, report.trim_frac, report.split_gap, report.split_cluster_frac,
         report.split_interleave_frac, report.switch_frame, pd.Timestamp.now().isoformat()))
    conn.commit()


def list_id_switch_flags(conn: sqlite3.Connection, unreviewed_only: bool = True) -> list[dict]:
    q = "SELECT * FROM id_switch_flags" + (" WHERE reviewed = 0" if unreviewed_only else "")
    return [dict(r) for r in conn.execute(q + " ORDER BY created_dt").fetchall()]


def mark_id_switch_reviewed(conn: sqlite3.Connection, flag_id: int) -> None:
    conn.execute("UPDATE id_switch_flags SET reviewed = 1 WHERE id = ?", (flag_id,))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# CLI. Same Drive discipline as reconcile.py: check flags, pull DB + galleries,
# write, then push the DB and the gallery files back. reconcile and the pose
# gallery class are imported lazily: reconcile.py imports this module.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_rebuilt(rebuilt: dict, conn: sqlite3.Connection, gallery_dir: str) -> set:
    """Write rebuilt vectors into the .npy galleries (+ the reid_registry
    mirror for flat). Returns the gallery file names it changed."""
    from reconcile import load_gallery, save_gallery, upsert_reid
    from vision_features.gallery import PoseGallery

    changed: set = set()
    flat = {(rid, m): v for (rid, m, g, s), v in rebuilt.items() if g == "flat"}
    pose = {(rid, m, s): v for (rid, m, g, s), v in rebuilt.items() if g == "pose"}
    for m in {m for (_, m) in flat}:
        gal = load_gallery(gallery_dir, m)
        for (rid, mm), (vec, n) in flat.items():
            if mm != m:
                continue
            if vec is None:
                gal.pop(rid, None)
            else:
                gal[rid] = vec
            upsert_reid(conn, rid, {
                f"gallery_embed_{m}": json.dumps(vec.tolist()) if vec is not None else None,
                f"gallery_n_{m}": n,
                f"last_updated_{m}_dt": pd.Timestamp.now().isoformat(),
            })
        save_gallery(gallery_dir, m, gal)
        changed.add(f"gallery_{m}.npy")
    for m in {m for (_, m, _) in pose}:
        pg = PoseGallery.load(gallery_dir, m)
        for (rid, mm, s), (vec, n) in pose.items():
            if mm == m:
                pg.set_slot_vector(rid, s, vec, n)
        pg.save(gallery_dir, m)
        changed.add(f"gallery_pose_{m}.npy")
    return changed


def _print_row(row: dict) -> None:
    drift = "—" if row["drift"] is None else f"{row['drift']:.3f}"
    slot = "" if row["gallery"] == "flat" else f"/slot{row['slot_idx']}"
    print(f"  id={row['id']:<6} {row['session_id']:<42} real_id={row['real_id']:<7} "
          f"{row['modality']:<5} {row['gallery']}{slot:<7} {row['source']:<19} "
          f"n_eff={row['n_effective_frames']:<7.1f} drift={drift:<6} {row['status']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_params_arg(ap)
    ap.add_argument("--bypass_upload_check", action="store_true",
                    help="Skip the dirty-flag check (proceeds with possibly stale data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list-pending"); p.add_argument("--real_id", type=int)
    for name in ("approve", "reject", "remove"):
        p = sub.add_parser(name)
        p.add_argument("id", type=int, nargs="?")
        p.add_argument("--session", default=None)
        p.add_argument("--real_id", type=int, default=None)
    p = sub.add_parser("show")
    p.add_argument("real_id", type=int)
    p.add_argument("--modality", choices=["day", "night"], required=True)
    p.add_argument("--gallery", choices=["flat", "pose"], required=True)
    p.add_argument("--slot", type=int, default=FLAT_SLOT)
    p = sub.add_parser("list-id-switch-flags")
    p.add_argument("--all", action="store_true", help="include reviewed flags")
    p = sub.add_parser("review-id-switch-flag"); p.add_argument("id", type=int)
    args = ap.parse_args()
    config.init(args.params)

    from drive_manager import (DriveManager, DriveNotSyncedError, DriveUnavailableError,
                               LOCAL_GALLERY_DIR, DRIVE_GALLERY_PREFIX)
    dm = DriveManager(bypass=args.bypass_upload_check, caller=__file__)
    writes_gallery = args.cmd in ("approve", "reject", "remove")
    writes_db = writes_gallery or args.cmd == "review-id-switch-flag"
    try:
        dm.check_flag("db")
        if writes_gallery:
            dm.check_flag("gallery")
        db = dm.pull_db(allow_stale=args.bypass_upload_check)
    except (DriveNotSyncedError, DriveUnavailableError) as e:
        print(f"ERROR: {e}\nUse --bypass_upload_check to proceed with current Drive state.")
        return
    if writes_gallery:
        dm.pull_gallery()
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    migrate_schema(conn)

    try:
        if args.cmd == "list-pending":
            rows = list_pending(conn, real_id=args.real_id)
            print(f"{len(rows)} contribution(s) held for review:")
            for r in rows:
                _print_row(r)
        elif writes_gallery:
            if (args.id is None) == (args.session is None or args.real_id is None):
                ap.error(f"{args.cmd}: give a row id OR --session and --real_id")
            res = edit_contributions(conn, args.cmd, contribution_id=args.id,
                                     session_id=args.session, real_id=args.real_id)
            for r in res.rows:
                _print_row(r)
            changed = _apply_rebuilt(res.rebuilt, conn, str(dm.get_gallery_dir()))
            for (rid, m, g, s), (vec, n) in res.rebuilt.items():
                print(f"  → real_id {rid} {m}/{g}" + (f"/slot{s}" if g == "pose" else "")
                      + f": {n} active contribution(s)" + ("" if vec is not None else " — cleared"))
            if args.cmd == "remove":
                print("  note: 'remove' is undone by the next re-run of that session; "
                      "use 'reject' to keep a contribution out.")
            dm.sync_db()
            for fname in sorted(changed):
                dm.write_file(LOCAL_GALLERY_DIR / fname, f"{DRIVE_GALLERY_PREFIX}/{fname}",
                              flag="gallery")
        elif args.cmd == "show":
            for r in conn.execute(
                    "SELECT * FROM gallery_contributions WHERE real_id = ? AND modality = ? "
                    "AND gallery = ? AND slot_idx = ? ORDER BY created_dt",
                    (args.real_id, args.modality, args.gallery, args.slot)).fetchall():
                _print_row(dict(r))
            vec, n, w = rebuild_vector(conn, args.real_id, args.modality, args.gallery, args.slot)
            print(f"rebuilt now: {n} active contribution(s), total weight {w:.2f}"
                  + ("" if vec is not None else " — no vector"))
        elif args.cmd == "list-id-switch-flags":
            rows = list_id_switch_flags(conn, unreviewed_only=not args.all)
            print(f"{len(rows)} flag(s):")
            for r in rows:
                sw = r.get("switch_frame")
                print(f"  id={r['id']:<6} {r['session_id']:<42} temp_id={r['temp_id']:<6} "
                      f"real_id={r['real_id'] if r['real_id'] is not None else 'unresolved'} "
                      f"{r['modality']:<5} switch@frame={sw if sw is not None else '?'} "
                      f"gap={r['split_gap']:.3f} smaller_group={r['split_cluster_frac']:.0%} "
                      f"trimmed={r['trim_frac']:.0%} reviewed={bool(r['reviewed'])}")
        elif args.cmd == "review-id-switch-flag":
            mark_id_switch_reviewed(conn, args.id)
            print(f"flag id={args.id} marked reviewed")
            dm.sync_db()
    finally:
        conn.close()


if __name__ == "__main__":
    main()