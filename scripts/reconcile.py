#!/usr/bin/env python3
"""
reconcile.py — Post-processing pipeline for identity resolution and feature extraction.

Runs after track_and_dump.py. Reads raw_tracks + collar_signals from SQLite, resolves
real_id for every temp_id in a session, and writes to resolved_cow_timeline.

Steps (in order):
  Tracklet Stitching     — join a cow's fragments across short tracker dropouts
  Noise Filter           — drop noise-level temp_ids
  Appearance Quality     — per-detection weight for every step that pools embeddings
  Kinetic Matcher        — bbox centroid speed ↔ ΔKineticsCountR (Pearson r)
  Manual Merge           — load operator overrides, manual beats kinetic on conflict
  Gallery preview        — in-memory blend of this session's manual/kinetic ids (for cosine queries)
  Cosine Resolver        — heal temp_id switches, cross-video/cross-camera continuity
  Boundary Rendezvous    — cross-camera match via shared-zone tracks (see cross_camera.py)
  Duplicate Resolver     — merge temp_ids assigned to the same AnimalId (tracker switches)
  Gallery commit         — the run's single flat-gallery update, weighted by id source
  Sensor Sequencer       — forward-fill behavior/kinetics to video time grid
  Vision Feature Extraction — posture/facing from keypoints, update pose-conditioned gallery
  Timeline Writer        — write resolved_cow_timeline rows

Usage:
    python3 reconcile.py --session <session_id> [--dry_run] [--params my.yaml]

All thresholds (kinetic matching, gallery, cosine, co-occurrence, vision,
cross-camera) come from config/params.yaml — see config.py.

Note: --db, --kinetics, --gallery_dir, and --embed_parquet are NOT arguments.
All data paths (DB, collar CSVs, gallery, parquet) are resolved exclusively
through drive_manager.py, keyed off --session. This guarantees every run
reads/writes the same canonical Drive-backed data regardless of who runs it
or from which machine.

Requirements: pip install pandas numpy scipy pyarrow
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vision_features import run_vision_features, migrate_timeline_schema
from vision_features.gallery import (
    TempPoseGallery, TempKey, PoseGallery,
    mint_synthetic_id, is_synthetic, backpropagate_resolution,
)
from vision_features.schema import EMBED_DIM

import config
import kinetic_match
import tracklet_stitch
import appearance_quality
import gallery_ledger
import identity_record
from appearance_quality import effective_frames

# Drive I/O layer
from drive_manager import DriveManager, DriveNotSyncedError, DriveUnavailableError


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[reconcile] {msg}", flush=True)


def section(title: str) -> None:
    print(f"\n{'─'*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─'*60}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Post-processing: kinetic match → gallery update → cosine resolve → timeline write"
    )

    # --- required ---
    ap.add_argument("--session",  required=True, help="session_id to process (from video_sessions)")
    # NOTE: --db, --kinetics, --gallery_dir, --embed_parquet are intentionally
    # NOT arguments. Every data path is resolved through drive_manager.py,
    # keyed off --session, so there is exactly one source of truth (Drive).

    # --- misc ---
    ap.add_argument("--dry_run", action="store_true",
                    help="Run all steps but do not write to the database")
    ap.add_argument("--bypass_upload_check", action="store_true",
                    help="Skip dirty-flag check when reading from Drive "
                         "(proceeds with potentially stale data — use with caution)")
    config.add_params_arg(ap)

    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS video_sessions (
    session_id      TEXT PRIMARY KEY,
    video_path      TEXT,
    camera_id       TEXT,
    start_dt        TEXT,
    end_dt          TEXT,
    collar_csv_path TEXT,
    is_night        INTEGER DEFAULT 0,  -- majority camera mode, summary only (written by
                                        -- track_and_dump.py; per-row raw_tracks.is_night
                                        -- is what reconcile actually uses)
    night_frame_fraction REAL,          -- fraction of frames in grayscale/IR mode
    frame_w         INTEGER,            -- video frame size in px (appearance_quality
    frame_h         INTEGER             -- edge check); NULL for older sessions
);

CREATE TABLE IF NOT EXISTS cow_registry (
    real_id          INTEGER PRIMARY KEY,
    breed            TEXT,
    parity           INTEGER,
    pen_id           TEXT,
    collar_id        TEXT,
    baseline_window  TEXT
);

CREATE TABLE IF NOT EXISTS collar_signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    animal_id   INTEGER,
    datetime    TEXT,
    signal_type TEXT,   -- 'behavior' | 'kinetic'
    f_1_2       REAL,
    f_2_3       REAL,
    v           REAL,
    kin_x       REAL,
    kin_y       REAL,
    kin_z       REAL,
    kin_r       REAL
);

CREATE TABLE IF NOT EXISTS raw_tracks (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT,
    frame_index    INTEGER,
    frame_datetime TEXT,
    temp_id        INTEGER,
    det_conf       REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    cx REAL, cy REAL,
    embed          TEXT,   -- JSON list[128] or NULL
    kps            TEXT,   -- JSON flat list[57] or NULL
    kps_norm       TEXT,
    kps_conf       REAL,
    kps_kconf      TEXT,
    is_night       INTEGER   -- camera mode at this frame (1=IR/grayscale, 0=colour)
);

CREATE TABLE IF NOT EXISTS reid_registry (
    real_id               INTEGER PRIMARY KEY,
    gallery_embed_day     TEXT,   -- JSON list[128] or NULL
    gallery_embed_night   TEXT,
    gallery_n_day         INTEGER DEFAULT 0,
    gallery_n_night       INTEGER DEFAULT 0,
    gallery_conf_day      REAL    DEFAULT 0.0,
    gallery_conf_night    REAL    DEFAULT 0.0,
    last_updated_day_dt   TEXT,
    last_updated_night_dt TEXT,
    known_temp_ids        TEXT,   -- DEPRECATED, no longer written: see session_assignments
                                  -- (identity_record.py); kept so old DBs still open
    first_seen_dt         TEXT,
    match_method          TEXT    -- best source that ever identified the cow (upgrade-only):
                                  -- manual | kinetic | boundary_rendezvous | cosine;
                                  -- 'synthetic' for synthetic ids; older rows may hold
                                  -- 'cosine_day' / 'cosine_night' (= cosine)
);

CREATE TABLE IF NOT EXISTS resolved_cow_timeline (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    real_id         INTEGER,
    session_id      TEXT,
    window_start_dt TEXT,
    modality_mask   INTEGER DEFAULT 0,  -- bitmask: 1=sensor_ok, 2=vision_ok, 4=reid_ok
    -- sensor features (forward-filled)
    d_f12   REAL, d_f23   REAL, d_v    REAL,
    d_kin_x REAL, d_kin_y REAL, d_kin_z REAL, d_kin_r REAL,
    -- vision features (from pose)
    spine_angle     REAL,
    pelvic_tilt     REAL,
    tail_elevation  REAL,
    limb_symmetry   REAL,
    head_drop       REAL,
    lying_flag      INTEGER,
    restlessness    REAL,
    kps_coverage    REAL,
    embed_mean      TEXT    -- JSON list[128] mean-pooled over window
);

CREATE TABLE IF NOT EXISTS calving_ledger (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    real_id     INTEGER,
    calving_dt  TEXT,
    outcome     TEXT    -- 'Unassisted'|'Assisted'|'Twin'|'Veterinarian-assisted'
);

CREATE TABLE IF NOT EXISTS manual_assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    temp_id     INTEGER NOT NULL,
    real_id     INTEGER NOT NULL,
    assigned_by TEXT    DEFAULT 'manual',
    assigned_dt TEXT,
    note        TEXT,
    UNIQUE(session_id, temp_id)
);

CREATE TABLE IF NOT EXISTS tracklet_stitches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    loser_tid   INTEGER NOT NULL,   -- raw temp_id renamed by stitching
    root_tid    INTEGER NOT NULL,   -- temp_id it now carries (first fragment of the chain)
    prev_tid    INTEGER NOT NULL,   -- the fragment it was linked to directly
    gap_s       REAL,               -- prev's last frame → loser's first frame
    dist        REAL,               -- end→start distance in box sizes
    size_ratio  REAL,
    created_dt  TEXT,
    UNIQUE(session_id, loser_tid)
);

CREATE TABLE IF NOT EXISTS temp_id_merges (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    winner_tid    INTEGER NOT NULL,   -- temp_id that survives
    loser_tid     INTEGER NOT NULL,   -- temp_id that was remapped to winner
    real_id       INTEGER NOT NULL,   -- the shared AnimalId
    winner_reason TEXT,               -- source of the winner's id: 'manual' | 'kinetic' | 'boundary_rendezvous' | 'cosine'
    loser_reason  TEXT,               -- source of the loser's id (same values)
    merged_dt     TEXT,
    action        TEXT DEFAULT 'merged',  -- 'merged' (same animal, rows pooled into winner)
                                          -- | 'unassigned_cooccurring' (on screen at the same time
                                          --   as the winner in a different spot, so a different
                                          --   animal: loser lost the id and stays unassigned)
    UNIQUE(session_id, loser_tid)     -- a loser can only be merged once per session
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    for sql in (
        # DBs created before temp_id_merges.action existed
        "ALTER TABLE temp_id_merges ADD COLUMN action TEXT DEFAULT 'merged'",
        # DBs created before per-frame camera-mode detection existed
        "ALTER TABLE raw_tracks ADD COLUMN is_night INTEGER",
        "ALTER TABLE video_sessions ADD COLUMN night_frame_fraction REAL",
        # DBs created before appearance-quality weighting existed
        "ALTER TABLE video_sessions ADD COLUMN frame_w INTEGER",
        "ALTER TABLE video_sessions ADD COLUMN frame_h INTEGER",
    ):
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass   # column already present
    conn.commit()
    log(f"Database ready: {db_path}")
    return conn


def get_session(conn: sqlite3.Connection, session_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM video_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    return dict(row) if row else None


def upsert_session(conn: sqlite3.Connection, session_id: str,
                   tracks_path: str, is_night: bool) -> None:
    """Register session if it doesn't exist yet (minimal fields). A session
    registered by track_and_dump.py is left untouched (INSERT OR IGNORE)."""
    conn.execute("""
        INSERT OR IGNORE INTO video_sessions (session_id, video_path, is_night)
        VALUES (?, ?, ?)
    """, (session_id, tracks_path, int(is_night)))
    conn.commit()


def get_reid_row(conn: sqlite3.Connection, real_id: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM reid_registry WHERE real_id = ?", (real_id,)
    ).fetchone()
    return dict(row) if row else None


def upsert_reid(conn: sqlite3.Connection, real_id: int, updates: dict) -> None:
    existing = get_reid_row(conn, real_id)
    if existing is None:
        cols = ["real_id"] + list(updates.keys())
        vals = [real_id] + list(updates.values())
        ph = ", ".join("?" * len(vals))
        conn.execute(
            f"INSERT INTO reid_registry ({', '.join(cols)}) VALUES ({ph})", vals
        )
    else:
        sets = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"UPDATE reid_registry SET {sets} WHERE real_id = ?",
            list(updates.values()) + [real_id]
        )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Camera mode (day colour / night IR) — per frame
#
# track_and_dump.py detects the mode from the pixels of every frame and stores
# it per row in raw_tracks.is_night; the cameras can switch mid-video. Every
# gallery step splits a session's rows by it, so one session can feed both the
# day and the night galleries without ever mixing the two.
# ─────────────────────────────────────────────────────────────────────────────

MODALITIES = ("day", "night")


def modality_of(is_night) -> str:
    return "night" if bool(is_night) else "day"


def detect_is_night_from_tracks(tracks_df: pd.DataFrame) -> bool:
    """
    LEGACY FALLBACK — clock-based, only for sessions tracked before per-frame
    camera-mode detection existed (raw_tracks.is_night is NULL).
    Night = more than session.night_fraction of frames fall in
    [night_start_hour, 24) ∪ [0, night_end_hour) local (params.yaml).
    """
    if "frame_datetime" not in tracks_df.columns:
        return False
    dt_col = pd.to_datetime(tracks_df["frame_datetime"], errors="coerce").dropna()
    if dt_col.empty:
        return False
    sp = config.params().session
    hours = dt_col.dt.hour
    night_mask = (hours >= sp.night_start_hour) | (hours < sp.night_end_hour)
    return bool(night_mask.mean() > sp.night_fraction)


def raw_tracks_is_night_select(conn: sqlite3.Connection) -> str:
    """SELECT expression for raw_tracks.is_night (NULL on a pre-migration table)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(raw_tracks)").fetchall()}
    return "is_night" if "is_night" in cols else "NULL AS is_night"


def resolve_camera_mode(tracks_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Turn tracks_df.is_night (raw per-row values, NULL for legacy sessions) into
    a bool column. Rows without a per-frame value get the clock-based fallback.
    Returns (tracks_df, fallback_is_night) — the fallback is also what
    run_vision_features uses for its own NULL rows.
    """
    fallback = detect_is_night_from_tracks(tracks_df)
    n_null = int(tracks_df["is_night"].isna().sum())
    if n_null:
        log(f"WARNING: {n_null}/{len(tracks_df)} raw_tracks rows have no per-frame "
            f"camera mode (tracked before detection existed) — using the legacy "
            f"clock-based fallback is_night={fallback}. Re-run track_and_dump.py on "
            f"this video for per-frame labels.")
    tracks_df = tracks_df.copy()
    tracks_df["is_night"] = tracks_df["is_night"].fillna(int(fallback)).astype(bool)
    return tracks_df, fallback


def split_by_modality(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{modality -> rows of df in that camera mode}, only modalities present."""
    if df.empty or "is_night" not in df.columns:
        return {}
    return {modality_of(v): g for v, g in df.groupby("is_night")}


def raw_tracks_optional_select(conn: sqlite3.Connection, col: str) -> str:
    """SELECT expression for a raw_tracks column that older DBs may lack."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(raw_tracks)").fetchall()}
    return col if col in cols else f"NULL AS {col}"


# ─────────────────────────────────────────────────────────────────────────────
# Tracklet stitching — join a cow's fragments across short tracker dropouts
# (implementation in tracklet_stitch.py; this wrapper renames and records)
# ─────────────────────────────────────────────────────────────────────────────

def stitch_tracklets(tracks_df: pd.DataFrame, manual: dict, conn: sqlite3.Connection,
                     session_id: str, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Rename every stitched fragment to its chain's first temp_id.
    Returns (tracks_df with renamed temp_ids, stitch_map {raw loser -> root}).

    Records each link in tracklet_stitches, replacing this session's rows from
    any earlier run. Every step that reads raw temp_ids straight from storage
    must apply stitch_map itself: embeds (load_embeds_for_session), the
    extractor (merge_map) and cross_camera.py's motion tier (via this table).
    """
    section("Tracklet Stitching")
    res = tracklet_stitch.stitch(tracks_df, manual=manual)
    n_before = tracks_df["temp_id"].nunique()
    if res.tid_map:
        tracks_df = tracks_df.copy()
        tracks_df["temp_id"] = tracks_df["temp_id"].replace(res.tid_map)
    log(f"{len(res.links)} links → {n_before} temp_ids become "
        f"{tracks_df['temp_id'].nunique()} ({res.n_chains} multi-fragment chains); "
        f"{res.n_candidates} candidate links, {res.n_ambiguous} skipped as ambiguous"
        + (f", {res.n_manual_veto} vetoed by conflicting manual assignments"
           if res.n_manual_veto else ""))
    if res.links:
        gaps = np.array([l.gap_s for l in res.links])
        log(f"  gaps bridged: median {np.median(gaps):.2f} s, max {gaps.max():.2f} s")

    if not dry_run:
        conn.execute("DELETE FROM tracklet_stitches WHERE session_id = ?", (session_id,))
        ts = pd.Timestamp.now().isoformat()
        conn.executemany(
            "INSERT INTO tracklet_stitches (session_id, loser_tid, root_tid, prev_tid, "
            "gap_s, dist, size_ratio, created_dt) VALUES (?,?,?,?,?,?,?,?)",
            [(session_id, int(l.next_tid), int(res.tid_map[l.next_tid]), int(l.prev_tid),
              l.gap_s, l.dist, l.size_ratio, ts) for l in res.links])
        conn.commit()
    return tracks_df, {int(k): int(v) for k, v in res.tid_map.items()}


def compose_tid_maps(first: dict, then: dict) -> dict:
    """raw temp_id → final temp_id through two renames (stitching, then the
    Duplicate Resolver's merges, which are keyed by post-stitch ids)."""
    out = {}
    for raw in set(first) | set(then):
        mid = first.get(raw, raw)
        final = then.get(mid, mid)
        if final != raw:
            out[raw] = final
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Noise filter — drop noise-level temp_ids before any identity step
#
# On the June 16 sessions roughly half of all temp_ids lasted under 5 s while
# carrying only 1-3% of the detections. Kept, each of them is queried by the
# Cosine Resolver (5 embeds is 0.2 s at 25 fps) and gets its own TempPoseGallery
# entry that later sessions can match — the seed of a junk synthetic id.
# A good share are fragments of real cows' broken tracks rather than false
# detections; short-gap tracklet stitching, when it exists, runs BEFORE this
# filter so fragments get the chance to be joined first.
# ─────────────────────────────────────────────────────────────────────────────

def noise_filter(tracks_df: pd.DataFrame,
                 exempt_tids: set) -> tuple[pd.DataFrame, set]:
    """
    Drop temp_ids lasting < noise_filter.min_duration_s (first → last detection)
    or with < noise_filter.min_detections rows. temp_ids in exempt_tids
    (manual assignments — operator decisions always stand) are kept regardless.

    Duration uses frame_time_sec when present, else frame_datetime.
    Returns (filtered tracks_df, set of dropped temp_ids).
    """
    nf = config.params().noise_filter
    if "frame_time_sec" in tracks_df.columns and tracks_df["frame_time_sec"].notna().all():
        t = tracks_df["frame_time_sec"].astype(float)
    else:
        t = (tracks_df["frame_datetime"] - tracks_df["frame_datetime"].min()).dt.total_seconds()
    per_tid = (pd.DataFrame({"temp_id": tracks_df["temp_id"], "t": t})
               .groupby("temp_id")["t"].agg(["min", "max", "size"]))
    duration = per_tid["max"] - per_tid["min"]
    noisy = (duration < nf.min_duration_s) | (per_tid["size"] < nf.min_detections)
    dropped = set(per_tid.index[noisy]) - set(exempt_tids)

    kept_exempt = set(per_tid.index[noisy]) & set(exempt_tids)
    n_rows = int(tracks_df["temp_id"].isin(dropped).sum())
    n_tids = tracks_df["temp_id"].nunique()
    log(f"Noise filter (< {nf.min_duration_s:g} s or < {nf.min_detections} detections): "
        f"dropped {len(dropped)}/{n_tids} temp_ids "
        f"({len(dropped) / max(n_tids, 1) * 100:.1f}%), "
        f"{n_rows}/{len(tracks_df)} rows ({n_rows / max(len(tracks_df), 1) * 100:.2f}%)")
    if kept_exempt:
        log(f"  kept {len(kept_exempt)} noise-level temp_id(s) because they are manually "
            f"assigned: {sorted(kept_exempt)}")
    return tracks_df[~tracks_df["temp_id"].isin(dropped)].copy(), dropped




# ─────────────────────────────────────────────────────────────────────────────
# Kinetic Matcher — the implementation lives in kinetic_match.py (shared with
# the match_identity.py CLI); this wrapper only logs.
# ─────────────────────────────────────────────────────────────────────────────

def kinetic_matcher(tracks_df: pd.DataFrame,
                    kinetics_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Returns (assignment, kinetic_corr):
        assignment   : {temp_id (int) -> AnimalId (int)}
        kinetic_corr : {temp_id (int) -> Pearson r}, the correlation that
                       produced each assignment (used by Boundary Rendezvous,
                       cross_camera.py, for its tier-1 / confidence-gap logic).
    """
    section("Kinetic Matcher")
    km = config.params().kinetic_match
    bin_minutes = config.params().timeline.bin_minutes

    res = kinetic_match.match(tracks_df, kinetics_df)

    log(f"Stable temp_ids (≥{km.min_temp_id_frames*100:.0f}% frames): {sorted(res.stable_tids)}")
    if res.bins is not None:
        log(f"Time bins: {len(res.bins)-1} × {bin_minutes}-min "
            f"from {res.bins[0]} to {res.bins[-1]}")
    for w in res.warnings:
        log(f"WARNING: {w}.")

    matrix = res.matrix()
    if not matrix.empty:
        print("\nCorrelation matrix (temp_id × AnimalId):")
        print(matrix.to_string())
        print()

    if res.assignment:
        for tid, aid in sorted(res.assignment.items()):
            row = res.scores[(res.scores["temp_id"] == tid)
                             & (res.scores["AnimalId"] == aid)].iloc[0]
            log(f"  temp_id {tid:>3}  →  AnimalId {aid}  "
                f"(r={row['correlation']:.3f}, n_bins={int(row['n_bins'])})")
    elif res.stable_tids:
        log("  No confident kinetic matches. "
            "Try lowering kinetic_match.corr_threshold in params.yaml or using a longer video.")

    if res.unmatched:
        log(f"  Unmatched temp_ids after kinetics: {res.unmatched}")

    return res.assignment, res.corr


# ─────────────────────────────────────────────────────────────────────────────
# Embed helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_embeds_for_session(tracks_df: pd.DataFrame,
                             embed_parquet: str,
                             session_id: str,
                             tid_map: dict | None = None) -> pd.DataFrame:
    """
    Returns DataFrame with columns [frame_index, temp_id, embed (np.ndarray
    shape 128), is_night (bool)]. Tries parquet first, falls back to an
    'embed' column in tracks_df.

    is_night comes from tracks_df (one row per committed (frame_index,
    temp_id) — the same rows the embed parquet points at), so it must already
    be resolved to bool by resolve_camera_mode().
    """
    QC = appearance_quality.QUALITY_COL
    mode_cols = ["frame_index", "temp_id", "is_night"] + ([QC] if QC in tracks_df.columns else [])
    mode = tracks_df[mode_cols].drop_duplicates(["frame_index", "temp_id"])
    kept_tids = set(tracks_df["temp_id"].unique())

    def _attach_mode(df: pd.DataFrame) -> pd.DataFrame:
        # stored embeds carry raw temp_ids: rename stitched fragments first
        if tid_map:
            df = df.copy()
            df["temp_id"] = df["temp_id"].replace(tid_map)
        # temp_ids removed upstream (noise filter) leave the embeds too
        df = df[df["temp_id"].isin(kept_tids)]
        df = df.merge(mode, on=["frame_index", "temp_id"], how="left")
        n_missing = int(df["is_night"].isna().sum())
        if n_missing:
            # Shouldn't happen (every embed row has a raw_tracks row); if it
            # does, fall back to the session's majority mode rather than drop.
            majority = bool(tracks_df["is_night"].mean() > 0.5)
            log(f"WARNING: {n_missing} embed rows have no matching raw_tracks row — "
                f"assigning majority camera mode is_night={majority}")
            df["is_night"] = df["is_night"].fillna(majority)
        df["is_night"] = df["is_night"].astype(bool)
        if QC in df.columns:
            # appearance-quality weight per frame; weight-0 frames (box below
            # appearance_quality.min_px) carry no usable appearance — drop them
            df[QC] = df[QC].fillna(0.0)
            n_zero = int((df[QC] <= 0).sum())
            if n_zero:
                log(f"  appearance quality: dropped {n_zero}/{len(df)} embed rows with "
                    f"weight 0 (too small or cut off); effective frames "
                    f"{df[QC].sum():.0f} of {len(df) - n_zero} kept rows")
            df = df[df[QC] > 0].copy()
        return df

    empty = pd.DataFrame(columns=["frame_index", "temp_id", "embed", "is_night", QC])

    if embed_parquet:
        ep = Path(embed_parquet)
        # Accept either a session directory (multi-part) or a single legacy file.
        if ep.is_dir():
            parts = sorted(ep.glob("embeds_part*.parquet"))
        elif ep.exists():
            parts = [ep]
        else:
            parts = []
        if parts:
            log(f"Loading embeddings from {len(parts)} parquet part(s): {ep}")
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            if "session_id" in df.columns:
                df = df[df["session_id"] == session_id]
            if "embed" in df.columns and not df.empty:
                if not isinstance(df["embed"].iloc[0], np.ndarray):
                    df["embed"] = df["embed"].apply(
                        lambda x: np.array(json.loads(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32)
                    )
                df = _attach_mode(df[["frame_index", "temp_id", "embed"]].copy())
                n_night = int(df["is_night"].sum())
                log(f"  {len(df)} embed rows  ({len(df) - n_night} day, {n_night} night)")
                return df

    # fall back to 'embed' column in tracks CSV
    if "embed" not in tracks_df.columns:
        log("WARNING: no 'embed' column in tracks and no parquet — gallery step will be skipped.")
        return empty

    rows = []
    for _, row in tracks_df[tracks_df["embed"].notna()].iterrows():
        raw = row["embed"]
        if not raw or raw == "[]":
            continue
        try:
            arr = np.array(json.loads(raw), dtype=np.float32)
            if arr.shape == (EMBED_DIM,):
                rows.append({"frame_index": row["frame_index"],
                             "temp_id": row["temp_id"], "embed": arr})
        except Exception:
            pass

    if not rows:
        log("WARNING: no valid embeds found in tracks CSV — gallery step will be skipped.")
        return empty

    log(f"Loaded {len(rows)} embed rows from tracks CSV.")
    return _attach_mode(pd.DataFrame(rows))


def compute_mean_embed(embed_rows: pd.DataFrame) -> np.ndarray:
    """Appearance-quality-weighted, outlier-trimmed mean of the embed arrays,
    L2-normalised (plain mean when no weights are attached) — see
    appearance_quality.py."""
    return appearance_quality.weighted_mean_embed(embed_rows)


def compute_mean_embed_report(embed_rows: pd.DataFrame):
    """Same as compute_mean_embed() but also returns the outlier-trim and
    ID-switch-split diagnostics (appearance_quality.MeanEmbedReport) — use
    this wherever a possible tracker ID switch should be surfaced."""
    return appearance_quality.mean_embed_report(embed_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Gallery I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_gallery(gallery_dir: str, modality: str) -> dict:
    """
    Load gallery_{modality}.npy → dict {real_id (int): np.ndarray shape (128,)}.
    modality: 'day' | 'night'
    Returns empty dict if file doesn't exist.
    """
    path = Path(gallery_dir) / f"gallery_{modality}.npy"
    if not path.exists():
        return {}
    data = np.load(str(path), allow_pickle=True).item()
    return {int(k): v.astype(np.float32) for k, v in data.items()}


def save_gallery(gallery_dir: str, modality: str, gallery: dict) -> None:
    """Save gallery dict to gallery_{modality}.npy. Creates directory if needed."""
    Path(gallery_dir).mkdir(parents=True, exist_ok=True)
    path = Path(gallery_dir) / f"gallery_{modality}.npy"
    np.save(str(path), gallery)
    log(f"Gallery saved: {path}  ({len(gallery)} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# Gallery Builder
# ─────────────────────────────────────────────────────────────────────────────

def _ema_update(gallery: dict, aid, session_mean: np.ndarray, alpha: float):
    """EMA-blend session_mean into gallery[aid] (new entry if absent).
    Returns cos(previous vector, session_mean), or None for a new entry."""
    if aid in gallery:
        old = gallery[aid]
        drift = float(np.dot(old, session_mean))
        upd = alpha * session_mean + (1 - alpha) * old
        norm = np.linalg.norm(upd)
        gallery[aid] = upd / norm if norm > 1e-8 else upd
        return drift
    gallery[aid] = session_mean
    return None


def preview_galleries(embed_df: pd.DataFrame, confirmed: dict,
                      provenance: dict, gallery_dir: str) -> dict:
    """
    In-memory only: both modalities' flat galleries as stored, plus this
    session's manual/kinetic temp_ids blended in — what the Cosine Resolver
    queries, so a fragment of a cow confirmed earlier in THIS session can
    still be matched to her (heals temp_id switches). Nothing is saved and
    reid_registry is not touched; the one real update is commit_galleries(),
    after every identity step.
    """
    section("Gallery preview (for the Cosine Resolver)")
    P = config.params()
    gp = P.gallery
    galleries = {m: load_gallery(gallery_dir, m) for m in MODALITIES}
    log("Loaded galleries: " + "  ".join(f"{m}={len(g)}" for m, g in galleries.items()))
    n = 0
    for modality, mdf in split_by_modality(embed_df).items():
        for tid, aid in confirmed.items():
            weight = getattr(gp.source_weights, provenance.get(tid, "kinetic"))
            tid_embeds = mdf[mdf["temp_id"] == tid]
            if weight <= 0 or effective_frames(tid_embeds) < gp.min_embeds:
                continue
            _ema_update(galleries[modality], aid, compute_mean_embed(tid_embeds),
                        gp.ema_alpha * weight)
            n += 1
    log(f"  {n} in-memory update(s) from this session's manual/kinetic ids")
    return galleries


def commit_galleries(embed_df: pd.DataFrame, assignment: dict, provenance: dict,
                     conn: sqlite3.Connection, gallery_dir: str, session_id: str,
                     dry_run: bool = False) -> tuple[dict, set | None]:
    """
    The run's single flat-gallery update, after the Cosine Resolver,
    Boundary Rendezvous and Duplicate Resolver have settled every temp_id's id
    (embed_df already has the Duplicate Resolver's merged fragments pooled
    under their winner).

    Per temp_id and camera mode, this session's outlier-trimmed, quality-
    weighted mean is recorded in the gallery contribution ledger
    (gallery_ledger.py) and the cow's vector becomes the ledger's rebuild —
    no running EMA. The session's rows from earlier runs are replaced as a
    whole, so a re-run never applies anything twice and a cow this session no
    longer contributes to is rebuilt without it. A contribution too far from
    the cow's current vector is held for review (gallery.contribution.
    drift_gate_cosine). Skipped: synthetic ids, a source weight of 0, fewer
    than gallery.min_embeds effective frames.

    Item 10: each contributing track is also checked for a two-group split
    (appearance_quality.mean_embed_report) and flagged in id_switch_flags —
    for review only, nothing is changed automatically.

    Item 9 (gallery.cosine_gate): a cosine id contributes only if the track's
    mean is at least cosine_resolver.threshold + threshold_margin from the
    cow's vector AND beats every other cow by second_best_margin. Vectors are
    as they stand for this run (gallery_ledger.SessionCommit.reference): other
    sessions plus this run's manual / kinetic / boundary ids, which are
    committed first, never this session's earlier runs.

    Returns (galleries, cosine_ok): cosine_ok is the set of (temp_id,
    modality) cosine ids allowed to update the galleries — the pose update
    in the Vision Feature Extractor follows it — or None when the gate is off.
    """
    section("Gallery commit")
    P = config.params()
    gp = P.gallery
    galleries = {m: load_gallery(gallery_dir, m) for m in MODALITIES}
    ledger = gallery_ledger.SessionCommit(conn, session_id, "flat", dry_run=dry_run)
    gallery_ledger.clear_id_switch_flags(conn, session_id, dry_run=dry_run)
    gate = gp.cosine_gate
    cosine_ok: set | None = set() if gate.enabled else None
    counts: dict = {}
    held: dict = {}
    gated: dict = {}
    touched: set = set()
    # manual / kinetic / boundary first, so cosine ids are gated against
    # galleries that already include this run's confirmed ids
    order = sorted(assignment.items(), key=lambda kv: provenance.get(kv[0], "cosine") == "cosine")
    for modality, mdf in split_by_modality(embed_df).items():
        gallery = galleries[modality]
        refs = None   # {real_id -> reference vector}, built at the first cosine id
        for tid, aid in order:
            if is_synthetic(aid):
                continue
            source = provenance.get(tid, "cosine")
            weight = getattr(gp.source_weights, source)
            tid_embeds = mdf[mdf["temp_id"] == tid]
            if len(tid_embeds) == 0 or weight <= 0:
                continue
            eff = effective_frames(tid_embeds)
            if eff < gp.min_embeds:
                log(f"  [{modality}] t{tid} ({source}): {eff:.1f} effective frames "
                    f"(< {gp.min_embeds}) — skip")
                continue

            report = compute_mean_embed_report(tid_embeds)
            if report.n_trimmed > 0:
                log(f"  [{modality}] t{tid}: trimmed {report.n_trimmed:.1f} of "
                    f"{report.n_total:.1f} effective frames as outliers "
                    f"({report.trim_frac:.0%})")
            if report.split_suspected:
                log(f"  [{modality}] t{tid}: possible ID SWITCH — frames split into two "
                    f"groups (gap={report.split_gap:.3f}, smaller group="
                    f"{report.split_cluster_frac:.0%} of {report.n_total:.0f} effective "
                    f"frames) — flagged for review, nothing changed")
                gallery_ledger.flag_id_switch(conn, session_id, tid, aid, modality,
                                              report, dry_run=dry_run)

            if source == "cosine" and gate.enabled:
                if refs is None:
                    cand = ledger.known_ids(modality) | {r for r in gallery if r > 0}
                    refs = {r: v for r in cand
                            if (v := ledger.reference(r, modality, fallback=gallery.get(r)))
                            is not None}
                own = refs.get(aid)
                score = float(report.vector @ own) if own is not None else None
                rival_id, rival = None, None
                for r, v in refs.items():
                    if r != aid:
                        c = float(report.vector @ v)
                        if rival is None or c > rival:
                            rival_id, rival = r, c
                need = P.cosine_resolver.threshold + gate.threshold_margin
                if score is None:
                    why = "cow has no gallery vector to check against"
                elif score < need:
                    why = f"cosine {score:.3f} < {need:.3f}"
                elif rival is not None and score - rival < gate.second_best_margin:
                    why = (f"cosine {score:.3f} only {score - rival:.3f} above AnimalId "
                           f"{rival_id} ({rival:.3f}); need {gate.second_best_margin}")
                else:
                    why = None
                if why:
                    log(f"  [{modality}] AnimalId {aid} (t{tid}, cosine): not used for the "
                        f"gallery — {why}")
                    gated[modality] = gated.get(modality, 0) + 1
                    continue
                cosine_ok.add((int(tid), modality))

            existing = get_reid_row(conn, aid) or {}
            res = ledger.record(
                real_id=aid, modality=modality, source=source, source_weight=weight,
                n_effective_frames=report.n_total - report.n_trimmed,
                vector=report.vector,
                current_vector=gallery.get(aid),
                current_dt=existing.get(f"last_updated_{modality}_dt"),
            )
            if res.seeded_legacy:
                log(f"  [{modality}] AnimalId {aid}: pre-ledger gallery vector seeded as a "
                    f"'legacy' contribution")
            if res.note:
                log(f"  [{modality}] AnimalId {aid}: WARNING {res.note}")
            drift_s = f"{res.drift:.3f}" if res.drift is not None else "—"
            if res.status != "active":
                why = (f"kept {res.status} (decision from an earlier run)"
                       if res.carried_decision else
                       f"drift {drift_s} < {gp.contribution.drift_gate_cosine} — HELD FOR "
                       f"REVIEW (python3 gallery_ledger.py list-pending)")
                log(f"  [{modality}] AnimalId {aid} (t{tid}, {source}): {why}; not applied")
                held[(modality, source)] = held.get((modality, source), 0) + 1
                continue

            gallery[aid] = res.vector_after
            touched.add((aid, modality))
            log(f"  [{modality}] AnimalId {aid} (t{tid}, {source}): drift={drift_s}, "
                f"rebuilt from {res.n_contributions} contribution(s)"
                + (" [approved earlier]" if res.carried_decision else ""))
            counts[(modality, source)] = counts.get((modality, source), 0) + 1
            if not dry_run:
                # match_method is set once per run from session_assignments'
                # sources (identity_record.upgrade_match_methods), not here
                upsert_reid(conn, aid, {
                    f"gallery_embed_{modality}": json.dumps(res.vector_after.tolist()),
                    f"gallery_n_{modality}":     res.n_contributions,
                    f"last_updated_{modality}_dt": pd.Timestamp.now().isoformat(),
                })

    # Cows this session contributed to on an earlier run but not on this one
    # (re-assigned, dropped below min_embeds, ...): rebuild them without it.
    for (aid, modality, _), (vec, n) in ledger.finish().items():
        gallery = galleries[modality]
        if vec is None:
            gallery.pop(aid, None)
        else:
            gallery[aid] = vec
        touched.add((aid, modality))
        log(f"  [{modality}] AnimalId {aid}: earlier run's contribution withdrawn — "
            + (f"rebuilt from {n} contribution(s)" if vec is not None else "no vector left"))
        if not dry_run:
            upsert_reid(conn, aid, {
                f"gallery_embed_{modality}": json.dumps(vec.tolist()) if vec is not None else None,
                f"gallery_n_{modality}": n,
                f"last_updated_{modality}_dt": pd.Timestamp.now().isoformat(),
            })

    if counts:
        log("Updated: " + ", ".join(f"{m}/{src}={n}" for (m, src), n in sorted(counts.items())))
    else:
        log("No gallery updates this session.")
    if held:
        log("Not applied (held/rejected): "
            + ", ".join(f"{m}/{src}={n}" for (m, src), n in sorted(held.items())))
    if gated:
        log("Cosine ids kept out of the gallery by the margin gate: "
            + ", ".join(f"{m}={n}" for m, n in sorted(gated.items())))
    if not dry_run:
        for m in {m for (_, m) in touched}:
            save_gallery(gallery_dir, m, galleries[m])
    return galleries, cosine_ok


# ─────────────────────────────────────────────────────────────────────────────
# Cosine Resolver
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tid_frame_arrays(tracks_df: pd.DataFrame) -> dict:
    """{temp_id -> sorted np.ndarray of frame_index values actually recorded
    for that tid} — used for real frame-level co-occurrence checks, not just
    an outer [min,max] interval (see _frames_cooccur)."""
    out = {}
    for tid, g in tracks_df.groupby("temp_id"):
        out[int(tid)] = np.sort(g["frame_index"].to_numpy())
    return out


def _tid_frame_boxes(tracks_df: pd.DataFrame) -> dict:
    """{temp_id -> DataFrame[frame_index, x1, y1, x2, y2] sorted by
    frame_index} — lets a frame-level co-occurrence hit be checked
    spatially too: same frame in different parts of the pen is two
    different animals; same frame in the same spot is one animal whose
    box briefly got two temp_ids (an ID-handoff artifact)."""
    cols = ["frame_index", "x1", "y1", "x2", "y2"]
    out = {}
    for tid, g in tracks_df.groupby("temp_id"):
        out[int(tid)] = g[cols].sort_values("frame_index").reset_index(drop=True)
    return out


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Standard IoU for two (x1,y1,x2,y2) boxes. 0 if either is degenerate
    or they don't overlap at all."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _nearest_box(frame_boxes: pd.DataFrame, target_frame: int):
    """Row (as a plain (x1,y1,x2,y2) tuple) whose frame_index is closest to
    target_frame. None if frame_boxes is empty."""
    if frame_boxes.empty:
        return None
    idx = (frame_boxes["frame_index"] - target_frame).abs().idxmin()
    row = frame_boxes.loc[idx]
    return (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))


def _estimate_frame_tolerance(tracks_df: pd.DataFrame) -> int:
    """Approximate the per-tid sampling cadence (roughly save_every from
    track_and_dump.py) directly from the data, so co-occurrence tolerance
    isn't a guessed constant. Uses the median consecutive-frame gap of
    whichever temp_id has the most rows (most likely to be continuously
    tracked, giving a clean estimate of the commit cadence).

    params.yaml cooccurrence: tolerance = max(frame_tolerance_min,
    gap_multiplier × median gap); frame_tolerance_min also when no estimate."""
    co = config.params().cooccurrence
    default = co.frame_tolerance_min
    counts = tracks_df["temp_id"].value_counts()
    if counts.empty:
        return default
    densest_tid = counts.idxmax()
    fr = np.sort(tracks_df.loc[tracks_df["temp_id"] == densest_tid, "frame_index"].to_numpy())
    if len(fr) < 2:
        return default
    gaps = np.diff(fr)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return default
    # Safety margin (gap_multiplier, 2x by default) for jitter in exactly
    # which frame within a window ends up being the one committed.
    return max(int(co.gap_multiplier * np.median(gaps)), default)


def _min_frame_dist(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    """Smallest gap between any recorded frame of arr_a and any recorded
    frame of arr_b. inf if either is empty."""
    if len(arr_a) == 0 or len(arr_b) == 0:
        return float("inf")
    idx = np.searchsorted(arr_b, arr_a)
    idx_lo = np.clip(idx - 1, 0, len(arr_b) - 1)
    idx_hi = np.clip(idx, 0, len(arr_b) - 1)
    nearest_dist = np.minimum(np.abs(arr_a - arr_b[idx_lo]), np.abs(arr_a - arr_b[idx_hi]))
    return float(nearest_dist.min())


def _frames_cooccur(arr_a: np.ndarray, arr_b: np.ndarray, tol: int) -> bool:
    """True if any recorded frame of arr_a is within `tol` frames of any
    recorded frame of arr_b — i.e. the two temp_ids were actually on screen
    at close to the same moment, not just somewhere within each other's
    overall active-duration interval."""
    return _min_frame_dist(arr_a, arr_b) <= tol


def cosine_resolver(
    tracks_df: pd.DataFrame,
    embed_df: pd.DataFrame,
    kinetic_assignment: dict,       # {temp_id -> AnimalId} — already confirmed
    galleries: dict,                # {"day"|"night": {real_id: np.ndarray}} from Gallery Builder
    conn: sqlite3.Connection,
    gallery_dir: str,
    session_id: str,
    camera_id: str = "cam0",
    dry_run: bool = False,
    foreign_links: list | None = None,   # out: [(other_session_id, other_temp_id, id)]
) -> tuple[dict, dict]:
    """
    Resolve unconfirmed temp_ids via cosine similarity.

    Returns (merged_assignment, cosine_scores): cosine_scores is
    {temp_id -> score}, parallel to the cosine-resolved subset of
    merged_assignment — added for Boundary Rendezvous (cross_camera.py).
    When a synthetic id is minted for a pair, the OTHER session's temp_id is
    appended to foreign_links; reconcile records it in session_assignments
    (identity_record.py) together with this session's own rows.

    Query order per unresolved temp_id:
      1. Flat real_id gallery        — confirmed identities across all sessions
      2. Pose-conditioned real_id gallery — slot-aware, higher discrimination
      3. TempPoseGallery (any camera, other sessions) — cross-session, cross-
         camera temp matching. Cows move freely between the cameras that
         monitor the enclosure, so this step is NOT restricted to the camera
         the current session was recorded on — it checks every camera's
         TempPoseGallery and keeps the best-scoring match.

    If a temp_id matches a real_id (steps 1/2) → assign that real_id.
    If a temp_id matches another temp_id (step 3) that already has a synthetic id
      → assign the same synthetic id.
    If a temp_id matches another temp_id with no identity yet → mint a new
      synthetic id and assign it to both.
    If no match → leave unresolved.

    Also runs backpropagation: if any confirmed real_id in this session was
    previously a synthetic id, all timeline rows are updated.

    Camera mode: embed_df.is_night is per frame, so a temp_id's embeds are
    split by mode and each mode's mean is only ever compared against that
    mode's galleries (a grayscale embedding never meets a colour gallery).
    A temp_id is queryable if at least one of its modes has ≥ min_embeds;
    modes are tried in order of how many embeds the temp_id has in each, and
    the first mode whose 3-step chain produces a match wins. Cosine matches
    reach the galleries later, in commit_galleries(), in every mode with
    enough frames — so a match made in daylight also teaches the night
    gallery that cow.
    """
    section("Cosine Resolver")
    P  = config.params()
    cr = P.cosine_resolver
    gp = P.gallery

    already_resolved = set(kinetic_assignment.keys())
    all_tids = set(tracks_df["temp_id"].unique())
    unresolved_tids = all_tids - already_resolved

    if embed_df.empty:
        log("No embeds available — cosine resolver skipped.")
        return dict(kinetic_assignment), {}

    embeds_by_mode = split_by_modality(embed_df)            # {mode -> rows}
    counts_by_mode = {m: (df.groupby("temp_id")[appearance_quality.QUALITY_COL].sum()
                          if appearance_quality.QUALITY_COL in df.columns
                          else df.groupby("temp_id").size())
                      for m, df in embeds_by_mode.items()}   # effective frames

    def _query_modes(tid) -> list[str]:
        """Modes in which tid has ≥ min_embeds, most embeds first."""
        have = [(int(c.get(tid, 0)), m) for m, c in counts_by_mode.items()]
        return [m for n, m in sorted(have, reverse=True) if n >= cr.min_embeds]

    queryable_tids = [tid for tid in unresolved_tids if _query_modes(tid)]
    log(f"Unresolved temp_ids: {len(unresolved_tids)}  "
        f"Queryable (≥{cr.min_embeds} embeds in at least one camera mode): "
        f"{len(queryable_tids)}")

    # Load pose galleries — both modes
    pose_galleries = {m: PoseGallery.load(gallery_dir, m) for m in MODALITIES}
    temp_galleries = {m: TempPoseGallery.load(gallery_dir, m) for m in MODALITIES}

    cosine_assignment = {}
    # {temp_id -> score}, parallel to cosine_assignment — needed by Boundary Rendezvous
    # (cross_camera.py) for its tier-1 / confidence-gap logic. Each of the
    # four assignment sites below records whichever score produced its match.
    cosine_scores = {}
    # {(other_session_id, other_temp_id) -> id minted for it in THIS run},
    # so two local tracks matching the same foreign track share one synthetic
    # id even under --dry_run (when nothing is written to reid_registry).
    foreign_ids: dict = {}

    # aid -> list of (claim_tid, frame_array) that currently hold this
    # real_id. Seeded from kinetic_assignment so we know which tids are
    # already "occupied" before any cosine matching runs. Keeping claim_tid
    # (not just the frame array) lets a blocked match be diagnosed: which
    # specific already-claimed tid caused the block, and by how many frames.
    tid_frames   = _tid_frame_arrays(tracks_df)
    tid_boxes    = _tid_frame_boxes(tracks_df)
    cooccur_tol  = _estimate_frame_tolerance(tracks_df)
    # above this, same screen location -> handoff artifact, not a real conflict
    IOU_SAME_ANIMAL = config.params().cooccurrence.iou_same_animal
    aid_claims: dict = {}
    for k_tid, k_aid in kinetic_assignment.items():
        aid_claims.setdefault(k_aid, []).append((k_tid, tid_frames.get(k_tid, np.array([]))))

    def _aid_block_reason(tid: int, aid):
        """Returns (blocked: bool, blocking_tid: int|None, min_dist: float|None,
        iou: float|None).
        An aid is only off-limits to `tid` if a temp_id that ALREADY holds it
        was actually on screen at close to the same moment as `tid` AND in a
        different part of the frame (temporal co-occurrence within
        cooccur_tol, spatial IoU below IOU_SAME_ANIMAL) — i.e. they're
        genuinely two different animals. Temporal co-occurrence alone isn't
        enough to conclude that: an ID-handoff (the old track dying and a new
        one starting for the SAME animal) commonly leaves 1-2 frames where
        both ids are recorded, but at essentially the SAME location — that's
        one animal, not two, and should NOT block. Two co-occurring ids in
        DIFFERENT parts of the pen are genuinely different animals and
        SHOULD block. A coarse [min,max] interval check was tried before this
        and over-blocked badly: when one animal dominates most of a session,
        its already-claimed fragments' combined intervals end up covering
        most of the timeline, so any further genuine fragment of the SAME
        animal would get flagged as 'overlapping' even without ever
        literally coinciding with another track in any single frame. If a
        high-cosine candidate still gets blocked here, inspect the reported
        blocking_tid, min_dist and iou directly against raw_tracks rather
        than assuming it's a true conflict."""
        my_frames = tid_frames.get(tid, np.array([]))
        my_boxes  = tid_boxes.get(tid)
        best_block_tid, best_dist, best_other_frames = None, float("inf"), None
        for claim_tid, other in aid_claims.get(aid, []):
            d = _min_frame_dist(my_frames, other)
            if d < best_dist:
                best_dist, best_block_tid, best_other_frames = d, claim_tid, other

        if best_dist > cooccur_tol or best_block_tid is None:
            return False, None, None, None

        # Temporally co-occurring — now check WHERE in the frame, to tell a
        # real conflict (different location) apart from an ID-handoff
        # artifact (same location, briefly two ids for the same box).
        # Must find the EXACT (my_frame, other_frame) pair that achieved
        # best_dist above -- symmetric nearest-neighbor (check both idx-1
        # and idx on both sides), not just the insertion point, or this can
        # pick a worse-matching pair and pull boxes from moments further
        # apart than best_dist implies, understating IoU for genuine
        # same-spot handoffs.
        best_iou = 0.0
        other_boxes = tid_boxes.get(best_block_tid)
        if (my_boxes is not None and len(my_boxes)
                and other_boxes is not None and len(other_boxes)
                and best_other_frames is not None and len(best_other_frames)):
            idx    = np.searchsorted(best_other_frames, my_frames)
            idx_lo = np.clip(idx - 1, 0, len(best_other_frames) - 1)
            idx_hi = np.clip(idx,     0, len(best_other_frames) - 1)
            dist_lo = np.abs(my_frames - best_other_frames[idx_lo])
            dist_hi = np.abs(my_frames - best_other_frames[idx_hi])
            use_lo  = dist_lo <= dist_hi
            chosen_other_frame = np.where(use_lo, best_other_frames[idx_lo], best_other_frames[idx_hi])
            chosen_dist         = np.where(use_lo, dist_lo, dist_hi)
            best_i = int(np.argmin(chosen_dist))
            my_frame_at_best    = int(my_frames[best_i])
            other_frame_at_best = int(chosen_other_frame[best_i])
            box_a = _nearest_box(my_boxes, my_frame_at_best)
            box_b = _nearest_box(other_boxes, other_frame_at_best)
            best_iou = _iou(box_a, box_b) if box_a and box_b else 0.0

        if best_iou >= IOU_SAME_ANIMAL:
            # Same spot, same moment -> one animal, handoff artifact.
            # Not a real conflict, don't block.
            return False, best_block_tid, best_dist, best_iou
        return True, best_block_tid, best_dist, best_iou

    def _aid_blocked_for(tid: int, aid) -> bool:
        return _aid_block_reason(tid, aid)[0]

    def _resolve_real_id(tid, mode: str, query_vec: np.ndarray, near_miss: list) -> bool:
        """Steps 1-2 (confirmed real_id galleries) for one temp_id against one
        camera mode's galleries. Records the assignment and returns True on a
        match; appends to near_miss and returns False otherwise."""
        gallery      = galleries[mode]
        pose_gallery = pose_galleries[mode]

        # ── Step 1: flat real_id gallery ─────────────────────────────────────
        best_aid, best_sim = None, -1.0
        for aid, gvec in gallery.items():
            if _aid_blocked_for(tid, aid):
                continue
            sim = cosine_sim(query_vec, gvec)
            if sim > best_sim:
                best_sim, best_aid = sim, aid

        if best_aid is not None and best_sim >= cr.threshold:
            cosine_assignment[tid] = best_aid
            cosine_scores[tid] = float(best_sim)
            aid_claims.setdefault(best_aid, []).append((tid, tid_frames.get(tid, np.array([]))))
            log(f"  t{tid} → real_id {best_aid}  ({mode} flat cosine={best_sim:.3f})")
            return True

        # Best score seen so far across steps (and modes), for near-miss
        # reporting if this tid ends up unresolved (Step 1's best_sim is
        # always a real number regardless of whether it cleared the threshold).
        near_miss.append(("flat", best_aid, best_sim))

        # ── Step 2: pose-conditioned real_id gallery ─────────────────────────
        p_id, p_cos, _ = pose_gallery.query(
            query_vec, posture=None, facing=None, threshold=cr.threshold
        )
        if p_cos is not None:
            near_miss.append(("pose-slot", p_id, p_cos))
        if p_id is not None and not _aid_blocked_for(tid, p_id):
            cosine_assignment[tid] = p_id
            cosine_scores[tid] = float(p_cos)
            aid_claims.setdefault(p_id, []).append((tid, tid_frames.get(tid, np.array([]))))
            log(f"  t{tid} → real_id {p_id}  ({mode} pose-slot cosine={p_cos:.3f})")
            return True
        return False

    def _resolve_temp(tid, mode: str, query_vec: np.ndarray, near_miss: list) -> bool:
        """Step 3 (TempPoseGallery — inherit or mint a synthetic id) for one
        temp_id against one camera mode's temp gallery."""
        temp_gallery = temp_galleries[mode]

        # ── Step 3: TempPoseGallery — cross-session, any camera ───────────────
        # Cows move between the cameras monitoring the enclosure, so we no
        # longer restrict this lookup to the camera the current session was
        # recorded on (`camera_id`). Instead we query every known camera's
        # TempPoseGallery and keep whichever candidate scores best overall —
        # an actual match (cleared threshold) always wins over a mere
        # near-miss, and among several actual matches the highest cosine wins.
        cam_results = [
            (cam,) + temp_gallery.query_for_camera(
                camera_id       = cam,
                query_embed     = query_vec,
                posture         = None,
                facing          = None,
                threshold       = cr.threshold,
                exclude_session = session_id,   # don't match against self
            )
            for cam in P.camera_ids
        ]

        matched = [r for r in cam_results if r[1] is not None]
        if matched:
            _, match_key, t_cos, _extra = max(matched, key=lambda r: r[2])
        else:
            scored = [r for r in cam_results if r[2] is not None]
            if scored:
                _, match_key, t_cos, _extra = max(scored, key=lambda r: r[2])
            else:
                match_key, t_cos = None, None

        if t_cos is not None:
            near_miss.append(("temp-key", match_key, t_cos))

        if match_key is not None:
            # match_key is a temp_id from ANOTHER session (exclude_session=
            # session_id above). Its temp_id number means nothing in this
            # session's dicts, and it is not a real_id, so look up any id it
            # already has by (session_id, temp_id) — in this run first, then
            # in session_assignments.
            foreign = (match_key.session_id, int(match_key.temp_id))
            matched_sid = foreign_ids.get(foreign)
            if matched_sid is None:
                matched_sid = find_real_id_for_temp(conn, *foreign)

            if matched_sid is not None and not _aid_blocked_for(tid, matched_sid):
                # Inherit the id the matched track already has
                cosine_assignment[tid] = matched_sid
                cosine_scores[tid] = float(t_cos)
                aid_claims.setdefault(matched_sid, []).append((tid, tid_frames.get(tid, np.array([]))))
                log(f"  t{tid} → id {matched_sid}  "
                    f"({mode} temp cosine={t_cos:.3f}, matched {match_key})")
                return True
            else:
                # Mint a new synthetic id for this pair. Only THIS session's
                # temp_id goes into cosine_assignment; the other session's
                # temp_id is recorded as a link row in session_assignments.
                new_synth = mint_synthetic_id(gallery_dir)
                cosine_assignment[tid] = new_synth
                cosine_scores[tid]     = float(t_cos)
                foreign_ids[foreign]   = new_synth
                aid_claims.setdefault(new_synth, []).append((tid, tid_frames.get(tid, np.array([]))))
                if foreign_links is not None:
                    foreign_links.append((match_key.session_id, int(match_key.temp_id), new_synth))
                # Register synthetic id in reid_registry
                if not dry_run:
                    upsert_reid(conn, new_synth, {
                        "match_method":  "synthetic",
                        "first_seen_dt": pd.Timestamp.now().isoformat(),
                    })
                log(f"  t{tid} + {match_key} → NEW synthetic_id {new_synth}  "
                    f"({mode} temp cosine={t_cos:.3f})")
                return True
        return False

    for tid in queryable_tids:
        # One query vector per camera mode the temp_id has enough embeds in,
        # most embeds first. Real-id galleries (steps 1-2) are tried in every
        # mode before any temp-gallery match (step 3), so a confirmed identity
        # in one mode always beats inheriting/minting a synthetic id in another
        # — the same real-before-temp priority as a single-mode session.
        query_vecs = {}
        for mode in _query_modes(tid):
            mode_embeds = embeds_by_mode[mode]
            query_vecs[mode] = compute_mean_embed(mode_embeds[mode_embeds["temp_id"] == tid])

        near_miss: list = []
        resolved_mode = None
        for step_fn in (_resolve_real_id, _resolve_temp):
            for mode, query_vec in query_vecs.items():
                if step_fn(tid, mode, query_vec, near_miss):
                    resolved_mode = mode
                    break
            if resolved_mode is not None:
                break
        if resolved_mode is not None:
            continue

        source, candidate, score = max(near_miss, key=lambda x: x[2])
        if candidate is None:
            log(f"  t{tid}: no match above {cr.threshold} — unresolved  "
                f"(closest: {source} @ {score:.3f}, no candidate id)")
        elif score >= cr.threshold and source in ("flat", "pose-slot"):
            # Cleared the cosine threshold but still ended up unresolved
            # -> it was blocked by the overlap check, not by similarity.
            # Report exactly what blocked it so a real conflict can be
            # told apart from an overlap-check false positive (see the
            # NOTE in _aid_block_reason).
            blocked, block_tid, block_dist, block_iou = _aid_block_reason(tid, candidate)
            if blocked:
                log(f"  t{tid}: scored {score:.3f} ({source}) on real_id {candidate} "
                    f"— BLOCKED: {block_dist:.0f} frames + IoU={block_iou:.2f} vs "
                    f"already-claimed t{block_tid} (tolerance={cooccur_tol}, "
                    f"same-animal IoU≥{IOU_SAME_ANIMAL}). Low IoU here means a different "
                    f"part of the pen -> likely a genuinely different animal.")
            else:
                # Shouldn't normally happen (if not blocked it should have
                # been accepted above) — log plainly rather than guess why.
                log(f"  t{tid}: no match above {cr.threshold} — unresolved  "
                    f"(closest: {source} candidate {candidate} @ {score:.3f})")
        else:
            log(f"  t{tid}: no match above {cr.threshold} — unresolved  "
                f"(closest: {source} candidate {candidate} @ {score:.3f})")

    if not cosine_assignment:
        log("  No cosine matches found.")

    # (No gallery update here: cosine matches reach the flat galleries in
    # commit_galleries(), once every identity step has run.)

    # ── Backpropagate: if any confirmed real_id was previously synthetic ──────
    all_assignments = {**kinetic_assignment, **cosine_assignment}
    for tid, aid in list(all_assignments.items()):
        if not is_synthetic(aid):
            continue
        # This temp_id now has a confirmed real_id via kinetics or manual?
        confirmed = kinetic_assignment.get(tid)
        if confirmed and not is_synthetic(confirmed):
            # Folds the synthetic id's slots in BOTH modes' pose galleries
            # (and saves each) — a synthetic id can have been built up from
            # day and night sightings alike.
            rows = backpropagate_resolution(
                synthetic_id   = aid,
                real_id        = confirmed,
                conn           = conn,
                pose_galleries = pose_galleries,
                gallery_dir    = gallery_dir,
                dry_run        = dry_run,
            )
            log(f"  Backpropagated synthetic_id {aid} → real_id {confirmed} "
                f"({rows} timeline rows updated)")
            all_assignments[tid] = confirmed

    # NOTE: duplicate real_id resolution (multiple temp_ids matched to the same
    # AnimalId — expected now that assigned_aids has been replaced with the
    # temporal-overlap check above) is intentionally NOT done here. A prior
    # "switch healing" pass used to pre-merge duplicates using only raw frame
    # count, with no manual-assignment priority and no audit trail — it ran
    # before resolve_duplicate_assignments (Duplicate Resolver) and silently deleted
    # every duplicate but the single largest fragment, so Duplicate Resolver (which has
    # the correct manual > cosine > kinetic priority, proper tiebreaking, and
    # writes temp_id_merges for traceability) only ever saw what that pass
    # left behind. Duplicate Resolver is the single authoritative merge step — leave
    # every raw match intact here and let it do that job once, correctly.
    merged = {**kinetic_assignment, **cosine_assignment}
    n_real    = sum(1 for v in merged.values() if not is_synthetic(v))
    n_synth   = sum(1 for v in merged.values() if is_synthetic(v))
    log(f"Final assignment: {len(merged)} temp_ids resolved  "
        f"({len(kinetic_assignment)} kinetic, {len(cosine_assignment)} cosine  "
        f"[{n_real} real_id, {n_synth} synthetic])")
    return merged, cosine_scores


# ─────────────────────────────────────────────────────────────────────────────
# Sensor Sequencer (forward-fill to video time grid)
# ─────────────────────────────────────────────────────────────────────────────

def sensor_sequencer(
    tracks_df: pd.DataFrame,
    kinetics_df: pd.DataFrame,
    behavior_df: pd.DataFrame | None,
    assignment: dict,
    bin_minutes: int,                 # params timeline.bin_minutes
) -> pd.DataFrame:
    """
    Forward-fill behavior (~90s) and kinetics (~15min) signals onto the video time grid.

    For each resolved (temp_id → AnimalId) pair:
      - Build a time grid from the video bins
      - Join and forward-fill kinetics deltas and behavior features
      - Return one row per (real_id, window_start_dt) suitable for resolved_cow_timeline

    Returns DataFrame with columns aligned to resolved_cow_timeline schema.
    """
    section("Sensor Sequencer")

    if assignment is None or len(assignment) == 0:
        log("No assignments — sensor sequencer skipped.")
        return pd.DataFrame()

    rows = []

    for tid, aid in assignment.items():
        tid_tracks = tracks_df[tracks_df["temp_id"] == tid].copy()
        if tid_tracks.empty:
            continue

        t_start = tid_tracks["frame_datetime"].min().floor(f"{bin_minutes}min")
        t_end   = tid_tracks["frame_datetime"].max().ceil(f"{bin_minutes}min")
        bins    = pd.date_range(start=t_start, end=t_end, freq=f"{bin_minutes}min")

        # ---- kinetics deltas ----
        kin_animal = kinetics_df[kinetics_df["AnimalId"] == aid].sort_values("datetime")

        kin_bins = []
        for i in range(len(bins) - 1):
            t0, t1 = bins[i], bins[i + 1]
            before = kin_animal[kin_animal["datetime"] < t0]
            after  = kin_animal[kin_animal["datetime"] < t1]
            if before.empty or after.empty:
                kin_bins.append({
                    "window_start_dt": t0,
                    "d_kin_x": np.nan, "d_kin_y": np.nan,
                    "d_kin_z": np.nan, "d_kin_r": np.nan,
                })
                continue
            deltas = {
                "window_start_dt": t0,
                "d_kin_x": max(0, after.iloc[-1]["KineticsCountX"] - before.iloc[-1]["KineticsCountX"]),
                "d_kin_y": max(0, after.iloc[-1]["KineticsCountY"] - before.iloc[-1]["KineticsCountY"]),
                "d_kin_z": max(0, after.iloc[-1]["KineticsCountZ"] - before.iloc[-1]["KineticsCountZ"]),
                "d_kin_r": max(0, after.iloc[-1]["KineticsCountR"] - before.iloc[-1]["KineticsCountR"]),
            }
            kin_bins.append(deltas)

        kin_grid = pd.DataFrame(kin_bins).set_index("window_start_dt")
        # forward-fill any NaN gaps (sensor dropouts)
        kin_grid = kin_grid.ffill()

        # ---- behavior features (forward-fill from ~90s intervals) ----
        beh_cols = ["d_f12", "d_f23", "d_v"]
        beh_grid = pd.DataFrame(index=bins[:-1], columns=beh_cols, dtype=float)
        beh_grid.index.name = "window_start_dt"

        if behavior_df is not None and not behavior_df.empty:
            beh_animal = behavior_df[behavior_df["AnimalId"] == aid].sort_values("datetime")
            if not beh_animal.empty:
                # assign each behavior row to nearest bin, then mean-aggregate
                beh_animal = beh_animal.copy()
                beh_animal["bin"] = pd.cut(
                    beh_animal["datetime"], bins=bins, right=False, labels=bins[:-1]
                )
                beh_animal = beh_animal.dropna(subset=["bin"])
                for b, grp in beh_animal.groupby("bin", observed=True):
                    # compute deltas of each feature over this bin
                    beh_grid.loc[b, "d_f12"] = grp["f_1_2"].diff().abs().sum() if len(grp) > 1 else grp["f_1_2"].iloc[0]
                    beh_grid.loc[b, "d_f23"] = grp["f_2_3"].diff().abs().sum() if len(grp) > 1 else grp["f_2_3"].iloc[0]
                    beh_grid.loc[b, "d_v"]   = grp["v"].diff().abs().sum()      if len(grp) > 1 else grp["v"].iloc[0]
                beh_grid = beh_grid.ffill()

        # ---- merge into one row per bin ----
        for t0 in bins[:-1]:
            krow = kin_grid.loc[t0] if t0 in kin_grid.index else pd.Series(dtype=float)
            brow = beh_grid.loc[t0] if t0 in beh_grid.index else pd.Series(dtype=float)

            sensor_ok = not (krow.isna().all() and brow.isna().all())
            modality_mask = 1 if sensor_ok else 0  # bit 0 = sensor_ok

            rows.append({
                "real_id":         int(aid),
                "window_start_dt": t0.isoformat(),
                "modality_mask":   modality_mask,
                "d_f12":   float(brow.get("d_f12",   np.nan)),
                "d_f23":   float(brow.get("d_f23",   np.nan)),
                "d_v":     float(brow.get("d_v",     np.nan)),
                "d_kin_x": float(krow.get("d_kin_x", np.nan)),
                "d_kin_y": float(krow.get("d_kin_y", np.nan)),
                "d_kin_z": float(krow.get("d_kin_z", np.nan)),
                "d_kin_r": float(krow.get("d_kin_r", np.nan)),
                # vision features filled by future pose extractor (step E)
                "spine_angle": None, "pelvic_tilt": None, "tail_elevation": None,
                "limb_symmetry": None, "head_drop": None, "lying_flag": None,
                "restlessness": None, "kps_coverage": None, "embed_mean": None,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        log(f"Sensor grid: {len(result)} rows for {result['real_id'].nunique()} animals")
    else:
        log("Sensor grid empty.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Timeline Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_timeline(
    timeline_df: pd.DataFrame,
    session_id: str,
    conn: sqlite3.Connection,
    dry_run: bool = False,
) -> None:
    section("Timeline Writer")

    # A run REPLACES this session's rows (re-runs, e.g. assign_identity.py
    # --run_reconcile after a manual fix, used to append a second copy).
    # The delete and the insert commit together, so an aborted run leaves the
    # previous rows in place.
    if not dry_run:
        n_old = conn.execute("SELECT COUNT(*) FROM resolved_cow_timeline WHERE session_id = ?",
                             (session_id,)).fetchone()[0]
        conn.execute("DELETE FROM resolved_cow_timeline WHERE session_id = ?", (session_id,))
        if n_old:
            log(f"Replacing {n_old} rows from an earlier run of this session")

    if timeline_df.empty:
        if not dry_run:
            conn.commit()
        log("Nothing to write.")
        return

    timeline_df = timeline_df.copy()
    timeline_df["session_id"] = session_id

    # Convert NaN embed_mean to None so SQLite stores NULL
    if "embed_mean" in timeline_df.columns:
        timeline_df["embed_mean"] = timeline_df["embed_mean"].where(
            timeline_df["embed_mean"].notna(), None
        )

    if dry_run:
        log(f"[dry_run] Would insert {len(timeline_df)} rows into resolved_cow_timeline")
        print(timeline_df.head(5).to_string())
        return

    cols = [
        "real_id", "session_id", "window_start_dt", "modality_mask",
        "d_f12", "d_f23", "d_v",
        "d_kin_x", "d_kin_y", "d_kin_z", "d_kin_r",
        "spine_angle", "pelvic_tilt", "tail_elevation",
        "limb_symmetry", "head_drop", "lying_flag",
        "restlessness", "kps_coverage", "embed_mean",
        # vision feature extractor columns (added by migrate_timeline_schema)
        "lying_fraction", "posture_transitions", "facing_dominant", "facing_entropy",
    ]
    # only include columns that exist in the df
    cols = [c for c in cols if c in timeline_df.columns]

    ph  = ", ".join("?" * len(cols))
    sql = f"INSERT INTO resolved_cow_timeline ({', '.join(cols)}) VALUES ({ph})"

    def _to_sqlite(v):
        """Coerce a value to a SQLite-safe type."""
        import pandas as _pd
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, _pd.Timestamp):
            return v.isoformat()
        if hasattr(v, "item"):          # numpy scalar → python scalar
            return v.item()
        return v

    inserted = 0
    for _, row in timeline_df.iterrows():
        vals = [_to_sqlite(row.get(c)) for c in cols]
        conn.execute(sql, vals)
        inserted += 1

    conn.commit()
    log(f"Inserted {inserted} rows into resolved_cow_timeline for session '{session_id}'")


# ─────────────────────────────────────────────────────────────────────────────
# Assignment provenance
# ─────────────────────────────────────────────────────────────────────────────

# Which step produced a temp_id's current AnimalId, and how much to trust it
# when two temp_ids in one session claim the same AnimalId: params.yaml
# identity.source_priority, highest trust first (lower rank wins) — operator >
# collar ground truth > cross-camera boundary match > appearance alone.
# identity.protected_sources: sources Boundary Rendezvous may not overwrite.
#
# Same-animal IoU: params.yaml cooccurrence.iou_same_animal, shared with the
# Cosine Resolver — two temp_ids on screen at the same moment with box IoU at
# or above it are one animal whose box briefly had two ids (handoff
# artifact); below it, they are two different animals.


def _pair_conflict(frames_a: np.ndarray, boxes_a, frames_b: np.ndarray, boxes_b,
                   tol: int, iou_same: float) -> tuple:
    """Are temp_ids A and B two different animals? Same rule as the Cosine
    Resolver's _aid_block_reason: they conflict only if they were on screen
    within `tol` frames of each other AND, at that closest moment, their
    boxes overlap less than `iou_same` (different spots in the pen).

    Returns (conflict: bool, min_frame_dist: float, iou: float|None).
    """
    d = _min_frame_dist(frames_a, frames_b)
    if d > tol:
        return False, d, None
    iou = 0.0
    if boxes_a is not None and len(boxes_a) and boxes_b is not None and len(boxes_b):
        idx    = np.searchsorted(frames_b, frames_a)
        idx_lo = np.clip(idx - 1, 0, len(frames_b) - 1)
        idx_hi = np.clip(idx,     0, len(frames_b) - 1)
        dist_lo = np.abs(frames_a - frames_b[idx_lo])
        dist_hi = np.abs(frames_a - frames_b[idx_hi])
        use_lo  = dist_lo <= dist_hi
        other   = np.where(use_lo, frames_b[idx_lo], frames_b[idx_hi])
        i       = int(np.argmin(np.where(use_lo, dist_lo, dist_hi)))
        box_a = _nearest_box(boxes_a, int(frames_a[i]))
        box_b = _nearest_box(boxes_b, int(other[i]))
        iou = _iou(box_a, box_b) if box_a and box_b else 0.0
    return iou < iou_same, d, iou


def find_real_id_for_temp(conn: sqlite3.Connection, session_id: str,
                          temp_id: int):
    """The id (real or synthetic) on record for a temp_id of a given session
    (session_assignments — identity_record.lookup). None if none.

    temp_ids are per-session counters, so a temp_id is only meaningful
    together with its session_id, and it is never a real_id itself.
    """
    return identity_record.lookup(conn, session_id, temp_id)


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_duplicate_assignments(
    assignment: dict,               # {temp_id -> real_id}  — may contain duplicates
    provenance: dict,               # {temp_id -> source}, see identity.source_priority
    tracks_df: pd.DataFrame,        # for frame counts and co-occurrence
    conn: sqlite3.Connection,
    session_id: str,
    dry_run: bool = False,
) -> tuple[dict, dict]:
    """
    Resolve AnimalIds claimed by more than one temp_id in this session.

    Each duplicate group is ranked by:
      1. Source of the id (params identity.source_priority): manual > kinetic >
         boundary_rendezvous > cosine
      2. More frames in raw_tracks
      3. Lower temp_id
    The top-ranked temp_id is the winner and keeps the id.

    Every other temp_id in the group is then checked against the winner and
    any fragments already merged into it:
      - Never on screen at the same time in a different spot -> the same
        animal split across tracker ids. Merged: remapped to the winner,
        recorded in temp_id_merges with action='merged'.
      - On screen at the same time in a different spot (_pair_conflict) ->
        a different animal; one of the two ids is wrong. The lower-ranked
        temp_id loses the id and stays unassigned, recorded with
        action='unassigned_cooccurring'. It is NOT added to merge_map, so
        its rows are never pooled into the winner's.

    Returns (clean_assignment, merge_map): every real_id appears at most
    once in clean_assignment; merge_map is {loser_tid -> winner_tid} for
    merged fragments only, for downstream steps to pool embed/track rows.
    """
    from collections import defaultdict
    aid_to_tids = defaultdict(list)
    for tid, aid in assignment.items():
        aid_to_tids[aid].append(tid)

    # This run's merge records replace any from an earlier run of the session
    # — including when this run finds no duplicates at all.
    if not dry_run:
        conn.execute("DELETE FROM temp_id_merges WHERE session_id = ?", (session_id,))
        conn.commit()

    duplicates = {aid: tids for aid, tids in aid_to_tids.items() if len(tids) > 1}
    if not duplicates:
        return assignment, {}

    section("Duplicate Resolver")

    frame_counts = tracks_df.groupby("temp_id")["frame_index"].nunique().to_dict()
    tid_frames   = _tid_frame_arrays(tracks_df)
    tid_boxes    = _tid_frame_boxes(tracks_df)
    cooccur_tol  = _estimate_frame_tolerance(tracks_df)
    P            = config.params()
    priority     = P.source_priority          # {source -> rank}, lower wins
    iou_same     = P.cooccurrence.iou_same_animal

    def source(tid):
        return provenance.get(tid, "unknown")

    def rank(tid):
        """Lower sorts first (wins)."""
        return (priority.get(source(tid), len(priority)),
                -frame_counts.get(tid, 0), tid)

    def describe(tid):
        return f"temp_id {tid} ({source(tid)}, {frame_counts.get(tid, 0)} frames)"

    def record(winner, loser, aid, action):
        if dry_run:
            return
        conn.execute("""
            INSERT INTO temp_id_merges
                (session_id, winner_tid, loser_tid, real_id,
                 winner_reason, loser_reason, merged_dt, action)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(session_id, loser_tid) DO UPDATE SET
                winner_tid    = excluded.winner_tid,
                real_id       = excluded.real_id,
                winner_reason = excluded.winner_reason,
                loser_reason  = excluded.loser_reason,
                merged_dt     = excluded.merged_dt,
                action        = excluded.action
        """, (session_id, winner, loser, aid,
              source(winner), source(loser), ts, action))

    merge_map  = {}   # {loser_tid -> winner_tid}, merged fragments only
    unassigned = []
    clean = dict(assignment)
    ts = pd.Timestamp.now().isoformat()

    for aid, tids in sorted(duplicates.items()):
        ranked = sorted(tids, key=rank)
        winner = ranked[0]
        kept   = [winner]   # winner + fragments merged into it = one animal

        log(f"  AnimalId {aid} → duplicate temp_ids {sorted(tids)}")
        log(f"    winner: {describe(winner)}")

        for tid in ranked[1:]:
            conflict_with = None
            for k in kept:
                conflict, dist, iou = _pair_conflict(
                    tid_frames.get(tid, np.array([])), tid_boxes.get(tid),
                    tid_frames.get(k,   np.array([])), tid_boxes.get(k),
                    cooccur_tol, iou_same,
                )
                if conflict:
                    conflict_with = (k, dist, iou)
                    break

            del clean[tid]
            if conflict_with:
                k, dist, iou = conflict_with
                iou_str = f"{iou:.2f}" if iou is not None else "n/a"
                log(f"    UNASSIGN: {describe(tid)} — on screen with temp_id {k} "
                    f"({int(dist)} frames apart, IoU {iou_str}), so a different "
                    f"animal; {source(winner)} outranks {source(tid)}")
                unassigned.append(tid)
                record(winner, tid, aid, "unassigned_cooccurring")
            else:
                log(f"    merge:  {describe(tid)} → remapped to temp_id {winner}")
                merge_map[tid] = winner
                kept.append(tid)
                record(winner, tid, aid, "merged")

    if not dry_run:
        conn.commit()

    log(f"  Resolved {len(merge_map) + len(unassigned)} duplicate(s): "
        f"{len(merge_map)} merged, {len(unassigned)} unassigned → "
        f"{len(clean)} unique assignments")
    return clean, merge_map


# ─────────────────────────────────────────────────────────────────────────────
# Manual Merge
# ─────────────────────────────────────────────────────────────────────────────

def load_manual_assignments(conn: sqlite3.Connection, session_id: str) -> dict:
    """
    Load manual identity assignments for a session from manual_assignments table.
    Returns {temp_id (int) -> real_id (int)}.
    """
    rows = conn.execute(
        "SELECT temp_id, real_id FROM manual_assignments WHERE session_id = ?",
        (session_id,)
    ).fetchall()
    if not rows:
        return {}
    result = {int(r[0]): int(r[1]) for r in rows}
    log(f"Manual assignments loaded: {len(result)} "
        f"({', '.join(f't{t}→{a}' for t,a in sorted(result.items()))})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(session: str, dry_run: bool = False, bypass_upload_check: bool = False) -> None:
    """
    Core pipeline entry point. Callable from CLI (via main()) or directly by
    track_and_dump.py / assign_identity.py. Every threshold comes from
    config.params() — the caller only says which session and the two one-shot
    safety switches. Tracks are loaded from the DB.
    """
    args = argparse.Namespace(session=session, dry_run=dry_run,
                              bypass_upload_check=bypass_upload_check)
    P = config.params()
    _, params_hash = config.params_snapshot()

    section("reconcile.py — ReID pipeline")
    log(f"session_id  : {args.session}")
    log("All data paths (db, kinetics, behavior, gallery, parquet) resolved via drive_manager")
    log(f"params      : {config.params_path()}  (sha256 {params_hash})")
    log(f"dry_run     : {args.dry_run}")

    # ── Drive setup ───────────────────────────────────────────────────────────
    bypass = args.bypass_upload_check
    dm = DriveManager(bypass=bypass, caller=__file__)

    # Dirty-flag check: reconcile reads from DB and gallery — both must be clean
    try:
        dm.check_flag("db")
        dm.check_flag("gallery")
        dm.check_flag("parquet")
    except DriveNotSyncedError as e:
        log(f"ERROR: {e}")
        log("Use --bypass_upload_check to proceed with current Drive state.")
        return

    # Pull the canonical DB from Drive before opening
    try:
        db_local = dm.pull_db(allow_stale=False)
    except DriveUnavailableError as e:
        log(f"ERROR pulling DB: {e}")
        return

    # Pull gallery files from Drive
    dm.pull_gallery()

    # Resolve gallery_dir to the local copy
    gallery_dir = str(dm.get_gallery_dir())
    log(f"Using local gallery_dir: {gallery_dir}")

    # ── database ──────────────────────────────────────────────────────────────
    conn = init_db(str(db_local))
    migrate_timeline_schema(conn)
    gallery_ledger.migrate_schema(conn)
    identity_record.ensure(conn, log=log)


    # ── load tracks from SQLite ───────────────────────────────────────────────
    log("Loading tracks from SQLite...")
    tracks_df = pd.read_sql(
        "SELECT frame_index, frame_datetime, temp_id, cx, cy, x1, y1, x2, y2, "
        f"{raw_tracks_optional_select(conn, 'frame_time_sec')}, "
        f"{raw_tracks_is_night_select(conn)} "
        "FROM raw_tracks WHERE session_id = ? ORDER BY frame_index",
        conn, params=(args.session,), parse_dates=["frame_datetime"],
    )
    if tracks_df.empty:
        log(f"No rows in raw_tracks for session '{args.session}' — aborting.")
        conn.close()
        return
    log(f"  {len(tracks_df)} rows, {tracks_df['temp_id'].nunique()} temp_ids, "
        f"frames {tracks_df['frame_index'].min()}–{tracks_df['frame_index'].max()}")


    # camera mode (day colour / night IR) — per row, detected by track_and_dump.py
    tracks_df, fallback_is_night = resolve_camera_mode(tracks_df)
    night_frac = float(tracks_df["is_night"].mean())
    log(f"Camera mode: {night_frac*100:.1f}% of track rows night/IR "
        f"({'mixed — both galleries' if 0 < night_frac < 1 else modality_of(night_frac > 0.5) + ' only'})")

    # ── Tracklet stitching, then noise filter — before every identity step ────
    # Manual assignments are keyed by raw temp_ids (what display_tracks.py
    # shows the operator); they veto conflicting stitches, then follow the rename.
    manual_assignment = load_manual_assignments(conn, args.session)
    tracks_df, stitch_map = stitch_tracklets(tracks_df, manual_assignment, conn,
                                             args.session, args.dry_run)
    manual_assignment = {stitch_map.get(t, t): aid for t, aid in manual_assignment.items()}
    tracks_df, noise_tids = noise_filter(tracks_df, exempt_tids=set(manual_assignment))
    if tracks_df.empty:
        log("Every temp_id was filtered as noise — nothing to reconcile.")
        conn.close()
        return

    # register the session if missing (video_path and is_night are written by
    # track_and_dump.py; this is just a fallback INSERT OR IGNORE for sessions
    # reconcile.py is run against standalone) and load camera_id
    upsert_session(conn, args.session, "", night_frac > 0.5)
    session_row = get_session(conn, args.session)
    camera_id   = (session_row.get("camera_id") or "cam0") if session_row else "cam0"
    log(f"camera_id={camera_id}")

    # appearance-quality weight per detection (box size, frame-edge cut-off),
    # used by every step that pools embeddings — never by positions/kinetics
    frame_size = ((session_row or {}).get("frame_w"), (session_row or {}).get("frame_h"))
    tracks_df[appearance_quality.QUALITY_COL] = appearance_quality.frame_weights(
        tracks_df, *frame_size)
    q = tracks_df[appearance_quality.QUALITY_COL]
    log(f"Appearance quality: mean weight {q.mean():.2f}, {(q == 0).mean() * 100:.1f}% "
        f"of detections at 0, {(q == 1).mean() * 100:.1f}% at full"
        + ("" if all(frame_size) else "  (frame size unknown — edge check skipped)"))

    # ── load kinetics + behavior — auto-discover from Drive unless --kinetics given ──
    session_start = pd.to_datetime(
        session_row.get("start_dt") if session_row else None, errors="coerce"
    )
    session_end = pd.to_datetime(
        session_row.get("end_dt") if session_row else None, errors="coerce"
    )
    # Fall back to track timestamps if session row has no datetimes
    if pd.isnull(session_start):
        session_start = tracks_df["frame_datetime"].min()
    if pd.isnull(session_end):
        session_end = tracks_df["frame_datetime"].max()

    log("Auto-discovering kinetics CSVs from Drive...")
    kinetics_df = dm.load_collar_data(session_start.to_pydatetime(),
                                      session_end.to_pydatetime(), kind="kinetic")
    if kinetics_df is None or kinetics_df.empty:
        log("WARNING: no kinetics data found for this session window — "
            "kinetic matching will be skipped.")
        kinetics_df = pd.DataFrame(columns=["AnimalId","datetime",
                                             "KineticsCountX","KineticsCountY",
                                             "KineticsCountZ","KineticsCountR"])
    else:
        log(f"  {len(kinetics_df)} rows, animals: "
            f"{sorted(kinetics_df['AnimalId'].unique().tolist())}")

    # behavior — auto-discover from Drive
    log("Auto-discovering behavior CSVs from Drive...")
    behavior_df = dm.load_collar_data(session_start.to_pydatetime(),
                                      session_end.to_pydatetime(), kind="behavior")
    if behavior_df is None or behavior_df.empty:
        log("No behavior data found for this session window — d_f12/f23/v will be NaN")
        behavior_df = None
    else:
        log(f"  {len(behavior_df)} rows")

    # ── Kinetic Matcher ───────────────────────────────────────────────────────
    kinetic_assignment, kinetic_corr = kinetic_matcher(tracks_df, kinetics_df)

    # provenance: {temp_id -> source of its current id}. Every step that sets
    # or changes an id records itself here, so the Duplicate Resolver can rank
    # competing claims by identity.source_priority and the audit trail is accurate.
    provenance = {tid: "kinetic" for tid in kinetic_assignment}

    # ── Manual Merge ──────────────────────────────────────────────────────────
    # (manual_assignment was loaded before the noise filter, which exempts it)
    if manual_assignment:
        conflicts = set(kinetic_assignment) & set(manual_assignment)
        if conflicts:
            log(f"  Note: manual overrides kinetic for temp_ids: {sorted(conflicts)}")
        kinetic_assignment.update(manual_assignment)
        for tid in manual_assignment:
            provenance[tid] = "manual"

    # ── load embeds from parquet (always via drive_manager — never a CLI override) ──
    # NOTE: embeds are written as multi-part files (embeds_part000.parquet, ...)
    # by track_and_dump.py's EmbedWriter, so we pass the session directory —
    # load_embeds_for_session() globs *_part*.parquet within it.
    embed_parquet_path = str(dm.get_session_dir(args.session))
    embed_df = load_embeds_for_session(tracks_df, embed_parquet_path, args.session,
                                       tid_map=stitch_map)

    # ── Gallery preview — in memory, for the Cosine Resolver's queries ───────
    galleries = preview_galleries(embed_df, kinetic_assignment, provenance, gallery_dir)

    # ── Cosine Resolver ───────────────────────────────────────────────────────
    merge_map = {}   # populated by Duplicate Resolver below
    foreign_links: list = []   # other sessions' temp_ids tied to a synthetic id this run
    full_assignment, cosine_scores = cosine_resolver(
        tracks_df          = tracks_df,
        embed_df           = embed_df,
        kinetic_assignment = kinetic_assignment,
        galleries          = galleries,
        conn               = conn,
        gallery_dir        = gallery_dir,
        session_id         = args.session,
        camera_id          = camera_id,
        dry_run            = args.dry_run,
        foreign_links      = foreign_links,
    )

    # ── Boundary Rendezvous — triggered cross-camera reconcile ───────────────
    # See cross_camera.py. Matches on the flat embedding (available now,
    # unlike the pose-conditioned one, which isn't computed until
    # run_vision_features runs later) so a boundary-triggered resolution
    # still makes it into THIS session's own Sensor Sequencer / timeline output.
    for tid in full_assignment:            # anything new came from the Cosine Resolver
        provenance.setdefault(tid, "cosine")

    from cross_camera import run_boundary_rendezvous, CrossCameraConfig, build_confidence_dict
    c5_confidence = build_confidence_dict(
        kinetic_corr=kinetic_corr,       # {temp_id -> Pearson r} — NEW, from Kinetic Matcher
        cosine_scores=cosine_scores,     # {temp_id -> cosine}    — NEW, from Cosine Resolver
    )
    before_boundary = dict(full_assignment)   # run_boundary_rendezvous mutates in place
    full_assignment, c5_log = run_boundary_rendezvous(
        conn=conn, session_id=args.session, camera_id=camera_id,
        tracks_df=tracks_df, embed_df=embed_df,
        assignment=full_assignment, confidence=c5_confidence,
        config=CrossCameraConfig.from_params(P), dry_run=args.dry_run,
        protected_tids={t for t, s in provenance.items()
                        if s in P.identity.protected_sources},
    )
    for line in c5_log:
        log(line)
    for tid, aid in full_assignment.items():
        if before_boundary.get(tid) != aid:
            provenance[tid] = "boundary_rendezvous"

    # ── Duplicate Resolver — resolve duplicates across all assignments (kinetic+manual+cosine)
    full_assignment, merge_map = resolve_duplicate_assignments(
        assignment   = full_assignment,
        provenance   = provenance,
        tracks_df    = tracks_df,
        conn         = conn,
        session_id   = args.session,
        dry_run      = args.dry_run,
    )

    # ── Gallery commit — the run's single flat-gallery update ────────────────
    # Merged losers' embeds are pooled under their winner first, so the update
    # uses every sighting of the animal.
    embed_df_final = embed_df
    if merge_map and not embed_df.empty:
        embed_df_final = embed_df.copy()
        embed_df_final["temp_id"] = embed_df_final["temp_id"].replace(merge_map)
    _, cosine_ok = commit_galleries(embed_df_final, full_assignment, provenance, conn,
                                    gallery_dir, session_id=args.session, dry_run=args.dry_run)

    # ── Identity record — session_assignments (identity_record.py) ───────────
    # raw temp_id → final temp_id (stitching, then Duplicate Resolver merges).
    # Everything that reads raw temp_ids from storage needs it.
    raw_to_final = compose_tid_maps(stitch_map, merge_map)
    # Replaces every row this session wrote before. display_tracks.py shows
    # raw temp_ids, so every raw fragment of an identified track gets a row.
    n_rows = identity_record.write_session(
        conn, args.session, full_assignment, provenance, raw_to_final,
        links=foreign_links, dry_run=args.dry_run)
    log(f"session_assignments: {n_rows} row(s) for this session"
        + (f" ({len(foreign_links)} link(s) to other sessions)" if foreign_links else "")
        + (" [dry_run]" if args.dry_run else ""))
    # match_method = best source that has ever identified the cow (upgrade-only)
    best_source: dict = {}
    for tid, aid in full_assignment.items():
        src = provenance.get(tid, "cosine")
        if aid not in best_source or (identity_record.method_rank(src)
                                      < identity_record.method_rank(best_source[aid])):
            best_source[aid] = src
    for rid, old, new in identity_record.upgrade_match_methods(conn, best_source, args.dry_run):
        log(f"  match_method AnimalId {rid}: {old or '(new)'} → {new}")

    # Duplicate Resolver deletes loser temp_ids from full_assignment (keeps only
    # {winner_tid: real_id}) — tracks_df still carries the losers' rows under
    # their original temp_id. Without this remap, sensor_sequencer's
    # per-row assignment.get(temp_id) lookup would silently treat every
    # merged-but-losing fragment as unassigned, even though Duplicate Resolver already
    # decided it's the same animal. Mirrors the embed_df remap above.
    if merge_map:
        tracks_df = tracks_df.copy()
        tracks_df["temp_id"] = tracks_df["temp_id"].replace(merge_map)

    # ── Sensor Sequencer + Vision Feature Extraction ─────────────────────────────────────────────
    timeline_df = sensor_sequencer(
        tracks_df   = tracks_df,
        kinetics_df = kinetics_df,
        behavior_df = behavior_df,
        assignment  = full_assignment,
        bin_minutes = P.timeline.bin_minutes,
    )
    if not timeline_df.empty:
        timeline_df["session_id"] = args.session

    # ── Vision Feature Extraction ─────────────────────────────────────────────
    # Called even with an empty timeline: it still has to replace this
    # session's pose-gallery contributions and temp-gallery entries from
    # earlier runs (it returns at once otherwise).
    timeline_df = run_vision_features(
        session_id    = args.session,
        timeline_df   = timeline_df,
        conn          = conn,
        assignment    = full_assignment,
        provenance    = provenance,   # {temp_id -> source} — gallery contribution ledger weighting
        cosine_ok     = cosine_ok,    # cosine ids that passed the gallery margin gate (None = gate off)
        fallback_is_night = fallback_is_night,   # only for NULL (legacy) rows
        exclude_tids  = noise_tids,
        frame_size    = frame_size,
        camera_id     = session_row.get("camera_id", "cam0") if session_row else "cam0",
        # kps is written as multi-part files (kps_part000.parquet, ...) by
        # track_and_dump.py's KpsWriter, same as embeds — pass the session
        # directory so extractor.py's glob-based loader can find them.
        # (dm.get_parquet_path(session, "kps") returns a single hardcoded
        # "kps.parquet" filename that never actually exists — see embeds
        # a few lines up for the correct pattern.)
        kps_parquet   = str(dm.get_session_dir(args.session)),
        embed_parquet = embed_parquet_path,
        gallery_dir   = gallery_dir,
        dry_run       = args.dry_run,
        merge_map     = raw_to_final,   # stitching + Duplicate Resolver, raw → final
    )

    # ── Timeline Writer — write to DB ─────────────────────────────────────────
    write_timeline(timeline_df, args.session, conn, dry_run=args.dry_run)

    # ── summary ───────────────────────────────────────────────────────────────
    section("Summary")
    if merge_map:
        log(f"Merged switches : {len(merge_map)} "
            f"({', '.join(f't{l}→t{w}' for l,w in sorted(merge_map.items()))})")
    by_source = {}
    for tid in full_assignment:
        src = provenance.get(tid, "cosine")
        by_source[src] = by_source.get(src, 0) + 1
    for src in P.identity.source_priority:
        log(f"{src + ' ids':<16}: {by_source.get(src, 0)}")
    log(f"Total resolved  : {len(full_assignment)} temp_ids → AnimalId")
    unresolved = set(tracks_df["temp_id"].unique()) - set(full_assignment.keys())
    if unresolved:
        log(f"Unresolved      : {sorted(int(t) for t in unresolved)}")
    log(f"Timeline rows   : {len(timeline_df)}")
    if args.dry_run:
        log("dry_run=True — no data written to DB or gallery files")

    # ── Upload updated files to Drive ─────────────────────────────────────────
    if not args.dry_run:
        section("Drive sync")
        # DB
        dm.sync_db(session_id=args.session)
        # Reproducibility: the exact params.yaml this run used
        config.save_params_snapshot(dm, args.session, "reconcile")
        log(f"params snapshot → outputs/{args.session}/params_reconcile.yaml")
        # Gallery .npy files — BOTH modalities. A session can span the
        # camera's colour/IR switch and update both; backprop folds synthetic
        # ids in both pose galleries; and the temp galleries are purged of
        # this session's old entries in both. Uploading only the files this
        # session's majority mode suggests would leave the rest local-only,
        # and the next run's pull_gallery() would overwrite them. Files this
        # run didn't touch are byte-identical to what was just pulled.
        gallery_files = [f"{prefix}_{m}.npy"
                         for m in MODALITIES
                         for prefix in ("gallery", "gallery_pose", "temp_gallery_pose")]
        gallery_files.append("synthetic_id_counter.npy")
        from drive_manager import LOCAL_GALLERY_DIR, DRIVE_GALLERY_PREFIX
        for fname in gallery_files:
            local_f = LOCAL_GALLERY_DIR / fname
            if local_f.exists():
                dm.write_file(local_f, f"{DRIVE_GALLERY_PREFIX}/{fname}", flag="gallery")

    conn.close()
    log("Done.")


def main() -> None:
    args = parse_args()
    config.init(args.params)
    run(args.session, dry_run=args.dry_run,
        bypass_upload_check=args.bypass_upload_check)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
#
# python3 reconcile.py --session cam33_6_S20260616100001_E20260616110000
#   thresholds: config/params.yaml   (sweep: --params config/params_strict.yaml)
#
# Dry run (no DB writes):
#   --dry_run
#
# After running, validate visually:
#   python3 display_tracks.py \
#     --video      ~/thesis_workspace/raw_data/calving/refet_33_S20241221070000_E20241221080000.mp4 \
#     --session_id cam33_6_S20260616100001_E20260616110000
#   (overlays such as draw_pose: config/cli.yaml → display_tracks:)
#
# Pipeline notes:
#   - Run once per session after track_and_dump.py completes
#   - DB, kinetics, behavior, gallery, and parquet are all resolved through
#     drive_manager.py — there is no local-path override for any of them
#   - Gallery files accumulate across sessions — more sessions = better cosine matching
#   - Kinetic matching requires kinetic_match.min_active_bins active bins
#     (default 3 × 15 min → 45+ minutes of video)
#   - Cosine resolver is a no-op on first run (empty gallery) — by design
#   - Vision features (spine_angle etc.) will be NULL until pose extractor is added