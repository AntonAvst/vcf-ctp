#!/usr/bin/env python3
"""
reconcile.py — Post-processing pipeline for identity resolution and feature extraction.

Runs after track_and_dump.py. Reads raw_tracks + collar_signals from SQLite, resolves
real_id for every temp_id in a session, and writes to resolved_cow_timeline.

Steps (in order):
  A. Kinetic matcher    — bbox centroid speed ↔ ΔKineticsCountR (Pearson r)
  B. Gallery builder    — group embeds by confirmed AnimalId → EMA mean → day/night galleries
  C. Cosine resolver    — heal temp_id switches, cross-video continuity
  D. Sensor sequencer   — forward-fill behavior/kinetics to video time grid
  E. Write output       — resolved_cow_timeline rows

Usage:
    python3 reconcile.py \\
        --db        calving_project.db \\
        --session   session_001 \\
        --kinetics  kinetic_data_6366_7507_7513.csv \\
        --tracks    tracks.csv \\
        [--gallery_dir   ./reid_gallery] \\
        [--embed_parquet session_001_embeds.parquet] \\
        [--corr_threshold 0.7] \\
        [--min_active_bins 3] \\
        [--cosine_threshold 0.75] \\
        [--ema_alpha 0.15] \\
        [--dry_run]

Requirements: pip install pandas numpy scipy pyarrow
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import product

from vision_features import run_vision_features, migrate_timeline_schema
from vision_features.gallery import (
    TempPoseGallery, TempKey, PoseGallery,
    mint_synthetic_id, is_synthetic, backpropagate_resolution,
)


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
    ap.add_argument("--db",       required=True, help="Path to calving_project.db (SQLite)")
    ap.add_argument("--session",  required=True, help="session_id to process (from video_sessions)")
    ap.add_argument("--kinetics", required=True, help="kinetic_data_*.csv for this session's animals")
    # --- optional paths ---
    ap.add_argument("--gallery_dir",   default="./reid_gallery",
                    help="Directory containing gallery_day.npy / gallery_night.npy (default: ./reid_gallery)")
    ap.add_argument("--embed_parquet", default="",
                    help="Parquet file with embed[128] columns keyed by (session_id, frame_index, temp_id). "
                         "If omitted, embeds are read from tracks.csv 'embed' column.")

    # --- step A: kinetic matching ---
    ap.add_argument("--corr_threshold",   type=float, default=0.7,
                    help="Min Pearson r for kinetic match (default: 0.7)")
    ap.add_argument("--min_active_bins",  type=int,   default=3,
                    help="Min active kinetics bins required (default: 3)")
    ap.add_argument("--min_temp_id_frames", type=float, default=0.10,
                    help="Min fraction of frames a temp_id must appear in (default: 0.10)")
    ap.add_argument("--activity_pct",     type=float, default=0.25,
                    help="Percentile threshold for 'active' kinetics bins (default: 0.25)")
    ap.add_argument("--bin_minutes",      type=int,   default=15,
                    help="Kinetics bin width in minutes (default: 15)")

    # --- step B: gallery builder ---
    ap.add_argument("--ema_alpha",       type=float, default=0.15,
                    help="EMA decay for gallery update — α for kinetic-confirmed, α/2 for cosine-only (default: 0.15)")
    ap.add_argument("--min_embeds_gallery", type=int, default=10,
                    help="Min embed rows per temp_id to contribute to gallery (default: 10)")

    # --- step C: cosine resolver ---
    ap.add_argument("--cosine_threshold",    type=float, default=0.75,
                    help="Min cosine similarity for cross-video / switch-healing match (default: 0.75)")
    ap.add_argument("--cosine_min_embeds",   type=int,   default=5,
                    help="Min embed rows for a temp_id to be queried via cosine (default: 5)")

    # --- misc ---
    ap.add_argument("--dry_run", action="store_true",
                    help="Run all steps but do not write to the database")
    ap.add_argument("--verbose", action="store_true",
                    help="Print extra diagnostic output")

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
    is_night        INTEGER DEFAULT 0   -- 0=day, 1=night (auto-detected)
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
    kps_kconf      TEXT
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
    known_temp_ids        TEXT,   -- JSON list of {session_id, temp_id} objects
    first_seen_dt         TEXT,
    match_method          TEXT    -- 'kinetic' | 'cosine_day' | 'cosine_night'
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

CREATE TABLE IF NOT EXISTS temp_id_merges (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    winner_tid    INTEGER NOT NULL,   -- temp_id that survives
    loser_tid     INTEGER NOT NULL,   -- temp_id that was remapped to winner
    real_id       INTEGER NOT NULL,   -- the shared AnimalId
    winner_reason TEXT,               -- 'manual' | 'more_frames' | 'kinetic'
    loser_reason  TEXT,               -- why the loser was assigned this animal
    merged_dt     TEXT,
    UNIQUE(session_id, loser_tid)     -- a loser can only be merged once per session
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
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
    """Register session if it doesn't exist yet (minimal fields)."""
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
# Detect is_night from tracks.csv (no video needed)
# ─────────────────────────────────────────────────────────────────────────────

def detect_is_night_from_tracks(tracks_df: pd.DataFrame) -> bool:
    """
    Heuristic: night/IR footage has near-zero variance between colour channels.
    Since we don't have pixel data here, fall back to time-of-day from frame_datetime.
    Night = any session where >50% of frames fall between 20:00 and 06:00 local.
    (Proper version uses per-channel variance on sampled frames — see architecture doc.)
    """
    if "frame_datetime" not in tracks_df.columns:
        return False
    dt_col = pd.to_datetime(tracks_df["frame_datetime"], errors="coerce").dropna()
    if dt_col.empty:
        return False
    hours = dt_col.dt.hour
    night_mask = (hours >= 20) | (hours < 6)
    return bool(night_mask.mean() > 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Step A — Kinetic Matcher (wraps match_identity.py logic inline)
# ─────────────────────────────────────────────────────────────────────────────

def _camera_displacement_per_bin(tracks: pd.DataFrame,
                                  bins: pd.DatetimeIndex) -> pd.DataFrame:
    df = tracks.sort_values(["temp_id", "frame_datetime"])
    df["bin"] = pd.cut(df["frame_datetime"], bins=bins, right=False, labels=bins[:-1])
    df = df.dropna(subset=["bin"])
    # cast Categorical bin labels → Timestamp so merge with kin_delta works
    df["bin"] = df["bin"].astype("datetime64[ns]")
    rows = []
    for (tid, b), grp in df.groupby(["temp_id", "bin"], observed=True):
        grp = grp.sort_values("frame_datetime")
        dx = grp["cx"].diff().abs()
        dy = grp["cy"].diff().abs()
        disp = np.sqrt(dx**2 + dy**2).sum()
        rows.append({"bin": b, "temp_id": tid, "displacement": disp})
    return pd.DataFrame(rows)


def _kinetics_delta_per_bin(kinetics: pd.DataFrame,
                             bins: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for aid, grp in kinetics.groupby("AnimalId"):
        grp = grp.sort_values("datetime")
        for i in range(len(bins) - 1):
            t0, t1 = bins[i], bins[i + 1]
            before = grp[grp["datetime"] < t0]["KineticsCountR"]
            after  = grp[grp["datetime"] < t1]["KineticsCountR"]
            if before.empty or after.empty:
                continue
            delta = after.iloc[-1] - before.iloc[-1]
            if delta < 0:
                delta = 0
            rows.append({"bin": t0, "AnimalId": aid, "delta": delta})
    return pd.DataFrame(rows)


def _compute_scores(cam_disp: pd.DataFrame, kin_delta: pd.DataFrame,
                    activity_pct: float, min_active_bins: int) -> pd.DataFrame:
    results = []
    for tid, aid in product(cam_disp["temp_id"].unique(), kin_delta["AnimalId"].unique()):
        cam = cam_disp[cam_disp["temp_id"] == tid][["bin", "displacement"]]
        kin = kin_delta[kin_delta["AnimalId"] == aid][["bin", "delta"]]
        merged = cam.merge(kin, on="bin", how="inner")
        if merged.empty:
            continue
        thresh = merged["delta"].quantile(activity_pct)
        active = merged[merged["delta"] >= thresh]
        n = len(active)
        if n < max(min_active_bins, 2):
            results.append({"temp_id": tid, "AnimalId": aid,
                             "correlation": np.nan, "n_bins": n,
                             "p_value": np.nan, "note": f"only {n} active bins"})
            continue
        if active["displacement"].std() == 0 or active["delta"].std() == 0:
            results.append({"temp_id": tid, "AnimalId": aid,
                             "correlation": np.nan, "n_bins": n,
                             "p_value": np.nan, "note": "zero variance"})
            continue
        r, p = pearsonr(active["displacement"], active["delta"])
        results.append({"temp_id": tid, "AnimalId": aid,
                         "correlation": round(r, 4), "n_bins": n,
                         "p_value": round(p, 4), "note": "ok"})
    return pd.DataFrame(results)


def _greedy_assign(scores: pd.DataFrame, corr_threshold: float) -> dict:
    assignment = {}
    used_animals, used_tids = set(), set()
    valid = scores[scores["correlation"] >= corr_threshold].sort_values(
        "correlation", ascending=False)
    for _, row in valid.iterrows():
        tid, aid = row["temp_id"], row["AnimalId"]
        if tid in used_tids or aid in used_animals:
            continue
        assignment[tid] = aid
        used_tids.add(tid)
        used_animals.add(aid)
    return assignment


def step_a_kinetic_match(tracks_df: pd.DataFrame,
                          kinetics_df: pd.DataFrame,
                          args: argparse.Namespace) -> dict:
    """
    Returns assignment dict: {temp_id (int) -> AnimalId (int)}
    """
    section("Step A — Kinetic Matcher")

    total_frames = tracks_df["frame_index"].nunique()
    tid_counts = tracks_df.groupby("temp_id")["frame_index"].nunique()
    stable_tids = tid_counts[
        tid_counts / total_frames >= args.min_temp_id_frames
    ].index.tolist()
    log(f"Stable temp_ids (≥{args.min_temp_id_frames*100:.0f}% frames): {sorted(stable_tids)}")

    tracks_filt = tracks_df[tracks_df["temp_id"].isin(stable_tids)].copy()

    t_start = tracks_filt["frame_datetime"].min().floor(f"{args.bin_minutes}min")
    t_end   = tracks_filt["frame_datetime"].max().ceil(f"{args.bin_minutes}min")
    bins    = pd.date_range(start=t_start, end=t_end, freq=f"{args.bin_minutes}min")
    log(f"Time bins: {len(bins)-1} × {args.bin_minutes}-min from {t_start} to {t_end}")

    if len(bins) < 3:
        log("WARNING: fewer than 2 complete bins — correlation will be unreliable.")

    cam_disp  = _camera_displacement_per_bin(tracks_filt, bins)
    kin_delta = _kinetics_delta_per_bin(kinetics_df, bins)

    if cam_disp.empty or kin_delta.empty:
        log("WARNING: empty displacement or kinetics signal — no matches possible.")
        return {}

    scores = _compute_scores(cam_disp, kin_delta, args.activity_pct, args.min_active_bins)

    # print correlation matrix
    if not scores.empty:
        pivot = scores.pivot_table(index="temp_id", columns="AnimalId",
                                   values="correlation", aggfunc="first")
        print("\nCorrelation matrix (temp_id × AnimalId):")
        print(pivot.to_string())
        print()

    assignment = _greedy_assign(scores, args.corr_threshold)

    if assignment:
        for tid, aid in sorted(assignment.items()):
            row = scores[(scores["temp_id"] == tid) & (scores["AnimalId"] == aid)].iloc[0]
            log(f"  temp_id {tid:>3}  →  AnimalId {aid}  "
                f"(r={row['correlation']:.3f}, n_bins={int(row['n_bins'])})")
    else:
        log("  No confident kinetic matches. "
            "Try lowering --corr_threshold or using a longer video.")

    unmatched = set(stable_tids) - set(assignment.keys())
    if unmatched:
        log(f"  Unmatched temp_ids after kinetics: {sorted(unmatched)}")

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# Embed helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_embeds_for_session(tracks_df: pd.DataFrame,
                             embed_parquet: str,
                             session_id: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns [temp_id, embed_array (np.ndarray shape 128)].
    Tries parquet first, falls back to 'embed' column in tracks_df.
    """
    if embed_parquet and Path(embed_parquet).exists():
        log(f"Loading embeddings from parquet: {embed_parquet}")
        df = pd.read_parquet(embed_parquet)
        if "session_id" in df.columns:
            df = df[df["session_id"] == session_id]
        if "embed" in df.columns and isinstance(df["embed"].iloc[0], np.ndarray):
            return df[["temp_id", "embed"]].copy()
        elif "embed" in df.columns:
            df["embed"] = df["embed"].apply(
                lambda x: np.array(json.loads(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32)
            )
            return df[["temp_id", "embed"]].copy()

    # fall back to 'embed' column in tracks CSV
    if "embed" not in tracks_df.columns:
        log("WARNING: no 'embed' column in tracks and no parquet — gallery step will be skipped.")
        return pd.DataFrame(columns=["temp_id", "embed"])

    rows = []
    for _, row in tracks_df[tracks_df["embed"].notna()].iterrows():
        raw = row["embed"]
        if not raw or raw == "[]":
            continue
        try:
            arr = np.array(json.loads(raw), dtype=np.float32)
            if arr.shape == (128,):
                rows.append({"temp_id": row["temp_id"], "embed": arr})
        except Exception:
            pass

    if not rows:
        log("WARNING: no valid embeds found in tracks CSV — gallery step will be skipped.")
        return pd.DataFrame(columns=["temp_id", "embed"])

    log(f"Loaded {len(rows)} embed rows from tracks CSV.")
    return pd.DataFrame(rows)


def compute_mean_embed(embed_rows: pd.DataFrame) -> np.ndarray:
    """Stack and mean-pool all embed arrays, return L2-normalised 128D vector."""
    stack = np.stack(embed_rows["embed"].values)     # (N, 128)
    mean  = stack.mean(axis=0)
    norm  = np.linalg.norm(mean)
    return (mean / norm) if norm > 1e-8 else mean


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
# Step B — Gallery Builder
# ─────────────────────────────────────────────────────────────────────────────

def step_b_gallery_builder(
    tracks_df: pd.DataFrame,
    embed_df: pd.DataFrame,
    kinetic_assignment: dict,       # {temp_id -> AnimalId}
    is_night: bool,
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    gallery_dir: str,
    dry_run: bool = False,
) -> dict:
    """
    For each kinetically-confirmed (temp_id → AnimalId) pair:
      1. Pool all embeds for that temp_id into a session mean vector
      2. EMA-blend with existing gallery entry (α = ema_alpha)
      3. Write updated gallery to .npy and to reid_registry in SQLite

    Returns updated gallery dict {real_id: np.ndarray}.
    """
    section("Step B — Gallery Builder")

    modality = "night" if is_night else "day"
    gallery = load_gallery(gallery_dir, modality)
    log(f"Loaded {modality} gallery: {len(gallery)} existing entries")

    if embed_df.empty:
        log("No embeds available — skipping gallery update.")
        return gallery

    updates = 0
    alpha = args.ema_alpha

    for tid, aid in kinetic_assignment.items():
        tid_embeds = embed_df[embed_df["temp_id"] == tid]
        if len(tid_embeds) < args.min_embeds_gallery:
            log(f"  temp_id {tid}: only {len(tid_embeds)} embeds (< {args.min_embeds_gallery}) — skip")
            continue

        session_mean = compute_mean_embed(tid_embeds)

        if aid in gallery:
            old_vec     = gallery[aid]
            updated_vec = alpha * session_mean + (1 - alpha) * old_vec
            norm        = np.linalg.norm(updated_vec)
            gallery[aid] = updated_vec / norm if norm > 1e-8 else updated_vec
            log(f"  AnimalId {aid} (t{tid}): EMA update  α={alpha}  "
                f"cosine(old,new)={float(np.dot(old_vec, session_mean)):.3f}")
        else:
            gallery[aid] = session_mean
            log(f"  AnimalId {aid} (t{tid}): new gallery entry from {len(tid_embeds)} embeds")

        updates += 1

        # update reid_registry in SQLite
        if not dry_run:
            col_emb   = f"gallery_embed_{modality}"
            col_n     = f"gallery_n_{modality}"
            col_upd   = f"last_updated_{modality}_dt"
            ts        = pd.Timestamp.now().isoformat()
            emb_json  = json.dumps(gallery[aid].tolist())

            existing  = get_reid_row(conn, aid)
            old_n     = (existing or {}).get(col_n, 0) or 0

            upsert_reid(conn, aid, {
                col_emb: emb_json,
                col_n:   old_n + 1,
                col_upd: ts,
                "match_method": "kinetic",
            })

    log(f"Gallery updated: {updates} entries  modality={modality}")

    if not dry_run and updates > 0:
        save_gallery(gallery_dir, modality, gallery)

    return gallery


# ─────────────────────────────────────────────────────────────────────────────
# Step C — Cosine Resolver
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def step_c_cosine_resolver(
    tracks_df: pd.DataFrame,
    embed_df: pd.DataFrame,
    kinetic_assignment: dict,       # {temp_id -> AnimalId} — already confirmed
    gallery: dict,                  # {real_id: np.ndarray} flat gallery from step B
    is_night: bool,
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    gallery_dir: str,
    session_id: str,
    camera_id: str = "cam0",
    dry_run: bool = False,
) -> dict:
    """
    Resolve unconfirmed temp_ids via cosine similarity.

    Query order per unresolved temp_id:
      1. Flat real_id gallery        — confirmed identities across all sessions
      2. Pose-conditioned real_id gallery — slot-aware, higher discrimination
      3. TempPoseGallery (same camera, other sessions) — cross-session temp matching

    If a temp_id matches a real_id (steps 1/2) → assign that real_id.
    If a temp_id matches another temp_id (step 3) that already has a synthetic id
      → assign the same synthetic id.
    If a temp_id matches another temp_id with no identity yet → mint a new
      synthetic id and assign it to both.
    If no match → leave unresolved.

    Also runs backpropagation: if any confirmed real_id in this session was
    previously a synthetic id, all timeline rows are updated.
    """
    section("Step C — Cosine Resolver")

    modality = "night" if is_night else "day"

    already_resolved = set(kinetic_assignment.keys())
    all_tids = set(tracks_df["temp_id"].unique())
    unresolved_tids = all_tids - already_resolved

    if embed_df.empty:
        log("No embeds available — cosine resolver skipped.")
        return dict(kinetic_assignment)

    embed_counts   = embed_df.groupby("temp_id").size()
    queryable_tids = [
        tid for tid in unresolved_tids
        if embed_counts.get(tid, 0) >= args.cosine_min_embeds
    ]
    log(f"Unresolved temp_ids: {len(unresolved_tids)}  "
        f"Queryable (≥{args.cosine_min_embeds} embeds): {len(queryable_tids)}")

    # Load pose galleries
    pose_gallery = PoseGallery.load(gallery_dir, modality)
    temp_gallery = TempPoseGallery.load(gallery_dir, modality)

    cosine_assignment = {}
    assigned_aids     = set(kinetic_assignment.values())

    for tid in queryable_tids:
        tid_embeds = embed_df[embed_df["temp_id"] == tid]
        query_vec  = compute_mean_embed(tid_embeds)

        # ── Step 1: flat real_id gallery ─────────────────────────────────────
        best_aid, best_sim = None, -1.0
        for aid, gvec in gallery.items():
            if aid in assigned_aids:
                continue
            sim = cosine_sim(query_vec, gvec)
            if sim > best_sim:
                best_sim, best_aid = sim, aid

        if best_aid is not None and best_sim >= args.cosine_threshold:
            cosine_assignment[tid] = best_aid
            assigned_aids.add(best_aid)
            log(f"  t{tid} → real_id {best_aid}  (flat cosine={best_sim:.3f})")
            continue

        # ── Step 2: pose-conditioned real_id gallery ─────────────────────────
        p_id, p_cos, _ = pose_gallery.query(
            query_vec, posture=None, facing=None, threshold=args.cosine_threshold
        )
        if p_id is not None and p_id not in assigned_aids:
            cosine_assignment[tid] = p_id
            assigned_aids.add(p_id)
            log(f"  t{tid} → real_id {p_id}  (pose-slot cosine={p_cos:.3f})")
            continue

        # ── Step 3: TempPoseGallery — cross-session same camera ──────────────
        match_key, t_cos, _ = temp_gallery.query_for_camera(
            camera_id       = camera_id,
            query_embed     = query_vec,
            posture         = None,
            facing          = None,
            threshold       = args.cosine_threshold,
            exclude_session = session_id,   # don't match against self
        )

        if match_key is not None:
            # The matched temp_id may already have a synthetic id assigned
            matched_sid = cosine_assignment.get(match_key.temp_id)
            if matched_sid is None:
                # check reid_registry for an existing assignment
                existing_row = get_reid_row(conn, match_key.temp_id)
                if existing_row:
                    matched_sid = existing_row.get("real_id")

            if matched_sid is not None and matched_sid not in assigned_aids:
                # Inherit the same synthetic id
                cosine_assignment[tid] = matched_sid
                assigned_aids.add(matched_sid)
                log(f"  t{tid} → synthetic_id {matched_sid}  "
                    f"(temp cosine={t_cos:.3f}, matched {match_key})")
            else:
                # Mint a new synthetic id for this pair
                new_synth = mint_synthetic_id(gallery_dir)
                cosine_assignment[tid]              = new_synth
                cosine_assignment[match_key.temp_id] = new_synth
                assigned_aids.add(new_synth)
                # Register synthetic id in reid_registry
                if not dry_run:
                    ts = pd.Timestamp.now().isoformat()
                    upsert_reid(conn, new_synth, {
                        "match_method":  "synthetic",
                        "first_seen_dt": ts,
                        "known_temp_ids": json.dumps([
                            {"session_id": session_id,       "temp_id": int(tid)},
                            {"session_id": match_key.session_id, "temp_id": match_key.temp_id},
                        ]),
                    })
                log(f"  t{tid} + {match_key} → NEW synthetic_id {new_synth}  "
                    f"(temp cosine={t_cos:.3f})")
        else:
            log(f"  t{tid}: no match above {args.cosine_threshold} — unresolved")

    if not cosine_assignment:
        log("  No cosine matches found.")

    # ── Soft gallery update for real_id cosine matches (α/2) ─────────────────
    if not dry_run:
        alpha_half = args.ema_alpha / 2
        for tid, aid in cosine_assignment.items():
            if is_synthetic(aid):
                continue   # don't update real_id gallery with unconfirmed embeddings
            tid_embeds = embed_df[embed_df["temp_id"] == tid]
            if len(tid_embeds) < args.min_embeds_gallery:
                continue
            session_mean = compute_mean_embed(tid_embeds)
            if aid in gallery:
                updated = gallery[aid] * (1 - alpha_half) + session_mean * alpha_half
                norm = np.linalg.norm(updated)
                gallery[aid] = updated / norm if norm > 1e-8 else updated
            else:
                gallery[aid] = session_mean
            ts      = pd.Timestamp.now().isoformat()
            col_emb = f"gallery_embed_{modality}"
            col_n   = f"gallery_n_{modality}"
            existing = get_reid_row(conn, aid)
            old_n    = (existing or {}).get(col_n, 0) or 0
            upsert_reid(conn, aid, {
                col_emb: json.dumps(gallery[aid].tolist()),
                col_n:   old_n + 1,
                f"last_updated_{modality}_dt": ts,
                "match_method": f"cosine_{modality}",
            })
        save_gallery(gallery_dir, modality, gallery)

    # ── Backpropagate: if any confirmed real_id was previously synthetic ──────
    all_assignments = {**kinetic_assignment, **cosine_assignment}
    for tid, aid in list(all_assignments.items()):
        if not is_synthetic(aid):
            continue
        # This temp_id now has a confirmed real_id via kinetics or manual?
        confirmed = kinetic_assignment.get(tid)
        if confirmed and not is_synthetic(confirmed):
            rows = backpropagate_resolution(
                synthetic_id = aid,
                real_id      = confirmed,
                conn         = conn,
                pose_gallery = pose_gallery,
                gallery_dir  = gallery_dir,
                modality     = modality,
                dry_run      = dry_run,
            )
            log(f"  Backpropagated synthetic_id {aid} → real_id {confirmed} "
                f"({rows} timeline rows updated)")
            all_assignments[tid] = confirmed
            pose_gallery.save(gallery_dir, modality)

    # ── Switch healing ────────────────────────────────────────────────────────
    aid_to_tids: dict[int, list] = {}
    for tid, aid in all_assignments.items():
        aid_to_tids.setdefault(aid, []).append(tid)

    for aid, tids in aid_to_tids.items():
        if len(tids) > 1:
            log(f"  [switch] id {aid} ← temp_ids {tids} — keeping most frames")
            frame_counts = {tid: (tracks_df["temp_id"] == tid).sum() for tid in tids}
            keep_tid  = max(frame_counts, key=frame_counts.get)
            drop_tids = [t for t in tids if t != keep_tid]
            for d in drop_tids:
                if d in cosine_assignment:
                    del cosine_assignment[d]
                elif d in kinetic_assignment:
                    log(f"    WARNING: kinetically-assigned t{d} conflicts — review")

    merged = {**kinetic_assignment, **cosine_assignment}
    n_real    = sum(1 for v in merged.values() if not is_synthetic(v))
    n_synth   = sum(1 for v in merged.values() if is_synthetic(v))
    log(f"Final assignment: {len(merged)} temp_ids resolved  "
        f"({len(kinetic_assignment)} kinetic, {len(cosine_assignment)} cosine  "
        f"[{n_real} real_id, {n_synth} synthetic])")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Step D — Sensor Sequencer (forward-fill to video time grid)
# ─────────────────────────────────────────────────────────────────────────────

def step_d_sensor_sequencer(
    tracks_df: pd.DataFrame,
    kinetics_df: pd.DataFrame,
    behavior_df: pd.DataFrame | None,
    assignment: dict,
    bin_minutes: int = 15,
) -> pd.DataFrame:
    """
    Forward-fill behavior (~90s) and kinetics (~15min) signals onto the video time grid.

    For each resolved (temp_id → AnimalId) pair:
      - Build a time grid from the video bins
      - Join and forward-fill kinetics deltas and behavior features
      - Return one row per (real_id, window_start_dt) suitable for resolved_cow_timeline

    Returns DataFrame with columns aligned to resolved_cow_timeline schema.
    """
    section("Step D — Sensor Sequencer")

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
# Step E — Write resolved_cow_timeline
# ─────────────────────────────────────────────────────────────────────────────

def step_e_write_timeline(
    timeline_df: pd.DataFrame,
    session_id: str,
    conn: sqlite3.Connection,
    dry_run: bool = False,
) -> None:
    section("Step E — Write resolved_cow_timeline")

    if timeline_df.empty:
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
# Update known_temp_ids in reid_registry
# ─────────────────────────────────────────────────────────────────────────────

def update_known_temp_ids(conn: sqlite3.Connection,
                           assignment: dict,
                           session_id: str,
                           dry_run: bool = False) -> None:
    """Append {session_id, temp_id} entries to reid_registry.known_temp_ids."""
    if dry_run:
        return
    for tid, aid in assignment.items():
        existing = get_reid_row(conn, aid)
        if existing is None:
            continue
        raw = existing.get("known_temp_ids") or "[]"
        try:
            known = json.loads(raw)
        except Exception:
            known = []
        entry = {"session_id": session_id, "temp_id": int(tid)}
        if entry not in known:
            known.append(entry)
        upsert_reid(conn, aid, {"known_temp_ids": json.dumps(known)})


# ─────────────────────────────────────────────────────────────────────────────
# Step A.6 — Duplicate assignment resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_duplicate_assignments(
    assignment: dict,               # {temp_id -> real_id}  — may contain duplicates
    manual_tids: set,               # temp_ids that came from manual assignments
    kinetic_tids: set,              # temp_ids that came from kinetic matching
    tracks_df: pd.DataFrame,        # for frame-count lookup
    conn: sqlite3.Connection,
    session_id: str,
    dry_run: bool = False,
) -> dict:
    """
    Detect AnimalIds assigned to multiple temp_ids and resolve by merging.

    Winner hierarchy (highest priority first):
      1. Manual assignment
      2. More frames in raw_tracks
      3. Kinetic assignment (lower temp_id breaks ties)

    The loser temp_id is remapped to the winner temp_id in the returned dict.
    A record is written to temp_id_merges for traceability.

    Returns a cleaned assignment dict where every real_id appears exactly once,
    plus a merge_map {loser_tid -> winner_tid} for downstream steps to use
    when pooling embed/track rows.
    """
    # group by real_id → find duplicates
    from collections import defaultdict
    aid_to_tids = defaultdict(list)
    for tid, aid in assignment.items():
        aid_to_tids[aid].append(tid)

    duplicates = {aid: tids for aid, tids in aid_to_tids.items() if len(tids) > 1}
    if not duplicates:
        return assignment, {}

    section("Step A.6 — Duplicate Assignment Resolver")

    # frame counts for all temp_ids in this session
    frame_counts = (
        tracks_df.groupby("temp_id")["frame_index"]
        .nunique()
        .to_dict()
    )

    def priority(tid, aid):
        """Lower number = higher priority (wins).
        Hierarchy: manual(0) > cosine(1) > kinetic(2)
        Tiebreak within same tier: more frames wins, then lower temp_id.
        """
        if tid in manual_tids:
            return (0, -frame_counts.get(tid, 0), tid)   # manual — highest
        if tid not in kinetic_tids:
            return (1, -frame_counts.get(tid, 0), tid)   # cosine — second
        return (2, -frame_counts.get(tid, 0), tid)       # kinetic — lowest

    def reason(tid):
        if tid in manual_tids:   return "manual"
        if tid in kinetic_tids:  return "kinetic"
        return "cosine"

    merge_map = {}   # {loser_tid -> winner_tid}
    clean = dict(assignment)
    ts = pd.Timestamp.now().isoformat()

    for aid, tids in sorted(duplicates.items()):
        ranked = sorted(tids, key=lambda t: priority(t, aid))
        winner = ranked[0]
        losers = ranked[1:]

        log(f"  AnimalId {aid} → duplicate temp_ids {sorted(tids)}")
        log(f"    winner: temp_id {winner} ({reason(winner)}, "
            f"{frame_counts.get(winner,0)} frames)")

        for loser in losers:
            log(f"    merge:  temp_id {loser} ({reason(loser)}, "
                f"{frame_counts.get(loser,0)} frames) → remapped to temp_id {winner}")
            merge_map[loser] = winner
            del clean[loser]

            if not dry_run:
                conn.execute("""
                    INSERT INTO temp_id_merges
                        (session_id, winner_tid, loser_tid, real_id,
                         winner_reason, loser_reason, merged_dt)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(session_id, loser_tid) DO UPDATE SET
                        winner_tid    = excluded.winner_tid,
                        real_id       = excluded.real_id,
                        winner_reason = excluded.winner_reason,
                        loser_reason  = excluded.loser_reason,
                        merged_dt     = excluded.merged_dt
                """, (session_id, winner, loser, aid,
                      reason(winner), reason(loser), ts))

    if not dry_run and merge_map:
        conn.commit()

    log(f"  Resolved {len(merge_map)} duplicate(s) → "
        f"{len(clean)} unique assignments")
    return clean, merge_map


# ─────────────────────────────────────────────────────────────────────────────
# Step A.5 — Manual assignment loader
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

def run(args) -> None:
    """
    Core pipeline entry point. Callable from CLI (via main()) or directly when
    imported by track_and_dump.py. args is an argparse.Namespace with all fields
    from parse_args() — no --tracks needed; tracks are loaded from the DB.
    """
    section("reconcile.py — ReID pipeline")
    log(f"session_id  : {args.session}")
    log(f"kinetics    : {args.kinetics}")
    log(f"db          : {args.db}")
    log(f"gallery_dir : {args.gallery_dir}")
    log(f"dry_run     : {args.dry_run}")

    # ── database ──────────────────────────────────────────────────────────────
    conn = init_db(args.db)
    migrate_timeline_schema(conn)


    # ── load tracks from SQLite ───────────────────────────────────────────────
    log("Loading tracks from SQLite...")
    tracks_df = pd.read_sql(
        "SELECT frame_index, frame_datetime, temp_id, cx, cy, x1, y1, x2, y2 "
        "FROM raw_tracks WHERE session_id = ? ORDER BY frame_index",
        conn, params=(args.session,), parse_dates=["frame_datetime"],
    )
    if tracks_df.empty:
        log(f"No rows in raw_tracks for session '{args.session}' — aborting.")
        conn.close()
        return
    log(f"  {len(tracks_df)} rows, {tracks_df['temp_id'].nunique()} temp_ids, "
        f"frames {tracks_df['frame_index'].min()}–{tracks_df['frame_index'].max()}")

    log("Loading kinetics CSV...")
    kinetics_df = pd.read_csv(args.kinetics, parse_dates=["datetime"])
    log(f"  {len(kinetics_df)} rows, animals: {sorted(kinetics_df['AnimalId'].unique())}")

    # optional behavior CSV alongside kinetics
    behavior_df = None
    beh_candidate = Path(args.kinetics).parent / Path(args.kinetics).name.replace(
        "kinetic_data", "behavior_data"
    )
    if beh_candidate.exists():
        log(f"Loading behavior CSV: {beh_candidate}")
        behavior_df = pd.read_csv(str(beh_candidate), parse_dates=["datetime"])
        log(f"  {len(behavior_df)} rows")
    else:
        log(f"No behavior CSV found (looked for {beh_candidate.name}) — d_f12/f23/v will be NaN")

    # detect day/night
    is_night = detect_is_night_from_tracks(tracks_df)
    log(f"is_night={is_night} (heuristic from frame_datetime hour distribution)")

    # update is_night on the session row and load camera_id
    upsert_session(conn, args.session, args.db, is_night)
    session_row = get_session(conn, args.session)
    camera_id   = (session_row.get("camera_id") or "cam0") if session_row else "cam0"
    log(f"camera_id={camera_id}")

    # ── Step A — kinetic matching ─────────────────────────────────────────────
    kinetic_assignment = step_a_kinetic_match(tracks_df, kinetics_df, args)

    # ── Step A.5 — merge manual assignments ──────────────────────────────────
    manual_assignment = load_manual_assignments(conn, args.session)
    if manual_assignment:
        conflicts = set(kinetic_assignment) & set(manual_assignment)
        if conflicts:
            log(f"  Note: manual overrides kinetic for temp_ids: {sorted(conflicts)}")
        kinetic_assignment.update(manual_assignment)

    # ── load embeds from parquet ──────────────────────────────────────────────
    embed_df = load_embeds_for_session(tracks_df, args.embed_parquet, args.session)

    # ── Step B — gallery builder ──────────────────────────────────────────────
    gallery = step_b_gallery_builder(
        tracks_df          = tracks_df,
        embed_df           = embed_df,
        kinetic_assignment = kinetic_assignment,
        is_night           = is_night,
        conn               = conn,
        args               = args,
        gallery_dir        = args.gallery_dir,
        dry_run            = args.dry_run,
    )

    # ── Step C — cosine resolver ──────────────────────────────────────────────
    merge_map = {}   # populated by Step A.6 below
    full_assignment = step_c_cosine_resolver(
        tracks_df          = tracks_df,
        embed_df           = embed_df,
        kinetic_assignment = kinetic_assignment,
        gallery            = gallery,
        is_night           = is_night,
        conn               = conn,
        args               = args,
        gallery_dir        = args.gallery_dir,
        session_id         = args.session,
        camera_id          = camera_id,
        dry_run            = args.dry_run,
    )

    # ── Step A.6 — resolve duplicates across all assignments (kinetic+manual+cosine)
    full_assignment, merge_map = resolve_duplicate_assignments(
        assignment   = full_assignment,
        manual_tids  = set(manual_assignment.keys()),
        kinetic_tids = set(
            tid for tid, aid in kinetic_assignment.items()
            if tid not in manual_assignment
        ),
        tracks_df    = tracks_df,
        conn         = conn,
        session_id   = args.session,
        dry_run      = args.dry_run,
    )

    # remap loser embed rows → winner so gallery update pools all sightings
    if merge_map and not embed_df.empty:
        embed_df_remapped = embed_df.copy()
        embed_df_remapped["temp_id"] = embed_df_remapped["temp_id"].replace(merge_map)
        # rebuild gallery with remapped embeds for any merged pairs
        gallery = step_b_gallery_builder(
            tracks_df          = tracks_df,
            embed_df           = embed_df_remapped,
            kinetic_assignment = full_assignment,
            is_night           = is_night,
            conn               = conn,
            args               = args,
            gallery_dir        = args.gallery_dir,
            dry_run            = args.dry_run,
        )

    # ── update known_temp_ids ─────────────────────────────────────────────────
    update_known_temp_ids(conn, full_assignment, args.session, args.dry_run)

    # ── Step D — sensor sequencer + Vision feature extractor ─────────────────────────────────────────────
    timeline_df = step_d_sensor_sequencer(
        tracks_df   = tracks_df,
        kinetics_df = kinetics_df,
        behavior_df = behavior_df,
        assignment  = full_assignment,
        bin_minutes = args.bin_minutes,
    )
    if not timeline_df.empty:
        timeline_df["session_id"] = args.session

    # ── Step B (vision) — feature extraction ─────────────────────────────────
    if not timeline_df.empty:
        timeline_df = run_vision_features(
            session_id    = args.session,
            timeline_df   = timeline_df,
            conn          = conn,
            assignment    = full_assignment,
            is_night      = is_night,
            camera_id     = session_row.get("camera_id", "cam0") if session_row else "cam0",
            kps_parquet   = str(Path(args.embed_parquet).parent / "kps.parquet") if args.embed_parquet else None,
            embed_parquet = args.embed_parquet or None,
            gallery_dir   = args.gallery_dir,
            ema_alpha     = args.ema_alpha,
            dry_run       = args.dry_run,
            bin_minutes   = args.bin_minutes,
        )

    # ── Step E — write to DB ──────────────────────────────────────────────────
    step_e_write_timeline(timeline_df, args.session, conn, dry_run=args.dry_run)

    # ── summary ───────────────────────────────────────────────────────────────
    section("Summary")
    n_manual  = len(manual_assignment)
    n_kinetic = len(kinetic_assignment) - n_manual
    if merge_map:
        log(f"Merged switches : {len(merge_map)} "
            f"({', '.join(f't{l}→t{w}' for l,w in sorted(merge_map.items()))})")
    log(f"Kinetic matches : {n_kinetic}")
    log(f"Manual matches  : {n_manual}")
    log(f"Cosine matches  : {len(full_assignment) - len(kinetic_assignment)}")
    log(f"Total resolved  : {len(full_assignment)} temp_ids → AnimalId")
    unresolved = set(tracks_df["temp_id"].unique()) - set(full_assignment.keys())
    if unresolved:
        log(f"Unresolved      : {sorted(int(t) for t in unresolved)}")
    log(f"Timeline rows   : {len(timeline_df)}")
    if args.dry_run:
        log("dry_run=True — no data written to DB or gallery files")

    conn.close()
    log("Done.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
#
# python3 reconcile.py \
#   --db         ~/thesis_workspace/outputs/tracks/refet33_2024-12-21/calving_project.db \
#   --session    refet33_20241221 \
#   --kinetics   "$KIN" \
#   --gallery_dir "$GAL" \
#   --corr_threshold 0.7 \
#   --min_active_bins 3 \
#   --cosine_threshold 0.75 \
#   --ema_alpha 0.15 \
#   --embed_parquet ~/thesis_workspace/outputs/tracks/refet33_2024-12-21/embeds.parquet 
#
# Dry run (no DB writes):
#   --dry_run
#
# After running, validate visually:
#   python3 display_tracks.py \
#     --video      ~/thesis_workspace/raw_data/calving/refet_33_S20241221070000_E20241221080000.mp4 \
#     --db         ~/thesis_workspace/outputs/tracks/refet33_2024-12-21/calving_project.db \
#     --session_id refet33_20241221 \
#     --kinetics   "$KIN" \
#     --draw_pose --show_fps --sink ffplay
#
# Pipeline notes:
#   - Run once per session after track_and_dump.py completes
#   - Gallery files accumulate across sessions — more sessions = better cosine matching
#   - Kinetic matching requires ≥3 active 15-min bins (45+ minutes of video)
#   - Cosine resolver is a no-op on first run (empty gallery) — by design
#   - Vision features (spine_angle etc.) will be NULL until pose extractor is added