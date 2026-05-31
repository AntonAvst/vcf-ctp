"""
vision_features/extractor.py
─────────────────────────────
Orchestrator for Stage 2-B: Vision Feature Extraction.

Called by reconcile.py after kinetic matching (Step A) and before gallery
building (Step B / gallery update) and sensor sequencer (Step D).

Responsibilities
────────────────
1. Load raw kps + bbox data for a session from the kps.parquet file
   (or from raw_tracks in SQLite if parquet is absent — slower).
2. Run all feature extractors in vision_features/features/ in sequence.
3. Aggregate per-frame labels into per-window scalars.
4. Merge scalars into the timeline_df produced by step_d_sensor_sequencer.
5. Optionally update the pose-conditioned gallery (step B extension).

How to add a new feature
─────────────────────────
1. Create vision_features/features/my_feature.py with:
       extract_my_feature(kps, kps_kconf, bbox, ...) -> dict[str, ndarray]
       aggregate_my_feature(per_frame_output) -> dict[str, scalar]
2. Add the new column(s) to vision_features/schema.py:
       TIMELINE_VISION_COLS and TIMELINE_ALTER_SQLS
3. Call it in _run_extractors() below.
That's it — no other files need changing.

Inputs (all per-frame, aligned):
    kps_df   : DataFrame with columns session_id, frame_index, temp_id,
               kps (list[57] flat), kps_kconf (list[19]), plus bbox (x1,y1,x2,y2)
               and optionally det_conf.
               Loaded from kps.parquet (preferred) or reconstructed from raw_tracks.

Outputs:
    Per-frame labels DataFrame  — stored in memory; used by gallery builder.
    Window scalar DataFrame     — merged into timeline_df.
    Updated PoseGallery         — saved to gallery_pose_{modality}.npy.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    TIMELINE_VISION_COLS, TIMELINE_ALTER_SQLS,
    Posture, Facing,
    slot_name,
)
from .features import (
    extract_posture, aggregate_posture,
    extract_facing,  aggregate_facing,
)
from .gallery.pose_conditioned import PoseGallery, build_slot_embeds


# ─────────────────────────────────────────────────────────────────────────────
# Logging (matches reconcile.py style)
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[vision] {msg}", flush=True)


def section(title: str) -> None:
    print(f"\n{'─'*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─'*60}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Schema migration — add new columns to existing DB
# ─────────────────────────────────────────────────────────────────────────────

def migrate_timeline_schema(conn: sqlite3.Connection) -> None:
    """
    Add vision feature columns to resolved_cow_timeline if they don't exist.
    Safe to call multiple times (silently ignores duplicate-column errors).
    """
    for sql in TIMELINE_ALTER_SQLS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_kps_for_session(
    session_id: str,
    kps_parquet: str | None,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Load per-frame kps + bbox for a session.

    Tries kps.parquet first (fast, float32 arrays). Falls back to
    raw_tracks in SQLite (slower, JSON strings to parse).

    Returns DataFrame with columns:
        frame_index, frame_datetime, temp_id, det_conf,
        x1, y1, x2, y2,
        kps_flat  (np.ndarray shape (57,)  — flat [x,y,conf] × 19),
        kps_kconf (np.ndarray shape (19,)),
    One row per detection.
    """
    # ── Parquet path ──────────────────────────────────────────────────────────
    if kps_parquet and Path(kps_parquet).exists():
        log(f"Loading kps from parquet: {kps_parquet}")
        kdf = pd.read_parquet(kps_parquet)
        if "session_id" in kdf.columns:
            kdf = kdf[kdf["session_id"] == session_id].copy()

        # Parquet stores kps as FixedSizeList(57) → pyarrow list → python list
        # Convert to numpy arrays
        kdf["kps_flat"]  = kdf["kps"].apply(
            lambda v: np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v
        )
        kdf["kps_kconf"] = kdf["kps_kconf"].apply(
            lambda v: np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v
        )

        # Ensure bbox cols exist (parquet may include x1..y2 or bbox column)
        if "bbox" in kdf.columns and "x1" not in kdf.columns:
            bbox_arr = np.stack(kdf["bbox"].values)
            kdf[["x1","y1","x2","y2"]] = bbox_arr[:, :4]

        log(f"  {len(kdf)} rows loaded from parquet")
        return kdf[["frame_index", "frame_datetime", "temp_id", "det_conf",
                    "x1", "y1", "x2", "y2", "kps_flat", "kps_kconf"]].copy()

    # ── SQLite fallback ────────────────────────────────────────────────────────
    log("kps parquet not found — loading from raw_tracks (SQLite)...")
    rdf = pd.read_sql(
        """
        SELECT frame_index, frame_datetime, temp_id, det_conf,
               x1, y1, x2, y2, kps, kps_kconf
        FROM   raw_tracks
        WHERE  session_id = ?
        ORDER  BY frame_index, temp_id
        """,
        conn, params=(session_id,), parse_dates=["frame_datetime"],
    )
    if rdf.empty:
        log("  No rows in raw_tracks — vision features will be NULL.")
        return pd.DataFrame()

    # Parse JSON strings to numpy arrays
    def _parse_json_arr(s, dtype=np.float32):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return None
        try:
            return np.array(json.loads(s), dtype=dtype)
        except Exception:
            return None

    rdf["kps_flat"]  = rdf["kps"].apply(lambda s: _parse_json_arr(s))
    rdf["kps_kconf"] = rdf["kps_kconf"].apply(lambda s: _parse_json_arr(s))

    # Drop rows where kps failed to parse (pose model not run)
    rdf = rdf.dropna(subset=["kps_flat"]).copy()
    log(f"  {len(rdf)} rows with valid kps loaded from SQLite")
    return rdf[["frame_index", "frame_datetime", "temp_id", "det_conf",
                "x1", "y1", "x2", "y2", "kps_flat", "kps_kconf"]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def _run_extractors(kps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature extractors on a DataFrame of per-frame kps rows.

    Returns the same DataFrame with additional columns:
        posture, posture_conf, facing, facing_conf
    (All per-frame labels as scalar values.)

    To add a new feature: call extract_*() here and add its output columns.
    """
    N = len(kps_df)
    if N == 0:
        return kps_df

    # Stack arrays for vectorised ops
    kps_3d   = np.stack(kps_df["kps_flat"].values).reshape(N, 19, 3)   # (N, 19, 3)
    kps_kc   = np.stack(kps_df["kps_kconf"].values)                     # (N, 19)
    bbox     = kps_df[["x1","y1","x2","y2"]].values.astype(np.float32) # (N, 4)
    det_conf = kps_df["det_conf"].values.astype(np.float32)             # (N,)

    # ── Posture ───────────────────────────────────────────────────────────────
    posture_out = extract_posture(
        kps=kps_3d, kps_kconf=kps_kc, bbox=bbox, det_conf=det_conf
    )
    kps_df = kps_df.copy()
    kps_df["posture"]      = posture_out["posture"]
    kps_df["posture_conf"] = posture_out["posture_conf"]

    # ── Facing ────────────────────────────────────────────────────────────────
    facing_out = extract_facing(
        kps=kps_3d, kps_kconf=kps_kc, bbox=bbox,
        posture=posture_out["posture"],
    )
    kps_df["facing"]      = facing_out["facing"]
    kps_df["facing_conf"] = facing_out["facing_conf"]

    # ── Future features go here ───────────────────────────────────────────────
    # Example:
    # gait_out = extract_gait(kps=kps_3d, kps_kconf=kps_kc, bbox=bbox)
    # kps_df["stride_length"] = gait_out["stride_length"]

    return kps_df


# ─────────────────────────────────────────────────────────────────────────────
# Window aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_window(window_rows: pd.DataFrame) -> dict:
    """
    Aggregate per-frame labels for a single (real_id, window_start_dt) group
    into scalar features for resolved_cow_timeline.

    Add new aggregate calls here as new features are introduced.
    """
    posture_feats = aggregate_posture(
        posture=window_rows["posture"].values.astype(np.int8),
        posture_conf=window_rows["posture_conf"].values.astype(np.float32),
    )
    facing_feats = aggregate_facing(
        facing=window_rows["facing"].values.astype(np.int8),
        facing_conf=window_rows["facing_conf"].values.astype(np.float32),
    )
    return {**posture_feats, **facing_feats}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (called by reconcile.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_vision_features(
    session_id:        str,
    timeline_df:       pd.DataFrame,          # output of step_d_sensor_sequencer
    conn:              sqlite3.Connection,
    assignment:        dict,                   # {temp_id -> real_id}
    is_night:          bool,
    kps_parquet:       str | None = None,
    embed_parquet:     str | None = None,
    gallery_dir:       str        = "./reid_gallery",
    ema_alpha:         float      = 0.15,
    min_slot_conf:     float      = 0.30,
    dry_run:           bool       = False,
    bin_minutes:       int        = 15,
) -> pd.DataFrame:
    """
    Stage 2-B: Vision Feature Extraction.

    Parameters
    ----------
    session_id      : session being processed.
    timeline_df     : sensor-only timeline from step_d_sensor_sequencer.
    conn            : SQLite connection.
    assignment      : {temp_id -> real_id} from kinetic + cosine resolution.
    is_night        : True if this is a night/IR session.
    kps_parquet     : path to kps.parquet for this session (optional).
    embed_parquet   : path to embeds.parquet (for gallery slot update).
    gallery_dir     : directory containing gallery_pose_*.npy files.
    ema_alpha       : EMA learning rate for gallery update.
    min_slot_conf   : min confidence to assign a frame to a gallery slot.
    dry_run         : if True, skip all writes.
    bin_minutes     : window size in minutes (must match Step D).

    Returns
    -------
    timeline_df with new vision feature columns filled in.
    """
    section("Step B (vision) — Feature Extraction")
    log(f"session_id: {session_id}  |  is_night: {is_night}  |  dry_run: {dry_run}")

    if timeline_df.empty:
        log("timeline_df is empty — nothing to do.")
        return timeline_df

    # ── Schema migration ──────────────────────────────────────────────────────
    if not dry_run:
        migrate_timeline_schema(conn)

    # ── Load kps data ─────────────────────────────────────────────────────────
    kps_df = load_kps_for_session(session_id, kps_parquet, conn)
    if kps_df.empty:
        log("No kps data available — vision columns will remain NULL.")
        return timeline_df

    # ── Map temp_id → real_id ─────────────────────────────────────────────────
    kps_df["real_id"] = kps_df["temp_id"].map(assignment)
    kps_df = kps_df.dropna(subset=["real_id"]).copy()
    kps_df["real_id"] = kps_df["real_id"].astype(int)

    if kps_df.empty:
        log("No kps rows have a resolved real_id — check assignment dict.")
        return timeline_df

    log(f"kps rows with resolved identity: {len(kps_df)}, "
        f"{kps_df['real_id'].nunique()} cows")

    # ── Per-frame extraction ──────────────────────────────────────────────────
    log("Running per-frame extractors...")
    kps_df = _run_extractors(kps_df)

    posture_counts = {
        "standing": int((kps_df["posture"] == int(Posture.STANDING)).sum()),
        "lying":    int((kps_df["posture"] == int(Posture.LYING)).sum()),
        "uncertain":int((kps_df["posture"] == int(Posture.UNCERTAIN)).sum()),
    }
    log(f"  Posture labels: {posture_counts}")

    # ── Assign frames to time windows ─────────────────────────────────────────
    if "frame_datetime" not in kps_df.columns:
        log("WARNING: frame_datetime missing from kps data — cannot align to windows.")
        return timeline_df

    kps_df["frame_datetime"] = pd.to_datetime(kps_df["frame_datetime"])
    kps_df["window_start_dt"] = kps_df["frame_datetime"].dt.floor(f"{bin_minutes}min")

    # ── Aggregate per (real_id, window) ───────────────────────────────────────
    log("Aggregating per window...")
    window_rows = []
    for (rid, win), grp in kps_df.groupby(["real_id", "window_start_dt"]):
        agg = aggregate_window(grp)
        agg["real_id"]         = int(rid)
        agg["window_start_dt"] = win
        window_rows.append(agg)

    if not window_rows:
        log("No window aggregates produced.")
        return timeline_df

    vision_df = pd.DataFrame(window_rows)
    log(f"  Produced {len(vision_df)} window rows")

    # ── Merge into timeline_df ────────────────────────────────────────────────
    timeline_df = timeline_df.copy()
    timeline_df["window_start_dt"] = pd.to_datetime(timeline_df["window_start_dt"])

    for col in TIMELINE_VISION_COLS:
        if col not in timeline_df.columns:
            timeline_df[col] = None

    # Left-join vision_df onto timeline_df
    merge_cols = ["real_id", "window_start_dt"] + TIMELINE_VISION_COLS
    vision_df  = vision_df[[c for c in merge_cols if c in vision_df.columns]]
    timeline_df = timeline_df.merge(
        vision_df, on=["real_id", "window_start_dt"], how="left", suffixes=("", "_new")
    )
    # Fill original NaN cols from _new cols (in case cols already existed in timeline_df)
    for col in TIMELINE_VISION_COLS:
        new_col = col + "_new"
        if new_col in timeline_df.columns:
            timeline_df[col] = timeline_df[col].where(
                timeline_df[col].notna(), timeline_df[new_col]
            )
            timeline_df.drop(columns=[new_col], inplace=True)

    # Update modality_mask bit 1 (vision_ok) for windows that have vision data
    has_vision = timeline_df["lying_fraction"].notna()
    timeline_df.loc[has_vision, "modality_mask"] = (
        timeline_df.loc[has_vision, "modality_mask"].fillna(0).astype(int) | 2
    )
    n_filled = int(has_vision.sum())
    log(f"  Vision features filled for {n_filled}/{len(timeline_df)} timeline rows")

    # ── Pose-conditioned gallery update ────────────────────────────────────────
    if embed_parquet and Path(embed_parquet).exists():
        _update_pose_gallery(
            session_id   = session_id,
            kps_df       = kps_df,
            embed_parquet= embed_parquet,
            assignment   = assignment,
            is_night     = is_night,
            gallery_dir  = gallery_dir,
            ema_alpha    = ema_alpha,
            min_slot_conf= min_slot_conf,
            dry_run      = dry_run,
        )
    else:
        log("No embed_parquet provided — pose gallery update skipped.")

    return timeline_df


# ─────────────────────────────────────────────────────────────────────────────
# Pose gallery update (internal)
# ─────────────────────────────────────────────────────────────────────────────

def _update_pose_gallery(
    session_id:    str,
    kps_df:        pd.DataFrame,    # per-frame rows with posture/facing columns
    embed_parquet: str,
    assignment:    dict,
    is_night:      bool,
    gallery_dir:   str,
    ema_alpha:     float,
    min_slot_conf: float,
    dry_run:       bool,
) -> None:
    """
    Update gallery_pose_{day|night}.npy with pose-conditioned embeddings.
    Only kinetic-confirmed temp_ids contribute (full alpha).
    """
    import json as _json

    log("Updating pose-conditioned gallery...")
    modality = "night" if is_night else "day"

    # Load embeddings
    edf = pd.read_parquet(embed_parquet)
    if "session_id" in edf.columns:
        edf = edf[edf["session_id"] == session_id].copy()
    if edf.empty:
        log("  No embeds in parquet for this session — gallery update skipped.")
        return

    # Ensure embed column is ndarray
    if not isinstance(edf["embed"].iloc[0], np.ndarray):
        edf["embed"] = edf["embed"].apply(
            lambda v: np.array(v, dtype=np.float32)
        )

    # Align embed rows with kps_df by (frame_index, temp_id)
    # kps_df already has posture/facing columns from _run_extractors
    merged = edf.merge(
        kps_df[["frame_index", "temp_id", "posture", "posture_conf",
                "facing", "facing_conf"]],
        on=["frame_index", "temp_id"], how="inner",
    )
    if merged.empty:
        log("  No rows after merging embeds with kps labels — gallery update skipped.")
        return

    log(f"  {len(merged)} frames available for slot assignment")

    # Map temp_id → real_id (only kinetic-confirmed = full alpha)
    merged["real_id"] = merged["temp_id"].map(assignment)
    merged = merged.dropna(subset=["real_id"]).copy()
    merged["real_id"] = merged["real_id"].astype(int)

    frame_temp_ids = merged["temp_id"].values
    per_frame_labels = {
        "posture": merged["posture"].values.astype(np.int8),
        "facing":  merged["facing"].values.astype(np.int8),
    }

    slot_embeds_by_tid = build_slot_embeds(
        embed_df          = merged[["temp_id", "embed"]].reset_index(drop=True),
        per_frame_labels  = per_frame_labels,
        frame_temp_ids    = frame_temp_ids,
        min_conf_posture  = merged["posture_conf"].values.astype(np.float32),
        min_conf_facing   = merged["facing_conf"].values.astype(np.float32),
        min_conf          = min_slot_conf,
    )

    if not slot_embeds_by_tid:
        log("  No confident slot frames found — gallery update skipped.")
        return

    # Load gallery, apply updates, save
    gallery = PoseGallery.load(gallery_dir, modality)
    total_updates = 0

    for tid, slot_embeds in slot_embeds_by_tid.items():
        real_id = int(assignment.get(tid, -1))
        if real_id < 0:
            continue
        cosines = gallery.update(real_id, slot_embeds, alpha=ema_alpha)
        for slot_idx, cos in cosines.items():
            cos_str = f"{cos:.3f}" if not (isinstance(cos, float) and np.isnan(cos)) else "new"
            log(f"  real_id {real_id}  slot [{slot_name(slot_idx)}]  "
                f"cosine(old,new)={cos_str}")
        total_updates += len(slot_embeds)

    log(f"  {total_updates} slot updates across {len(slot_embeds_by_tid)} temp_ids")

    if not dry_run:
        gallery.save(gallery_dir, modality)
        log(f"  Saved gallery_pose_{modality}.npy  ({len(gallery.embeds)} cows)")
        if gallery.embeds:
            log(gallery.summary())
    else:
        log("  [dry_run] gallery not saved")
