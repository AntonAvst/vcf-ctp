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
from .gallery.pose_conditioned import (
    PoseGallery, TempPoseGallery, TempKey,
    build_slot_embeds, mint_synthetic_id, is_synthetic,
)


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

    kps.parquet stores only (session_id, frame_index, temp_id, kps, kps_kconf).
    bbox (x1/y1/x2/y2), frame_datetime, and det_conf live in SQLite raw_tracks.
    When parquet is available, kps arrays are loaded from there and the scalar
    columns are fetched from SQLite and merged in by (frame_index, temp_id).

    Returns DataFrame with columns:
        frame_index, frame_datetime, temp_id, det_conf,
        x1, y1, x2, y2,
        kps_flat  (np.ndarray shape (57,)  — flat [x,y,conf] × 19),
        kps_kconf (np.ndarray shape (19,)),
    One row per detection.
    """
    # ── Always load scalar columns from SQLite (bbox, datetime, det_conf) ────
    log("Loading scalar columns from SQLite (bbox, datetime, det_conf)...")
    scalar_df = pd.read_sql(
        """
        SELECT frame_index, frame_datetime, temp_id, det_conf,
               x1, y1, x2, y2
        FROM   raw_tracks
        WHERE  session_id = ?
        ORDER  BY frame_index, temp_id
        """,
        conn, params=(session_id,), parse_dates=["frame_datetime"],
    )
    log(f"  {len(scalar_df)} scalar rows from SQLite")

    # ── Parquet path — load kps arrays and merge with scalar_df ──────────────
    if kps_parquet and Path(kps_parquet).exists():
        log(f"Loading kps arrays from parquet: {kps_parquet}")
        kdf = pd.read_parquet(kps_parquet)
        if "session_id" in kdf.columns:
            kdf = kdf[kdf["session_id"] == session_id].copy()

        # Convert FixedSizeList → numpy arrays
        kdf["kps_flat"]  = kdf["kps"].apply(
            lambda v: np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v
        )
        kdf["kps_kconf"] = kdf["kps_kconf"].apply(
            lambda v: np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v
        )
        kdf = kdf[["frame_index", "temp_id", "kps_flat", "kps_kconf"]].copy()

        # Merge kps arrays onto scalar rows
        merged = scalar_df.merge(kdf, on=["frame_index", "temp_id"], how="inner")
        n_before = len(scalar_df)
        n_after  = len(merged)
        if n_after < n_before:
            log(f"  {n_before - n_after} rows dropped (no kps in parquet for those detections)")
        log(f"  {n_after} rows after merge")
        return merged[["frame_index", "frame_datetime", "temp_id", "det_conf",
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

    # ── Diagnostic: log distributions to help tune thresholds ───────────────
    back_kp_indices = [2, 3, 4, 5]   # WITHERS, SPINE_MID, SACRUM, TAIL_BASE
    back_confs = kps_kc[:, back_kp_indices]
    log(f"  kps_kconf back KPs (2,3,4,5) — "
        f"min={back_confs.min():.3f}  p25={np.percentile(back_confs,25):.3f}  "
        f"median={np.median(back_confs):.3f}  p75={np.percentile(back_confs,75):.3f}  "
        f"max={back_confs.max():.3f}")
    log(f"  det_conf — "
        f"min={det_conf.min():.3f}  median={np.median(det_conf):.3f}  max={det_conf.max():.3f}")
    w = bbox[:,2] - bbox[:,0]; h = bbox[:,3] - bbox[:,1]
    ar = w / np.maximum(h, 1e-6)
    log(f"  bbox aspect ratio (w/h) — "
        f"min={ar.min():.2f}  p25={np.percentile(ar,25):.2f}  "
        f"median={np.median(ar):.2f}  p75={np.percentile(ar,75):.2f}  "
        f"max={ar.max():.2f}  "
        f"[standing AR<0.85: {(ar<0.85).sum()}  lying AR>1.10: {(ar>1.10).sum()}  uncertain: {((ar>=0.85)&(ar<=1.10)).sum()}]")

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
    camera_id:         str        = "cam0",
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

    # ── Per-frame extraction (all temp_ids — no identity filter yet) ─────────
    log(f"Running per-frame extractors on all {len(kps_df)} rows "
        f"({kps_df['temp_id'].nunique()} temp_ids)...")
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

    kps_df["frame_datetime"]  = pd.to_datetime(kps_df["frame_datetime"])
    kps_df["window_start_dt"] = kps_df["frame_datetime"].dt.floor(f"{bin_minutes}min")

    # ── Map temp_id → real_id (after extraction — identity filter happens here) 
    # Rows with no real_id are kept for the pose gallery update (uses temp_id
    # directly) but excluded from the timeline merge (which requires real_id).
    kps_df["real_id"] = kps_df["temp_id"].map(assignment)

    resolved_df   = kps_df[kps_df["real_id"].notna()].copy()
    unresolved_df = kps_df[kps_df["real_id"].isna()].copy()

    log(f"  Resolved:   {len(resolved_df)} rows, "
        f"{resolved_df['temp_id'].nunique()} temp_ids → real_id")
    log(f"  Unresolved: {len(unresolved_df)} rows, "
        f"{unresolved_df['temp_id'].nunique()} temp_ids (features extracted, not written to timeline)")

    # ── Aggregate per (real_id, window) and merge into timeline_df ───────────
    if resolved_df.empty:
        log("No resolved rows — timeline vision columns will remain NULL.")
    else:
        resolved_df["real_id"] = resolved_df["real_id"].astype(int)

        log("Aggregating resolved rows per window...")
        window_rows = []
        for (rid, win), grp in resolved_df.groupby(["real_id", "window_start_dt"]):
            agg = aggregate_window(grp)
            agg["real_id"]         = int(rid)
            agg["window_start_dt"] = win
            window_rows.append(agg)

        if window_rows:
            vision_df = pd.DataFrame(window_rows)
            log(f"  Produced {len(vision_df)} window rows")

            timeline_df = timeline_df.copy()
            timeline_df["window_start_dt"] = pd.to_datetime(timeline_df["window_start_dt"])

            for col in TIMELINE_VISION_COLS:
                if col not in timeline_df.columns:
                    timeline_df[col] = None

            merge_cols = ["real_id", "window_start_dt"] + TIMELINE_VISION_COLS
            vision_df  = vision_df[[c for c in merge_cols if c in vision_df.columns]]
            timeline_df = timeline_df.merge(
                vision_df, on=["real_id", "window_start_dt"], how="left", suffixes=("", "_new")
            )
            for col in TIMELINE_VISION_COLS:
                new_col = col + "_new"
                if new_col in timeline_df.columns:
                    timeline_df[col] = timeline_df[col].where(
                        timeline_df[col].notna(), timeline_df[new_col]
                    )
                    timeline_df.drop(columns=[new_col], inplace=True)

            has_vision = timeline_df["lying_fraction"].notna()
            timeline_df.loc[has_vision, "modality_mask"] = (
                timeline_df.loc[has_vision, "modality_mask"].fillna(0).astype(int) | 2
            )
            log(f"  Vision features filled for {int(has_vision.sum())}/{len(timeline_df)} timeline rows")

    # ── Pose-conditioned gallery update (uses full kps_df incl. unresolved) ──
    if embed_parquet and Path(embed_parquet).exists():
        _update_pose_gallery(
            session_id   = session_id,
            camera_id    = camera_id,
            kps_df       = kps_df,       # ALL temp_ids — temp gallery needs unresolved too
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
    camera_id:     str,
    kps_df:        pd.DataFrame,    # ALL temp_ids, with posture/facing columns
    embed_parquet: str,
    assignment:    dict,            # {temp_id -> real_id} — resolved only
    is_night:      bool,
    gallery_dir:   str,
    ema_alpha:     float,
    min_slot_conf: float,
    dry_run:       bool,
) -> None:
    """
    Update both pose galleries after feature extraction:

      1. Build slot embeddings for ALL temp_ids (resolved + unresolved)
      2. Resolved temp_ids  → update PoseGallery (real_id keyed, full alpha)
      3. All temp_ids       → update TempPoseGallery (TempKey keyed, full alpha)
      4. Resolved temp_ids  → delete from TempPoseGallery (they have graduated)
    """
    log("Updating pose-conditioned galleries...")
    modality = "night" if is_night else "day"

    # ── Load embeddings ───────────────────────────────────────────────────────
    edf = pd.read_parquet(embed_parquet)
    if "session_id" in edf.columns:
        edf = edf[edf["session_id"] == session_id].copy()
    if edf.empty:
        log("  No embeds in parquet for this session — gallery update skipped.")
        return

    if not isinstance(edf["embed"].iloc[0], np.ndarray):
        edf["embed"] = edf["embed"].apply(
            lambda v: np.array(v, dtype=np.float32)
        )

    # ── Merge embeds with per-frame labels ────────────────────────────────────
    merged = edf.merge(
        kps_df[["frame_index", "temp_id", "posture", "posture_conf",
                "facing", "facing_conf"]],
        on=["frame_index", "temp_id"], how="inner",
    )
    if merged.empty:
        log("  No rows after merging embeds with kps labels — gallery update skipped.")
        return

    log(f"  {len(merged)} frames available for slot assignment "
        f"({merged['temp_id'].nunique()} temp_ids)")

    # ── Build slot embeddings for ALL temp_ids ────────────────────────────────
    slot_embeds_by_tid = build_slot_embeds(
        embed_df         = merged[["temp_id", "embed"]].reset_index(drop=True),
        per_frame_labels = {
            "posture": merged["posture"].values.astype(np.int8),
            "facing":  merged["facing"].values.astype(np.int8),
        },
        frame_temp_ids   = merged["temp_id"].values,
        min_conf_posture = merged["posture_conf"].values.astype(np.float32),
        min_conf_facing  = merged["facing_conf"].values.astype(np.float32),
        min_conf         = min_slot_conf,
    )

    if not slot_embeds_by_tid:
        log("  No confident slot frames found — gallery update skipped.")
        return

    log(f"  Slot embeddings built for {len(slot_embeds_by_tid)} temp_ids")

    # ── Update PoseGallery (resolved temp_ids only, full alpha) ───────────────
    pose_gallery  = PoseGallery.load(gallery_dir, modality)
    pose_updates  = 0
    for tid, slot_embeds in slot_embeds_by_tid.items():
        real_id = assignment.get(int(tid))
        if real_id is None:
            continue
        cosines = pose_gallery.update(int(real_id), slot_embeds, alpha=ema_alpha)
        for slot_idx, cos in cosines.items():
            cos_str = f"{cos:.3f}" if not (isinstance(cos, float) and np.isnan(cos)) else "new"
            log(f"  real_id {real_id}  slot [{slot_name(slot_idx)}]  cos={cos_str}")
        pose_updates += len(slot_embeds)
    log(f"  PoseGallery: {pose_updates} slot updates ({len(pose_gallery.embeds)} total entries)")

    # ── Update TempPoseGallery (ALL temp_ids, full alpha) ─────────────────────
    temp_gallery  = TempPoseGallery.load(gallery_dir, modality)
    temp_updates  = 0
    for tid, slot_embeds in slot_embeds_by_tid.items():
        key     = TempKey(camera_id=camera_id, session_id=session_id, temp_id=int(tid))
        cosines = temp_gallery.update(key, slot_embeds, alpha=ema_alpha)
        temp_updates += len(slot_embeds)
    log(f"  TempPoseGallery: {temp_updates} slot updates")

    # ── Delete resolved temp_ids from TempPoseGallery ─────────────────────────
    resolved_tids = set(int(t) for t in assignment.keys())
    deleted = temp_gallery.delete_resolved(session_id, resolved_tids)
    if deleted:
        log(f"  TempPoseGallery: deleted {len(deleted)} resolved entries: "
            f"{[f't{k.temp_id}' for k in deleted]}")

    # ── Save both galleries ───────────────────────────────────────────────────
    if not dry_run:
        pose_gallery.save(gallery_dir, modality)
        temp_gallery.save(gallery_dir, modality)
        log(f"  Saved gallery_pose_{modality}.npy  ({len(pose_gallery.embeds)} entries)")
        log(f"  Saved temp_gallery_pose_{modality}.npy  "
            f"({len(temp_gallery.embeds)} entries, "
            f"camera {camera_id})")
        if pose_gallery.embeds:
            log(pose_gallery.summary())
        cam_count = len(temp_gallery.entries_for_camera(camera_id))
        if cam_count:
            log(temp_gallery.summary(camera_id))
    else:
        log("  [dry_run] galleries not saved")