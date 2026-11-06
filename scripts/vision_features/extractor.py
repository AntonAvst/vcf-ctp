"""
vision_features/extractor.py
─────────────────────────────
Orchestrator for Stage 2-B: Vision Feature Extraction.

Called by reconcile.py after Kinetic Matcher and before Gallery Builder
(gallery update) and Sensor Sequencer.

Responsibilities
────────────────
1. Load raw kps + bbox data for a session from the kps.parquet file
   (or from raw_tracks in SQLite if parquet is absent — slower).
2. Run all feature extractors in vision_features/features/ in sequence.
3. Aggregate per-frame labels into per-window scalars.
4. Merge scalars into the timeline_df produced by sensor_sequencer.
5. Optionally update the pose-conditioned gallery (Gallery Builder extension).

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
    Updated PoseGallery         — saved to gallery_pose_{modality}.npy, one
                                  update per modality present in the session
                                  (raw_tracks.is_night is per frame).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import config
from .schema import (
    BACK_KPS, N_KP,
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

def _is_night_select(conn: sqlite3.Connection) -> str:
    """SELECT expression for raw_tracks.is_night — NULL AS is_night on a DB
    whose raw_tracks predates per-frame camera-mode detection."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(raw_tracks)").fetchall()}
    return "is_night" if "is_night" in cols else "NULL AS is_night"


def _fill_is_night(df: pd.DataFrame, fallback_is_night: bool) -> pd.DataFrame:
    """raw_tracks.is_night → bool column; NULL (legacy rows) → session fallback."""
    n_null = int(df["is_night"].isna().sum())
    if n_null:
        log(f"  {n_null} rows have no per-frame camera mode (tracked before "
            f"detection existed) — using session fallback is_night={fallback_is_night}")
    df["is_night"] = df["is_night"].fillna(int(fallback_is_night)).astype(bool)
    return df


def load_kps_for_session(
    session_id: str,
    kps_parquet: str | None,
    conn: sqlite3.Connection,
    fallback_is_night: bool = False,
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
        is_night  (bool — camera mode at that frame; NULLs → fallback_is_night)
    One row per detection.
    """
    is_night_sel = _is_night_select(conn)
    # ── Always load scalar columns from SQLite (bbox, datetime, det_conf) ────
    log("Loading scalar columns from SQLite (bbox, datetime, det_conf)...")
    scalar_df = pd.read_sql(
        f"""
        SELECT frame_index, frame_datetime, temp_id, det_conf,
               x1, y1, x2, y2, {is_night_sel}
        FROM   raw_tracks
        WHERE  session_id = ?
        ORDER  BY frame_index, temp_id
        """,
        conn, params=(session_id,), parse_dates=["frame_datetime"],
    )
    scalar_df = _fill_is_night(scalar_df, fallback_is_night)
    log(f"  {len(scalar_df)} scalar rows from SQLite")

    # ── Parquet path — load kps arrays and merge with scalar_df ──────────────
    if kps_parquet:
        _kp = Path(kps_parquet)
        _kps_parts = sorted(_kp.glob("kps_part*.parquet")) if _kp.is_dir() else ([_kp] if _kp.exists() else [])
    else:
        _kps_parts = []
    if _kps_parts:
        log(f"Loading kps arrays from {len(_kps_parts)} parquet part(s): {kps_parquet}")
        kdf = pd.concat([pd.read_parquet(p) for p in _kps_parts], ignore_index=True)
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
                        "x1", "y1", "x2", "y2", "kps_flat", "kps_kconf",
                        "is_night"]].copy()

    # ── SQLite fallback ────────────────────────────────────────────────────────
    # NOTE: track_and_dump.py's raw_tracks only ever stores kps_conf (scalar)
    # + kps_parquet_row (an integer pointer into kps_part*.parquet) — it never
    # writes kps/kps_kconf inline. So this fallback can only work against a
    # raw_tracks table that was created with an (incompatible) inline-array
    # schema; against the real production schema, those columns never exist.
    # If we get here, the kps parquet for this session is genuinely missing
    # (not just a path-resolution bug — check that first), and there is no
    # valid full-fidelity fallback: fail soft rather than crash the run.
    log("kps parquet not found — checking for an inline-kps raw_tracks fallback...")
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(raw_tracks)").fetchall()}
    except Exception:
        cols = set()
    if not {"kps", "kps_kconf"} <= cols:
        log("  raw_tracks has no inline kps/kps_kconf columns in this DB "
            "(expected — kps data lives only in kps_part*.parquet for this "
            "pipeline). No fallback is possible for this session.")
        log("  Vision features will be NULL for this session. If the parquet "
            "should exist, check the session's output directory / Drive sync "
            "before re-running.")
        return pd.DataFrame()

    rdf = pd.read_sql(
        f"""
        SELECT frame_index, frame_datetime, temp_id, det_conf,
               x1, y1, x2, y2, kps, kps_kconf, {is_night_sel}
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
    rdf = _fill_is_night(rdf, fallback_is_night)
    log(f"  {len(rdf)} rows with valid kps loaded from SQLite")
    return rdf[["frame_index", "frame_datetime", "temp_id", "det_conf",
                "x1", "y1", "x2", "y2", "kps_flat", "kps_kconf", "is_night"]].copy()


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
    kps_3d   = np.stack(kps_df["kps_flat"].values).reshape(N, N_KP, 3)  # (N, 19, 3)
    kps_kc   = np.stack(kps_df["kps_kconf"].values)                     # (N, 19)
    bbox     = kps_df[["x1","y1","x2","y2"]].values.astype(np.float32) # (N, 4)
    det_conf = kps_df["det_conf"].values.astype(np.float32)             # (N,)

    # ── Diagnostic: log distributions to help tune thresholds ───────────────
    back_kp_indices = [int(k) for k in BACK_KPS]
    back_confs = kps_kc[:, back_kp_indices]
    log(f"  kps_kconf back KPs {tuple(back_kp_indices)} — "
        f"min={back_confs.min():.3f}  p25={np.percentile(back_confs,25):.3f}  "
        f"median={np.median(back_confs):.3f}  p75={np.percentile(back_confs,75):.3f}  "
        f"max={back_confs.max():.3f}")
    log(f"  det_conf — "
        f"min={det_conf.min():.3f}  median={np.median(det_conf):.3f}  max={det_conf.max():.3f}")
    w = bbox[:,2] - bbox[:,0]; h = bbox[:,3] - bbox[:,1]
    ar = w / np.maximum(h, 1e-6)
    _pc = config.params().vision.posture
    smax, lmin = _pc.standing_max_ar, _pc.lying_min_ar
    log(f"  bbox aspect ratio (w/h) — "
        f"min={ar.min():.2f}  p25={np.percentile(ar,25):.2f}  "
        f"median={np.median(ar):.2f}  p75={np.percentile(ar,75):.2f}  "
        f"max={ar.max():.2f}  "
        f"[standing AR<={smax}: {(ar<=smax).sum()}  lying AR>={lmin}: {(ar>=lmin).sum()}  "
        f"uncertain: {((ar>smax)&(ar<lmin)).sum()}]")

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
    min_conf = config.params().vision.aggregate_min_conf
    posture_feats = aggregate_posture(
        posture=window_rows["posture"].values.astype(np.int8),
        posture_conf=window_rows["posture_conf"].values.astype(np.float32),
        min_conf=min_conf,
    )
    facing_feats = aggregate_facing(
        facing=window_rows["facing"].values.astype(np.int8),
        facing_conf=window_rows["facing_conf"].values.astype(np.float32),
        min_conf=min_conf,
    )
    return {**posture_feats, **facing_feats}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (called by reconcile.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_vision_features(
    session_id:        str,
    timeline_df:       pd.DataFrame,          # output of sensor_sequencer
    conn:              sqlite3.Connection,
    assignment:        dict,                   # {temp_id -> real_id}
    fallback_is_night: bool       = False,     # only for legacy rows with NULL is_night
    camera_id:         str        = "cam0",
    kps_parquet:       str | None = None,
    embed_parquet:     str | None = None,
    gallery_dir:       str        = "./reid_gallery",
    dry_run:           bool       = False,
    merge_map:         dict | None = None,   # {loser_tid -> winner_tid} from Duplicate Resolver
) -> pd.DataFrame:
    """
    Stage 2-B: Vision Feature Extraction.

    Parameters
    ----------
    session_id      : session being processed.
    timeline_df     : sensor-only timeline from sensor_sequencer.
    conn            : SQLite connection.
    assignment      : {temp_id -> real_id} from kinetic + cosine resolution.
    fallback_is_night : camera mode to assume for raw_tracks rows whose
                      is_night is NULL (sessions tracked before per-frame
                      detection existed). Rows that have it use their own.
    kps_parquet     : path to kps.parquet for this session (optional).
    embed_parquet   : path to embeds.parquet (for gallery slot update).
    gallery_dir     : directory containing gallery_pose_*.npy files.
    dry_run         : if True, skip all writes.

    Thresholds come from config/params.yaml, not arguments, so this step can
    never disagree with the rest of reconcile:
        gallery.ema_alpha      EMA learning rate for the pose-gallery update
        vision.min_slot_conf   min posture AND facing conf to feed a slot
        timeline.bin_minutes   window size (same grid as Sensor Sequencer)

    Returns
    -------
    timeline_df with new vision feature columns filled in.
    """
    section("Vision Feature Extraction")
    log(f"session_id: {session_id}  |  dry_run: {dry_run}")
    P = config.params()
    ema_alpha     = P.gallery.ema_alpha
    min_slot_conf = P.vision.min_slot_conf
    bin_minutes   = P.timeline.bin_minutes

    if timeline_df.empty:
        log("timeline_df is empty — nothing to do.")
        return timeline_df

    # ── Schema migration ──────────────────────────────────────────────────────
    if not dry_run:
        migrate_timeline_schema(conn)

    # ── Load kps data ─────────────────────────────────────────────────────────
    kps_df = load_kps_for_session(session_id, kps_parquet, conn,
                                  fallback_is_night=fallback_is_night)
    n_night = int(kps_df["is_night"].sum()) if not kps_df.empty else 0
    if not kps_df.empty:
        log(f"  Camera mode: {len(kps_df) - n_night} day rows, {n_night} night rows")
    if kps_df.empty:
        log("No kps data available — vision columns will remain NULL.")
        return timeline_df

    # Duplicate Resolver (resolve_duplicate_assignments) deletes loser temp_ids from
    # `assignment` — it only keeps {winner_tid: real_id}. kps_df still has
    # the losers' rows under their original temp_id, so without this remap
    # `kps_df["temp_id"].map(assignment)` below would silently classify every
    # merged-but-losing fragment as unresolved (NaN real_id), dropping it from
    # the timeline/gallery even though Duplicate Resolver already decided it belongs to
    # the same animal. Mirrors the equivalent embed_df remap in reconcile.py.
    if merge_map:
        n_remapped = int(kps_df["temp_id"].isin(merge_map.keys()).sum())
        if n_remapped:
            log(f"  Remapping {n_remapped} rows from {len(merge_map)} "
                f"merged (loser→winner) temp_id(s) before identity lookup")
            kps_df["temp_id"] = kps_df["temp_id"].replace(merge_map)

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
    _ep = Path(embed_parquet) if embed_parquet else None
    _embed_parts = (sorted(_ep.glob("embeds_part*.parquet")) if _ep and _ep.is_dir()
                    else ([_ep] if _ep and _ep.exists() else []))
    if _embed_parts:
        _update_pose_gallery(
            session_id   = session_id,
            camera_id    = camera_id,
            kps_df       = kps_df,       # ALL temp_ids — temp gallery needs unresolved too
            embed_parquet= embed_parquet,
            assignment   = assignment,
            gallery_dir  = gallery_dir,
            ema_alpha    = ema_alpha,
            min_slot_conf= min_slot_conf,
            dry_run      = dry_run,
            merge_map    = merge_map,
        )
    else:
        log("No embed_parquet provided or no parts found — pose gallery update skipped.")

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
    gallery_dir:   str,
    ema_alpha:     float,
    min_slot_conf: float,
    dry_run:       bool,
    merge_map:     dict | None = None,   # {loser_tid -> winner_tid} from Duplicate Resolver
) -> None:
    """
    Update both pose galleries after feature extraction:

      1. Build slot embeddings for ALL temp_ids (resolved + unresolved)
      2. Resolved temp_ids  → update PoseGallery (real_id keyed, full alpha)
      3. All temp_ids       → update TempPoseGallery (TempKey keyed, full alpha)
      4. Resolved temp_ids  → delete from TempPoseGallery (they have graduated)

    Each frame carries its own camera mode (kps_df.is_night), so the session's
    frames are split by it and steps 1-3 run once per modality present, each
    against that modality's files. A temp_id seen on both sides of the
    camera's colour↔IR switch feeds both galleries — which is how the night
    gallery learns identities confirmed in daylight. Modalities never mix:
    a day frame only ever updates a day slot.
    """
    log("Updating pose-conditioned galleries...")

    # ── Load embeddings (multi-part or single file) ───────────────────────────
    _ep = Path(embed_parquet)
    _parts = sorted(_ep.glob("embeds_part*.parquet")) if _ep.is_dir() else ([_ep] if _ep.exists() else [])
    if not _parts:
        log("  embed_parquet path not found — gallery update skipped.")
        return
    edf = pd.concat([pd.read_parquet(p) for p in _parts], ignore_index=True)
    if "session_id" in edf.columns:
        edf = edf[edf["session_id"] == session_id].copy()
    if edf.empty:
        log("  No embeds in parquet for this session — gallery update skipped.")
        return

    if not isinstance(edf["embed"].iloc[0], np.ndarray):
        edf["embed"] = edf["embed"].apply(
            lambda v: np.array(v, dtype=np.float32)
        )

    # edf is loaded fresh from the embed parquet here and still has RAW
    # (pre-merge) temp_id values. kps_df, passed in by the caller, has
    # already been remapped via merge_map. Without remapping edf the same
    # way, the frame_index+temp_id join below would silently drop every
    # merged-loser frame (edf says e.g. 313, kps_df says 1 for the same
    # frame_index — no match), so none of that pooled data would reach the
    # slot-embedding/gallery update.
    if merge_map:
        edf["temp_id"] = edf["temp_id"].replace(merge_map)

    # ── Merge embeds with per-frame labels ────────────────────────────────────
    merged = edf.merge(
        kps_df[["frame_index", "temp_id", "posture", "posture_conf",
                "facing", "facing_conf", "is_night"]],
        on=["frame_index", "temp_id"], how="inner",
    )
    if merged.empty:
        log("  No rows after merging embeds with kps labels — gallery update skipped.")
        return

    log(f"  {len(merged)} frames available for slot assignment "
        f"({merged['temp_id'].nunique()} temp_ids)")

    # Resolved temp_ids graduate out of the temp gallery. Include merge_map's
    # losers (e.g. t313, t263...), not just the winner (t1) that remains as
    # assignment's key — they graduated into the same real_id and their
    # raw-keyed TempPoseGallery entries are now stale.
    resolved_tids = set(int(t) for t in assignment.keys())
    if merge_map:
        resolved_tids |= set(int(t) for t in merge_map.keys())

    # Both modalities' temp galleries are always loaded: this session's old
    # entries are purged from BOTH first, so a re-run replaces them — including
    # entries left in the other modality's file by an earlier run whose
    # day/night labelling differed (e.g. the old clock-based rule).
    pose_galleries = {}
    temp_galleries = {m: TempPoseGallery.load(gallery_dir, m) for m in ("day", "night")}
    for m, tg in temp_galleries.items():
        purged = tg.delete_session(session_id)
        if purged:
            log(f"  TempPoseGallery[{m}]: purged {len(purged)} entries from an "
                f"earlier run of this session")

    for is_night_val, mdf in merged.groupby("is_night"):
        modality = "night" if bool(is_night_val) else "day"
        mdf = mdf.reset_index(drop=True)

        # ── Build slot embeddings for ALL temp_ids (this modality) ────────────
        slot_embeds_by_tid = build_slot_embeds(
            embed_df         = mdf[["temp_id", "embed"]],
            per_frame_labels = {
                "posture": mdf["posture"].values.astype(np.int8),
                "facing":  mdf["facing"].values.astype(np.int8),
            },
            frame_temp_ids   = mdf["temp_id"].values,
            min_conf_posture = mdf["posture_conf"].values.astype(np.float32),
            min_conf_facing  = mdf["facing_conf"].values.astype(np.float32),
            min_conf         = min_slot_conf,
        )
        if not slot_embeds_by_tid:
            log(f"  [{modality}] {len(mdf)} frames, no confident slot frames — skipped")
            continue
        log(f"  [{modality}] {len(mdf)} frames → slot embeddings for "
            f"{len(slot_embeds_by_tid)} temp_ids")

        # ── Update PoseGallery (resolved temp_ids only, full alpha) ───────────
        pose_gallery = PoseGallery.load(gallery_dir, modality)
        pose_updates = 0
        for tid, slot_embeds in slot_embeds_by_tid.items():
            real_id = assignment.get(int(tid))
            if real_id is None:
                continue
            cosines = pose_gallery.update(int(real_id), slot_embeds, alpha=ema_alpha)
            for slot_idx, cos in cosines.items():
                cos_str = f"{cos:.3f}" if not (isinstance(cos, float) and np.isnan(cos)) else "new"
                log(f"  [{modality}] real_id {real_id}  slot [{slot_name(slot_idx)}]  cos={cos_str}")
            pose_updates += len(slot_embeds)
        pose_galleries[modality] = pose_gallery
        log(f"  [{modality}] PoseGallery: {pose_updates} slot updates "
            f"({len(pose_gallery.embeds)} total entries)")

        # ── Update TempPoseGallery (ALL temp_ids, full alpha) ─────────────────
        temp_gallery = temp_galleries[modality]
        temp_updates = 0
        for tid, slot_embeds in slot_embeds_by_tid.items():
            key = TempKey(camera_id=camera_id, session_id=session_id, temp_id=int(tid))
            temp_gallery.update(key, slot_embeds, alpha=ema_alpha)
            temp_updates += len(slot_embeds)
        log(f"  [{modality}] TempPoseGallery: {temp_updates} slot updates")

    # ── Delete resolved temp_ids from BOTH TempPoseGalleries ──────────────────
    for m, tg in temp_galleries.items():
        deleted = tg.delete_resolved(session_id, resolved_tids)
        if deleted:
            log(f"  TempPoseGallery[{m}]: deleted {len(deleted)} resolved entries: "
                f"{[f't{k.temp_id}' for k in deleted]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    # Pose galleries only for modalities that were updated; temp galleries
    # always (both), since the purge above may have changed either.
    if not dry_run:
        for m, pg in pose_galleries.items():
            pg.save(gallery_dir, m)
            log(f"  Saved gallery_pose_{m}.npy  ({len(pg.embeds)} entries)")
            if pg.embeds:
                log(pg.summary())
        for m, tg in temp_galleries.items():
            tg.save(gallery_dir, m)
            log(f"  Saved temp_gallery_pose_{m}.npy  ({len(tg.embeds)} entries)")
            if tg.entries_for_camera(camera_id):
                log(tg.summary(camera_id))
    else:
        log("  [dry_run] galleries not saved")