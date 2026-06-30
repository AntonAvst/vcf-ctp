#!/usr/bin/env python3
"""
match_identity.py — match camera temp_ids to sensor AnimalIds using kinetics correlation.

Note on scope: this is the one script in the pipeline that intentionally does
NOT go through drive_manager.py. It's a standalone correlation utility for
ad-hoc debugging against any exported tracks.csv / kinetics.csv pair — by
design it never touches the database (see Databases section of the project
README). Inside the actual pipeline, reconcile.py and display_tracks.py never
invoke this CLI; they call score_up_to()/compute_scores() directly with
in-memory DataFrames that *they* already sourced from drive_manager. If you
want this CLI mode itself to stop accepting raw file paths too, it would need
to take --session and call dm.load_collar_data(...) / read tracks from the DB
instead of --tracks/--kinetics — ask if you want that version instead.

Strategy:
  - Divide the overlapping time window into 15-minute bins (aligned to kinetics intervals)
  - For each bin: compute camera displacement per temp_id and kinetics delta per animal
  - For each (temp_id, AnimalId) pair: compute Pearson correlation across all bins
  - Only assign if correlation >= --corr_threshold AND active bins >= --min_active_bins
  - Output: tracks.csv with a new `animal_id` column filled where confident

Usage:
    python3 match_identity.py \\
        --tracks   tracks.csv \\
        --kinetics kinetic_data_6558_7509_7774.csv \\
        --output   tracks_identified.csv \\
        --corr_threshold 0.7 \\
        --min_active_bins 3 \\
        --min_temp_id_frames 0.10 \\
        --activity_pct 0.25

Requirements: pip install pandas numpy scipy
"""

import argparse
import json
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import pearsonr
from pathlib import Path


# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks",   required=True, help="tracks.csv with frame_datetime column")
    ap.add_argument("--kinetics", required=True, help="kinetic_data_*.csv")
    ap.add_argument("--output",   required=True, help="output CSV path")

    ap.add_argument(
        "--corr_threshold", type=float, default=0.7,
        help="Min Pearson r to accept a temp_id → AnimalId assignment (default: 0.7)"
    )
    ap.add_argument(
        "--min_active_bins", type=int, default=3,
        help="Min number of kinetics bins where the animal was active (default: 3). "
             "Prevents matching during long lying/inactive periods."
    )
    ap.add_argument(
        "--min_temp_id_frames", type=float, default=0.10,
        help="Min fraction of total frames a temp_id must appear in to be considered "
             "a real cow (default: 0.10 = 10%%). Filters tracker noise."
    )
    ap.add_argument(
        "--activity_pct", type=float, default=0.25,
        help="A kinetics bin is 'active' if its delta is >= this percentile of all "
             "deltas for that animal (default: 0.25). Filters lying periods."
    )
    ap.add_argument(
        "--bin_minutes", type=int, default=15,
        help="Bin width in minutes, should match kinetics sampling interval (default: 15)"
    )
    ap.add_argument(
        "--scores_json", default="",
        help="Optional path to write the full score table as JSON (for display_tracks.py)"
    )
    return ap.parse_args()


# -------------------- helpers --------------------

def log(msg):
    print(f"[match] {msg}", flush=True)


def camera_displacement_per_bin(tracks: pd.DataFrame, bins: pd.DatetimeIndex) -> pd.DataFrame:
    """
    For each (temp_id, time_bin), compute total centroid displacement in pixels.
    Returns a DataFrame with columns: [bin, temp_id, displacement]
    """
    df = tracks.copy()
    df = df.sort_values(["temp_id", "frame_datetime"])
    df["bin"] = pd.cut(df["frame_datetime"], bins=bins, right=False, labels=bins[:-1])
    df = df.dropna(subset=["bin"])

    rows = []
    for (tid, b), grp in df.groupby(["temp_id", "bin"], observed=True):
        grp = grp.sort_values("frame_datetime")
        dx = grp["cx"].diff().abs()
        dy = grp["cy"].diff().abs()
        disp = np.sqrt(dx**2 + dy**2).sum()
        rows.append({"bin": b, "temp_id": tid, "displacement": disp})

    return pd.DataFrame(rows)


def kinetics_delta_per_bin(kinetics: pd.DataFrame, bins: pd.DatetimeIndex) -> pd.DataFrame:
    """
    For each (AnimalId, time_bin), compute KineticsCountR delta.
    Uses the last reading before bin_end minus last reading before bin_start.
    Returns a DataFrame with columns: [bin, AnimalId, delta]
    """
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
                delta = 0   # sensor reset or gap — treat as unknown, skip below
            rows.append({"bin": t0, "AnimalId": aid, "delta": delta})

    return pd.DataFrame(rows)


def compute_scores(cam_disp: pd.DataFrame, kin_delta: pd.DataFrame,
                   activity_pct: float, min_active_bins: int):
    """
    For every (temp_id, AnimalId) pair compute Pearson r across shared bins.
    Only use bins where the animal's delta >= activity_pct percentile (active bins).
    Returns a DataFrame: [temp_id, AnimalId, correlation, n_bins, p_value]
    """
    results = []

    temp_ids  = cam_disp["temp_id"].unique()
    animal_ids = kin_delta["AnimalId"].unique()

    for tid, aid in product(temp_ids, animal_ids):
        cam = cam_disp[cam_disp["temp_id"] == tid][["bin", "displacement"]]
        kin = kin_delta[kin_delta["AnimalId"] == aid][["bin", "delta"]]

        merged = cam.merge(kin, on="bin", how="inner")
        if merged.empty:
            continue

        # activity filter — drop bins where the animal was lying/inactive
        thresh = merged["delta"].quantile(activity_pct)
        active = merged[merged["delta"] >= thresh]

        n_active = len(active)
        if n_active < min_active_bins:
            results.append({
                "temp_id": tid, "AnimalId": aid,
                "correlation": np.nan, "n_bins": n_active,
                "p_value": np.nan, "note": f"only {n_active} active bins"
            })
            continue

        if active["displacement"].std() == 0 or active["delta"].std() == 0:
            results.append({
                "temp_id": tid, "AnimalId": aid,
                "correlation": np.nan, "n_bins": n_active,
                "p_value": np.nan, "note": "zero variance in active bins"
            })
            continue

        r, p = pearsonr(active["displacement"], active["delta"])
        results.append({
            "temp_id": tid, "AnimalId": aid,
            "correlation": round(r, 4), "n_bins": n_active,
            "p_value": round(p, 4), "note": "ok"
        })

    return pd.DataFrame(results)


def assign_identities(scores: pd.DataFrame, corr_threshold: float) -> dict:
    """
    Greedy assignment: sort by correlation descending, assign best unambiguous matches.
    A temp_id and AnimalId can each only be assigned once.
    Only assigns if correlation >= corr_threshold.
    Returns dict: {temp_id -> AnimalId}
    """
    assignment = {}
    used_animals = set()
    used_tids = set()

    valid = scores[scores["correlation"] >= corr_threshold].copy()
    valid = valid.sort_values("correlation", ascending=False)

    for _, row in valid.iterrows():
        tid = row["temp_id"]
        aid = row["AnimalId"]
        if tid in used_tids or aid in used_animals:
            continue
        assignment[tid] = aid
        used_tids.add(tid)
        used_animals.add(aid)

    return assignment


# -------------------- main --------------------

def run(tracks_path, kinetics_path,
        corr_threshold=0.7, min_active_bins=3,
        min_temp_id_frames=0.10, activity_pct=0.25,
        bin_minutes=15):
    """
    Callable entry point for display_tracks.py.
    Returns (assignment dict {temp_id -> AnimalId}, scores DataFrame).
    """
    import types
    args = types.SimpleNamespace(
        tracks=tracks_path,
        kinetics=kinetics_path,
        output=None,
        corr_threshold=corr_threshold,
        min_active_bins=min_active_bins,
        min_temp_id_frames=min_temp_id_frames,
        activity_pct=activity_pct,
        bin_minutes=bin_minutes,
        scores_json="",
    )
    return _run(args)


def main():
    args = parse_args()
    assignment, scores = _run(args)

    # write output CSV
    if args.output:
        tracks_out = pd.read_csv(args.tracks, parse_dates=["frame_datetime"])
        tracks_out["animal_id"] = tracks_out["temp_id"].map(assignment).astype("Int64")
        out_path = Path(args.output)
        tracks_out.to_csv(out_path, index=False)
        log(f"Saved: {out_path}  ({tracks_out['animal_id'].notna().sum()} rows assigned)")
        matched_pct = tracks_out["animal_id"].notna().sum() / len(tracks_out) * 100
        log(f"Coverage: {matched_pct:.1f}% of rows assigned an animal_id")

    if args.scores_json:
        scores.to_json(args.scores_json, orient="records", indent=2)
        log(f"Scores saved to: {args.scores_json}")


def _run(args):
    """Core logic shared by main() and run(). Returns (assignment, scores_df)."""
    args = parse_args() if args is None else args

    log(f"Loading tracks:   {args.tracks}")
    tracks = pd.read_csv(args.tracks, parse_dates=["frame_datetime"])

    log(f"Loading kinetics: {args.kinetics}")
    kinetics = pd.read_csv(args.kinetics, parse_dates=["datetime"])

    # ---- filter stable temp_ids ----
    total_frames = tracks["frame_index"].nunique()
    tid_counts = tracks.groupby("temp_id")["frame_index"].nunique()
    stable_tids = tid_counts[tid_counts / total_frames >= args.min_temp_id_frames].index.tolist()
    log(f"Stable temp_ids (>={args.min_temp_id_frames*100:.0f}% frames): {sorted(stable_tids)}")
    tracks = tracks[tracks["temp_id"].isin(stable_tids)]

    # ---- define time bins covering the overlap window ----
    t_start = tracks["frame_datetime"].min().floor(f"{args.bin_minutes}min")
    t_end   = tracks["frame_datetime"].max().ceil(f"{args.bin_minutes}min")
    bins = pd.date_range(start=t_start, end=t_end, freq=f"{args.bin_minutes}min")
    log(f"Time bins: {len(bins)-1} x {args.bin_minutes}-min windows from {t_start} to {t_end}")

    if len(bins) < 3:
        log("WARNING: fewer than 2 complete bins — correlation will be unreliable.")
        log("         Provide a longer video (ideally 45-60+ minutes) for robust matching.")

    # ---- compute signals ----
    log("Computing camera displacement per bin...")
    cam_disp = camera_displacement_per_bin(tracks, bins)

    log("Computing kinetics delta per bin...")
    kin_delta = kinetics_delta_per_bin(kinetics, bins)

    # ---- score all pairs ----
    log("Scoring all (temp_id, AnimalId) pairs...")
    scores = compute_scores(cam_disp, kin_delta, args.activity_pct, args.min_active_bins)

    # ---- print score matrix ----
    print("\n=== Correlation matrix (temp_id x AnimalId) ===")
    pivot = scores.pivot_table(index="temp_id", columns="AnimalId",
                               values="correlation", aggfunc="first")
    print(pivot.to_string())
    print()

    print("=== Full score table ===")
    print(scores.sort_values("correlation", ascending=False).to_string(index=False))
    print()

    # ---- assign ----
    assignment = assign_identities(scores, args.corr_threshold)

    print(f"=== Assignments (threshold={args.corr_threshold}) ===")
    if assignment:
        for tid, aid in sorted(assignment.items()):
            row = scores[(scores["temp_id"]==tid) & (scores["AnimalId"]==aid)].iloc[0]
            print(f"  temp_id {tid:>3}  →  AnimalId {aid}  "
                  f"(r={row['correlation']:.3f}, n_bins={int(row['n_bins'])})")
    else:
        print("  No confident assignments found. "
              "Try lowering --corr_threshold or providing a longer video.")

    unmatched_tids = set(stable_tids) - set(assignment.keys())
    if unmatched_tids:
        print(f"  Unmatched temp_ids: {sorted(unmatched_tids)} "
              f"(below threshold or insufficient active bins)")
    print()

    return assignment, scores



def score_up_to(tracks_df: "pd.DataFrame",
                kinetics_df: "pd.DataFrame",
                up_to_datetime,
                bin_minutes: int = 15,
                activity_pct: float = 0.25,
                min_active_bins: int = 1,
                min_temp_id_frames: float = 0.10) -> "tuple[dict, pd.DataFrame]":
    """
    Compute matching scores using only data up to `up_to_datetime`.
    Called on each new kinetics interval boundary during live playback.
    Returns (assignment dict, scores DataFrame) — same format as run().
    min_active_bins defaults to 1 here so early intervals still produce scores.
    """
    # filter tracks to window seen so far
    tracks_window = tracks_df[tracks_df["frame_datetime"] <= up_to_datetime].copy()
    if tracks_window.empty:
        return {}, pd.DataFrame()

    # filter stable temp_ids within this window
    total_frames = tracks_window["frame_index"].nunique()
    if total_frames == 0:
        return {}, pd.DataFrame()
    tid_counts = tracks_window.groupby("temp_id")["frame_index"].nunique()
    stable_tids = tid_counts[tid_counts / total_frames >= min_temp_id_frames].index.tolist()
    tracks_window = tracks_window[tracks_window["temp_id"].isin(stable_tids)]

    # bins from start of video up to now
    t_start = tracks_window["frame_datetime"].min().floor(f"{bin_minutes}min")
    t_end   = up_to_datetime.ceil(f"{bin_minutes}min")
    bins    = pd.date_range(start=t_start, end=t_end, freq=f"{bin_minutes}min")
    if len(bins) < 2:
        return {}, pd.DataFrame()

    cam_disp  = camera_displacement_per_bin(tracks_window, bins)
    kin_delta = kinetics_delta_per_bin(kinetics_df, bins)

    if cam_disp.empty or kin_delta.empty:
        return {}, pd.DataFrame()

    scores = compute_scores(cam_disp, kin_delta, activity_pct, min_active_bins)
    if scores.empty:
        return {}, scores

    assignment = assign_identities(scores, corr_threshold=0.7)
    return assignment, scores


if __name__ == "__main__":
    main()


# Example:
# python3 match_identity.py \
#   --tracks   tracks.csv \
#   --kinetics kinetic_data_6558_7509_7774.csv \
#   --output   tracks_identified.csv \
#   --corr_threshold 0.7 \
#   --min_active_bins 3 \
#   --min_temp_id_frames 0.10 \
#   --activity_pct 0.25
#
# Key parameters to tune:
#   --corr_threshold   lower (e.g. 0.5) if too few matches; raise (e.g. 0.85) for stricter matching
#   --min_active_bins  requires this many active kinetics intervals — needs longer video to satisfy
#   --activity_pct     fraction of bins considered "active" per animal — raise if cows are very inactive
#   --bin_minutes      should match your kinetics sampling interval (typically 15)