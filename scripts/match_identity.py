#!/usr/bin/env python3
"""
match_identity.py — standalone CLI: match camera temp_ids to collar AnimalIds
by kinetics correlation, from exported CSVs.

The matching itself is kinetic_match.match() — the same code reconcile.py's
Kinetic Matcher runs, so results are identical for the same tracks, kinetics
and params. This script only adds CSV I/O and printing.

Scope: the one script that intentionally does NOT go through drive_manager.py
and never touches the database — an ad-hoc debugging tool for any exported
tracks.csv / kinetics.csv pair (see the Databases section of the README).

Thresholds come from config/params.yaml (kinetic_match:, timeline.bin_minutes).
Use --params to try other values.

Usage:
    python3 match_identity.py \\
        --tracks   tracks.csv \\
        --kinetics kinetic_data_6558_7509_7774.csv \\
        --output   tracks_identified.csv \\
        [--scores_json scores.json] [--params my_params.yaml]

tracks.csv needs: frame_index, frame_datetime, temp_id, cx, cy
kinetics CSV needs: AnimalId, datetime, KineticsCountR
"""

import argparse
from pathlib import Path

import pandas as pd

import config
import kinetic_match


def log(msg):
    print(f"[match] {msg}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks",   required=True, help="tracks.csv with frame_datetime column")
    ap.add_argument("--kinetics", required=True, help="kinetic_data_*.csv")
    ap.add_argument("--output",   required=True, help="output CSV path")
    ap.add_argument("--scores_json", default="",
                    help="Optional path to write the full score table as JSON")
    config.add_params_arg(ap)
    return ap.parse_args()


def run(tracks_path, kinetics_path):
    """Library entry point. Returns (assignment {temp_id -> AnimalId}, scores DataFrame)."""
    tracks   = pd.read_csv(tracks_path, parse_dates=["frame_datetime"])
    kinetics = pd.read_csv(kinetics_path, parse_dates=["datetime"])
    res = kinetic_match.match(tracks, kinetics)
    return res.assignment, res.scores


def main():
    args = parse_args()
    config.init(args.params)
    km = config.params().kinetic_match
    bin_minutes = config.params().timeline.bin_minutes

    log(f"Loading tracks:   {args.tracks}")
    tracks = pd.read_csv(args.tracks, parse_dates=["frame_datetime"])
    log(f"Loading kinetics: {args.kinetics}")
    kinetics = pd.read_csv(args.kinetics, parse_dates=["datetime"])

    res = kinetic_match.match(tracks, kinetics)

    log(f"Stable temp_ids (>={km.min_temp_id_frames*100:.0f}% frames): {sorted(res.stable_tids)}")
    if res.bins is not None:
        log(f"Time bins: {len(res.bins)-1} x {bin_minutes}-min windows "
            f"from {res.bins[0]} to {res.bins[-1]}")
    for w in res.warnings:
        log(f"WARNING: {w}.")
        if w.startswith("fewer than 2"):
            log("         Provide a longer video (ideally 45-60+ minutes) for robust matching.")

    print("\n=== Correlation matrix (temp_id x AnimalId) ===")
    print(res.matrix().to_string())
    print()
    print("=== Full score table ===")
    print(res.scores.sort_values("correlation", ascending=False).to_string(index=False))
    print()

    print(f"=== Assignments (threshold={km.corr_threshold}) ===")
    if res.assignment:
        for tid, aid in sorted(res.assignment.items()):
            row = res.scores[(res.scores["temp_id"] == tid)
                             & (res.scores["AnimalId"] == aid)].iloc[0]
            print(f"  temp_id {tid:>3}  →  AnimalId {aid}  "
                  f"(r={row['correlation']:.3f}, n_bins={int(row['n_bins'])})")
    else:
        print("  No confident assignments found. Try lowering kinetic_match.corr_threshold "
              "in params.yaml or providing a longer video.")
    if res.unmatched:
        print(f"  Unmatched temp_ids: {res.unmatched} "
              f"(below threshold or insufficient active bins)")
    print()

    out = tracks.copy()
    out["animal_id"] = out["temp_id"].map(res.assignment).astype("Int64")
    out_path = Path(args.output)
    out.to_csv(out_path, index=False)
    n_assigned = int(out["animal_id"].notna().sum())
    log(f"Saved: {out_path}  ({n_assigned} rows assigned)")
    if len(out):
        log(f"Coverage: {n_assigned / len(out) * 100:.1f}% of rows assigned an animal_id")

    if args.scores_json:
        res.scores.to_json(args.scores_json, orient="records", indent=2)
        log(f"Scores saved to: {args.scores_json}")


if __name__ == "__main__":
    main()