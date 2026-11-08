"""
kinetic_match.py — match camera temp_ids to collar AnimalIds by kinetics correlation.

The ONE implementation of the Kinetic Matcher. Used by:
  reconcile.py       — kinetic_matcher(), Stage 2 identity (DB-backed)
  match_identity.py  — standalone CLI over exported tracks.csv / kinetics.csv

Strategy:
  - Keep "stable" temp_ids: present in ≥ min_temp_id_frames of the session's frames
  - Divide the session into bin_minutes bins aligned to the kinetics interval
  - Per bin: camera centroid displacement per temp_id, KineticsCountR delta per animal
  - Per (temp_id, AnimalId): Pearson r over the animal's ACTIVE bins
    (delta ≥ activity_pct quantile), needing ≥ max(min_active_bins, 2) bins
  - Greedy one-to-one assignment by r, only where r ≥ corr_threshold

Thresholds come from config/params.yaml (kinetic_match:, timeline.bin_minutes).
Nothing here logs or prints — callers decide what to show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import config


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def stable_temp_ids(tracks: pd.DataFrame, min_frac: float) -> list:
    """temp_ids present in at least min_frac of the session's distinct frames."""
    total_frames = tracks["frame_index"].nunique()
    if total_frames == 0:
        return []
    counts = tracks.groupby("temp_id")["frame_index"].nunique()
    return counts[counts / total_frames >= min_frac].index.tolist()


def make_bins(frame_datetimes: pd.Series, bin_minutes: int) -> pd.DatetimeIndex:
    """bin_minutes-aligned bin edges covering the given timestamps."""
    t_start = frame_datetimes.min().floor(f"{bin_minutes}min")
    t_end   = frame_datetimes.max().ceil(f"{bin_minutes}min")
    return pd.date_range(start=t_start, end=t_end, freq=f"{bin_minutes}min")


def camera_displacement_per_bin(tracks: pd.DataFrame,
                                bins: pd.DatetimeIndex) -> pd.DataFrame:
    """[bin, temp_id, displacement] — summed centroid step length (px) per bin."""
    df = tracks.sort_values(["temp_id", "frame_datetime"]).copy()
    df["bin"] = pd.cut(df["frame_datetime"], bins=bins, right=False, labels=bins[:-1])
    df = df.dropna(subset=["bin"])
    # Categorical bin labels → Timestamp, or the merge with the kinetics deltas
    # (plain Timestamp bins) in compute_scores finds nothing.
    df["bin"] = df["bin"].astype("datetime64[ns]")
    rows = []
    for (tid, b), grp in df.groupby(["temp_id", "bin"], observed=True):
        grp = grp.sort_values("frame_datetime")
        dx = grp["cx"].diff().abs()
        dy = grp["cy"].diff().abs()
        rows.append({"bin": b, "temp_id": tid,
                     "displacement": np.sqrt(dx**2 + dy**2).sum()})
    return pd.DataFrame(rows, columns=["bin", "temp_id", "displacement"])


def kinetics_delta_per_bin(kinetics: pd.DataFrame,
                           bins: pd.DatetimeIndex) -> pd.DataFrame:
    """[bin, AnimalId, delta] — KineticsCountR increase per bin (last reading
    before bin end minus last reading before bin start; negative → 0)."""
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
                delta = 0   # sensor reset or gap
            rows.append({"bin": t0, "AnimalId": aid, "delta": delta})
    return pd.DataFrame(rows, columns=["bin", "AnimalId", "delta"])


def compute_scores(cam_disp: pd.DataFrame, kin_delta: pd.DataFrame,
                   activity_pct: float, min_active_bins: int) -> pd.DataFrame:
    """[temp_id, AnimalId, correlation, n_bins, p_value, note] for every pair
    that shares at least one bin. correlation is NaN when not computable."""
    results = []
    for tid, aid in product(cam_disp["temp_id"].unique(), kin_delta["AnimalId"].unique()):
        cam = cam_disp[cam_disp["temp_id"] == tid][["bin", "displacement"]]
        kin = kin_delta[kin_delta["AnimalId"] == aid][["bin", "delta"]]
        merged = cam.merge(kin, on="bin", how="inner")
        if merged.empty:
            continue
        # activity filter — drop bins where the animal was lying/inactive
        thresh = merged["delta"].quantile(activity_pct)
        active = merged[merged["delta"] >= thresh]
        n = len(active)
        # Pearson r needs ≥ 2 points whatever min_active_bins says
        if n < max(min_active_bins, 2):
            results.append({"temp_id": tid, "AnimalId": aid, "correlation": np.nan,
                            "n_bins": n, "p_value": np.nan,
                            "note": f"only {n} active bins"})
            continue
        if active["displacement"].std() == 0 or active["delta"].std() == 0:
            results.append({"temp_id": tid, "AnimalId": aid, "correlation": np.nan,
                            "n_bins": n, "p_value": np.nan, "note": "zero variance"})
            continue
        r, p = pearsonr(active["displacement"], active["delta"])
        results.append({"temp_id": tid, "AnimalId": aid,
                        "correlation": round(float(r), 4), "n_bins": n,
                        "p_value": round(float(p), 4), "note": "ok"})
    return pd.DataFrame(results, columns=["temp_id", "AnimalId", "correlation",
                                          "n_bins", "p_value", "note"])


def greedy_assign(scores: pd.DataFrame, corr_threshold: float) -> dict:
    """One-to-one: highest r first; each temp_id and AnimalId used once."""
    assignment, used_aids, used_tids = {}, set(), set()
    valid = scores[scores["correlation"] >= corr_threshold].sort_values(
        "correlation", ascending=False)
    for _, row in valid.iterrows():
        tid, aid = row["temp_id"], row["AnimalId"]
        if tid in used_tids or aid in used_aids:
            continue
        assignment[tid] = aid
        used_tids.add(tid)
        used_aids.add(aid)
    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# Whole match
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KineticMatchResult:
    assignment: dict                      # {temp_id -> AnimalId}
    corr: dict                            # {temp_id -> r that produced its assignment}
    scores: pd.DataFrame                  # every scored pair (compute_scores columns)
    stable_tids: list                     # temp_ids that passed min_temp_id_frames
    bins: pd.DatetimeIndex | None = None  # bin edges used (None if nothing to bin)
    warnings: list = field(default_factory=list)

    @property
    def unmatched(self) -> list:
        return sorted(set(self.stable_tids) - set(self.assignment))

    def matrix(self) -> pd.DataFrame:
        """temp_id × AnimalId correlation pivot (empty if nothing was scored)."""
        if self.scores.empty:
            return pd.DataFrame()
        return self.scores.pivot_table(index="temp_id", columns="AnimalId",
                                       values="correlation", aggfunc="first")


def match(tracks: pd.DataFrame, kinetics: pd.DataFrame) -> KineticMatchResult:
    """
    tracks   : frame_index, frame_datetime (datetime64), temp_id, cx, cy
    kinetics : AnimalId, datetime (datetime64), KineticsCountR
    """
    km = config.params().kinetic_match
    bin_minutes = config.params().timeline.bin_minutes
    empty = pd.DataFrame(columns=["temp_id", "AnimalId", "correlation",
                                  "n_bins", "p_value", "note"])

    stable = stable_temp_ids(tracks, km.min_temp_id_frames)
    if not stable:
        return KineticMatchResult({}, {}, empty, [],
                                  warnings=["no stable temp_ids"])
    tracks = tracks[tracks["temp_id"].isin(stable)]

    bins = make_bins(tracks["frame_datetime"], bin_minutes)
    warnings = []
    if len(bins) < 3:
        warnings.append("fewer than 2 complete bins — correlation will be unreliable")

    cam_disp  = camera_displacement_per_bin(tracks, bins)
    kin_delta = kinetics_delta_per_bin(kinetics, bins)
    if cam_disp.empty or kin_delta.empty:
        warnings.append("empty displacement or kinetics signal — no matches possible")
        return KineticMatchResult({}, {}, empty, stable, bins, warnings)

    scores = compute_scores(cam_disp, kin_delta, km.activity_pct, km.min_active_bins)
    assignment = greedy_assign(scores, km.corr_threshold)

    corr = {}
    for tid, aid in assignment.items():
        row = scores[(scores["temp_id"] == tid) & (scores["AnimalId"] == aid)]
        if not row.empty:
            corr[tid] = float(row.iloc[0]["correlation"])

    return KineticMatchResult(assignment, corr, scores, stable, bins, warnings)
