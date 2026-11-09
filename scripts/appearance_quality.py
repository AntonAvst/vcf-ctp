"""
appearance_quality.py — per-frame weight for how much a detection's APPEARANCE
(its embedding) can be trusted.

A small box is a low-resolution crop that the embedder upscales to its input
size, so its embedding is mostly noise; a box cut off by the frame border
shows only part of the cow. Every step that pools embeddings weights each
frame by this — gallery means, cosine queries, pose slots, the boundary
embedding — and minimum-frame thresholds count EFFECTIVE frames (the sum of
weights). Positions and kinetics never use it: a small far-away cow still
moves exactly as her collar says.

    size  = sqrt(w · h) of the box, in px (geometric size, so a lying cow seen
            from the side isn't penalised for a short box)
    w     = 0                                   if size <  min_px   (hard cutoff)
          = (size - min_px) / (full_px - min_px) if min_px <= size < full_px
          = 1                                   if size >= full_px
    × edge_weight if the box is within edge_margin_px of the frame border
      (only when the frame size is known)

Parameters: params.yaml appearance_quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

QUALITY_COL = "q"     # column name used for the weight wherever it is attached


def frame_weights(boxes: pd.DataFrame,
                  frame_w: float | None = None,
                  frame_h: float | None = None) -> np.ndarray:
    """Weight in [0, 1] per row of boxes (columns x1, y1, x2, y2, pixels)."""
    aq = config.params().appearance_quality
    x1 = boxes["x1"].to_numpy(float)
    y1 = boxes["y1"].to_numpy(float)
    x2 = boxes["x2"].to_numpy(float)
    y2 = boxes["y2"].to_numpy(float)
    size = np.sqrt(np.clip((x2 - x1) * (y2 - y1), 0.0, None))
    q = np.clip((size - aq.min_px) / (aq.full_px - aq.min_px), 0.0, 1.0)
    q[size < aq.min_px] = 0.0

    if frame_w and frame_h and not (pd.isna(frame_w) or pd.isna(frame_h)):
        m = aq.edge_margin_px
        at_edge = ((x1 <= m) | (y1 <= m) |
                   (x2 >= float(frame_w) - m) | (y2 >= float(frame_h) - m))
        q = np.where(at_edge, q * aq.edge_weight, q)
    return q


def effective_frames(rows: pd.DataFrame) -> float:
    """Sum of quality weights (raw row count when no weights are attached)."""
    if QUALITY_COL in rows.columns:
        return float(rows[QUALITY_COL].sum())
    return float(len(rows))


def _wmean(stack: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Quality-weighted mean, L2-normalised. Falls back to a plain mean when
    every weight is 0 or absent. Returns a zero vector for an empty stack
    (caller decides whether that's reachable)."""
    if len(stack) == 0:
        return np.zeros(128, dtype=np.float32)
    if w is None or w.sum() <= 0:
        mean = stack.mean(axis=0)
    else:
        mean = (stack * w[:, None]).sum(axis=0) / w.sum()
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 1e-8 else mean.astype(np.float32)


def _l2_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


@dataclass
class MeanEmbedReport:
    """Full result of mean_embed_report(): the (now outlier-trimmed) session
    mean, plus diagnostics for a likely tracker ID switch. See params.yaml
    appearance_quality.robust_mean / id_switch."""
    vector: np.ndarray
    n_total: float              # effective frames before trimming
    n_trimmed: float            # effective frames excluded as outliers
                                 # (n_total - n_trimmed = frames behind `vector`)
    trim_frac: float            # n_trimmed / n_total (0 if n_total == 0)
    split_suspected: bool
    split_gap: float | None            # within-cluster cosine - between-cluster cosine
    split_cluster_frac: float | None   # size of the smaller cluster, as a frame-weight fraction
    split_interleave_frac: float | None = None  # frame weight on the wrong side of the best
                                                # single switch point (0 = clean before/after)
    switch_frame: int | None = None    # frame_index of the first frame after that point


def _trim_outliers(stack: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Boolean keep-mask: drop frames whose cosine to the preliminary
    (untrimmed) mean falls more than trim_mad_k robust SDs below the median,
    worst-first, capped at max_trim_frac of the effective-frame weight."""
    rm = config.params().appearance_quality.robust_mean
    keep = np.ones(len(stack), dtype=bool)
    if not rm.enabled or w.sum() <= 0 or w.sum() < rm.min_frames_for_trim or len(stack) < 3:
        return keep

    prelim = _wmean(stack, w)
    cos_to_prelim = stack @ prelim
    med = np.median(cos_to_prelim)
    # 1.4826 × MAD estimates the standard deviation for normal data, so
    # trim_mad_k reads as "robust SDs below the median".
    sd = 1.4826 * np.median(np.abs(cos_to_prelim - med))
    if sd <= 1e-9:
        return keep   # every frame already agrees — nothing to trim
    floor = med - rm.trim_mad_k * sd
    candidate = cos_to_prelim < floor
    if not candidate.any():
        return keep

    max_drop_w = rm.max_trim_frac * w.sum()
    dropped_w = 0.0
    for i in np.argsort(cos_to_prelim):     # worst (lowest cosine) first
        if not candidate[i]:
            break
        if dropped_w + w[i] > max_drop_w:
            break
        keep[i] = False
        dropped_w += w[i]
    return keep


def _two_groups(stack: np.ndarray, w: np.ndarray):
    """Spherical 2-means. Returns (labels, gap, smaller_group_frac) or None
    when the frames don't separate into two groups at all."""
    # Seeds in O(n) (no n×n similarity matrix — a long track has tens of
    # thousands of frames): the frame least like the mean, then the frame
    # least like that one.
    i = int(np.argmin(stack @ _wmean(stack, w)))
    j = int(np.argmin(stack @ stack[i]))
    if i == j:
        return None
    c0, c1 = _l2_unit(stack[i]), _l2_unit(stack[j])
    labels = None
    for _ in range(10):
        new_labels = (stack @ c1) > (stack @ c0)       # True → group 1
        if new_labels.all() or (~new_labels).all():
            return None
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        c0 = _wmean(stack[~labels], w[~labels])
        c1 = _wmean(stack[labels], w[labels])
    total = w.sum()
    if total <= 0:
        return None
    frac = float(min(w[~labels].sum(), w[labels].sum()) / total)
    within = (np.sum((stack[~labels] @ c0) * w[~labels])
              + np.sum((stack[labels] @ c1) * w[labels])) / total
    return labels, float(within - float(c0 @ c1)), frac


def _best_switch_point(labels: np.ndarray, w: np.ndarray, frames: np.ndarray):
    """Frames in time order. Finds the single split point t that best explains
    the labels as "one group before t, the other after" (either way round).
    Returns (wrong-side weight fraction, frame_index at t). O(n)."""
    order = np.argsort(frames, kind="stable")
    lab, ww, fr = labels[order], w[order], frames[order]
    g1 = np.concatenate([[0.0], np.cumsum(ww * lab)])        # group-1 weight before t
    g0 = np.concatenate([[0.0], np.cumsum(ww * ~lab)])       # group-0 weight before t
    tot1, tot0 = g1[-1], g0[-1]
    # 0-then-1: group-1 frames before t and group-0 frames after t are wrong
    cost_01 = g1 + (tot0 - g0)
    cost_10 = g0 + (tot1 - g1)
    costs = np.minimum(cost_01, cost_10)
    t = int(np.argmin(costs))
    total = tot0 + tot1
    frame_at_t = int(fr[t]) if t < len(fr) else int(fr[-1])
    return (float(costs[t] / total) if total > 0 else 1.0), frame_at_t


def _check_split(stack: np.ndarray, w: np.ndarray, frames: np.ndarray | None):
    """Likely tracker ID switch: the frames form two well-separated, non-
    trivial groups AND, in time order, one group comes before a single frame
    and the other after it. One cow whose two sides embed differently gives
    two groups too, but they alternate through the track and fail the time
    test. Without frame indices the time test can't run → never flags.
    A pure diagnostic for gallery_ledger.id_switch_flags.
    Returns (suspected, gap, smaller_frac, interleave_frac, switch_frame)."""
    isc = config.params().appearance_quality.id_switch
    none = (False, None, None, None, None)
    if not isc.enabled or len(stack) < 4 or w.sum() < isc.min_frames:
        return none
    groups = _two_groups(stack, w)
    if groups is None:
        return none
    labels, gap, frac = groups
    if frac < isc.min_cluster_frac or gap < isc.min_gap:
        return False, gap, frac, None, None
    if frames is None:
        return False, gap, frac, None, None
    inter, switch_frame = _best_switch_point(labels, w, frames)
    return inter <= isc.max_interleave_frac, gap, frac, inter, switch_frame


def mean_embed_report(rows: pd.DataFrame, col: str = "embed",
                      check_split: bool = True) -> MeanEmbedReport:
    """Quality-weighted, outlier-trimmed mean embedding for one track's
    frames in one camera mode, plus (check_split) a two-cluster split
    diagnostic. Use this where the diagnostics are acted on (gallery commit);
    weighted_mean_embed() elsewhere."""
    stack = np.stack(rows[col].values).astype(np.float32)
    w = (rows[QUALITY_COL].to_numpy(float) if QUALITY_COL in rows.columns
         else np.ones(len(stack), dtype=np.float32))
    n_total = float(w.sum()) if w.sum() > 0 else float(len(stack))

    keep = _trim_outliers(stack, w)
    vector = _wmean(stack[keep], w[keep]) if keep.any() else _wmean(stack, w)
    n_trimmed = float(w[~keep].sum())

    frames = (rows["frame_index"].to_numpy() if "frame_index" in rows.columns else None)
    if check_split:
        split_suspected, split_gap, split_frac, inter, switch_frame = _check_split(
            stack[keep], w[keep], frames[keep] if frames is not None else None)
    else:
        split_suspected, split_gap, split_frac, inter, switch_frame = (False, None, None, None, None)

    return MeanEmbedReport(
        vector=vector, n_total=n_total, n_trimmed=n_trimmed,
        trim_frac=(n_trimmed / n_total) if n_total > 0 else 0.0,
        split_suspected=split_suspected, split_gap=split_gap,
        split_cluster_frac=split_frac, split_interleave_frac=inter,
        switch_frame=switch_frame,
    )


def weighted_mean_embed(rows: pd.DataFrame, col: str = "embed") -> np.ndarray:
    """Quality-weighted, outlier-trimmed mean of the embed arrays, L2-
    normalised. For callers that only want the vector (cosine queries,
    boundary embeddings, ...) — no split check."""
    return mean_embed_report(rows, col, check_split=False).vector