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


def weighted_mean_embed(rows: pd.DataFrame, col: str = "embed") -> np.ndarray:
    """Quality-weighted mean of the embed arrays, L2-normalised. Falls back to
    a plain mean when no weights are attached or every weight is 0."""
    stack = np.stack(rows[col].values).astype(np.float32)
    w = rows[QUALITY_COL].to_numpy(float) if QUALITY_COL in rows.columns else None
    if w is not None and w.sum() > 0:
        mean = (stack * w[:, None]).sum(axis=0) / w.sum()
    else:
        mean = stack.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm) if norm > 1e-8 else mean
