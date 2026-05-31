"""
vision_features/features/posture.py
─────────────────────────────────────
Per-frame posture classification: STANDING | LYING | UNCERTAIN.

Primary signal : bounding-box aspect ratio
                 A lying cow's bbox is much wider than it is tall.
                 Aspect ratio = width / height.
                 standing → aspect ≈ 0.3–0.7 (taller than wide)
                 lying    → aspect > ~1.2     (wider than tall)

Supporting signal : kps_coverage
                    Low mean keypoint confidence often co-occurs with lying
                    (legs tuck under, rear keypoints disappear). Used to
                    detect occlusion / ambiguous frames, not as a primary
                    classifier.

Per-frame outputs (returned as dict of 1D arrays, length = N_frames):
    "posture"        : np.ndarray[int8]   — Posture enum values (0/1/2)
    "posture_conf"   : np.ndarray[float32]— confidence of the classification [0,1]

Design note
───────────
UNCERTAIN is returned when the aspect ratio falls in the ambiguous band
[STANDING_MAX_AR, LYING_MIN_AR) or when the detection confidence is too
low. Downstream aggregation should skip UNCERTAIN frames rather than
casting them as a vote.
"""

import numpy as np
from ..schema import Posture

# ─────────────────────────────────────────────────────────────────────────────
# Tunable thresholds (easy to override from extractor.py)
# ─────────────────────────────────────────────────────────────────────────────

# Bounding-box aspect ratio thresholds
STANDING_MAX_AR    = 0.85   # below this → STANDING
LYING_MIN_AR       = 1.10   # above this → LYING
                            # [0.85, 1.10] band → UNCERTAIN

# Minimum mean keypoint confidence to trust the frame at all.
# Frames with kps_coverage < this threshold are always UNCERTAIN.
# YOLOv8-Pose often outputs low floats — 0.05 accepts any attempted placement.
MIN_KPS_COVERAGE   = 0.05

# Minimum detection confidence to include a frame in classification.
MIN_DET_CONF       = 0.10


def extract_posture(
    kps: np.ndarray,        # (N, 19, 3) — x, y, conf  (pixel coords, conf∈[0,1])
    kps_kconf: np.ndarray,  # (N, 19)    — per-keypoint confidence
    bbox: np.ndarray,       # (N, 4)     — x1, y1, x2, y2
    det_conf: np.ndarray | None = None,  # (N,) — optional detection confidences
) -> dict[str, np.ndarray]:
    """
    Classify posture for N frames.

    Parameters
    ----------
    kps        : (N, 19, 3)  raw keypoint array; third dimension is [x, y, conf].
    kps_kconf  : (N, 19)     per-keypoint confidence scores.
    bbox       : (N, 4)      bounding boxes [x1, y1, x2, y2].
    det_conf   : (N,)        optional; detection confidence per frame.
                             If None, det_conf check is skipped.

    Returns
    -------
    dict with:
        "posture"       : int8 array of Posture enum values  (N,)
        "posture_conf"  : float32 confidence array           (N,)
    """
    N = len(bbox)
    posture  = np.full(N, Posture.UNCERTAIN, dtype=np.int8)
    conf_out = np.zeros(N, dtype=np.float32)

    if N == 0:
        return {"posture": posture, "posture_conf": conf_out}

    # ── bbox aspect ratio ─────────────────────────────────────────────────────
    w = bbox[:, 2] - bbox[:, 0]   # x2 - x1
    h = bbox[:, 3] - bbox[:, 1]   # y2 - y1

    # Guard against degenerate boxes (height = 0)
    valid_box = h > 2.0
    aspect    = np.where(valid_box, w / np.maximum(h, 1e-6), np.nan)

    # ── mean keypoint coverage ────────────────────────────────────────────────
    kps_coverage = kps_kconf.mean(axis=1)   # (N,)

    # ── detection confidence gate ─────────────────────────────────────────────
    if det_conf is not None:
        low_det = det_conf < MIN_DET_CONF
    else:
        low_det = np.zeros(N, dtype=bool)

    # ── classify ─────────────────────────────────────────────────────────────
    # Frames that fail basic quality gates stay UNCERTAIN
    quality_ok = valid_box & ~low_det & (kps_coverage >= MIN_KPS_COVERAGE)

    # STANDING: narrow bbox, kps look normal
    standing_mask = quality_ok & (aspect <= STANDING_MAX_AR)
    # LYING: wide bbox
    lying_mask    = quality_ok & (aspect >= LYING_MIN_AR)
    # Ambiguous band stays UNCERTAIN

    posture[standing_mask] = int(Posture.STANDING)
    posture[lying_mask]    = int(Posture.LYING)

    # ── per-frame confidence ──────────────────────────────────────────────────
    # Confidence is a combination of:
    #   - how far the aspect ratio is from the ambiguous boundary
    #   - mean keypoint coverage (vision quality)
    # Confidence is 0 for UNCERTAIN frames.

    # Distance from the nearest threshold, normalised to [0, 1].
    ar_conf = np.zeros(N, dtype=np.float32)
    # For STANDING: standing is clearer the lower the AR is below STANDING_MAX_AR
    band    = LYING_MIN_AR - STANDING_MAX_AR   # width of uncertain band
    ar_conf[standing_mask] = np.clip(
        (STANDING_MAX_AR - aspect[standing_mask]) / (STANDING_MAX_AR * 0.5), 0, 1
    ).astype(np.float32)
    ar_conf[lying_mask] = np.clip(
        (aspect[lying_mask] - LYING_MIN_AR) / (LYING_MIN_AR * 0.5), 0, 1
    ).astype(np.float32)

    # Blend with kps quality
    kps_weight = np.clip(kps_coverage, 0, 1)
    conf_out   = (0.7 * ar_conf + 0.3 * kps_weight).astype(np.float32)
    conf_out[posture == int(Posture.UNCERTAIN)] = 0.0

    return {
        "posture":      posture,
        "posture_conf": conf_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Window aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_posture(
    posture: np.ndarray,        # (N,) int8 — per-frame Posture values
    posture_conf: np.ndarray,   # (N,) float32 — per-frame confidence
    min_conf: float = 0.3,
) -> dict[str, float | int]:
    """
    Aggregate per-frame posture labels into window-level scalars for
    resolved_cow_timeline.

    Parameters
    ----------
    posture      : per-frame Posture labels (N,)
    posture_conf : per-frame classification confidence (N,)
    min_conf     : frames below this confidence are excluded from aggregation.

    Returns
    -------
    dict:
        "lying_fraction"      : float  — fraction of confident frames that are LYING
        "posture_transitions" : int    — number of STANDING↔LYING state changes
    """
    # Only use confident, non-uncertain frames
    confident = (posture_conf >= min_conf) & (posture != int(Posture.UNCERTAIN))

    if confident.sum() == 0:
        return {"lying_fraction": float("nan"), "posture_transitions": 0}

    confident_posture = posture[confident]
    lying_fraction    = float((confident_posture == int(Posture.LYING)).mean())

    # Count transitions: find consecutive pairs where label changes
    # between STANDING and LYING only (ignore uncertain edges)
    transitions = 0
    prev = None
    for label in confident_posture:
        if prev is not None and label != prev:
            transitions += 1
        prev = label

    return {
        "lying_fraction":      lying_fraction,
        "posture_transitions": transitions,
    }