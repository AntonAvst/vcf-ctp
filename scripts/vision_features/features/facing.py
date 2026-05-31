"""
vision_features/features/facing.py
────────────────────────────────────
Per-frame facing direction classification: LEFT | RIGHT | TOWARD | AWAY | UNCERTAIN.

Primary signal : nose → sacrum vector (KP.NOSE → KP.SACRUM)
                 Projection onto frame axes tells us orientation.
                 The body axis vector points from sacrum → nose.

Axis logic
──────────
Let dx = nose.x − sacrum.x  (positive = nose is to the right in the frame)
Let dy = nose.y − sacrum.y  (positive = nose is lower in the frame — image coords)

If |dx| > |dy| × LATERAL_DOMINANCE_RATIO → lateral (LEFT or RIGHT)
    dx > 0 → RIGHT (nose is to the right)
    dx < 0 → LEFT  (nose is to the left)
Else → depth (TOWARD or AWAY):
    We cannot directly distinguish depth from a single 2D frame, but:
    A small bbox area relative to a short body axis vector → cow is foreshortened
    (nose-to-sacrum distance is compressed), meaning it faces toward or away.
    Bbox width/height ratio can help:
      - Facing TOWARD/AWAY: cow is narrow in one axis because body is end-on
    Without stereo depth, we default to an ambiguous TOWARD/AWAY heuristic
    based on which keypoints are visible:
      - If nose confidence is high but sacrum confidence is low → TOWARD
        (nose close to camera, sacrum occluded behind body)
      - If sacrum confidence is high but nose confidence is low → AWAY
      - If both visible → use the body-axis foreshortening heuristic

Per-frame outputs (dict of 1D arrays, length = N_frames):
    "facing"       : np.ndarray[int8]    — Facing enum values
    "facing_conf"  : np.ndarray[float32] — confidence [0,1]
"""

import numpy as np
from ..schema import KP, Facing, Posture

# ─────────────────────────────────────────────────────────────────────────────
# Tunable thresholds
# ─────────────────────────────────────────────────────────────────────────────

# dx must be this many times larger than dy to call it a lateral direction.
LATERAL_DOMINANCE_RATIO = 1.4

# Minimum keypoint confidence to trust a nose or sacrum reading.
MIN_KP_CONF = 0.30

# If the nose-to-sacrum pixel distance is very short the cow is likely
# end-on. Below this distance (in pixels) we call depth classification.
MIN_BODY_AXIS_PX = 15.0


def extract_facing(
    kps: np.ndarray,             # (N, 19, 3) — x, y, conf
    kps_kconf: np.ndarray,       # (N, 19)    — per-keypoint confidence
    bbox: np.ndarray,            # (N, 4)     — x1, y1, x2, y2
    posture: np.ndarray | None = None,   # (N,) int8 Posture values — optional context
) -> dict[str, np.ndarray]:
    """
    Classify facing direction for N frames.

    Parameters
    ----------
    kps        : (N, 19, 3)  keypoint array [x, y, conf].
    kps_kconf  : (N, 19)     per-keypoint confidence.
    bbox       : (N, 4)      bounding boxes.
    posture    : (N,)        optional posture array; LYING frames get lower
                             facing confidence since the body axis is less
                             informative when lying.

    Returns
    -------
    dict with:
        "facing"       : int8 array of Facing enum values (N,)
        "facing_conf"  : float32 confidence array (N,)
    """
    N = len(kps)
    facing   = np.full(N, Facing.UNCERTAIN, dtype=np.int8)
    conf_out = np.zeros(N, dtype=np.float32)

    if N == 0:
        return {"facing": facing, "facing_conf": conf_out}

    # Extract nose and sacrum keypoints
    nose_x    = kps[:, KP.NOSE, 0]
    nose_y    = kps[:, KP.NOSE, 1]
    nose_c    = kps_kconf[:, KP.NOSE]

    sacrum_x  = kps[:, KP.SACRUM, 0]
    sacrum_y  = kps[:, KP.SACRUM, 1]
    sacrum_c  = kps_kconf[:, KP.SACRUM]

    # Both anchor points must be sufficiently confident to classify
    both_visible = (nose_c >= MIN_KP_CONF) & (sacrum_c >= MIN_KP_CONF)
    nose_only    = (nose_c >= MIN_KP_CONF) & (sacrum_c < MIN_KP_CONF)
    sacrum_only  = (sacrum_c >= MIN_KP_CONF) & (nose_c < MIN_KP_CONF)

    # Body axis vector: sacrum → nose  (positive x = nose is to the right)
    dx = nose_x - sacrum_x
    dy = nose_y - sacrum_y
    body_len = np.sqrt(dx**2 + dy**2)

    # ── Case 1: both nose and sacrum visible ──────────────────────────────────
    # Use the body-axis vector direction to classify lateral vs depth.

    long_enough = body_len >= MIN_BODY_AXIS_PX
    lateral_dom = np.abs(dx) > np.abs(dy) * LATERAL_DOMINANCE_RATIO

    # Lateral classification (LEFT / RIGHT)
    lateral_frames = both_visible & long_enough & lateral_dom
    facing[lateral_frames & (dx > 0)] = int(Facing.RIGHT)
    facing[lateral_frames & (dx < 0)] = int(Facing.LEFT)

    # Depth classification (TOWARD / AWAY) — body axis is mostly vertical
    depth_frames = both_visible & long_enough & ~lateral_dom
    # Sub-classify by which keypoint is higher (lower y in image = higher in frame)
    # If nose is higher (lower y) than sacrum → cow faces away (tail toward camera)
    # If sacrum is higher (lower y) than nose → cow faces toward (nose toward camera)
    facing[depth_frames & (nose_y < sacrum_y)] = int(Facing.AWAY)
    facing[depth_frames & (nose_y >= sacrum_y)] = int(Facing.TOWARD)

    # Short body axis (foreshortened end-on cow) with both points visible
    end_on = both_visible & ~long_enough
    # Disambiguate using nose vs sacrum relative confidence as soft depth cue
    facing[end_on & (nose_c > sacrum_c)] = int(Facing.TOWARD)
    facing[end_on & (nose_c <= sacrum_c)] = int(Facing.AWAY)

    # ── Case 2: only nose visible ─────────────────────────────────────────────
    # We know where the head is but not the tail — likely facing toward camera.
    facing[nose_only] = int(Facing.TOWARD)

    # ── Case 3: only sacrum visible ───────────────────────────────────────────
    # We see the tail end — likely facing away.
    facing[sacrum_only] = int(Facing.AWAY)

    # ── Confidence ───────────────────────────────────────────────────────────
    # Base confidence from keypoint visibility quality
    mean_anchor_conf = (nose_c + sacrum_c) / 2.0

    # Boost for strong lateral dominance (clear direction)
    lateral_clarity = np.zeros(N, dtype=np.float32)
    denom = np.maximum(np.abs(dy) * LATERAL_DOMINANCE_RATIO, 1e-6)
    lateral_clarity = np.clip(np.abs(dx) / denom - 1.0, 0, 1).astype(np.float32)

    # For lateral frames: blend anchor quality + directional clarity
    conf_out[lateral_frames] = (
        0.6 * mean_anchor_conf[lateral_frames]
        + 0.4 * lateral_clarity[lateral_frames]
    )
    # For depth frames: rely more on anchor conf (depth is inherently less certain)
    non_lateral_classified = facing != int(Facing.UNCERTAIN)
    conf_out[non_lateral_classified & ~lateral_frames] = (
        0.5 * mean_anchor_conf[non_lateral_classified & ~lateral_frames]
    )
    # Penalise lying frames: body axis is less reliable when lying
    if posture is not None:
        lying_mask = posture == int(Posture.LYING)
        conf_out[lying_mask] *= 0.7

    conf_out[facing == int(Facing.UNCERTAIN)] = 0.0
    conf_out = conf_out.astype(np.float32)

    return {"facing": facing, "facing_conf": conf_out}


# ─────────────────────────────────────────────────────────────────────────────
# Window aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_facing(
    facing: np.ndarray,       # (N,) int8 — per-frame Facing values
    facing_conf: np.ndarray,  # (N,) float32 — per-frame confidence
    min_conf: float = 0.3,
) -> dict[str, str | float]:
    """
    Aggregate per-frame facing labels into window-level scalars.

    Parameters
    ----------
    facing       : per-frame Facing labels (N,)
    facing_conf  : per-frame classification confidence (N,)
    min_conf     : frames below this confidence are excluded.

    Returns
    -------
    dict:
        "facing_dominant" : str   — modal facing direction among confident frames
                                    (e.g. "left", "right", "toward", "away")
                                    None if no confident frames.
        "facing_entropy"  : float — normalised entropy of facing distribution [0,1]
                                    0 = all frames same direction
                                    1 = perfectly uniform across 4 directions
    """
    from ..schema import FACING_NAMES, Facing

    confident = (facing_conf >= min_conf) & (facing != int(Facing.UNCERTAIN))
    if confident.sum() == 0:
        return {"facing_dominant": None, "facing_entropy": float("nan")}

    confident_facing = facing[confident]

    # Modal direction (dominant)
    non_uncertain_labels = [int(f) for f in Facing if f != Facing.UNCERTAIN]
    counts = {label: int((confident_facing == label).sum())
              for label in non_uncertain_labels}
    dominant_label = max(counts, key=counts.get)
    dominant_name  = FACING_NAMES[Facing(dominant_label)]

    # Normalised entropy over 4 directions
    total = confident.sum()
    probs = np.array([counts[label] / total for label in non_uncertain_labels],
                     dtype=np.float64)
    # Shannon entropy, normalised by log2(4) = 2 so result ∈ [0,1]
    log_max = np.log2(len(non_uncertain_labels))  # log2(4) = 2
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(probs > 0, -probs * np.log2(probs), 0.0)
    entropy_norm = float(entropy_terms.sum() / log_max) if log_max > 0 else 0.0

    return {
        "facing_dominant": dominant_name,
        "facing_entropy":  round(entropy_norm, 4),
    }
