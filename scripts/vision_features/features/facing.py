"""
vision_features/features/facing.py
────────────────────────────────────
Per-frame facing direction classification: LEFT | RIGHT | TOWARD | AWAY | UNCERTAIN.

Primary signal : spine axis derived from the two best back keypoints in
                 {WITHERS(2), SPINE_MID(3), SACRUM(4), TAIL_BASE(5)}.

                 Head/neck keypoints (NOSE=0, FOREHEAD=1, NECK=18) are
                 intentionally excluded — they are too mobile and occlude
                 easily, making them unreliable direction anchors.

Spine axis selection (per frame)
─────────────────────────────────
Candidate back KPs: WITHERS(2), SPINE_MID(3), SACRUM(4), TAIL_BASE(5).

Two selection strategies are tried in order:

  Strategy A — highest-confidence pair
      Take the two KPs with the highest kps_kconf scores, provided
      both exceed MIN_KP_CONF.  This maximises signal quality.

  Strategy B — farthest-apart pair
      Among all KPs exceeding MIN_KP_CONF, take the pair with the
      greatest pixel distance.  Longer lever arm → more stable angle.

  The strategy that yields the longer axis vector is used (if both pass
  the MIN_BODY_AXIS_PX check).  If neither passes, frame is UNCERTAIN.

The axis vector always points from the rear KP (higher index = more
posterior) toward the front KP (lower index = more anterior), so the
vector tip nominally points in the direction the cow faces.

Direction from angle (degrees, image coords)
─────────────────────────────────────────────
0 deg   = pointing right  (+x in image)
90 deg  = pointing up     (−y in image — image y increases downward)
180 deg = pointing left   (−x in image)
270 deg = pointing down   (+y in image)

Segment boundaries (configurable below, all in degrees [0, 360)):

    TOWARD  (up)   : angle in [TOWARD_CENTER ± DEPTH_HALF_WIDTH]
    AWAY    (down) : angle in [AWAY_CENTER   ± DEPTH_HALF_WIDTH]
    RIGHT          : fills remaining arc on the right half
    LEFT           : fills remaining arc on the left half

With the defaults (TOWARD=90°±10°, AWAY=270°±10°) the layout is:
    up   :  80° – 100°   → TOWARD
    down : 260° – 280°   → AWAY
    right:   0° –  80°  ∪  280° – 360°  (minus tiny depth bands) → RIGHT
    left : 100° – 260°   (minus tiny depth band) → LEFT

This is deliberately very left/right dominant.  The narrow ±10° depth
bands ensure that near-vertical spines still get a depth label while
almost everything else is classified lateral.

Per-frame outputs (dict of 1D arrays, length = N_frames):
    "facing"      : np.ndarray[int8]    — Facing enum values
    "facing_conf" : np.ndarray[float32] — confidence [0,1]
"""

import numpy as np
from ..schema import KP, Facing, Posture

# ─────────────────────────────────────────────────────────────────────────────
# Back keypoints used for spine axis (head/neck explicitly excluded)
# ─────────────────────────────────────────────────────────────────────────────

BACK_KPS = [KP.WITHERS, KP.SPINE_MID, KP.SACRUM, KP.TAIL_BASE]  # indices 2,3,4,5

# ─────────────────────────────────────────────────────────────────────────────
# Quality thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Minimum per-keypoint confidence to consider a back KP usable.
# YOLOv8-Pose outputs low float values even for clearly visible points;
# 0.05 accepts anything the pose model attempted to place.
# Raise to 0.20–0.30 only if you see many spurious placements.
MIN_KP_CONF = 0.05

# Minimum pixel distance between the two chosen back KPs.
# Shorter axes are too noisy to trust for angle estimation.
MIN_BODY_AXIS_PX = 20.0

# ─────────────────────────────────────────────────────────────────────────────
# Direction segment boundaries  (degrees, 0=right, CCW positive in math space)
#
# In image coordinates y increases downward, so:
#   np.arctan2(-dy, dx) maps image vectors to math angles correctly:
#   pure +x (right)  → 0°
#   pure -y (up)     → 90°   ← TOWARD (cow faces camera from above)
#   pure -x (left)   → 180°
#   pure +y (down)   → 270°  ← AWAY   (cow faces away from camera)
#
# Change TOWARD_CENTER / AWAY_CENTER to rotate the depth zones.
# Change DEPTH_HALF_WIDTH to widen or narrow them (both use the same width).
# ─────────────────────────────────────────────────────────────────────────────

TOWARD_CENTER    = 90.0   # degrees — spine pointing up   → cow faces toward camera
AWAY_CENTER      = 270.0  # degrees — spine pointing down → cow faces away from camera
DEPTH_HALF_WIDTH = 10.0   # degrees — half-width of each depth zone (±10° = 20° total band)

# Derived boundaries (computed once at import; do not edit these directly)
_TOWARD_LO = TOWARD_CENTER - DEPTH_HALF_WIDTH   #  80°
_TOWARD_HI = TOWARD_CENTER + DEPTH_HALF_WIDTH   # 100°
_AWAY_LO   = AWAY_CENTER   - DEPTH_HALF_WIDTH   # 260°
_AWAY_HI   = AWAY_CENTER   + DEPTH_HALF_WIDTH   # 280°

# Confidence penalty applied when the cow is classified LYING
# (spine axis still works but is less informative in lateral recumbency)
LYING_CONF_PENALTY = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Spine axis selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_pair_highest_conf(
    kp_xy: np.ndarray,    # (4, 2) — x,y for the 4 back KPs
    kp_c:  np.ndarray,    # (4,)   — confidence for the 4 back KPs
) -> tuple[int, int] | None:
    """
    Return (i, j) indices into BACK_KPS for the two KPs with the highest
    confidence, both above MIN_KP_CONF.  Returns None if fewer than 2 qualify.
    i < j always (i is more anterior in BACK_KPS ordering).
    """
    usable = np.where(kp_c >= MIN_KP_CONF)[0]
    if len(usable) < 2:
        return None
    top2 = usable[np.argsort(kp_c[usable])[::-1][:2]]
    i, j = int(top2.min()), int(top2.max())
    return (i, j)


def _best_pair_farthest(
    kp_xy: np.ndarray,    # (4, 2)
    kp_c:  np.ndarray,    # (4,)
) -> tuple[int, int] | None:
    """
    Return (i, j) indices into BACK_KPS for the pair with the greatest
    pixel distance among KPs above MIN_KP_CONF.
    i < j always.
    """
    usable = np.where(kp_c >= MIN_KP_CONF)[0]
    if len(usable) < 2:
        return None
    best_dist = -1.0
    best_pair = None
    for a in range(len(usable)):
        for b in range(a + 1, len(usable)):
            ia, ib = usable[a], usable[b]
            d = float(np.linalg.norm(kp_xy[ia] - kp_xy[ib]))
            if d > best_dist:
                best_dist = d
                best_pair = (int(min(ia, ib)), int(max(ia, ib)))
    return best_pair


def _spine_vector(
    kp_xy: np.ndarray,    # (4, 2)
    kp_c:  np.ndarray,    # (4,)
) -> tuple[float, float, float, float] | None:
    """
    Select the best spine axis vector for a single frame.

    Returns (dx, dy, axis_len, mean_conf) where the vector points from the
    more-posterior KP toward the more-anterior KP (i.e. front of the cow).

    Returns None if no valid pair exists or axis is too short.
    """
    # Try both strategies; keep the one with the longer axis
    pair_hc = _best_pair_highest_conf(kp_xy, kp_c)
    pair_fd = _best_pair_farthest(kp_xy, kp_c)

    candidates = []
    for pair in (pair_hc, pair_fd):
        if pair is None:
            continue
        i, j = pair   # i < j  → i is more anterior (lower KP index = more forward)
        # vector: posterior (j) → anterior (i) = direction cow faces
        dxv = float(kp_xy[i, 0] - kp_xy[j, 0])
        dyv = float(kp_xy[i, 1] - kp_xy[j, 1])
        length = float(np.sqrt(dxv**2 + dyv**2))
        if length >= MIN_BODY_AXIS_PX:
            mean_c = float((kp_c[i] + kp_c[j]) / 2.0)
            candidates.append((dxv, dyv, length, mean_c))

    if not candidates:
        return None
    # prefer longer axis (more stable angle)
    return max(candidates, key=lambda t: t[2])


# ─────────────────────────────────────────────────────────────────────────────
# Angle → direction assignment
# ─────────────────────────────────────────────────────────────────────────────

def _angle_to_facing(angle_deg: float) -> int:
    """
    Map a spine axis angle (degrees, math convention: 0=right, CCW positive)
    to a Facing enum value.

    Zones (with defaults):
        TOWARD :  80° – 100°
        AWAY   : 260° – 280°
        RIGHT  : everything in right half not covered by depth zones
        LEFT   : everything in left half not covered by depth zones
    """
    a = angle_deg % 360.0

    if _TOWARD_LO <= a <= _TOWARD_HI:
        return int(Facing.TOWARD)
    if _AWAY_LO <= a <= _AWAY_HI:
        return int(Facing.AWAY)
    # right half: 0–80° and 280–360°
    if a < _TOWARD_LO or a > _AWAY_HI:
        return int(Facing.RIGHT)
    # left half: 100°–260°
    return int(Facing.LEFT)


def _angle_confidence(angle_deg: float, facing_label: int) -> float:
    """
    Returns a [0,1] confidence score based on how far the angle is from the
    nearest zone boundary.  Angles deep inside their zone score high;
    angles near a boundary score lower.

    For LEFT/RIGHT (wide zones): confidence scales with distance from the
    nearest depth-zone boundary, normalised by the zone half-width.
    For TOWARD/AWAY (narrow zones): confidence scales with distance from
    the zone center, normalised by DEPTH_HALF_WIDTH.
    """
    a = angle_deg % 360.0

    if facing_label == int(Facing.TOWARD):
        dist_from_center = abs(a - TOWARD_CENTER)
        return float(np.clip(1.0 - dist_from_center / DEPTH_HALF_WIDTH, 0.0, 1.0))

    if facing_label == int(Facing.AWAY):
        dist_from_center = min(abs(a - AWAY_CENTER), 360.0 - abs(a - AWAY_CENTER))
        return float(np.clip(1.0 - dist_from_center / DEPTH_HALF_WIDTH, 0.0, 1.0))

    # LEFT or RIGHT — how far are we from the nearest depth-zone boundary?
    # The narrower the remaining arc, the lower the confidence near the edge.
    if facing_label == int(Facing.RIGHT):
        # Confidence = distance to the nearest depth-zone boundary.
        # The two depth boundaries that border the RIGHT zone are
        # _TOWARD_LO (80°) and _AWAY_HI (280°).
        # Distance is measured on the circle (shortest arc).
        dist_to_toward = min(abs(a - _TOWARD_LO), 360.0 - abs(a - _TOWARD_LO))
        dist_to_away   = min(abs(a - _AWAY_HI),   360.0 - abs(a - _AWAY_HI))
        dist = min(dist_to_toward, dist_to_away)
        lateral_half = (360.0 - 2 * DEPTH_HALF_WIDTH) / 4.0   # ~85°
        return float(np.clip(dist / lateral_half, 0.0, 1.0))

    if facing_label == int(Facing.LEFT):
        # left zone spans [_TOWARD_HI, _AWAY_LO]: boundaries at both ends
        dist = min(a - _TOWARD_HI, _AWAY_LO - a)
        lateral_half = (360.0 - 2 * DEPTH_HALF_WIDTH) / 4.0
        return float(np.clip(dist / lateral_half, 0.0, 1.0))

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_facing(
    kps: np.ndarray,                      # (N, 19, 3) — x, y, conf
    kps_kconf: np.ndarray,                # (N, 19)    — per-keypoint confidence
    bbox: np.ndarray,                     # (N, 4)     — x1, y1, x2, y2 (unused, kept for API compat)
    posture: np.ndarray | None = None,    # (N,) int8 Posture values — optional
) -> dict[str, np.ndarray]:
    """
    Classify facing direction for N frames using the spine axis.

    Parameters
    ----------
    kps        : (N, 19, 3)  keypoint array [x, y, conf].
    kps_kconf  : (N, 19)     per-keypoint confidence.
    bbox       : (N, 4)      bounding boxes — not used, kept for API compatibility.
    posture    : (N,)        optional; LYING frames receive a confidence penalty.

    Returns
    -------
    dict with:
        "facing"      : int8 array of Facing enum values (N,)
        "facing_conf" : float32 confidence array (N,)
    """
    N = len(kps)
    facing   = np.full(N, Facing.UNCERTAIN, dtype=np.int8)
    conf_out = np.zeros(N, dtype=np.float32)

    if N == 0:
        return {"facing": facing, "facing_conf": conf_out}

    # Extract only the 4 back KPs for all frames at once
    back_indices = [int(k) for k in BACK_KPS]
    back_xy  = kps[:, back_indices, :2]   # (N, 4, 2)
    back_c   = kps_kconf[:, back_indices] # (N, 4)

    for i in range(N):
        result = _spine_vector(back_xy[i], back_c[i])
        if result is None:
            continue   # stays UNCERTAIN

        dx, dy, axis_len, mean_kp_conf = result

        # Math angle: arctan2(-dy, dx) corrects for image y-axis inversion
        angle_rad = float(np.arctan2(-dy, dx))
        angle_deg = float(np.degrees(angle_rad)) % 360.0

        label = _angle_to_facing(angle_deg)
        facing[i] = label

        # Confidence = geometric clarity × keypoint quality
        angle_conf  = _angle_confidence(angle_deg, label)
        conf_out[i] = float(angle_conf * mean_kp_conf)

    # Penalise lying frames — spine axis less reliable in lateral recumbency
    if posture is not None:
        lying_mask = (posture == int(Posture.LYING))
        conf_out[lying_mask] *= LYING_CONF_PENALTY

    conf_out[facing == int(Facing.UNCERTAIN)] = 0.0
    conf_out = conf_out.astype(np.float32)

    return {"facing": facing, "facing_conf": conf_out}


# ─────────────────────────────────────────────────────────────────────────────
# Window aggregation  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_facing(
    facing: np.ndarray,       # (N,) int8 — per-frame Facing values
    facing_conf: np.ndarray,  # (N,) float32 — per-frame confidence
    min_conf: float = 0.3,
) -> dict[str, str | float]:
    """
    Aggregate per-frame facing labels into window-level scalars.

    Returns
    -------
    dict:
        "facing_dominant" : str   — modal facing direction among confident frames
        "facing_entropy"  : float — normalised entropy [0,1]
                                    0 = all frames same direction
                                    1 = perfectly uniform across 4 directions
    """
    from ..schema import FACING_NAMES, Facing

    confident = (facing_conf >= min_conf) & (facing != int(Facing.UNCERTAIN))
    if confident.sum() == 0:
        return {"facing_dominant": None, "facing_entropy": float("nan")}

    confident_facing = facing[confident]

    non_uncertain_labels = [int(f) for f in Facing if f != Facing.UNCERTAIN]
    counts = {label: int((confident_facing == label).sum())
              for label in non_uncertain_labels}
    dominant_label = max(counts, key=counts.get)
    dominant_name  = FACING_NAMES[Facing(dominant_label)]

    total  = confident.sum()
    probs  = np.array([counts[label] / total for label in non_uncertain_labels],
                      dtype=np.float64)
    log_max = np.log2(len(non_uncertain_labels))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(probs > 0, -probs * np.log2(probs), 0.0)
    entropy_norm = float(entropy_terms.sum() / log_max) if log_max > 0 else 0.0

    return {
        "facing_dominant": dominant_name,
        "facing_entropy":  round(entropy_norm, 4),
    }