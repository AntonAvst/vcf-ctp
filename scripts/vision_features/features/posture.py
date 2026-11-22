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

Secondary signal : limb extension ratio
                    For each leg (shoulder_L/R→hoof_fore_L/R, hip_L/R→hoof_back_L/R),
                    the pixel distance between its two joints, normalised by a
                    body-scale reference independent of posture. Standing legs are
                    extended → ratio close to the leg's full anatomical proportion;
                    lying legs fold under or out to the side → ratio collapses
                    toward zero.

                    The reference is withers↔sacrum spine length when those two
                    keypoints are confidently visible, falling back to the bbox
                    diagonal otherwise. The fallback matters: spine keypoints are
                    exactly the ones most likely to be low-confidence when the cow
                    IS lying (body collapsed, rear obscured), so without it the
                    signal would be unavailable for a disproportionate share of the
                    frames it's most needed for. The bbox diagonal is always
                    available once detection succeeds at all, at the cost of being
                    a cruder, less purely-anatomical reference. Spine- and
                    bbox-normalized ratios are on different numeric scales and use
                    separate threshold pairs (`*_spine` / `*_bbox` in params.yaml)
                    — never compared against one shared threshold.

                    Used ONLY to resolve frames that bbox AR leaves UNCERTAIN (the
                    ambiguous band between standing_max_ar and lying_min_ar) — it
                    does not override a clear AR-based classification. This keeps
                    it a fallback for borderline cases, not a competing primary
                    signal, since AR is the better-validated of the two.

                    Needs at least one leg's both joints + a usable reference
                    (spine or bbox) above min_kp_conf; frames with legs too
                    occluded to measure stay UNCERTAIN rather than guessing.

                    All four limb_*_ratio_{spine,bbox} thresholds in params.yaml
                    are starting estimates, NOT calibrated against labeled data
                    yet — plot each ratio's distribution for a labeled
                    standing/lying validation set before trusting them. The bbox
                    variant in particular is a rougher guess than the spine one,
                    since it has no direct anatomical precedent to reason from.

Per-frame outputs (returned as dict of 1D arrays, length = N_frames):
    "posture"        : np.ndarray[int8]   — Posture enum values (0/1/2)
    "posture_conf"   : np.ndarray[float32]— confidence of the classification [0,1]

Design note
───────────
UNCERTAIN is returned when the aspect ratio falls in the ambiguous band
[standing_max_ar, lying_min_ar] (params.yaml vision.posture) or when the detection confidence is too
low. Downstream aggregation should skip UNCERTAIN frames rather than
casting them as a vote.
"""

import numpy as np

import config
from ..schema import KP, Posture

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds — config/params.yaml → vision.posture
#   standing_max_ar   bbox w/h at or below → STANDING
#   lying_min_ar      at or above → LYING; [standing_max_ar, lying_min_ar] = UNCERTAIN
#   min_kps_coverage  mean keypoint conf below → always UNCERTAIN
#   min_det_conf      detection conf below → always UNCERTAIN
#   ar_conf_weight    share of aspect-ratio clarity in posture_conf (rest = kp quality)
# Read at call time (never at import) so --params always takes effect.
# ─────────────────────────────────────────────────────────────────────────────

def _cfg():
    return config.params().vision.posture


# ─────────────────────────────────────────────────────────────────────────────
# Limb-extension geometry — see module docstring "Secondary signal" above.
# ─────────────────────────────────────────────────────────────────────────────

# (joint, hoof) pairs — one per leg. Order doesn't matter, all four are pooled.
_LEG_PAIRS: tuple[tuple[KP, KP], ...] = (
    (KP.SHOULDER_L, KP.HOOF_FORE_L),
    (KP.SHOULDER_R, KP.HOOF_FORE_R),
    (KP.HIP_L,      KP.HOOF_BACK_L),
    (KP.HIP_R,      KP.HOOF_BACK_R),
)
_SPINE_REF: tuple[KP, KP] = (KP.WITHERS, KP.SACRUM)


def _limb_extension(
    kps: np.ndarray,        # (N, 26, 3)
    kps_kconf: np.ndarray,  # (N, 26)
    bbox: np.ndarray,       # (N, 4) — x1, y1, x2, y2; bbox-diagonal fallback reference
    cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-frame limb-extension ratio, its evidence weight, and which scale
    reference was used.

    Normalizing reference, in priority order:
      1. withers↔sacrum spine length — preferred: purely anatomical, doesn't
         move when the cow's posture/bbox shape changes.
      2. bbox diagonal (sqrt(w²+h²)) — fallback for when the spine keypoints
         aren't confidently visible. This matters more than an edge case: a
         lying cow's spine keypoints are exactly the ones most likely to be
         low-confidence (body collapsed, rear obscured), so without this
         fallback the signal would be unavailable for a disproportionate
         share of the frames it's most needed for.

    Spine-normalized and bbox-normalized ratios are NOT on the same numeric
    scale (a bbox diagonal spans the whole animal; the spine segment is just
    the torso) — callers must use the separate `*_spine` / `*_bbox` threshold
    pairs in params.yaml, keyed off the returned `ref_is_spine` flag, rather
    than one threshold for both.

    Returns
    -------
    limb_score   : (N,) float32 — confidence-weighted mean, across available
                   legs, of (joint→hoof distance) / reference_length.
                   NaN where no leg + usable reference was found.
    limb_weight  : (N,) float32 in [0,1] — how much evidence backs limb_score.
                   0.0 where limb_score is NaN. Scales with both per-leg
                   keypoint confidence AND how many of the 4 legs were usable,
                   so a frame with one barely-confident leg visible scores low
                   even if that one leg's confidence alone looks OK.
    ref_is_spine : (N,) bool — True where the spine reference was used,
                   False where it fell back to the bbox diagonal. Meaningless
                   where limb_score is NaN.
    """
    N = len(kps)
    w_i, s_i = int(_SPINE_REF[0]), int(_SPINE_REF[1])

    spine_c   = np.minimum(kps_kconf[:, w_i], kps_kconf[:, s_i])       # (N,)
    spine_len = np.linalg.norm(kps[:, w_i, :2] - kps[:, s_i, :2], axis=1)  # (N,)
    spine_ok  = (spine_c >= cfg.min_kp_conf) & (spine_len > 1.0)

    bw = bbox[:, 2] - bbox[:, 0]
    bh = bbox[:, 3] - bbox[:, 1]
    bbox_diag = np.sqrt(bw ** 2 + bh ** 2)
    bbox_ok   = (bw > 2.0) & (bh > 2.0)

    ref_is_spine = spine_ok                                   # prefer spine whenever usable
    ref_len      = np.where(ref_is_spine, spine_len, bbox_diag)
    ref_ok       = ref_is_spine | bbox_ok                      # need SOME usable reference

    n_legs = len(_LEG_PAIRS)
    leg_ratio  = np.full((N, n_legs), np.nan, dtype=np.float32)
    leg_weight = np.zeros((N, n_legs), dtype=np.float32)

    for li, (joint, hoof) in enumerate(_LEG_PAIRS):
        ji, hi = int(joint), int(hoof)
        d = np.linalg.norm(kps[:, ji, :2] - kps[:, hi, :2], axis=1)
        c = np.minimum(kps_kconf[:, ji], kps_kconf[:, hi])
        usable = ref_ok & (c >= cfg.min_kp_conf)

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = d / np.maximum(ref_len, 1e-6)
        leg_ratio[usable, li]  = ratio[usable]
        leg_weight[usable, li] = c[usable]

    total_w = leg_weight.sum(axis=1)
    has_any = total_w > 0

    with np.errstate(invalid="ignore", divide="ignore"):
        weighted_sum = np.nansum(np.nan_to_num(leg_ratio) * leg_weight, axis=1)
        limb_score = np.where(has_any, weighted_sum / np.maximum(total_w, 1e-6), np.nan)

    n_legs_usable = (leg_weight > 0).sum(axis=1)
    # Mean per-leg confidence, scaled down further by how few legs were visible
    # at all — one confident leg is weaker evidence than four confident legs.
    limb_weight = np.where(
        has_any,
        (total_w / n_legs) * (n_legs_usable / n_legs),
        0.0,
    ).astype(np.float32)

    return limb_score.astype(np.float32), limb_weight, ref_is_spine


def extract_posture(
    kps: np.ndarray,        # (N, 26, 3) — x, y, conf  (pixel coords, conf∈[0,1])
    kps_kconf: np.ndarray,  # (N, 26)    — per-keypoint confidence
    bbox: np.ndarray,       # (N, 4)     — x1, y1, x2, y2
    det_conf: np.ndarray | None = None,  # (N,) — optional detection confidences
) -> dict[str, np.ndarray]:
    """
    Classify posture for N frames.

    Parameters
    ----------
    kps        : (N, 26, 3)  raw keypoint array; third dimension is [x, y, conf].
    kps_kconf  : (N, 26)     per-keypoint confidence scores.
    bbox       : (N, 4)      bounding boxes [x1, y1, x2, y2].
    det_conf   : (N,)        optional; detection confidence per frame.
                             If None, det_conf check is skipped.

    Returns
    -------
    dict with:
        "posture"       : int8 array of Posture enum values  (N,)
        "posture_conf"  : float32 confidence array           (N,)
    """
    cfg = _cfg()
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
        low_det = det_conf < cfg.min_det_conf
    else:
        low_det = np.zeros(N, dtype=bool)

    # ── classify ─────────────────────────────────────────────────────────────
    # Frames that fail basic quality gates stay UNCERTAIN
    quality_ok = valid_box & ~low_det & (kps_coverage >= cfg.min_kps_coverage)

    # STANDING: narrow bbox, kps look normal
    standing_mask = quality_ok & (aspect <= cfg.standing_max_ar)
    # LYING: wide bbox
    lying_mask    = quality_ok & (aspect >= cfg.lying_min_ar)
    # Ambiguous band stays UNCERTAIN

    posture[standing_mask] = int(Posture.STANDING)
    posture[lying_mask]    = int(Posture.LYING)

    # ── limb-extension fallback for AR-ambiguous frames ─────────────────────
    # Only touches frames that passed quality gates but fell in the ambiguous
    # AR band — never overrides a clear AR-based call above.
    ambiguous_mask = quality_ok & ~standing_mask & ~lying_mask
    limb_resolved_mask = np.zeros(N, dtype=bool)
    limb_score = limb_weight = None

    if ambiguous_mask.any():
        limb_score, limb_weight, ref_is_spine = _limb_extension(kps, kps_kconf, bbox, cfg)
        resolvable = ambiguous_mask & (limb_weight >= cfg.min_limb_weight) & ~np.isnan(limb_score)

        # Spine- and bbox-normalized ratios live on different scales, so each
        # gets its own threshold pair — never compare limb_score to a single
        # shared threshold regardless of which reference produced it.
        resolvable_spine = resolvable & ref_is_spine
        resolvable_bbox  = resolvable & ~ref_is_spine

        limb_standing = (
            (resolvable_spine & (limb_score >= cfg.limb_extended_min_ratio_spine)) |
            (resolvable_bbox  & (limb_score >= cfg.limb_extended_min_ratio_bbox))
        )
        limb_lying = (
            (resolvable_spine & (limb_score <= cfg.limb_folded_max_ratio_spine)) |
            (resolvable_bbox  & (limb_score <= cfg.limb_folded_max_ratio_bbox))
        )
        # limb_score strictly between the two applicable thresholds: genuinely
        # ambiguous even by this signal, so it's left as UNCERTAIN rather than
        # guessing.

        posture[limb_standing] = int(Posture.STANDING)
        posture[limb_lying]    = int(Posture.LYING)
        limb_resolved_mask = limb_standing | limb_lying

    # ── per-frame confidence ──────────────────────────────────────────────────
    # Confidence is a combination of:
    #   - how far the aspect ratio is from the ambiguous boundary
    #   - mean keypoint coverage (vision quality)
    # Confidence is 0 for UNCERTAIN frames.

    # Distance from the nearest threshold, normalised to [0, 1].
    ar_conf = np.zeros(N, dtype=np.float32)
    # For STANDING: standing is clearer the lower the AR is below standing_max_ar
    smax, lmin = cfg.standing_max_ar, cfg.lying_min_ar
    ar_conf[standing_mask] = np.clip(
        (smax - aspect[standing_mask]) / (smax * 0.5), 0, 1
    ).astype(np.float32)
    ar_conf[lying_mask] = np.clip(
        (aspect[lying_mask] - lmin) / (lmin * 0.5), 0, 1
    ).astype(np.float32)

    # Blend with kps quality (ar_conf_weight · geometry + rest · keypoint quality)
    w_ar       = cfg.ar_conf_weight
    kps_weight = np.clip(kps_coverage, 0, 1)
    conf_out   = (w_ar * ar_conf + (1.0 - w_ar) * kps_weight).astype(np.float32)

    # Limb-resolved frames: confidence = limb_weight, scaled down by
    # limb_resolved_conf_scale since this signal only ever activates where
    # AR already couldn't decide — treat it as less certain than a clean
    # AR-based call even when limb_weight itself is high.
    if limb_resolved_mask.any():
        conf_out[limb_resolved_mask] = (
            cfg.limb_resolved_conf_scale * limb_weight[limb_resolved_mask]
        ).astype(np.float32)

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
    min_conf: float,            # params vision.aggregate_min_conf
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