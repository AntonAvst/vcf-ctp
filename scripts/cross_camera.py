"""
cross_camera.py
────────────────
Boundary Rendezvous — triggered cross-camera reconcile.

Design doc: "Cross-Camera Cow Tracking — Design Scope"

Where this sits in reconcile.py's step sequence:

    Kinetic Matcher
    Gallery Builder
    Cosine Resolver                     <- existing, UNCHANGED, UNRESTRICTED
    *** Boundary Rendezvous (this module) ***    <- NEW
    Duplicate Resolver (resolve_duplicate_assignments)
    Sensor Sequencer
    run_vision_features (posture/facing/pose-gallery update)
    Timeline Writer

Boundary Rendezvous matches on the FLAT embedding (the same one Cosine
Resolver's "Step 1" flat-gallery lookup uses), not the pose-conditioned
8-slot one — that embedding isn't computed until run_vision_features runs,
which is after Sensor Sequencer, and Boundary Rendezvous has to finish
before Sensor Sequencer so a boundary-triggered resolution still makes it
into this session's own resolved_cow_timeline rows. Pose-conditioned
gallery updates need no special handling here: each session updates the
gallery normally in vision_features/extractor.py's _update_pose_gallery
(see README, Key Design Decision 22, for why a same-slot "keep the better
view" rule was dropped).

Integration (reconcile.py, after Cosine Resolver, before Duplicate Resolver):

    from cross_camera import run_boundary_rendezvous, CrossCameraConfig

    boundary_config = CrossCameraConfig.from_params(config.params())
    full_assignment, boundary_log = run_boundary_rendezvous(
        conn=conn, session_id=args.session, camera_id=camera_id,
        tracks_df=tracks_df, embed_df=embed_df,
        assignment=full_assignment,     # {**kinetic_assignment, **cosine_assignment}
        confidence=assignment_confidence,  # {temp_id -> float}, see note below
        config=boundary_config,
    )
    for line in boundary_log:
        log(line)

`confidence` isn't produced anywhere yet as a clean {temp_id: float} dict —
reconcile.py currently has per-step local confidence values (corr, cosine
score) that never get collected into one place. Something has to build that
dict before calling run_boundary_rendezvous; see build_confidence_dict()
below for a drop-in example you can adapt to reconcile.py's actual local
variable names.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

# Imported as _config: several functions here take a parameter named
# `config` (a CrossCameraConfig), which would shadow the module.
import config as _config


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CrossCameraConfig:
    """Flat view of config/params.yaml for this module — cross_camera.* plus
    the one cosine threshold shared with the Cosine Resolver. No defaults:
    always build it with from_params(), so it can never drift from the file."""
    boundary_dwell_seconds: float
    boundary_dwell_min_frames: int
    conflict_confidence_gap: float
    # Tier 3 (motion-magnitude correlation) — see make_motion_corr_fn
    motion_corr_threshold: float
    motion_bin_seconds: float         # speed is computed per bin of this width
    motion_window_pad_seconds: float  # look this far before/after the dwell overlap
    motion_min_samples: int           # min shared speed bins to trust a correlation
    motion_min_std: float             # min speed std (body-heights/s) — below this
                                      # the cow barely moved and r is just noise
    motion_max_lag_bins: int          # tolerate this much clock offset between cameras
    single_cow_sanity_floor: float
    cosine_threshold: float           # = params cosine_resolver.threshold (tier 2)

    @classmethod
    def from_params(cls, p) -> "CrossCameraConfig":
        cc, m = p.cross_camera, p.cross_camera.motion
        return cls(
            boundary_dwell_seconds    = cc.boundary_dwell_seconds,
            boundary_dwell_min_frames = cc.boundary_dwell_min_frames,
            conflict_confidence_gap   = cc.conflict_confidence_gap,
            motion_corr_threshold     = m.corr_threshold,
            motion_bin_seconds        = m.bin_seconds,
            motion_window_pad_seconds = m.window_pad_seconds,
            motion_min_samples        = m.min_samples,
            motion_min_std            = m.min_std,
            motion_max_lag_bins       = m.max_lag_bins,
            single_cow_sanity_floor   = cc.single_cow_sanity_floor,
            cosine_threshold          = p.cosine_resolver.threshold,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Camera topology — hand-annotated zone polygons, config/params.yaml
# camera_topology.
#
# direction_relation: 'same' if the two cameras face the same way (so a cow
# walking "away" in one is walking "toward" in the other in a consistent
# sense), 'opposite' if not. Only used to decide whether raw directional
# motion can be compared at all — Boundary Rendezvous doesn't do that (it uses motion
# MAGNITUDE, which doesn't care), but keeping the field for any future
# geometric work (e.g. real homography calibration) that would need it.
#
# zone_a_poly / zone_b_poly: list of [x, y] pixel points, camera_a's and
# camera_b's own frame respectively.
# ─────────────────────────────────────────────────────────────────────────────

def camera_topology() -> list[dict]:
    """params.yaml camera_topology as the list-of-dicts shape this module uses."""
    return [
        {
            "camera_a": z.camera_a, "camera_b": z.camera_b,
            "zone_a_poly": [tuple(pt) for pt in z.zone_a_poly],
            "zone_b_poly": [tuple(pt) for pt in z.zone_b_poly],
            "direction_relation": z.direction_relation,
        }
        for z in _config.params().camera_topology
    ]


def _zone_pairs_for_camera(camera_id: str, topology: list[dict] | None = None) -> list[dict]:
    """All topology rows involving camera_id, normalized so 'this_poly'/'this_cam'
    always refers to camera_id and 'partner_poly'/'partner_cam' to the other side.
    topology defaults to params.yaml camera_topology."""
    if topology is None:
        topology = camera_topology()
    out = []
    for row in topology:
        if row["camera_a"] == camera_id:
            out.append({
                "zone_pair_id": f"{row['camera_a']}|{row['camera_b']}",
                "this_cam": row["camera_a"], "this_poly": row["zone_a_poly"],
                "partner_cam": row["camera_b"], "partner_poly": row["zone_b_poly"],
                "direction_relation": row["direction_relation"],
            })
        elif row["camera_b"] == camera_id:
            out.append({
                "zone_pair_id": f"{row['camera_a']}|{row['camera_b']}",
                "this_cam": row["camera_b"], "this_poly": row["zone_b_poly"],
                "partner_cam": row["camera_a"], "partner_poly": row["zone_a_poly"],
                "direction_relation": row["direction_relation"],
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — point-in-polygon (ray casting), no external deps
# ─────────────────────────────────────────────────────────────────────────────

def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test. poly is a list of (x, y)
    vertices, not necessarily closed (first point need not repeat at the end)."""
    n = len(poly)
    inside = False
    x1, y1 = poly[-1]
    for i in range(n):
        x2, y2 = poly[i]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def _bbox_center(row: pd.Series) -> tuple[float, float]:
    return ((row["x1"] + row["x2"]) / 2.0, (row["y1"] + row["y2"]) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Dwell computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DwellInfo:
    zone_pair_id: str
    partner_cam: str
    enter_time: datetime
    exit_time: datetime
    frames: int
    seconds: float


def compute_dwell_in_known_zones(
    tracks_df: pd.DataFrame,      # this session's raw_tracks rows (all temp_ids)
    temp_id: int,
    camera_id: str,
    config: CrossCameraConfig,
) -> Optional[DwellInfo]:
    """
    For one temp_id, find the longest qualifying dwell inside any zone shared
    with another camera. Returns None if no zone clears the dwell threshold.
    If a track dwells in more than one zone (shouldn't happen given the
    confirmed topology has no triple-overlap, but handled defensively),
    the zone with the most frames wins.
    """
    tid_rows = tracks_df[tracks_df["temp_id"] == temp_id]
    if tid_rows.empty:
        return None

    best: Optional[DwellInfo] = None
    for zp in _zone_pairs_for_camera(camera_id):
        in_zone_mask = tid_rows.apply(
            lambda r: _point_in_polygon(*_bbox_center(r), zp["this_poly"]), axis=1
        )
        in_zone = tid_rows[in_zone_mask]
        if len(in_zone) < config.boundary_dwell_min_frames:
            continue

        in_zone = in_zone.sort_values("frame_datetime")
        enter_time = in_zone["frame_datetime"].iloc[0]
        exit_time = in_zone["frame_datetime"].iloc[-1]
        seconds = (exit_time - enter_time).total_seconds()
        if seconds < config.boundary_dwell_seconds:
            continue

        if best is None or len(in_zone) > best.frames:
            best = DwellInfo(
                zone_pair_id=zp["zone_pair_id"], partner_cam=zp["partner_cam"],
                enter_time=enter_time, exit_time=exit_time,
                frames=len(in_zone), seconds=seconds,
            )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

BOUNDARY_OBSERVATIONS_DDL = """
CREATE TABLE IF NOT EXISTS boundary_observations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id           TEXT NOT NULL,
    zone_pair_id        TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    temp_id             INTEGER NOT NULL,
    enter_time          TEXT NOT NULL,
    exit_time           TEXT NOT NULL,
    frames_in_zone      INTEGER NOT NULL,
    flat_embedding      TEXT NOT NULL,      -- JSON-encoded list[float]
    resolved_animal_id  INTEGER,
    resolved_confidence REAL,
    status              TEXT NOT NULL DEFAULT 'pending',  -- pending|matched|no_match_found
    created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CROSS_CAMERA_CONFLICTS_DDL = """
CREATE TABLE IF NOT EXISTS cross_camera_conflicts (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    real_id                INTEGER NOT NULL,
    session_a              TEXT NOT NULL,
    session_b              TEXT NOT NULL,
    camera_a               TEXT NOT NULL,
    camera_b               TEXT NOT NULL,
    overlap_window_start   TEXT NOT NULL,
    overlap_window_end     TEXT NOT NULL,
    zone_check_passed      INTEGER NOT NULL,
    confidence_a           REAL,
    confidence_b           REAL,
    status                 TEXT NOT NULL,   -- resolved_by_confidence|needs_review
    created_at             TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def migrate_cross_camera_schema(conn: sqlite3.Connection) -> None:
    """Idempotent — safe to call every run, same pattern as
    vision_features.migrate_timeline_schema."""
    conn.executescript(BOUNDARY_OBSERVATIONS_DDL)
    conn.executescript(CROSS_CAMERA_CONFLICTS_DDL)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Staging write + pending lookup
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PendingObservation:
    id: int
    camera_id: str
    session_id: str
    temp_id: int
    enter_time: datetime
    exit_time: datetime
    flat_embedding: np.ndarray
    resolved_animal_id: Optional[int]
    resolved_confidence: Optional[float]
    consumed: bool = field(default=False, compare=False)


def write_boundary_observation(
    conn: sqlite3.Connection,
    camera_id: str, zone_pair_id: str, session_id: str, temp_id: int,
    dwell: DwellInfo, flat_embedding: np.ndarray,
    resolved_animal_id: Optional[int], resolved_confidence: Optional[float],
    dry_run: bool = False,
) -> int:
    import json
    if dry_run:
        return -1
    cur = conn.execute(
        """
        INSERT INTO boundary_observations
            (camera_id, zone_pair_id, session_id, temp_id,
             enter_time, exit_time, frames_in_zone, flat_embedding,
             resolved_animal_id, resolved_confidence, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        (camera_id, zone_pair_id, session_id, temp_id,
         dwell.enter_time.isoformat(), dwell.exit_time.isoformat(), dwell.frames,
         json.dumps(flat_embedding.tolist()),
         resolved_animal_id, resolved_confidence),
    )
    conn.commit()
    return cur.lastrowid


def find_pending_partner_observations(
    conn: sqlite3.Connection,
    zone_pair_id: str, partner_camera_id: str,
    enter_time: datetime, exit_time: datetime,
) -> list[PendingObservation]:
    import json
    rows = conn.execute(
        """
        SELECT id, camera_id, session_id, temp_id, enter_time, exit_time,
               flat_embedding, resolved_animal_id, resolved_confidence
        FROM boundary_observations
        WHERE zone_pair_id = ? AND camera_id = ? AND status = 'pending'
          AND NOT (exit_time < ? OR enter_time > ?)
        """,
        (zone_pair_id, partner_camera_id, enter_time.isoformat(), exit_time.isoformat()),
    ).fetchall()
    out = []
    for r in rows:
        out.append(PendingObservation(
            id=r[0], camera_id=r[1], session_id=r[2], temp_id=r[3],
            enter_time=pd.Timestamp(r[4]), exit_time=pd.Timestamp(r[5]),
            flat_embedding=np.array(json.loads(r[6]), dtype=np.float32),
            resolved_animal_id=r[7], resolved_confidence=r[8],
        ))
    return out


def _mark_status(conn: sqlite3.Connection, obs_id: int, status: str,
                  resolved_animal_id: Optional[int] = None,
                  resolved_confidence: Optional[float] = None) -> None:
    if obs_id < 0:   # dry_run sentinel
        return
    if resolved_animal_id is not None:
        conn.execute(
            "UPDATE boundary_observations SET status=?, resolved_animal_id=?, "
            "resolved_confidence=? WHERE id=?",
            (status, resolved_animal_id, resolved_confidence, obs_id),
        )
    else:
        conn.execute("UPDATE boundary_observations SET status=? WHERE id=?", (status, obs_id))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Tier hierarchy matching
# ─────────────────────────────────────────────────────────────────────────────

# Tier numbers: lower = higher priority (sorted ascending, score descending
# within a tier).
TIER_KINETIC_BOTH = 1
TIER_COSINE = 2
TIER_MOTION_MAGNITUDE = 3
TIER_SINGLE_COW = 4


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def match_tier(
    a: PendingObservation, b: PendingObservation,
    config: CrossCameraConfig,
    n_candidates_a: int, n_candidates_b: int,
    motion_corr: Optional[float] = None,
) -> Optional[tuple[int, float]]:
    """
    Returns (tier, score) for this candidate pair, or None if no tier
    accepts it. Higher-priority tiers are tried first; a pair can only be
    scored at the HIGHEST tier it qualifies for (not all of them), so the
    global best-pairing step compares like with like within a tier before
    falling back across tiers.
    """
    # Tier 1 — both independently kinetically confirmed to the same AnimalId.
    if (a.resolved_animal_id is not None and b.resolved_animal_id is not None
            and a.resolved_animal_id == b.resolved_animal_id
            and a.resolved_animal_id > 0):   # positive = real, not synthetic
        conf = min(a.resolved_confidence or 1.0, b.resolved_confidence or 1.0)
        return TIER_KINETIC_BOTH, conf

    # Tier 2 — flat-embedding cosine similarity.
    cos = _cosine(a.flat_embedding, b.flat_embedding)
    if cos >= config.cosine_threshold:
        return TIER_COSINE, cos

    # Tier 3 — motion-magnitude correlation (caller computes this; expensive
    # enough, and dependent on enough raw_tracks history, that it's not
    # worth computing for every candidate pair up front).
    if motion_corr is not None and motion_corr >= config.motion_corr_threshold:
        return TIER_MOTION_MAGNITUDE, motion_corr

    # Tier 4 — only one cow present on each side of the zone at this time.
    if n_candidates_a == 1 and n_candidates_b == 1 and cos >= config.single_cow_sanity_floor:
        return TIER_SINGLE_COW, cos

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — motion-magnitude correlation
# ─────────────────────────────────────────────────────────────────────────────

def _load_track(conn: sqlite3.Connection, session_id: str, temp_id: int) -> pd.DataFrame:
    """All raw_tracks rows for one track, with frame_datetime parsed.
    Columns: t (Timestamp), cx, cy, h."""
    rows = conn.execute(
        "SELECT frame_datetime, x1, y1, x2, y2 FROM raw_tracks "
        "WHERE session_id = ? AND temp_id = ?",
        (session_id, int(temp_id)),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["t", "cx", "cy", "h"])
    df = pd.DataFrame(rows, columns=["t", "x1", "y1", "x2", "y2"])
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    return pd.DataFrame({
        "t":  df["t"],
        "cx": (df["x1"] + df["x2"]) / 2.0,
        "cy": (df["y1"] + df["y2"]) / 2.0,
        "h":  (df["y2"] - df["y1"]).clip(lower=1.0),
    })


def _speed_series(track: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp,
                  bin_s: float) -> pd.Series:
    """
    Speed in body-heights per second, one value per time bin.

    Bins are fixed wall-clock intervals starting at t0, so the same index
    means the same moment for both cameras. Per bin: mean box center and mean
    box height. speed[k] = distance between bin k and bin k+1 centers, divided
    by the mean height of the two bins and by bin_s. Indexed by bin number;
    only bins where the track is present in both k and k+1 are included.

    Dividing by box height makes the value comparable within a camera (a cow
    near the camera has a bigger box and moves more pixels per step) — the
    raw pixel scale doesn't matter for Pearson r, but uneven scale within the
    window would distort the shape of the curve.
    """
    w = track[(track["t"] >= t0) & (track["t"] < t1)]
    if w.empty:
        return pd.Series(dtype=float)
    k = ((w["t"] - t0).dt.total_seconds() // bin_s).astype(int)
    per_bin = w.groupby(k)[["cx", "cy", "h"]].mean()
    nxt = per_bin.shift(-1)
    consecutive = pd.Series(per_bin.index, index=per_bin.index).shift(-1) == per_bin.index + 1
    step = np.hypot(nxt["cx"] - per_bin["cx"], nxt["cy"] - per_bin["cy"])
    height = (per_bin["h"] + nxt["h"]) / 2.0
    speed = step / height / bin_s
    return speed[consecutive].dropna()


def motion_magnitude_corr(track_a: pd.DataFrame, track_b: pd.DataFrame,
                          t0: pd.Timestamp, t1: pd.Timestamp,
                          config: "CrossCameraConfig") -> "float | None":
    """
    Pearson r between two tracks' speed curves over [t0, t1), maximized over
    small time shifts (±motion_max_lag_bins) to absorb clock offset between
    cameras. Uses speed magnitude only, never direction: cameras 26/38 and 33
    face opposite ways, so "moving right" in one is not "moving right" in the
    other, but "speeding up" and "stopping" look the same in both.

    Returns None — meaning "no evidence either way", not "no match" — when:
      - fewer than motion_min_samples shared speed bins, or
      - either cow barely moved (speed std < motion_min_std). Two near-still
        curves are just detection jitter, and jitter can correlate by chance.
    """
    sa = _speed_series(track_a, t0, t1, config.motion_bin_seconds)
    sb = _speed_series(track_b, t0, t1, config.motion_bin_seconds)
    if len(sa) < config.motion_min_samples or len(sb) < config.motion_min_samples:
        return None
    if sa.std() < config.motion_min_std or sb.std() < config.motion_min_std:
        return None

    best = None
    for lag in range(-config.motion_max_lag_bins, config.motion_max_lag_bins + 1):
        shifted = sb.copy()
        shifted.index = shifted.index + lag
        common = sa.index.intersection(shifted.index)
        if len(common) < config.motion_min_samples:
            continue
        x, y = sa.loc[common].to_numpy(), shifted.loc[common].to_numpy()
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if best is None or r > best:
            best = r
    return best


def make_motion_corr_fn(conn: sqlite3.Connection, config: "CrossCameraConfig"):
    """
    Build the tier-3 scorer used by pair_candidates: (a, b) -> r | None.

    Both tracks are read from raw_tracks by (session_id, temp_id) — every
    camera's tracks live in the same DB, so the partner camera's track is
    available however long ago its session was processed. The comparison
    window is the overlap of the two dwell windows, padded on both sides by
    motion_window_pad_seconds: a few seconds of dwell alone is too few
    samples, and both cameras usually see the cow before and after the zone.
    Tracks are cached, since one track can be scored against several
    candidates.
    """
    cache: dict = {}

    def track(obs: PendingObservation) -> pd.DataFrame:
        key = (obs.session_id, int(obs.temp_id))
        if key not in cache:
            cache[key] = _load_track(conn, *key)
        return cache[key]

    def fn(a: PendingObservation, b: PendingObservation) -> "float | None":
        pad = pd.Timedelta(seconds=config.motion_window_pad_seconds)
        t0 = max(pd.Timestamp(a.enter_time), pd.Timestamp(b.enter_time)) - pad
        t1 = min(pd.Timestamp(a.exit_time),  pd.Timestamp(b.exit_time))  + pad
        if t1 <= t0:
            return None
        try:
            ta, tb = track(a), track(b)
        except sqlite3.Error:
            return None   # can't read the tracks: no evidence, never a crash
        return motion_magnitude_corr(ta, tb, t0, t1, config)

    return fn


def pair_candidates(
    candidates_a: list[PendingObservation],
    candidates_b: list[PendingObservation],
    config: CrossCameraConfig,
    motion_corr_fn=None,   # Optional[(a, b) -> float | None]; see make_motion_corr_fn
) -> list[tuple[PendingObservation, PendingObservation, int, float]]:
    """
    Score every (a, b) candidate pair, then take the globally best set of
    non-conflicting pairings: sort by (tier ascending, score descending),
    greedily accept a pair if neither side has been consumed yet.

    motion_corr_fn is only called for pairs that tiers 1-2 didn't already
    match (it reads both tracks from raw_tracks, so it isn't free).
    """
    n_a, n_b = len(candidates_a), len(candidates_b)
    scored = []
    for a, b in product(candidates_a, candidates_b):
        result = match_tier(a, b, config, n_a, n_b)
        if motion_corr_fn and (result is None or result[0] > TIER_MOTION_MAGNITUDE):
            result = match_tier(a, b, config, n_a, n_b,
                                motion_corr=motion_corr_fn(a, b))
        if result is not None:
            tier, score = result
            scored.append((tier, score, a, b))

    scored.sort(key=lambda x: (x[0], -x[1]))

    accepted = []
    for tier, score, a, b in scored:
        if a.consumed or b.consumed:
            continue
        a.consumed = b.consumed = True
        accepted.append((a, b, tier, score))
    return accepted


# ─────────────────────────────────────────────────────────────────────────────
# Resolution: zone-consistency / confidence conflict rule
# ─────────────────────────────────────────────────────────────────────────────

def resolve_cross_camera_pair(
    conn: sqlite3.Connection,
    a: PendingObservation, b: PendingObservation,
    tier: int, score: float,
    config: CrossCameraConfig,
    dry_run: bool = False,
) -> dict:
    """
    A pairing has already been decided (by pair_candidates) — this just
    applies the zone-consistency / confidence rule to it and writes the
    outcome. Both observations are, by construction, in the SAME shared
    zone with overlapping dwell windows, so zone-consistency here reduces
    to: do the two sides already carry conflicting *resolved* identities?

        - Neither resolved, or both resolved to the SAME AnimalId:
              zone-consistent -> corroboration.
        - Resolved to DIFFERENT AnimalIds:
              real conflict -> confidence rule.

    Returns a dict describing the outcome; caller uses it to update the
    live `assignment` dict for Boundary Rendezvous's own session (see run_boundary_rendezvous).

    NOTE: does not check this pairing's result against identity claims
    recorded elsewhere in the DB (e.g. a non-adjacent camera's independent
    kinetic claim on the same AnimalId) — that's the deferred cross-claim
    check from the design doc, intentionally out of scope here.
    """
    aid_a, aid_b = a.resolved_animal_id, b.resolved_animal_id

    if aid_a is None and aid_b is None:
        # Neither side resolved — nothing to corroborate or conflict with,
        # just a confirmed physical link. Caller may still want to mint a
        # synthetic id for the pair; left to the integration layer since
        # that needs gallery_dir / mint_synthetic_id from reconcile.py.
        outcome = {"kind": "linked_unresolved", "tier": tier, "score": score}

    elif aid_a is None or aid_b is None or aid_a == aid_b:
        winner_aid = aid_a if aid_a is not None else aid_b
        outcome = {"kind": "corroboration", "animal_id": winner_aid,
                   "tier": tier, "score": score}

    else:
        conf_a = a.resolved_confidence or 0.0
        conf_b = b.resolved_confidence or 0.0
        gap = abs(conf_a - conf_b)
        if gap >= config.conflict_confidence_gap:
            winner, loser, winner_aid = (a, b, aid_a) if conf_a > conf_b else (b, a, aid_b)
            outcome = {"kind": "resolved_by_confidence", "animal_id": winner_aid,
                       "superseded_session": loser.session_id,
                       "superseded_temp_id": loser.temp_id, "tier": tier, "score": score}
            if not dry_run:
                conn.execute(
                    "INSERT INTO cross_camera_conflicts "
                    "(real_id, session_a, session_b, camera_a, camera_b, "
                    " overlap_window_start, overlap_window_end, zone_check_passed, "
                    " confidence_a, confidence_b, status) "
                    "VALUES (?,?,?,?,?,?,?,1,?,?,'resolved_by_confidence')",
                    (winner_aid, a.session_id, b.session_id, a.camera_id, b.camera_id,
                     max(a.enter_time, b.enter_time).isoformat(),
                     min(a.exit_time, b.exit_time).isoformat(),
                     conf_a, conf_b),
                )
                conn.commit()
        else:
            outcome = {"kind": "needs_review", "tier": tier, "score": score}
            if not dry_run:
                conn.execute(
                    "INSERT INTO cross_camera_conflicts "
                    "(real_id, session_a, session_b, camera_a, camera_b, "
                    " overlap_window_start, overlap_window_end, zone_check_passed, "
                    " confidence_a, confidence_b, status) "
                    "VALUES (?,?,?,?,?,?,?,1,?,?,'needs_review')",
                    (aid_a, a.session_id, b.session_id, a.camera_id, b.camera_id,
                     max(a.enter_time, b.enter_time).isoformat(),
                     min(a.exit_time, b.exit_time).isoformat(),
                     conf_a, conf_b),
                )
                conn.commit()

    if not dry_run:
        _mark_status(conn, a.id, "matched")
        _mark_status(conn, b.id, "matched")
    return outcome


# ─────────────────────────────────────────────────────────────────────────────
# Coverage-based finalization (replaces a timeout — see design doc)
# ─────────────────────────────────────────────────────────────────────────────

def finalize_no_match_observations(
    conn: sqlite3.Connection,
    camera_id: str, session_start: datetime, session_end: datetime,
    dry_run: bool = False,
) -> list[int]:
    """
    This session just finished processing [session_start, session_end] on
    camera_id. For every zone where camera_id is the PARTNER camera, any
    still-pending observation from the other side whose time range is
    fully covered by this session's range — and that this session's own
    staging pass didn't match — is now provably unmatched: this camera had
    the chance to see it and didn't. Finalize those as 'no_match_found'
    rather than leaving them pending forever or guessing with a timeout.
    """
    finalized = []
    for zp in _zone_pairs_for_camera(camera_id):
        rows = conn.execute(
            """
            SELECT id FROM boundary_observations
            WHERE zone_pair_id = ? AND camera_id = ? AND status = 'pending'
              AND enter_time >= ? AND exit_time <= ?
            """,
            (zp["zone_pair_id"], zp["partner_cam"],
             session_start.isoformat(), session_end.isoformat()),
        ).fetchall()
        for (obs_id,) in rows:
            finalized.append(obs_id)
            if not dry_run:
                _mark_status(conn, obs_id, "no_match_found")
    return finalized


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator — call this from reconcile.py
# ─────────────────────────────────────────────────────────────────────────────

def run_boundary_rendezvous(
    conn: sqlite3.Connection,
    session_id: str,
    camera_id: str,
    tracks_df: pd.DataFrame,          # this session's raw_tracks (all temp_ids)
    embed_df: pd.DataFrame,           # this session's flat embeddings (temp_id, embed)
    assignment: dict,                 # {temp_id -> real_id or synthetic_id}, mutated in place
    confidence: dict,                 # {temp_id -> float}, see module docstring
    config: CrossCameraConfig | None = None,   # None → from params.yaml
    dry_run: bool = False,
    motion_corr_fn=None,
    protected_tids: set | None = None,   # temp_ids whose id came from manual or kinetic
) -> tuple[dict, list[str]]:
    """
    Runs Boundary Rendezvous for one session. Returns (updated_assignment, log_lines).
    `assignment` is also mutated in place for convenience.

    `protected_tids` are never overwritten: a manual or collar-confirmed id
    outranks a cross-camera match (params identity.source_priority). If a boundary
    match points such a temp_id at a different id, the existing id is kept
    and the disagreement is logged. Unassigned temp_ids and ones resolved by
    appearance alone (cosine) can still be set or changed.

    Does NOT re-run resolve_duplicate_assignments — the caller (reconcile.py)
    should do that after this returns, same as the existing Duplicate Resolver call,
    so same-session conflict protection also covers anything Boundary Rendezvous
    changed. See the design doc's Priority rule section.
    """
    if config is None:
        config = CrossCameraConfig.from_params(_config.params())
    log: list[str] = []
    migrate_cross_camera_schema(conn)

    session_start = tracks_df["frame_datetime"].min()
    session_end = tracks_df["frame_datetime"].max()

    # ── Stage this session's qualifying boundary tracks ──────────────────────
    staged_ids: dict[int, int] = {}   # temp_id -> boundary_observations.id
    for temp_id in tracks_df["temp_id"].unique():
        dwell = compute_dwell_in_known_zones(tracks_df, int(temp_id), camera_id, config)
        if dwell is None:
            continue

        tid_embeds = embed_df[embed_df["temp_id"] == temp_id]
        if tid_embeds.empty:
            log.append(f"  [boundary_rendezvous] t{temp_id}: qualifies for zone {dwell.zone_pair_id} "
                       f"but has no embedding — skipped")
            continue
        flat_embed = np.stack(tid_embeds["embed"].values).mean(axis=0)
        norm = np.linalg.norm(flat_embed)
        if norm > 1e-8:
            flat_embed = flat_embed / norm

        obs_id = write_boundary_observation(
            conn, camera_id=camera_id, zone_pair_id=dwell.zone_pair_id,
            session_id=session_id, temp_id=int(temp_id), dwell=dwell,
            flat_embedding=flat_embed,
            resolved_animal_id=assignment.get(int(temp_id)),
            resolved_confidence=confidence.get(int(temp_id)),
            dry_run=dry_run,
        )
        staged_ids[int(temp_id)] = obs_id
        log.append(f"  [boundary_rendezvous] t{temp_id}: staged in zone {dwell.zone_pair_id} "
                   f"({dwell.frames} frames, {dwell.seconds:.1f}s)")

    if not staged_ids:
        log.append("  [boundary_rendezvous] No boundary candidates this session.")

    # ── For each zone this camera participates in, look for a partner match ──
    by_zone: dict[str, list[PendingObservation]] = {}
    for temp_id, obs_id in staged_ids.items():
        dwell = compute_dwell_in_known_zones(tracks_df, temp_id, camera_id, config)
        by_zone.setdefault(dwell.zone_pair_id, []).append(PendingObservation(
            id=obs_id, camera_id=camera_id, session_id=session_id, temp_id=temp_id,
            enter_time=dwell.enter_time, exit_time=dwell.exit_time,
            flat_embedding=np.stack(
                embed_df[embed_df["temp_id"] == temp_id]["embed"].values
            ).mean(axis=0),
            resolved_animal_id=assignment.get(temp_id),
            resolved_confidence=confidence.get(temp_id),
        ))

    for zone_pair_id, this_side in by_zone.items():
        zp = next(z for z in _zone_pairs_for_camera(camera_id) if z["zone_pair_id"] == zone_pair_id)
        window_start = min(o.enter_time for o in this_side)
        window_end = max(o.exit_time for o in this_side)
        partner_side = find_pending_partner_observations(
            conn, zone_pair_id, zp["partner_cam"], window_start, window_end
        )
        if not partner_side:
            continue

        pairs = pair_candidates(this_side, partner_side, config,
                                motion_corr_fn=motion_corr_fn or make_motion_corr_fn(conn, config))
        for a, b, tier, score in pairs:
            outcome = resolve_cross_camera_pair(conn, a, b, tier, score, config, dry_run=dry_run)
            log.append(f"  [boundary_rendezvous] t{a.temp_id}({a.camera_id}) <-> "
                       f"t{b.temp_id}({b.camera_id})  tier={tier} score={score:.3f}  "
                       f"-> {outcome['kind']}")

            if outcome["kind"] in ("corroboration", "resolved_by_confidence"):
                this_tid = a.temp_id if a.camera_id == camera_id else b.temp_id
                old = assignment.get(this_tid)
                new = outcome["animal_id"]
                if old == new:
                    continue
                if protected_tids and this_tid in protected_tids:
                    log.append(f"  [boundary_rendezvous] KEPT t{this_tid}={old}: partner "
                               f"camera suggests {new}, but a manual/kinetic id "
                               f"outranks a boundary match ({outcome['kind']})")
                    continue
                log.append(f"  [boundary_rendezvous] OVERRIDE t{this_tid}: {old} -> {new} "
                           f"(reason={outcome['kind']})")
                assignment[this_tid] = new

    # ── Coverage-based finalization ───────────────────────────────────────────
    finalized = finalize_no_match_observations(conn, camera_id, session_start, session_end,
                                                dry_run=dry_run)
    if finalized:
        log.append(f"  [boundary_rendezvous] Finalized {len(finalized)} pending observation(s) "
                   f"as no_match_found (this session's coverage proved no match)")

    return assignment, log


# ─────────────────────────────────────────────────────────────────────────────
# Helper: building the `confidence` dict reconcile.py needs to pass in
# ─────────────────────────────────────────────────────────────────────────────

def build_confidence_dict(
    kinetic_corr: dict,      # {temp_id -> Pearson r}, from Kinetic Matcher
    cosine_scores: dict,     # {temp_id -> cosine}, from Cosine Resolver
) -> dict:
    """
    reconcile.py doesn't currently collect per-temp_id confidence into one
    dict — Kinetic Matcher's `corr` and Cosine Resolver's cosine scores are local variables
    scoped to their own loops. This is a drop-in example of how to build
    the `confidence` dict run_boundary_rendezvous needs; adapt the two input dicts'
    construction to reconcile.py's actual variable names at the call site.
    """
    out = dict(kinetic_corr)
    for tid, cos in cosine_scores.items():
        out.setdefault(tid, cos)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Minimal self-test with synthetic data (no DB required for the geometry/
# pairing logic — exercises the parts that don't need a real session)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Point-in-polygon sanity check:")
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert _point_in_polygon(5, 5, square) is True
    assert _point_in_polygon(15, 5, square) is False
    print("  OK")

    print("\nTier hierarchy sanity check:")
    cfg = CrossCameraConfig.from_params(_config.params())
    a = PendingObservation(id=1, camera_id="רפת 26", session_id="s1", temp_id=21,
                            enter_time=pd.Timestamp("2026-06-16 09:07:00"),
                            exit_time=pd.Timestamp("2026-06-16 09:07:05"),
                            flat_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                            resolved_animal_id=6366, resolved_confidence=0.89)
    b = PendingObservation(id=2, camera_id="רפת 38", session_id="s2", temp_id=5,
                            enter_time=pd.Timestamp("2026-06-16 09:07:02"),
                            exit_time=pd.Timestamp("2026-06-16 09:07:08"),
                            flat_embedding=np.array([0.98, 0.1, 0.0], dtype=np.float32),
                            resolved_animal_id=None, resolved_confidence=None)
    result = match_tier(a, b, cfg, n_candidates_a=1, n_candidates_b=1)
    print(f"  a vs b -> {result} (expect tier 2, cosine ~0.995)")
    assert result is not None and result[0] == TIER_COSINE

    print("\nAll self-tests passed.")