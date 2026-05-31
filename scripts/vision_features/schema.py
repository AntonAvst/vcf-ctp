"""
vision_features/schema.py
─────────────────────────
Canonical definitions for posture labels, facing labels, pose-conditioned gallery
slot indices, and the column names added to resolved_cow_timeline.

Everything else in the vision_features package imports from here — never hard-codes
strings or indices directly.
"""

from enum import IntEnum


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint index map  (19 kp, matches track_and_dump.py EXPECTED_KP = 19)
# ─────────────────────────────────────────────────────────────────────────────

class KP(IntEnum):
    NOSE             = 0
    FOREHEAD         = 1
    WITHERS          = 2
    SPINE_MID        = 3
    SACRUM           = 4
    TAIL_BASE        = 5
    TAIL_TIP         = 6
    SHOULDER_L       = 7
    ELBOW_L          = 8
    FETLOCK_FORE_L   = 9
    SHOULDER_R       = 10
    ELBOW_R          = 11
    FETLOCK_FORE_R   = 12
    HOCK_R           = 13
    HOCK_L           = 14
    FETLOCK_HIND_L   = 15
    FETLOCK_HIND_R   = 16
    UDDER_CENTER     = 17
    NECK             = 18


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame label enums
# ─────────────────────────────────────────────────────────────────────────────

class Posture(IntEnum):
    UNCERTAIN = 0
    STANDING  = 1
    LYING     = 2


class Facing(IntEnum):
    UNCERTAIN = 0
    LEFT      = 1
    RIGHT     = 2
    TOWARD    = 3   # nose pointed toward camera (small bbox, facing viewer)
    AWAY      = 4   # nose pointed away from camera


# Human-readable names — used for logging and gallery .npy keys
POSTURE_NAMES = {
    Posture.UNCERTAIN: "uncertain",
    Posture.STANDING:  "standing",
    Posture.LYING:     "lying",
}

FACING_NAMES = {
    Facing.UNCERTAIN: "uncertain",
    Facing.LEFT:      "left",
    Facing.RIGHT:     "right",
    Facing.TOWARD:    "toward",
    Facing.AWAY:      "away",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pose-conditioned gallery slots
#
# 8 slots = {standing, lying} × {left, right, toward, away}
# Slot index is the canonical key used in gallery .npy files
#   gallery_pose_day.npy  → shape (N_cows, 8, 128)
#   gallery_pose_night.npy → shape (N_cows, 8, 128)
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list of (posture, facing) pairs — the index in this list IS the slot index.
# UNCERTAIN posture/facing frames are never assigned to a slot.
GALLERY_SLOTS: list[tuple[Posture, Facing]] = [
    (Posture.STANDING, Facing.LEFT),    # slot 0
    (Posture.STANDING, Facing.RIGHT),   # slot 1
    (Posture.STANDING, Facing.TOWARD),  # slot 2
    (Posture.STANDING, Facing.AWAY),    # slot 3
    (Posture.LYING,    Facing.LEFT),    # slot 4
    (Posture.LYING,    Facing.RIGHT),   # slot 5
    (Posture.LYING,    Facing.TOWARD),  # slot 6
    (Posture.LYING,    Facing.AWAY),    # slot 7
]

N_SLOTS = len(GALLERY_SLOTS)   # 8

# Quick lookup: (posture, facing) → slot index
SLOT_INDEX: dict[tuple[Posture, Facing], int] = {
    pair: idx for idx, pair in enumerate(GALLERY_SLOTS)
}

# Human-readable slot names — used for logging
SLOT_NAMES: list[str] = [
    f"{POSTURE_NAMES[p]}_{FACING_NAMES[f]}" for p, f in GALLERY_SLOTS
]   # e.g. ["standing_left", "standing_right", ..., "lying_away"]


def slot_name(slot_idx: int) -> str:
    """Return human-readable name for a slot index, e.g. 'standing_left'."""
    return SLOT_NAMES[slot_idx]


def slot_index(posture: Posture, facing: Facing) -> int | None:
    """
    Return the slot index for a (posture, facing) pair.
    Returns None if either is UNCERTAIN (no slot should be assigned).
    """
    if posture == Posture.UNCERTAIN or facing == Facing.UNCERTAIN:
        return None
    return SLOT_INDEX.get((posture, facing))


# ─────────────────────────────────────────────────────────────────────────────
# resolved_cow_timeline — new columns added by this module
#
# These are the scalar, per-window aggregates written into the timeline.
# Raw per-frame labels are NOT stored in the timeline — only the aggregates.
# ─────────────────────────────────────────────────────────────────────────────

TIMELINE_VISION_COLS: list[str] = [
    # posture
    "lying_fraction",       # float [0,1]  — fraction of confident frames that were LYING
    "posture_transitions",  # int          — number of STANDING↔LYING switches in window
    # facing
    "facing_dominant",      # str          — modal facing direction (non-uncertain frames)
    "facing_entropy",       # float [0,1]  — normalised entropy of facing distribution
]

# SQLite ALTER TABLE statements to add the new columns to an existing DB.
# reconcile.py runs these on startup (IF NOT EXISTS via try/except).
TIMELINE_ALTER_SQLS: list[str] = [
    "ALTER TABLE resolved_cow_timeline ADD COLUMN lying_fraction      REAL",
    "ALTER TABLE resolved_cow_timeline ADD COLUMN posture_transitions INTEGER",
    "ALTER TABLE resolved_cow_timeline ADD COLUMN facing_dominant     TEXT",
    "ALTER TABLE resolved_cow_timeline ADD COLUMN facing_entropy      REAL",
]
