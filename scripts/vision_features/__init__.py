"""
vision_features/__init__.py
Place at: vision_features/__init__.py  (i.e. make a folder called vision_features/ next to reconcile.py)

vision_features — Stage 2-B vision feature extraction for the calving prediction pipeline.

Public API (all reconcile.py needs to import):
    run_vision_features(session_id, timeline_df, conn, assignment, is_night, ...)
    migrate_timeline_schema(conn)

Internal subpackages:
    features/   — per-frame extractors (posture, facing, ...)
    gallery/    — pose-conditioned 8-slot ReID gallery
    schema.py   — enums, column names, KP index map
"""

from .extractor import run_vision_features, migrate_timeline_schema

__all__ = ["run_vision_features", "migrate_timeline_schema"]
