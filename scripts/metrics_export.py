#!/usr/bin/env python3
"""
metrics_export.py — offline metrics exporter for the VCF-CTP development dashboard.

Reads from:
  - calving_project.db  (SQLite)
  - collar_data/        (kinetic_data_*.csv / behavior_data_*.csv)
  - processing_log.csv  (Drive-synced batch log)
  - reid_gallery/       (.npy files — optional, for gallery slot counts)
  - .buffer/            (dirty flags, buffer queue)

Writes:
  - metrics.json        (consumed by coverage_timeline.html / metrics_dashboard.html)

Usage:
  python3 metrics_export.py                        # always uses drive_manager paths
  python3 metrics_export.py --out path/to/out.json  # override OUTPUT path only (not a data source)
  python3 metrics_export.py --watch                 # re-export every 30 s (dev mode)

Note: there is no --db, --collar, --gallery, --proclog, or --buffer flag.
Every data path is imported directly from drive_manager.py — never
redefined or overridable here — so this script can't silently drift from
the single source of truth.
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Paths — imported directly from drive_manager.py, never redefined here ────
from drive_manager import (
    LOCAL_ROOT, LOCAL_DB_PATH, LOCAL_GALLERY_DIR, LOCAL_COLLAR_DIR, BUFFER_DIR,
    DriveManager,
)

SCRIPTS_DIR  = Path(__file__).parent

DEFAULT_DB           = LOCAL_DB_PATH
DEFAULT_COLLAR_DIR   = LOCAL_COLLAR_DIR
DEFAULT_GALLERY_DIR  = LOCAL_GALLERY_DIR
DEFAULT_PROC_LOG     = LOCAL_ROOT / "processing_log.csv"          # synced from Drive
DEFAULT_OUT          = SCRIPTS_DIR.parent / "metrics" / "metrics.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(dt_str: str | None) -> str | None:
    """Normalise a datetime string to ISO-8601 with Z suffix, or return None."""
    if not dt_str:
        return None
    try:
        # Accept both "2024-12-21 07:00:00" and ISO variants
        dt = pd.to_datetime(dt_str)
        return dt.isoformat(timespec="seconds") + "Z" if not dt_str.endswith("Z") else dt_str
    except Exception:
        return dt_str


def _safe_query(conn: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    """Run a query; return [] if the table/column doesn't exist yet."""
    try:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        return []


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    rows = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchall()
    return bool(rows)


# ── Section exporters ─────────────────────────────────────────────────────────

def export_pipeline(conn: sqlite3.Connection, proc_log_path: Path) -> dict:
    """Processing log summary + reconcile step pass rates."""

    # --- processing_log.csv ---
    recent_videos = []
    proc_summary  = {"total": 0, "ok": 0, "error": 0, "interrupted": 0, "skipped": 0,
                     "avg_fps": None, "total_frames": 0, "total_duration_s": 0}

    if proc_log_path.exists():
        try:
            pl = pd.read_csv(proc_log_path, parse_dates=["timestamp"])
            proc_summary["total"]         = len(pl)
            proc_summary["ok"]            = int((pl["status"] == "ok").sum())
            proc_summary["error"]         = int((pl["status"] == "error").sum())
            proc_summary["interrupted"]   = int((pl["status"] == "interrupted").sum())
            proc_summary["skipped"]       = int((pl["status"] == "skipped").sum())
            proc_summary["total_frames"]  = int(pl["frames"].sum())
            proc_summary["total_duration_s"] = float(pl["duration_s"].sum())
            ok_fps = pl.loc[pl["status"] == "ok", "fps_processed"].dropna()
            if len(ok_fps):
                proc_summary["avg_fps"] = round(float(ok_fps.mean()), 1)

            for _, row in pl.sort_values("timestamp", ascending=False).head(6).iterrows():
                recent_videos.append({
                    "timestamp":  _ts(str(row.get("timestamp", ""))),
                    "session_id": str(row.get("session_id", "")),
                    "filename":   str(row.get("filename", "")),
                    "status":     str(row.get("status", "")),
                    "frames":     int(row["frames"]) if pd.notna(row.get("frames")) else None,
                    "fps":        round(float(row["fps_processed"]), 1)
                               if pd.notna(row.get("fps_processed")) else None,
                    "error":      str(row["error"]) if pd.notna(row.get("error")) else None,
                })
        except Exception as e:
            proc_summary["read_error"] = str(e)

    # --- reconcile step pass rates (derived from resolved_cow_timeline) ---
    step_stats = {}

    # Step A — kinetic match: count rows where match_method includes 'kinetic'
    rows = _safe_query(conn,
        "SELECT COUNT(*) as n FROM reid_registry WHERE match_method='kinetic'")
    step_stats["A_kinetic_match"] = rows[0]["n"] if rows else 0

    # Step A.5 — manual assignments present
    rows = _safe_query(conn, "SELECT COUNT(*) as n FROM manual_assignments")
    step_stats["A5_manual_merge"] = rows[0]["n"] if rows else 0

    # Step B — gallery entries
    rows = _safe_query(conn,
        "SELECT COUNT(*) as n FROM reid_registry WHERE gallery_n_day > 0 OR gallery_n_night > 0")
    step_stats["B_gallery_builder"] = rows[0]["n"] if rows else 0

    # Step C — cosine-resolved
    rows = _safe_query(conn,
        "SELECT COUNT(*) as n FROM reid_registry WHERE match_method LIKE 'cosine%'")
    step_stats["C_cosine_resolver"] = rows[0]["n"] if rows else 0

    # Step A.6 — tracker switches merged
    rows = _safe_query(conn, "SELECT COUNT(*) as n FROM temp_id_merges")
    step_stats["A6_duplicate_resolver"] = rows[0]["n"] if rows else 0

    # Step D — sensor sequenced rows
    rows = _safe_query(conn,
        "SELECT COUNT(*) as n FROM resolved_cow_timeline WHERE d_kin_r IS NOT NULL")
    step_stats["D_sensor_sequencer"] = rows[0]["n"] if rows else 0

    # Step B-V — vision features populated
    rows = _safe_query(conn,
        "SELECT COUNT(*) as n FROM resolved_cow_timeline WHERE lying_fraction IS NOT NULL")
    step_stats["BV_vision_extractor"] = rows[0]["n"] if rows else 0

    # Step E — total timeline rows
    rows = _safe_query(conn, "SELECT COUNT(*) as n FROM resolved_cow_timeline")
    step_stats["E_write_timeline"] = rows[0]["n"] if rows else 0

    return {
        "proc_summary":   proc_summary,
        "recent_videos":  recent_videos,
        "step_stats":     step_stats,
    }


def export_identity(conn: sqlite3.Connection) -> dict:
    """Identity resolution rates, gallery state, modality mask."""

    # Resolution method breakdown
    method_rows = _safe_query(conn,
        "SELECT match_method, COUNT(*) as n FROM reid_registry GROUP BY match_method")
    by_method = {r["match_method"]: r["n"] for r in method_rows}

    # Unresolved temp_ids in latest sessions (real_id IS NULL)
    unresolved = _safe_query(conn,
        """SELECT COUNT(DISTINCT temp_id) as n FROM raw_tracks
           WHERE session_id NOT IN (
               SELECT DISTINCT session_id FROM resolved_cow_timeline WHERE real_id IS NOT NULL
           )""")
    unresolved_count = unresolved[0]["n"] if unresolved else 0

    # Tracker switches
    switches = _safe_query(conn, "SELECT COUNT(*) as n FROM temp_id_merges")
    switch_count = switches[0]["n"] if switches else 0

    # Gallery state per cow
    gallery_rows = _safe_query(conn,
        """SELECT real_id, gallery_n_day, gallery_n_night,
                  gallery_conf_day, gallery_conf_night,
                  last_updated_day_dt, last_updated_night_dt,
                  match_method
           FROM reid_registry ORDER BY real_id""")

    # Modality mask distribution
    mask_rows = _safe_query(conn,
        """SELECT modality_mask, COUNT(*) as n
           FROM resolved_cow_timeline
           GROUP BY modality_mask ORDER BY modality_mask""")

    # Pearson r scores — stored in a scores table if reconcile was run with --verbose
    # Fallback: reconstruct approximate distribution from timeline confidence columns
    pearson_rows = _safe_query(conn,
        "SELECT corr_score FROM kinetic_match_scores")  # may not exist yet
    pearson_hist = [0] * 7  # bins: 0-.2, .2-.4, .4-.6, .6-.7, .7-.8, .8-.9, .9-1.0
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    for r in pearson_rows:
        v = r.get("corr_score", 0) or 0
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= v < bin_edges[i + 1]:
                pearson_hist[i] += 1
                break
        else:
            if v >= 1.0:
                pearson_hist[-1] += 1

    return {
        "by_method":       by_method,
        "unresolved":      unresolved_count,
        "switch_count":    switch_count,
        "gallery":         gallery_rows,
        "modality_masks":  mask_rows,
        "pearson_hist":    pearson_hist,
        "pearson_bins":    ["0–0.2","0.2–0.4","0.4–0.6","0.6–0.7","0.7–0.8","0.8–0.9","0.9–1.0"],
    }


def export_sensor(collar_dir: Path) -> dict:
    """
    Parse collar CSV filenames for coverage windows.
    Also compute per-animal data health from the actual CSV contents.
    """
    kinetic_files  = []
    behavior_files = []

    fname_re = re.compile(
        r"^(kinetic|behavior)_data_s(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})"
        r"-e(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})__(.+?)\.csv$"
    )
    legacy_re = re.compile(
        r"^(kinetic|behavior)_data_([\d_]+)\.csv$"
    )

    def parse_ts(s: str) -> str:
        """'2024_12_21-07_00_00' → ISO string"""
        try:
            dt = datetime.strptime(s, "%Y_%m_%d-%H_%M_%S")
            return dt.isoformat(timespec="seconds") + "Z"
        except Exception:
            return s

    def infer_window_from_csv(path: Path) -> tuple[str | None, str | None, list[str]]:
        """Read first/last datetime and animal IDs from a collar CSV."""
        try:
            df = pd.read_csv(path, usecols=["AnimalId", "datetime"],
                             parse_dates=["datetime"], nrows=None)
            if df.empty:
                return None, None, []
            df["datetime"] = pd.to_datetime(df["datetime"])
            start = df["datetime"].min().isoformat(timespec="seconds") + "Z"
            end   = df["datetime"].max().isoformat(timespec="seconds") + "Z"
            ids   = sorted(df["AnimalId"].dropna().astype(int).unique().tolist())
            return start, end, [str(i) for i in ids]
        except Exception:
            return None, None, []

    if collar_dir.exists():
        for f in sorted(collar_dir.glob("*.csv")):
            m = fname_re.match(f.name)
            if m:
                kind, s_raw, e_raw, ids_raw = m.groups()
                entry = {
                    "file":       f.name,
                    "start":      parse_ts(s_raw),
                    "end":        parse_ts(e_raw),
                    "animal_ids": ids_raw.split("_"),
                }
            else:
                # Legacy filename — infer from CSV content
                m2 = legacy_re.match(f.name)
                kind = m2.group(1) if m2 else ("kinetic" if "kinetic" in f.name else "behavior")
                start, end, ids = infer_window_from_csv(f)
                entry = {
                    "file":       f.name,
                    "start":      start,
                    "end":        end,
                    "animal_ids": ids,
                    "legacy":     True,
                }

            if kind == "kinetic":
                kinetic_files.append(entry)
            else:
                behavior_files.append(entry)

    # Gap analysis — find gaps > 30 min between consecutive kinetic windows per animal
    gaps = []
    all_kinetic = sorted(kinetic_files, key=lambda x: x.get("start") or "")
    for i in range(1, len(all_kinetic)):
        prev_end   = all_kinetic[i - 1].get("end")
        curr_start = all_kinetic[i].get("start")
        if prev_end and curr_start:
            try:
                delta = pd.to_datetime(curr_start) - pd.to_datetime(prev_end)
                if delta.total_seconds() > 1800:
                    gaps.append({
                        "between": [all_kinetic[i - 1]["file"], all_kinetic[i]["file"]],
                        "gap_minutes": round(delta.total_seconds() / 60),
                    })
            except Exception:
                pass

    return {
        "kinetic_files":  kinetic_files,
        "behavior_files": behavior_files,
        "gaps":           gaps,
    }


def export_sessions(conn: sqlite3.Connection) -> dict:
    """Video sessions with start/end datetimes — for the coverage timeline."""
    rows = _safe_query(conn,
        """SELECT session_id, video_path, camera_id,
                  start_dt, end_dt, is_night
           FROM video_sessions
           ORDER BY start_dt""")
    for r in rows:
        r["start"] = _ts(r.pop("start_dt", None))
        r["end"]   = _ts(r.pop("end_dt",   None))
    return {"sessions": rows}


def export_dataset(conn: sqlite3.Connection) -> dict:
    """Calving ledger + vision feature fill rates."""

    calving_rows = _safe_query(conn,
        """SELECT cl.event_id, cl.real_id, cl.calving_dt, cl.outcome,
                  cr.breed, cr.parity
           FROM calving_ledger cl
           LEFT JOIN cow_registry cr ON cl.real_id = cr.real_id
           ORDER BY cl.calving_dt""")

    # Normalise calving_dt
    for r in calving_rows:
        r["calving_dt"] = _ts(r.get("calving_dt"))

    # Class balance
    class_counts = {}
    for r in calving_rows:
        o = r.get("outcome", "unknown")
        class_counts[o] = class_counts.get(o, 0) + 1

    # Unique cows
    unique_cows = len({r["real_id"] for r in calving_rows})

    # Average timeline coverage per event (hours)
    avg_hrs = None
    if calving_rows:
        hrs_rows = _safe_query(conn,
            """SELECT cl.real_id, cl.calving_dt,
                      COUNT(*) * 15.0 / 60 as approx_hrs
               FROM calving_ledger cl
               LEFT JOIN resolved_cow_timeline rct
                   ON cl.real_id = rct.real_id
               GROUP BY cl.event_id""")
        if hrs_rows:
            vals = [r["approx_hrs"] for r in hrs_rows if r["approx_hrs"] is not None]
            avg_hrs = round(sum(vals) / len(vals), 1) if vals else None

    # Cows in ledger that have no collar data
    ledger_ids = {r["real_id"] for r in calving_rows}
    collar_ids_rows = _safe_query(conn,
        "SELECT DISTINCT AnimalId as rid FROM collar_signals")
    collar_ids = {r["rid"] for r in collar_ids_rows}
    missing_collar = sorted(ledger_ids - collar_ids)
    missing_video  = []
    if _table_exists(conn, "video_sessions"):
        vid_ids_rows = _safe_query(conn,
            """SELECT DISTINCT real_id FROM resolved_cow_timeline
               WHERE modality_mask & 2 > 0""")   # bit 2 = vision_ok
        vid_ids = {r["real_id"] for r in vid_ids_rows}
        missing_video = sorted(ledger_ids - vid_ids)

    # Vision feature fill rates
    vision_cols = [
        "lying_fraction", "posture_transitions", "facing_dominant", "facing_entropy",
        "spine_angle", "pelvic_tilt", "tail_elevation", "limb_symmetry",
        "head_drop", "kps_coverage", "restlessness", "embed_mean",
    ]
    fill_rates = {}
    total_rows_res = _safe_query(conn, "SELECT COUNT(*) as n FROM resolved_cow_timeline")
    total_rows = total_rows_res[0]["n"] if total_rows_res else 0

    if total_rows > 0:
        for col in vision_cols:
            res = _safe_query(conn,
                f"SELECT COUNT(*) as n FROM resolved_cow_timeline WHERE {col} IS NOT NULL")
            fill_rates[col] = round(res[0]["n"] / total_rows * 100, 1) if res else 0.0
    else:
        fill_rates = {col: 0.0 for col in vision_cols}

    # Modality completeness per calving event
    modality_completeness = {"sensor_only": 0, "vision_only": 0,
                              "sensor_vision": 0, "all": 0, "none": 0}
    for r in calving_rows:
        masks = _safe_query(conn,
            "SELECT modality_mask FROM resolved_cow_timeline WHERE real_id=?",
            (r["real_id"],))
        combined = 0
        for m in masks:
            combined |= (m.get("modality_mask") or 0)
        sensor = bool(combined & 1)
        vision = bool(combined & 2)
        reid   = bool(combined & 4)
        if sensor and vision and reid:
            modality_completeness["all"] += 1
        elif sensor and vision:
            modality_completeness["sensor_vision"] += 1
        elif sensor:
            modality_completeness["sensor_only"] += 1
        elif vision:
            modality_completeness["vision_only"] += 1
        else:
            modality_completeness["none"] += 1

    return {
        "calving_events":         calving_rows,
        "class_counts":           class_counts,
        "unique_cows":            unique_cows,
        "avg_timeline_hrs":       avg_hrs,
        "missing_collar":         missing_collar,
        "missing_video":          missing_video,
        "vision_fill_rates":      fill_rates,
        "modality_completeness":  modality_completeness,
        "total_timeline_rows":    total_rows,
    }


def export_drive_status(buffer_dir: Path) -> dict:
    """Dirty flags + upload buffer state."""
    flags = {}
    for flag in ("db", "parquet", "gallery"):
        flag_file = buffer_dir / f"{flag}_sync_status.json"
        if flag_file.exists():
            try:
                data = json.loads(flag_file.read_text())
                flags[flag] = data
            except Exception:
                flags[flag] = {"dirty": False}
        else:
            flags[flag] = {"dirty": False, "missing": True}

    pending = []
    pending_dir = buffer_dir / "pending"
    if pending_dir.exists():
        for f in sorted(pending_dir.glob("*.meta.json")):
            try:
                meta = json.loads(f.read_text())
                pending.append(meta)
            except Exception:
                pass

    return {
        "flags":   flags,
        "pending": pending,
        "pending_count": len(pending),
        "abandoned_count": sum(1 for p in pending if p.get("abandoned")),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def export_all(db_path: Path, collar_dir: Path, gallery_dir: Path,
               proc_log: Path, buffer_dir: Path) -> dict:
    """Produce the full metrics dict."""
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db_path":      str(db_path),
    }

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        out["pipeline"] = export_pipeline(conn, proc_log)
        out["identity"] = export_identity(conn)
        out["sessions"] = export_sessions(conn)
        out["dataset"]  = export_dataset(conn)

        conn.close()
    else:
        out["db_missing"] = True
        out["pipeline"]   = {}
        out["identity"]   = {}
        out["sessions"]   = {"sessions": []}
        out["dataset"]    = {}

    out["sensor"]       = export_sensor(collar_dir)
    out["drive_status"] = export_drive_status(buffer_dir)

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # NOTE: no --db / --collar / --gallery / --proclog / --buffer flags.
    # All data paths come directly from drive_manager.py's constants — see
    # the module docstring for why.
    parser.add_argument("--out",     default=str(DEFAULT_OUT),
                        help="Output path for metrics.json (the only overridable path — "
                             "it's a destination, not a data source)")
    parser.add_argument("--watch",   action="store_true",
                        help="Re-export every 30 s (dev mode)")
    parser.add_argument("--pretty",  action="store_true", default=True,
                        help="Pretty-print JSON (default: on)")
    parser.add_argument("--bypass_upload_check", action="store_true",
                        help="Skip dirty-flag check when pulling the DB from Drive")
    args = parser.parse_args()

    # Pull the canonical DB from Drive (dev dashboard reads — never writes)
    dm = DriveManager(bypass=args.bypass_upload_check, caller=__file__)
    try:
        db_path = dm.pull_db(allow_stale=args.bypass_upload_check)
    except Exception as exc:
        print(f"WARNING: could not pull DB from Drive ({exc}); "
              f"metrics will report db_missing if no local copy exists.")
        db_path = DEFAULT_DB

    collar_dir  = DEFAULT_COLLAR_DIR
    gallery_dir = DEFAULT_GALLERY_DIR
    proc_log    = DEFAULT_PROC_LOG
    buffer_dir  = BUFFER_DIR
    out_path    = Path(args.out).expanduser()

    def run_once():
        data = export_all(db_path, collar_dir, gallery_dir, proc_log, buffer_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if args.pretty else None
        out_path.write_text(json.dumps(data, indent=indent, default=str))
        size = out_path.stat().st_size
        print(f"[{datetime.now().strftime('%H:%M:%S')}] metrics.json written "
              f"({size:,} bytes) → {out_path}")
        return data

    if args.watch:
        print(f"Watch mode — refreshing every 30 s. Ctrl+C to stop.")
        while True:
            try:
                run_once()
            except Exception as e:
                print(f"  export error: {e}", file=sys.stderr)
            time.sleep(30)
    else:
        data = run_once()

        # Print a brief human-readable summary
        print()
        if data.get("db_missing"):
            print("  ⚠  DB not found — only sensor/Drive status sections populated.")
        else:
            p = data.get("pipeline", {}).get("proc_summary", {})
            print(f"  Videos:         {p.get('total', 0)} total  "
                  f"({p.get('ok', 0)} ok · {p.get('error', 0)} error)")
            i = data.get("identity", {})
            bm = i.get("by_method", {})
            print(f"  Identity:       {bm.get('kinetic', 0)} kinetic  "
                  f"{bm.get('cosine_day', 0) + bm.get('cosine_night', 0)} cosine  "
                  f"{i.get('unresolved', 0)} unresolved")
            ds = data.get("dataset", {})
            cc = ds.get("class_counts", {})
            print(f"  Calving events: {sum(cc.values())} labeled  "
                  f"({', '.join(f'{k}={v}' for k, v in cc.items()) or 'none'})")
            sens = data.get("sensor", {})
            print(f"  Collar files:   {len(sens.get('kinetic_files', []))} kinetic  "
                  f"{len(sens.get('behavior_files', []))} behavior  "
                  f"({len(sens.get('gaps', []))} gaps detected)")
            drv = data.get("drive_status", {})
            dirty = [k for k, v in drv.get("flags", {}).items() if v.get("dirty")]
            print(f"  Drive flags:    {'dirty: ' + ', '.join(dirty) if dirty else 'all clean'}")
            print(f"  Buffer pending: {drv.get('pending_count', 0)}")


if __name__ == "__main__":
    main()