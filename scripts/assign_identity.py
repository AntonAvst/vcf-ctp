#!/usr/bin/env python3
"""
assign_identity.py — Manual identity assignment tool.

Use when kinetics data is unavailable or doesn't cover the video's date range.
Watch the video in display_tracks.py to identify which temp_id is which cow,
then run this script to write the assignment and optionally run reconcile.

Note: there is no --db, --kinetics, --gallery_dir, or --embed_parquet flag.
The DB is always the canonical copy pulled from Drive via drive_manager.py,
and kinetics/gallery/parquet (used only by --run_reconcile) are resolved the
same way reconcile.py resolves them — keyed off --session.

Usage:
    python3 assign_identity.py \\
        --session   refet33_20241221 \\
        --assign    2:7507  1:6366  71:7513 \\
        --note      "manual — no kinetics coverage for Dec 21" \\
        --run_reconcile

    # Just write assignments, don't reconcile yet:
    python3 assign_identity.py \\
        --session ... --assign 2:7507 1:6366

    # Show current assignments for a session:
    python3 assign_identity.py \\
        --session ... --list

    # Remove a specific temp_id assignment:
    python3 assign_identity.py \\
        --session ... --remove 71
"""

import argparse
import importlib.util
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Drive I/O layer
from drive_manager import DriveManager, DriveNotSyncedError, DriveUnavailableError


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[assign] {msg}", flush=True)

def section(title: str) -> None:
    print(f"\n{'─'*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─'*60}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Manually assign temp_id → real_id for a session, "
                    "then optionally run reconcile.py."
    )

    # required
    ap.add_argument("--session", required=True,
                    help="session_id to assign identities for")

    # assignment actions (mutually exclusive-ish — list/remove don't need kinetics)
    ap.add_argument("--assign", nargs="+", metavar="TEMP_ID:REAL_ID",
                    help="One or more temp_id:real_id pairs, e.g. 2:7507 1:6366 71:7513")
    ap.add_argument("--remove", nargs="+", type=int, metavar="TEMP_ID",
                    help="Remove manual assignment(s) for these temp_ids")
    ap.add_argument("--list",   action="store_true",
                    help="Print current manual assignments for this session and exit")

    # metadata
    ap.add_argument("--note", default="",
                    help="Optional note stored alongside the assignment (e.g. reason)")

    # reconcile passthrough — only needed with --run_reconcile
    ap.add_argument("--run_reconcile", action="store_true",
                    help="Run reconcile.py immediately after writing assignments")
    # NOTE: kinetics, gallery_dir, and embed_parquet are NOT arguments — when
    # --run_reconcile fires, reconcile.py resolves all of these via
    # drive_manager.py, keyed off --session.
    ap.add_argument("--corr_threshold",   type=float, default=0.7)
    ap.add_argument("--cosine_threshold", type=float, default=0.75)
    ap.add_argument("--ema_alpha",        type=float, default=0.15)
    ap.add_argument("--dry_run",          action="store_true",
                    help="Pass --dry_run to reconcile (no DB writes during reconcile)")
    ap.add_argument("--bypass_upload_check", action="store_true",
                    help="Skip dirty-flag check when reading from Drive "
                         "(proceeds with potentially stale data — use with caution)")

    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_ASSIGNMENTS_DDL = """
CREATE TABLE IF NOT EXISTS manual_assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    temp_id     INTEGER NOT NULL,
    real_id     INTEGER NOT NULL,
    assigned_by TEXT    DEFAULT 'manual',
    assigned_dt TEXT,
    note        TEXT,
    UNIQUE(session_id, temp_id)
);
"""

def open_db(bypass: bool = False) -> sqlite3.Connection:
    """
    Pull the canonical DB from Drive, then open the local copy.
    Raises DriveNotSyncedError if unsynced writes exist (unless bypass=True).
    There is no local-file fallback: if Drive is unreachable, this fails loudly
    rather than silently operating on a possibly-stale local copy.
    """
    dm = DriveManager(bypass=bypass, caller=__file__)
    # Dirty-flag check — assign_identity both reads and writes the DB
    try:
        dm.check_flag("db")
    except DriveNotSyncedError as e:
        raise DriveNotSyncedError(str(e))

    # Pull canonical DB from Drive — no local fallback
    local_db = dm.pull_db(allow_stale=False)

    conn = sqlite3.connect(str(local_db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(MANUAL_ASSIGNMENTS_DDL)
    conn.commit()
    return conn, dm


def validate_session(conn: sqlite3.Connection, session_id: str) -> None:
    row = conn.execute(
        "SELECT session_id FROM video_sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    if row is None:
        existing = [r[0] for r in conn.execute(
            "SELECT session_id FROM video_sessions ORDER BY session_id"
        ).fetchall()]
        raise ValueError(
            f"Session '{session_id}' not found in video_sessions.\n"
            f"  Available sessions: {existing or ['(none)']}"
        )


def get_session_temp_ids(conn: sqlite3.Connection, session_id: str) -> set:
    rows = conn.execute(
        "SELECT DISTINCT temp_id FROM raw_tracks WHERE session_id = ?",
        (session_id,)
    ).fetchall()
    return {int(r[0]) for r in rows}


def get_cow_registry_ids(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT real_id FROM cow_registry").fetchall()
    return {int(r[0]) for r in rows}


def get_kinetic_assignments(conn: sqlite3.Connection, session_id: str) -> dict:
    """Return temp_ids already kinetically confirmed in reid_registry for this session."""
    rows = conn.execute(
        "SELECT known_temp_ids FROM reid_registry WHERE match_method = 'kinetic'"
    ).fetchall()
    result = {}
    import json
    for row in rows:
        if not row["known_temp_ids"]:
            continue
        try:
            entries = json.loads(row["known_temp_ids"])
            for e in entries:
                if e.get("session_id") == session_id:
                    result[int(e["temp_id"])] = row["real_id"] if "real_id" in row.keys() else None
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Parse --assign pairs
# ─────────────────────────────────────────────────────────────────────────────

def parse_assign_pairs(assign_list: list[str]) -> dict:
    """
    Parse ['2:7507', '1:6366', '71:7513'] → {2: 7507, 1: 6366, 71: 7513}
    Raises ValueError on bad format.
    """
    result = {}
    for item in assign_list:
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Bad format '{item}' — expected TEMP_ID:REAL_ID (e.g. 2:7507)"
            )
        try:
            tid  = int(parts[0])
            aid  = int(parts[1])
        except ValueError:
            raise ValueError(
                f"Bad format '{item}' — both values must be integers (e.g. 2:7507)"
            )
        result[tid] = aid
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core actions
# ─────────────────────────────────────────────────────────────────────────────

def list_assignments(conn: sqlite3.Connection, session_id: str) -> None:
    rows = conn.execute("""
        SELECT temp_id, real_id, assigned_by, assigned_dt, note
        FROM   manual_assignments
        WHERE  session_id = ?
        ORDER  BY temp_id
    """, (session_id,)).fetchall()

    if not rows:
        log(f"No manual assignments for session '{session_id}'.")
        return

    log(f"Manual assignments for session '{session_id}' ({len(rows)} entries):")
    print(f"  {'temp_id':>8}  {'real_id':>8}  {'by':>8}  {'assigned_dt':>22}  note")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*22}  {'─'*30}")
    for r in rows:
        print(f"  {r['temp_id']:>8}  {r['real_id']:>8}  "
              f"{r['assigned_by']:>8}  {str(r['assigned_dt'] or ''):>22}  "
              f"{r['note'] or ''}")


def write_assignments(conn: sqlite3.Connection,
                      session_id: str,
                      pairs: dict,
                      note: str) -> None:
    """
    Write {temp_id -> real_id} to manual_assignments.
    Uses INSERT OR REPLACE so re-running updates existing entries.
    """
    ts = datetime.now().isoformat(timespec="seconds")
    for tid, aid in sorted(pairs.items()):
        conn.execute("""
            INSERT INTO manual_assignments
                (session_id, temp_id, real_id, assigned_by, assigned_dt, note)
            VALUES (?, ?, ?, 'manual', ?, ?)
            ON CONFLICT(session_id, temp_id) DO UPDATE SET
                real_id     = excluded.real_id,
                assigned_by = excluded.assigned_by,
                assigned_dt = excluded.assigned_dt,
                note        = excluded.note
        """, (session_id, tid, aid, ts, note or None))
        log(f"  Written: temp_id {tid:>4}  →  real_id {aid}")
    conn.commit()


def remove_assignments(conn: sqlite3.Connection,
                       session_id: str,
                       temp_ids: list[int]) -> None:
    for tid in temp_ids:
        n = conn.execute("""
            DELETE FROM manual_assignments
            WHERE session_id = ? AND temp_id = ?
        """, (session_id, tid)).rowcount
        if n:
            log(f"  Removed assignment for temp_id {tid}")
        else:
            log(f"  temp_id {tid}: no assignment found to remove")
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_pairs(pairs: dict,
                   session_temp_ids: set,
                   registry_ids: set,
                   kinetic_assignment: dict) -> None:
    """Warn (not error) on suspicious assignments."""
    for tid, aid in pairs.items():
        if tid not in session_temp_ids:
            log(f"  WARNING: temp_id {tid} not found in raw_tracks for this session "
                f"— double-check the temp_id from display_tracks.py")
        if registry_ids and aid not in registry_ids:
            log(f"  WARNING: real_id {aid} not in cow_registry "
                f"— animal may not be registered yet (will still be written)")
        if tid in kinetic_assignment:
            log(f"  WARNING: temp_id {tid} already has a kinetic assignment "
                f"(→ {kinetic_assignment[tid]}) — manual will override it")


# ─────────────────────────────────────────────────────────────────────────────
# Run reconcile
# ─────────────────────────────────────────────────────────────────────────────

def run_reconcile(args: argparse.Namespace, dm: DriveManager) -> None:
    section("Running reconcile.py")

    # locate reconcile.py alongside this script
    reconcile_path = Path(__file__).parent / "reconcile.py"
    if not reconcile_path.exists():
        raise FileNotFoundError(
            f"reconcile.py not found at {reconcile_path}\n"
            f"  Make sure reconcile.py is in the same directory as assign_identity.py"
        )

    spec  = importlib.util.spec_from_file_location("reconcile", reconcile_path)
    r_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r_mod)

    import argparse as _ap
    r_args = _ap.Namespace(
        session            = args.session,
        corr_threshold     = args.corr_threshold,
        min_active_bins    = 3,
        min_temp_id_frames = 0.10,
        activity_pct       = 0.25,
        bin_minutes        = 15,
        ema_alpha          = args.ema_alpha,
        min_embeds_gallery = 10,
        cosine_threshold   = args.cosine_threshold,
        cosine_min_embeds  = 5,
        dry_run            = args.dry_run,
        verbose            = False,
        bypass_upload_check = args.bypass_upload_check,
    )

    # reconcile.py resolves kinetics/behavior/gallery/parquet itself via
    # drive_manager, keyed off r_args.session — nothing to pass in here.
    r_mod.run(r_args)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    section("assign_identity.py")
    log(f"session : {args.session}")
    log("db: always the canonical copy pulled from Drive via drive_manager.py")

    try:
        conn, dm = open_db(bypass=args.bypass_upload_check)
    except (FileNotFoundError, DriveNotSyncedError, DriveUnavailableError) as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    try:
        validate_session(conn, args.session)
    except ValueError as e:
        log(f"ERROR: {e}")
        conn.close()
        sys.exit(1)

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        list_assignments(conn, args.session)
        conn.close()
        return

    # ── --remove ──────────────────────────────────────────────────────────────
    if args.remove:
        section("Removing assignments")
        remove_assignments(conn, args.session, args.remove)
        conn.close()
        return

    # ── --assign ──────────────────────────────────────────────────────────────
    if not args.assign:
        log("Nothing to do — provide --assign, --remove, or --list.")
        conn.close()
        sys.exit(0)

    # parse
    try:
        pairs = parse_assign_pairs(args.assign)
    except ValueError as e:
        log(f"ERROR: {e}")
        conn.close()
        sys.exit(1)

    log(f"Assignments to write: {len(pairs)}")

    # validate (warns only, never blocks)
    session_temp_ids  = get_session_temp_ids(conn, args.session)
    registry_ids      = get_cow_registry_ids(conn)
    kinetic_assignment = get_kinetic_assignments(conn, args.session)
    validate_pairs(pairs, session_temp_ids, registry_ids, kinetic_assignment)

    # write
    section("Writing assignments")
    write_assignments(conn, args.session, pairs, args.note)
    log(f"Done — {len(pairs)} assignment(s) written for session '{args.session}'")

    conn.close()

    # Sync DB back to Drive after writing manual assignments
    dm.sync_db(session_id=args.session)

    # ── --run_reconcile ───────────────────────────────────────────────────────
    if args.run_reconcile:
        try:
            run_reconcile(args, dm)
        except Exception as e:
            log(f"ERROR in reconcile: {e}")
            log("Assignments are saved — run reconcile.py manually to retry.")
            sys.exit(1)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
#
# # 1. Watch video to identify temp_ids:
# python3 display_tracks.py \
#   --video      ~/thesis_workspace/raw_data/videos/refet_33_S20241221070000.mp4 \
#   --session_id refet33_20241221 \
#   --show_fps --sink ffplay
#
# # 2. Assign identities and run reconcile:
# python3 assign_identity.py \
#   --session    refet33_20241221 \
#   --assign     2:7507  1:6366  71:7513 \
#   --note       "manual — no kinetics coverage for Dec 21" \
#   --run_reconcile
#
# # List current assignments:
# python3 assign_identity.py --session refet33_20241221 --list
#
# # Remove a wrong assignment and redo:
# python3 assign_identity.py --session refet33_20241221 --remove 71
# python3 assign_identity.py --session refet33_20241221 \
#   --assign 71:7513 --run_reconcile
#
# Note: db, kinetics, and gallery are never passed as arguments — they're
# always resolved through drive_manager.py, keyed off --session.