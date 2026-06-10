"""
calving_ledger_ingest.py
------------------------
Orchestrator for ingesting calving ledger files into the project database.

What it does
------------
1.  Selects the right farm adapter (via --farm or filename heuristic)
2.  Reads and parses the source file via adapter.load_and_parse()
3.  Derives temporal features (calving_hour, calving_month, calving_season)
4.  Derives is_breech from notes_raw
5.  Computes cross-event features per cow (days_since_last_calving,
    prior_outcome, prior_n_calves) — requires sorting by (real_id, calving_dt)
6.  Ensures calving_ledger and calving_features tables exist with correct schema
7.  Auto-migrates: any extra column returned by adapter triggers
    ALTER TABLE ADD COLUMN (fully automatic, no manual migrations)
8.  Writes rows — skips already-ingested events by default (idempotent);
    --overwrite replaces them
9.  Uploads DB to Drive via drive_manager
10. Prints ingestion report

Idempotency key
---------------
(real_id, calving_dt) — a cow cannot calve twice at the exact same datetime.

Usage
-----
  python3 calving_ledger_ingest.py \\
      --file  gazit_calving_events_202505_202605.xlsx \\
      --farm  gazit \\
      --db    ~/thesis_workspace/vcf-ctp/data/calving_project.db

  # Parse + validate only, write nothing:
  python3 calving_ledger_ingest.py --file ... --farm gazit --db ... --dry_run

  # Re-ingest (replace existing rows):
  python3 calving_ledger_ingest.py --file ... --farm gazit --db ... --overwrite

Adding a new farm
-----------------
1.  Create adapters/my_farm_adapter.py, subclass BaseAdapter
2.  Add an entry to ADAPTERS dict below
3.  Optionally add filename patterns to FARM_FILENAME_HINTS
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import warnings
from datetime import datetime
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Adapter registry — add new farms here
# ---------------------------------------------------------------------------
from adapters.gazit_adapter import GazitAdapter

ADAPTERS = {
    "gazit": GazitAdapter,
    # "farm_b": FarmBAdapter,
}

# Filename substrings → farm name (case-insensitive)
FARM_FILENAME_HINTS = {
    "gazit": "gazit",
    # "farm_b": "farm_b",
}


# ---------------------------------------------------------------------------
# Season helper
# ---------------------------------------------------------------------------

def _month_to_season(month: int) -> str:
    """Northern Hemisphere seasons (Israel)."""
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "fall"


# ---------------------------------------------------------------------------
# Breech detection (notes_raw)
# ---------------------------------------------------------------------------

_BREECH_KEYWORDS = {
    "הפוכה",      # Hebrew: "reversed / inverted" (breech presentation)
    "breech",
    "backwards",
}

def _detect_breech(notes_raw) -> bool:
    if pd.isna(notes_raw) or not str(notes_raw).strip():
        return False
    note_lower = str(notes_raw).lower()
    return any(kw in note_lower for kw in _BREECH_KEYWORDS)


# ---------------------------------------------------------------------------
# Temporal feature derivation
# ---------------------------------------------------------------------------

def _derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["calving_hour"]   = df["calving_dt"].dt.hour + \
                           df["calving_dt"].dt.minute / 60.0
    df["calving_month"]  = df["calving_dt"].dt.month
    df["calving_season"] = df["calving_month"].map(_month_to_season)
    df["is_breech"]      = df["notes_raw"].apply(_detect_breech)
    return df


# ---------------------------------------------------------------------------
# Cross-event feature computation
# ---------------------------------------------------------------------------

def _compute_cross_event_features(
    df: pd.DataFrame,
    existing_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each row in df, look back at prior calvings for the same cow —
    both from df itself (earlier rows in this batch) and from
    existing_events (already in the DB).

    Computes:
        days_since_last_calving   int   nullable
        prior_outcome             str   nullable
        prior_n_calves            int   nullable

    Parameters
    ----------
    df              : canonical DataFrame sorted by (real_id, calving_dt),
                      temporal features already derived
    existing_events : DataFrame of already-ingested events with columns
                      [real_id, calving_dt, outcome, n_calves].
                      May be empty.
    """
    df = df.copy()

    # Combine existing + new into one lookup table
    lookup_cols = ["real_id", "calving_dt", "outcome", "n_calves"]
    if not existing_events.empty:
        hist = pd.concat(
            [
                existing_events[lookup_cols],
                df[lookup_cols],
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["real_id", "calving_dt"])
    else:
        hist = df[lookup_cols].copy()

    hist = hist.sort_values(["real_id", "calving_dt"]).reset_index(drop=True)

    days_since   = []
    prior_out    = []
    prior_calves = []

    for _, row in df.iterrows():
        cow_hist = hist[
            (hist["real_id"]    == row["real_id"]) &
            (hist["calving_dt"] <  row["calving_dt"])
        ].sort_values("calving_dt")

        if cow_hist.empty:
            days_since.append(pd.NA)
            prior_out.append(None)
            prior_calves.append(pd.NA)
        else:
            last = cow_hist.iloc[-1]
            delta = (row["calving_dt"] - last["calving_dt"]).days
            days_since.append(delta)
            prior_out.append(last["outcome"])
            prior_calves.append(int(last["n_calves"]))

    df["days_since_last_calving"] = pd.array(days_since,   dtype="Int64")
    df["prior_outcome"]           = prior_out
    df["prior_n_calves"]          = pd.array(prior_calves, dtype="Int64")

    return df


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

# Core schema — all feature columns are nullable
_CALVING_LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS calving_ledger (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    real_id      TEXT    NOT NULL,
    calving_dt   TEXT    NOT NULL,
    outcome      TEXT    NOT NULL,
    source_farm  TEXT,
    source_file  TEXT,
    ingested_at  TEXT,
    UNIQUE(real_id, calving_dt)
)
"""

_CALVING_FEATURES_DDL = """
CREATE TABLE IF NOT EXISTS calving_features (
    event_id                  INTEGER PRIMARY KEY
                              REFERENCES calving_ledger(event_id),
    parity                    INTEGER,
    gestation_days            INTEGER,
    days_in_milk              INTEGER,
    dry_days                  INTEGER,
    milk_at_dryoff            REAL,
    dry_off_scc               REAL,
    n_calves                  INTEGER,
    calving_hour              REAL,
    calving_month             INTEGER,
    calving_season            TEXT,
    is_breech                 INTEGER,   -- stored as 0/1
    notes_raw                 TEXT,
    days_since_last_calving   INTEGER,
    prior_outcome             TEXT,
    prior_n_calves            INTEGER
)
"""

# Columns that belong to calving_ledger (not calving_features)
_LEDGER_COLS = {
    "real_id", "calving_dt", "outcome", "source_farm", "source_file"
}

# Columns that are internal / not stored in either table
_DROP_COLS = set()


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(_CALVING_LEDGER_DDL)
    conn.execute(_CALVING_FEATURES_DDL)
    conn.commit()


def _get_existing_feature_columns(conn: sqlite3.Connection) -> set[str]:
    """Return current column names of calving_features."""
    cur = conn.execute("PRAGMA table_info(calving_features)")
    return {row[1] for row in cur.fetchall()}


def _auto_migrate(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """
    For any feature column in df that doesn't yet exist in calving_features,
    issue ALTER TABLE ADD COLUMN. SQLite type is inferred from pandas dtype.
    """
    existing = _get_existing_feature_columns(conn)
    for col in feature_cols:
        if col == "event_id" or col in existing:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "REAL"
        else:
            sql_type = "TEXT"
        conn.execute(
            f"ALTER TABLE calving_features ADD COLUMN {col} {sql_type}"
        )
        print(f"  [migrate] Added column calving_features.{col} ({sql_type})")
    conn.commit()


def _load_existing_events(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load real_id, calving_dt, outcome, n_calves from calving_ledger + features."""
    try:
        df = pd.read_sql(
            """
            SELECT l.real_id, l.calving_dt, l.outcome, f.n_calves
            FROM calving_ledger l
            LEFT JOIN calving_features f ON l.event_id = f.event_id
            """,
            conn,
            parse_dates=["calving_dt"],
        )
        return df
    except Exception:
        return pd.DataFrame(
            columns=["real_id", "calving_dt", "outcome", "n_calves"]
        )


def _insert_rows(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    """
    Insert rows from df into calving_ledger and calving_features.
    Returns stats dict: {inserted, skipped, overwritten, errors}.
    """
    stats = {"inserted": 0, "skipped": 0, "overwritten": 0, "errors": 0}
    now_str = datetime.utcnow().isoformat()

    feature_cols = [
        c for c in df.columns
        if c not in _LEDGER_COLS and c != "event_id"
    ]

    if not dry_run:
        _auto_migrate(conn, df, feature_cols)

    for _, row in df.iterrows():
        real_id    = str(row["real_id"])
        calving_dt = row["calving_dt"].isoformat() \
                     if hasattr(row["calving_dt"], "isoformat") \
                     else str(row["calving_dt"])

        # Check if already exists
        cur = conn.execute(
            "SELECT event_id FROM calving_ledger "
            "WHERE real_id = ? AND calving_dt = ?",
            (real_id, calving_dt),
        )
        existing_row = cur.fetchone()

        if existing_row and not overwrite:
            stats["skipped"] += 1
            continue

        if dry_run:
            stats["inserted"] += 1
            continue

        try:
            if existing_row and overwrite:
                event_id = existing_row[0]
                # Update ledger row
                conn.execute(
                    """UPDATE calving_ledger
                       SET outcome=?, source_farm=?, source_file=?,
                           ingested_at=?
                       WHERE event_id=?""",
                    (
                        row["outcome"],
                        row.get("source_farm"),
                        row.get("source_file"),
                        now_str,
                        event_id,
                    ),
                )
                # Delete old features row — will be replaced below
                conn.execute(
                    "DELETE FROM calving_features WHERE event_id=?",
                    (event_id,),
                )
                stats["overwritten"] += 1
            else:
                # Insert new ledger row
                conn.execute(
                    """INSERT INTO calving_ledger
                       (real_id, calving_dt, outcome,
                        source_farm, source_file, ingested_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        real_id,
                        calving_dt,
                        row["outcome"],
                        row.get("source_farm"),
                        row.get("source_file"),
                        now_str,
                    ),
                )
                event_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                stats["inserted"] += 1

            # Build features row
            feat_values = {"event_id": event_id}
            for col in feature_cols:
                val = row.get(col)
                # Convert pandas NA / numpy types to Python native
                if pd.isna(val) if not isinstance(val, str) else False:
                    feat_values[col] = None
                elif isinstance(val, bool):
                    feat_values[col] = int(val)
                elif hasattr(val, "item"):   # numpy scalar
                    feat_values[col] = val.item()
                else:
                    feat_values[col] = val

            cols_str  = ", ".join(feat_values.keys())
            placeholders = ", ".join(["?"] * len(feat_values))
            conn.execute(
                f"INSERT INTO calving_features ({cols_str}) "
                f"VALUES ({placeholders})",
                list(feat_values.values()),
            )

        except Exception as exc:
            warnings.warn(
                f"Error inserting real_id={real_id} calving_dt={calving_dt}: "
                f"{exc}",
                UserWarning,
                stacklevel=2,
            )
            stats["errors"] += 1
            conn.rollback()
            continue

    if not dry_run:
        conn.commit()

    return stats


# ---------------------------------------------------------------------------
# Farm detection heuristic
# ---------------------------------------------------------------------------

def _detect_farm(filepath: str) -> Optional[str]:
    """Try to guess farm name from filename."""
    name = os.path.basename(filepath).lower()
    for farm, hint in FARM_FILENAME_HINTS.items():
        if hint in name:
            return farm
    return None


# ---------------------------------------------------------------------------
# Ingestion report
# ---------------------------------------------------------------------------

def _print_report(
    df: pd.DataFrame,
    stats: dict,
    farm: str,
    filepath: str,
    dry_run: bool,
) -> None:
    total = len(df)
    print()
    print("=" * 60)
    print(f"  Calving Ledger Ingestion Report")
    print("=" * 60)
    print(f"  Farm        : {farm}")
    print(f"  File        : {os.path.basename(filepath)}")
    print(f"  Rows parsed : {total}")
    if dry_run:
        print(f"  Mode        : DRY RUN — nothing written")
    else:
        print(f"  Inserted    : {stats['inserted']}")
        print(f"  Overwritten : {stats['overwritten']}")
        print(f"  Skipped     : {stats['skipped']}  (already in DB)")
        print(f"  Errors      : {stats['errors']}")
    print()
    print("  Outcome distribution:")
    outcome_counts = df["outcome"].value_counts()
    for outcome, count in outcome_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"    {outcome:<20} {count:>4}  ({pct:.1f}%)")
    print()
    print("  Missing value summary (feature columns):")
    feature_cols = [
        "parity", "gestation_days", "days_in_milk", "dry_days",
        "milk_at_dryoff", "dry_off_scc",
    ]
    for col in feature_cols:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                print(f"    {col:<28} {n_null:>4} null ({100*n_null/total:.1f}%)")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest a calving ledger file into the project database."
    )
    parser.add_argument(
        "--file", required=True,
        help="Path to the calving ledger file (.xlsx or .csv)",
    )
    parser.add_argument(
        "--farm",
        choices=list(ADAPTERS.keys()),
        help=(
            "Farm identifier. If omitted, guessed from filename. "
            f"Known farms: {list(ADAPTERS.keys())}"
        ),
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Parse and validate only — write nothing to the database.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Re-ingest rows that already exist (keyed on real_id + calving_dt). "
            "Default: skip existing rows."
        ),
    )
    parser.add_argument(
        "--bypass_upload_check", action="store_true",
        help="Skip dirty-flag check and proceed with whatever DB is on Drive.",
    )
    args = parser.parse_args()

    # ---- Resolve farm -------------------------------------------------------
    farm = args.farm or _detect_farm(args.file)
    if farm is None:
        print(
            f"ERROR: Could not detect farm from filename '{args.file}'. "
            f"Please pass --farm. Known farms: {list(ADAPTERS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    if farm not in ADAPTERS:
        print(
            f"ERROR: Unknown farm '{farm}'. "
            f"Known farms: {list(ADAPTERS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Pull DB from Drive -------------------------------------------------
    sys.path.insert(0, os.path.dirname(__file__))
    dm = None
    try:
        from drive_manager import DriveManager
        dm = DriveManager(caller=__file__, bypass=args.bypass_upload_check)
        if not args.dry_run:
            print("[ingest] Pulling DB from Drive...")
            dm.pull_db()
        db_path = dm.get_db_path()
    except ImportError:
        print(
            "WARNING: drive_manager not found. "
            "Falling back to local DB at default path.",
            file=sys.stderr,
        )
        db_path = os.path.join(
            os.path.expanduser("~"),
            "thesis_workspace", "vcf-ctp", "data", "calving_project.db",
        )

    print(f"[ingest] Farm      : {farm}")
    print(f"[ingest] File      : {args.file}")
    print(f"[ingest] DB        : {db_path}")
    if args.dry_run:
        print("[ingest] Mode      : DRY RUN")
    if args.overwrite:
        print("[ingest] Overwrite : enabled")

    # ---- Parse --------------------------------------------------------------
    adapter = ADAPTERS[farm]()
    print(f"[ingest] Parsing...")
    df = adapter.load_and_parse(args.file)
    print(f"[ingest] Parsed {len(df)} rows.")

    # ---- Derive temporal features -------------------------------------------
    df = _derive_temporal_features(df)

    # ---- Open DB ------------------------------------------------------------
    # In dry_run mode skip DB entirely — just parse, derive, and report.
    if args.dry_run:
        df = df.sort_values(["real_id", "calving_dt"]).reset_index(drop=True)
        df = _compute_cross_event_features(df, pd.DataFrame(
            columns=["real_id", "calving_dt", "outcome", "n_calves"]
        ))
        _print_report(df, {}, farm, args.file, dry_run=True)
        return

    conn = sqlite3.connect(db_path)
    _ensure_tables(conn)

    # ---- Load existing events for cross-event feature computation -----------
    existing_events = _load_existing_events(conn)
    print(f"[ingest] Existing events in DB: {len(existing_events)}")

    # ---- Sort before cross-event computation --------------------------------
    df = df.sort_values(["real_id", "calving_dt"]).reset_index(drop=True)

    # ---- Compute cross-event features ---------------------------------------
    print("[ingest] Computing cross-event features...")
    df = _compute_cross_event_features(df, existing_events)


    # ---- Insert rows --------------------------------------------------------
    print(f"[ingest] Writing to DB (dry_run={args.dry_run}, "
          f"overwrite={args.overwrite})...")
    stats = _insert_rows(conn, df, overwrite=args.overwrite, dry_run=args.dry_run)

    conn.close()

    # ---- Sync DB back to Drive ----------------------------------------------
    if not args.dry_run and dm is not None:
        try:
            dm.sync_db(session_id="ledger_ingest")
            print("[ingest] DB synced to Drive.")
        except Exception as exc:
            print(f"[ingest] Drive sync failed: {exc}")

    # ---- Report -------------------------------------------------------------
    _print_report(df, stats, farm, args.file, args.dry_run)


if __name__ == "__main__":
    main()