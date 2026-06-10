"""
gazit_adapter.py
----------------
Calving ledger adapter for Gazit farm.

Source format
-------------
  File type : Excel (.xlsx)
  Language  : Hebrew
  Sheet     : גיליון1
  Header row: 1 (row 0 is a title — skipped via skiprows=1)

Column mapping (Hebrew → canonical)
------------------------------------
  זיהוי בן-בקר   → real_id            cow ID (mother)
  תאריך המלטה    → _date_serial        Excel date serial number
  שעת המלטה      → _time_frac          Fraction of day (0.0–1.0); combined
                                        with _date_serial → calving_dt
  מס המל.        → parity
  אורך הריון     → gestation_days
  חלב ביובש      → milk_at_dryoff
  ימי יובש       → dry_days
  ימי חליבה      → days_in_milk
  צ"ג יובש       → dry_off_scc
  אופן המלטה     → _calving_type_raw   outcome source (see rules below)
  הערה           → notes_raw
  מס ולדות.      → n_calves

Outcome mapping rules (Gazit-specific, in priority order)
----------------------------------------------------------
  1. n_calves >= 2              → Twin          (overrides everything)
  2. ווטרינר / vet in notes     → Veterinarian  (not yet seen in this farm's
                                                  data but kept for robustness)
  3. קשה   (hard)               → Assisted
  4. קלה   (easy / light)       → Unassisted    (Gazit uses "easy" for
                                                  minor assists; treated as
                                                  unassisted per project spec)
  5. ללא התערבות (no intervention)→ Unassisted
  6. anything else              → logged as warning; row set to Unassisted

Notes
-----
  - The first row of the sheet is a Hebrew title ("רשימת ארועי המלטות")
    and is skipped via skiprows=1.
  - calving_dt is constructed from the Excel date serial (integer part)
    plus the time fraction (float). If time is missing/null, midnight is used.
  - real_id is stored as a string to match reid_registry convention.
  - Calf identity columns (זיהוי ולד 1/2, גורל 1/2, מין 1/2) and sire
    columns are read but NOT included in the canonical output — they are
    not used as model features and are intentionally dropped here.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import pandas as pd

from adapters.base_adapter import (
    BaseAdapter,
    OUTCOME_UNASSISTED,
    OUTCOME_ASSISTED,
    OUTCOME_TWIN,
    OUTCOME_VETERINARIAN,
)


# ------------------------------------------------------------------
# Hebrew column names as they appear in the source file
# ------------------------------------------------------------------
_COL_ROW_NUM      = "#"                    # unnamed index column (dropped)
_COL_COW_ID       = "זיהוי בן-בקר"
_COL_DATE         = "תאריך המלטה"
_COL_TIME         = "שעת המלטה"
_COL_PARITY       = "מס המל."
_COL_GESTATION    = "אורך הריון"
_COL_MILK_DRYOFF  = "חלב ביבוש"           # note: slightly different spelling
_COL_DRY_DAYS     = "ימי יובש"
_COL_DIM          = "ימי חליבה"
_COL_SCC          = "צ\"ג יובש"
_COL_TYPE         = "אופן המלטה"
_COL_NOTES        = "הערה"
_COL_N_CALVES     = "מס ולדות."

# Outcome raw string values
_RAW_UNASSISTED   = "ללא התערבות"
_RAW_EASY         = "קלה"
_RAW_HARD         = "קשה"

# Keywords that suggest veterinarian involvement (lowercased match)
_VET_KEYWORDS     = {"ווטרינר", "vet", "veterinarian", "וטרינר"}


# ------------------------------------------------------------------
# Excel date helpers
# ------------------------------------------------------------------

_EXCEL_EPOCH = datetime(1899, 12, 30)   # Excel's day-0 baseline

def _excel_serial_to_date(serial) -> datetime | None:
    """Convert an Excel integer date serial to a Python date."""
    try:
        serial = int(float(serial))
        return _EXCEL_EPOCH + timedelta(days=serial)
    except (TypeError, ValueError):
        return None

def _combine_date_time(date_serial, time_frac) -> datetime | None:
    """
    Combine an Excel date serial and a time fraction (0.0–1.0)
    into a single datetime. If time_frac is null/invalid, midnight is used.
    """
    base = _excel_serial_to_date(date_serial)
    if base is None:
        return None
    try:
        frac = float(time_frac)
        total_seconds = round(frac * 86400)
        return base + timedelta(seconds=total_seconds)
    except (TypeError, ValueError):
        return base   # fall back to midnight

def _combine_date_time_flexible(date_val, time_val) -> datetime | None:
    """
    Flexible date+time combiner that handles two cases:
      - openpyxl-parsed: date_val is datetime.datetime, time_val is datetime.time
      - Excel serial:    date_val is int/float, time_val is float fraction
    """
    import datetime as dt_module

    # Case 1: openpyxl already parsed the date as a datetime/date object
    if isinstance(date_val, (dt_module.datetime, dt_module.date)):
        base = dt_module.datetime(date_val.year, date_val.month, date_val.day)
        if isinstance(time_val, dt_module.time):
            return base.replace(
                hour=time_val.hour,
                minute=time_val.minute,
                second=time_val.second,
            )
        else:
            # try float fraction fallback
            try:
                frac = float(time_val)
                total_seconds = round(frac * 86400)
                return base + timedelta(seconds=total_seconds)
            except (TypeError, ValueError):
                return base

    # Case 2: Excel serial number
    return _combine_date_time(date_val, time_val)


# ------------------------------------------------------------------
# Outcome parsing
# ------------------------------------------------------------------

def _parse_outcome(calving_type_raw: str, n_calves: int, notes: str) -> str:
    """
    Apply Gazit-specific outcome rules in priority order.

    Priority:
      1. Twin (n_calves >= 2)
      2. Veterinarian (vet keyword in calving_type or notes)
      3. Hard (assisted)
      4. Easy / No-intervention (unassisted)
      5. Unknown → warn + unassisted
    """
    # Rule 1 — Twin overrides everything
    if n_calves >= 2:
        return OUTCOME_TWIN

    raw  = str(calving_type_raw).strip() if pd.notna(calving_type_raw) else ""
    note = str(notes).strip().lower()    if pd.notna(notes)            else ""

    # Rule 2 — Veterinarian keyword in type field or notes
    raw_lower = raw.lower()
    if any(kw in raw_lower for kw in _VET_KEYWORDS) or \
       any(kw in note       for kw in _VET_KEYWORDS):
        return OUTCOME_VETERINARIAN

    # Rule 3 — Hard → Assisted
    if raw == _RAW_HARD:
        return OUTCOME_ASSISTED

    # Rule 4 — Easy or No-intervention → Unassisted
    if raw in (_RAW_EASY, _RAW_UNASSISTED):
        return OUTCOME_UNASSISTED

    # Rule 5 — Unknown
    if raw:
        warnings.warn(
            f"[GazitAdapter] Unknown calving type value: '{raw}'. "
            f"Defaulting to Unassisted.",
            UserWarning,
            stacklevel=2,
        )
    return OUTCOME_UNASSISTED


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------

class GazitAdapter(BaseAdapter):

    FARM_NAME = "gazit"

    def read_options(self) -> dict:
        # Row 0: Hebrew title "רשימת ארועי המלטות"
        # Row 1: empty
        # Row 2: actual column headers
        return {"skiprows": 2}

    def parse(self, df_raw: pd.DataFrame, source_file: str) -> pd.DataFrame:
        """
        Parse the Gazit xlsx and return a canonical DataFrame.
        """
        df = df_raw.copy()

        # ---- 1. Rename columns we care about --------------------------------
        rename_map = {
            _COL_COW_ID    : "real_id",
            _COL_DATE      : "_date_serial",
            _COL_TIME      : "_time_frac",
            _COL_PARITY    : "parity",
            _COL_GESTATION : "gestation_days",
            _COL_MILK_DRYOFF: "milk_at_dryoff",
            _COL_DRY_DAYS  : "dry_days",
            _COL_DIM       : "days_in_milk",
            _COL_SCC       : "dry_off_scc",
            _COL_TYPE      : "_calving_type_raw",
            _COL_NOTES     : "notes_raw",
            _COL_N_CALVES  : "n_calves",
        }
        # Only rename columns that exist (guards against minor spelling drift)
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        missing = [k for k in rename_map if k not in df.columns]
        if missing:
            warnings.warn(
                f"[GazitAdapter] Expected columns not found in file, will be "
                f"set to null: {missing}",
                UserWarning,
                stacklevel=2,
            )
        df = df.rename(columns=actual_rename)

        # ---- 2. Build calving_dt from date + time --------------------------
        # openpyxl returns datetime.datetime for date cells and datetime.time
        # for time cells. Fall back to serial-number parsing if needed.
        df["calving_dt"] = df.apply(
            lambda r: _combine_date_time_flexible(
                r.get("_date_serial"), r.get("_time_frac")
            ),
            axis=1,
        )
        df["calving_dt"] = pd.to_datetime(df["calving_dt"])

        # ---- 3. real_id as string -------------------------------------------
        df["real_id"] = df["real_id"].astype(str).str.strip()

        # ---- 4. n_calves — fill missing with 1 (single calf assumed) --------
        df["n_calves"] = pd.to_numeric(df.get("n_calves"), errors="coerce") \
                           .fillna(1).astype(int)

        # ---- 5. parity -------------------------------------------------------
        df["parity"] = pd.to_numeric(df.get("parity"), errors="coerce") \
                         .astype("Int64")   # nullable int

        # ---- 6. Numeric feature columns -------------------------------------
        for col in ("gestation_days", "dry_days", "days_in_milk"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") \
                            .astype("Int64")

        for col in ("milk_at_dryoff", "dry_off_scc"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ---- 7. Outcome ------------------------------------------------------
        df["outcome"] = df.apply(
            lambda r: _parse_outcome(
                r.get("_calving_type_raw"),
                r.get("n_calves", 1),
                r.get("notes_raw"),
            ),
            axis=1,
        )

        # ---- 8. notes_raw — keep as clean string, empty → None --------------
        if "notes_raw" in df.columns:
            df["notes_raw"] = df["notes_raw"].astype(str).str.strip()
            df["notes_raw"] = df["notes_raw"].replace(
                {"nan": None, "": None, "None": None}
            )
        else:
            df["notes_raw"] = None

        # ---- 9. Drop internal / non-canonical columns -----------------------
        drop_cols = [c for c in df.columns if c.startswith("_")]
        # Also drop calf and sire columns — not used as model features
        drop_prefixes = ("זיהוי ולד", "גורל", "מין", "מספר אב", "שם אב",
                         "תאריך יציאה", "Unnamed")
        drop_cols += [c for c in df.columns if c == "#"]
        drop_cols += [
            c for c in df.columns
            if any(c.startswith(p) for p in drop_prefixes)
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                     errors="ignore")

        # ---- 10. Drop rows with no calving_dt (unparseable date) ------------
        bad_dt = df["calving_dt"].isna()
        if bad_dt.any():
            warnings.warn(
                f"[GazitAdapter] {bad_dt.sum()} row(s) dropped: "
                f"could not parse calving_dt.",
                UserWarning,
                stacklevel=2,
            )
            df = df[~bad_dt]

        # ---- 11. Drop rows with no real_id ----------------------------------
        bad_id = df["real_id"].isna() | (df["real_id"] == "")
        if bad_id.any():
            warnings.warn(
                f"[GazitAdapter] {bad_id.sum()} row(s) dropped: "
                f"missing real_id.",
                UserWarning,
                stacklevel=2,
            )
            df = df[~bad_id]

        return df.reset_index(drop=True)