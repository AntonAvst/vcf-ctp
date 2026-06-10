"""
base_adapter.py
---------------
Abstract base class for calving ledger adapters.

Each farm gets one adapter subclass that knows how to parse its specific
file format and produce a canonical DataFrame. The orchestrator
(calving_ledger_ingest.py) only ever calls parse() and never looks inside
the adapter.

Canonical output columns
------------------------
Required (must never be null):
    real_id         str     cow identifier — FK to cow_registry / reid_registry
    calving_dt      datetime
    outcome         str     one of: Unassisted | Assisted | Twin | Veterinarian
    parity          int     which calving number for this cow
    n_calves        int     number of calves born

Nullable (missing data is acceptable):
    gestation_days  int
    days_in_milk    int
    dry_days        int
    milk_at_dryoff  float
    dry_off_scc     float
    notes_raw       str     cleaned free-text notes

Injected by orchestrator (not the adapter's responsibility):
    calving_hour    float   derived from calving_dt
    calving_month   int     derived from calving_dt
    calving_season  str     derived from calving_dt
    is_breech       bool    derived from notes_raw
    source_farm     str     adapter must set self.FARM_NAME
    source_file     str     passed in by orchestrator

Adding a new farm
-----------------
1. Create adapters/my_farm_adapter.py
2. Subclass BaseAdapter
3. Set FARM_NAME = "my_farm"
4. Implement parse(df_raw, source_file) -> pd.DataFrame
5. Register in calving_ledger_ingest.py ADAPTERS dict
"""

from __future__ import annotations

import abc
from typing import final

import pandas as pd

# Canonical outcome values — shared across all adapters
OUTCOME_UNASSISTED   = "Unassisted"
OUTCOME_ASSISTED     = "Assisted"
OUTCOME_TWIN         = "Twin"
OUTCOME_VETERINARIAN = "Veterinarian"

VALID_OUTCOMES = {
    OUTCOME_UNASSISTED,
    OUTCOME_ASSISTED,
    OUTCOME_TWIN,
    OUTCOME_VETERINARIAN,
}

# Required columns the orchestrator expects in the canonical DataFrame.
# Adapters may return additional columns — the orchestrator will
# ALTER TABLE to accommodate them automatically.
REQUIRED_COLS = [
    "real_id",
    "calving_dt",
    "outcome",
    "parity",
    "n_calves",
]

NULLABLE_CORE_COLS = [
    "gestation_days",
    "days_in_milk",
    "dry_days",
    "milk_at_dryoff",
    "dry_off_scc",
    "notes_raw",
]

ALL_CANONICAL_COLS = REQUIRED_COLS + NULLABLE_CORE_COLS


class BaseAdapter(abc.ABC):
    """
    Abstract base for all farm-specific calving ledger adapters.

    Subclasses must:
      - Set FARM_NAME class attribute
      - Implement parse()
    """

    FARM_NAME: str = ""   # e.g. "gazit" — set in every subclass

    # ------------------------------------------------------------------
    # Public API — called by orchestrator
    # ------------------------------------------------------------------

    @final
    def load_and_parse(self, filepath: str) -> pd.DataFrame:
        """
        Read the raw file, call parse(), validate, inject source columns.
        This is the only method the orchestrator calls.
        """
        df_raw = self._read_file(filepath)
        df     = self.parse(df_raw, source_file=filepath)
        df     = self._inject_source(df, filepath)
        self._validate(df)
        return df

    # ------------------------------------------------------------------
    # Must override
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def parse(self, df_raw: pd.DataFrame, source_file: str) -> pd.DataFrame:
        """
        Parse a raw farm-specific DataFrame and return a canonical DataFrame.

        Parameters
        ----------
        df_raw      Raw DataFrame as read from the source file.
        source_file Path to the original file (for logging/error messages).

        Returns
        -------
        pd.DataFrame with at minimum all REQUIRED_COLS present and non-null,
        and any subset of NULLABLE_CORE_COLS. Additional farm-specific columns
        are allowed and will be stored automatically.
        """
        ...

    # ------------------------------------------------------------------
    # May override (sensible defaults provided)
    # ------------------------------------------------------------------

    def read_options(self) -> dict:
        """
        Extra kwargs forwarded to pd.read_excel / pd.read_csv.
        Override to handle encodings, sheet names, header rows, etc.
        """
        return {}

    # ------------------------------------------------------------------
    # Internal helpers available to subclasses
    # ------------------------------------------------------------------

    def _read_file(self, filepath: str) -> pd.DataFrame:
        """Read xlsx or csv based on file extension."""
        fp = filepath.lower()
        opts = self.read_options()
        if fp.endswith((".xlsx", ".xlsm", ".xls")):
            return pd.read_excel(filepath, **opts)
        elif fp.endswith(".csv"):
            return pd.read_csv(filepath, **opts)
        else:
            raise ValueError(
                f"[{self.FARM_NAME}] Unsupported file type: {filepath}. "
                "Expected .xlsx, .xlsm, .xls, or .csv"
            )

    def _inject_source(self, df: pd.DataFrame, filepath: str) -> pd.DataFrame:
        """Stamp source_farm and source_file onto every row."""
        import os
        df = df.copy()
        df["source_farm"] = self.FARM_NAME
        df["source_file"] = os.path.basename(filepath)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Validate the canonical DataFrame.
        Raises ValueError with a descriptive message on any violation.
        """
        if not self.FARM_NAME:
            raise ValueError("Adapter must set FARM_NAME")

        # Required columns present
        missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"[{self.FARM_NAME}] parse() is missing required columns: "
                f"{missing_cols}"
            )

        # Required columns non-null
        for col in REQUIRED_COLS:
            null_count = df[col].isna().sum()
            if null_count > 0:
                raise ValueError(
                    f"[{self.FARM_NAME}] Required column '{col}' has "
                    f"{null_count} null value(s). Fix in adapter or source data."
                )

        # calving_dt is actually datetime
        if not pd.api.types.is_datetime64_any_dtype(df["calving_dt"]):
            raise ValueError(
                f"[{self.FARM_NAME}] 'calving_dt' must be datetime64, "
                f"got {df['calving_dt'].dtype}"
            )

        # outcome values are valid
        bad_outcomes = df[~df["outcome"].isin(VALID_OUTCOMES)]["outcome"].unique()
        if len(bad_outcomes) > 0:
            raise ValueError(
                f"[{self.FARM_NAME}] Unknown outcome value(s): {bad_outcomes}. "
                f"Valid values: {VALID_OUTCOMES}"
            )

        # real_id is string-like
        if not pd.api.types.is_object_dtype(df["real_id"]) and \
           not pd.api.types.is_string_dtype(df["real_id"]):
            # coerce silently — ints are fine as IDs
            pass

        # parity and n_calves are positive integers
        for col in ("parity", "n_calves"):
            if (df[col] < 1).any():
                raise ValueError(
                    f"[{self.FARM_NAME}] Column '{col}' contains values < 1."
                )