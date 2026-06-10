#!/usr/bin/env python3
"""
drive_manager.py — Google Drive I/O abstraction layer for vcf-ctp.

Drive is the single source of truth for all data.
Local (data/) is a write buffer for the current session only.

All scripts call this module instead of touching paths directly.

Usage (library mode — imported by other scripts):
    from drive_manager import DriveManager
    dm = DriveManager()
    dm.pull_db()                                         # pull DB from Drive before writing
    dm.get_db_path()                                     # → local path to calving_project.db
    dm.write_file(local_path, drive_dest, caller=__file__)
    dm.get_video_path("/path/to/local/video.mp4")        # pass-through, validates existence
    dm.get_kinetics_path()                               # → local path to kinetics CSV
    dm.find_collar_files(start_dt, end_dt, 'kinetic')   # → list of local paths overlapping session
    dm.load_collar_data(start_dt, end_dt, 'kinetic')    # → merged DataFrame of all matching CSVs
    dm.get_gallery_path("day")                           # → local path to gallery_day.npy
    dm.get_parquet_path(session_id, "embeds")            # → local path to embeds.parquet
    dm.get_session_dir(session_id)                       # → local output dir for session
    dm.check_flag("db")                                  # raises DriveNotSyncedError if dirty
    dm.mark_dirty("db", session_id, caller=__file__)
    dm.mark_clean("db")

Usage (CLI mode — run directly from terminal):
    python3 drive_manager.py status
    python3 drive_manager.py retry-buffer
    python3 drive_manager.py pull-db
    python3 drive_manager.py list-sessions
    python3 drive_manager.py upload-kinetics /path/to/collar_data/
    python3 drive_manager.py upload-behavior /path/to/collar_data/
    python3 drive_manager.py upload-ledger   calving_ledger.csv
    python3 drive_manager.py clear-flag db|parquet|gallery

Bypass flag (for scripts that read Drive data):
    All get_*() calls accept bypass=True to skip the dirty-flag check.
    Also exposed as --bypass_upload_check on scripts that call them.

Requirements: rclone configured with remote named "thesis_google_drive"
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Graceful stop (Ctrl+C)
# ─────────────────────────────────────────────────────────────────────────────

_STOP = False   # set to True by SIGINT handler; checked at loop boundaries

def _handle_sigint(sig, frame):
    global _STOP
    if not _STOP:
        _STOP = True
        print("\n[drive_manager] Ctrl+C received — finishing current file then cleaning up...",
              flush=True)

signal.signal(signal.SIGINT, _handle_sigint)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these if your paths change
# ─────────────────────────────────────────────────────────────────────────────

# rclone remote name (from `rclone listremotes`)
RCLONE_REMOTE = "thesis_google_drive"

# Root folder on Google Drive
DRIVE_ROOT = f"{RCLONE_REMOTE}:vcf_ctp_data"

# Local working directory (write buffer)
LOCAL_ROOT = Path("~/thesis_workspace/vcf-ctp/data").expanduser()

# Buffer directory (local only, never uploaded)
BUFFER_DIR = Path("~/thesis_workspace/vcf-ctp/.buffer").expanduser()

# Max upload attempts per file (across all sessions)
MAX_ATTEMPTS = 5

# Retry backoff in seconds (attempt 1→2s, 2→8s, 3→32s)
BACKOFF_BASE = 2

# ─────────────────────────────────────────────────────────────────────────────
# Internal paths (derived from constants above — don't edit)
# ─────────────────────────────────────────────────────────────────────────────

PENDING_DIR         = BUFFER_DIR / "pending"
FAILURES_LOG        = BUFFER_DIR / "upload_failures.log"
BYPASS_LOG          = BUFFER_DIR / "bypass_warnings.log"
DRIVE_UPLOAD_LOG    = "upload_log.csv"    # relative path on Drive
LOCAL_UPLOAD_LOG    = BUFFER_DIR / "upload_log_staging.csv"  # local append-only staging

# Sync-status flag files
_FLAG_FILES = {
    "db":      BUFFER_DIR / "db_sync_status.json",
    "parquet": BUFFER_DIR / "parquet_sync_status.json",
    "gallery": BUFFER_DIR / "gallery_sync_status.json",
}

# Drive path constants
DRIVE_DB_PATH        = "calving_project.db"
DRIVE_GALLERY_PREFIX = "reid_gallery"
DRIVE_COLLAR_PREFIX  = "collar_data"
DRIVE_OUTPUTS_PREFIX = "outputs"
DRIVE_MODELS_PREFIX  = "models"

# Local path constants
LOCAL_DB_PATH        = LOCAL_ROOT / "calving_project.db"
LOCAL_GALLERY_DIR    = LOCAL_ROOT / "reid_gallery"
LOCAL_COLLAR_DIR     = LOCAL_ROOT / "collar_data"
LOCAL_OUTPUTS_DIR    = LOCAL_ROOT / "outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class DriveNotSyncedError(RuntimeError):
    """Raised when a script tries to read from Drive while local data is unsynced."""
    pass


class DriveUnavailableError(RuntimeError):
    """Raised when rclone cannot reach Google Drive."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[drive_manager] {msg}", flush=True)


def _log_failure(caller: str, dest: str, size_bytes: int, error: str) -> None:
    """Append one line to the local failure log."""
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    size_mb = size_bytes / 1e6 if size_bytes else 0
    line = (f"[{ts}] FAILED  size={size_mb:.2f}MB  caller={caller}  "
            f"dest={dest}  error={error}\n")
    with open(FAILURES_LOG, "a") as f:
        f.write(line)


def _log_bypass(caller: str, flag: str) -> None:
    """Append one line to the local bypass warnings log."""
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] BYPASS  flag={flag}  caller={caller}\n"
    with open(BYPASS_LOG, "a") as f:
        f.write(line)


def _log_upload_success(caller: str, filename: str, drive_dest: str,
                         size_bytes: int, attempt: int) -> None:
    """
    Append one row to the local staging file (upload_log_staging.csv).
    Never touches Drive directly — call _flush_upload_log() once at end of
    a batch (or after a single upload) to push the staged rows to Drive.
    """
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    header_needed = (not LOCAL_UPLOAD_LOG.exists()
                     or LOCAL_UPLOAD_LOG.stat().st_size == 0)
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = f"{ts},{caller},{filename},{drive_dest},{size_bytes},success,{attempt}\n"
    with open(LOCAL_UPLOAD_LOG, "a") as f:
        if header_needed:
            f.write("timestamp,caller,filename,drive_dest,size_bytes,status,attempt_number\n")
        f.write(row)


def _flush_upload_log() -> None:
    """
    Merge the local staging file into upload_log.csv on Drive.
    One Drive round-trip regardless of how many rows were staged.
    Called once at the end of a batch or after a single non-batch upload.
    Safe to call even if the staging file is empty — exits early.
    """
    if not LOCAL_UPLOAD_LOG.exists() or LOCAL_UPLOAD_LOG.stat().st_size == 0:
        return

    drive_log_path = f"{DRIVE_ROOT}/{DRIVE_UPLOAD_LOG}"
    local_full     = BUFFER_DIR / "_upload_log_full.csv"
    header         = "timestamp,caller,filename,drive_dest,size_bytes,status,attempt_number\n"

    try:
        # Pull existing Drive log (may not exist on first run — that's fine)
        subprocess.run(
            ["rclone", "copyto", drive_log_path, str(local_full)],
            capture_output=True, timeout=30
        )
        if not local_full.exists() or local_full.stat().st_size == 0:
            with open(local_full, "w") as f:
                f.write(header)

        # Append staged rows (strip duplicate header if staging file has one)
        staged = LOCAL_UPLOAD_LOG.read_text()
        lines  = staged.splitlines()
        # Drop header line from staged rows if present
        data_lines = [l for l in lines
                      if l and not l.startswith("timestamp,")]
        if data_lines:
            with open(local_full, "a") as f:
                f.write("\n".join(data_lines) + "\n")

        # Push merged log back to Drive
        subprocess.run(
            ["rclone", "copyto", str(local_full), drive_log_path],
            capture_output=True, timeout=60
        )

        # Clear the staging file after successful push
        LOCAL_UPLOAD_LOG.write_text("")
        _log(f"Upload log synced to Drive ({len(data_lines)} new row(s))")

    except Exception as e:
        _log(f"Upload log sync failed (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Dirty flag helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_flag(flag: str) -> dict:
    path = _FLAG_FILES[flag]
    if not path.exists():
        return {"status": "clean"}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"status": "clean"}


def _write_flag(flag: str, data: dict) -> None:
    path = _FLAG_FILES[flag]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def mark_dirty(flag: str, session_id: str = "", caller: str = "") -> None:
    """Mark a data type as having unsynced local writes."""
    _write_flag(flag, {
        "status":                "dirty",
        "session_id":            session_id,
        "last_local_write":      datetime.now().isoformat(),
        "last_successful_upload": _read_flag(flag).get("last_successful_upload", ""),
        "pending_since":         datetime.now().isoformat(),
        "caller":                str(Path(caller).name) if caller else "",
    })


def mark_clean(flag: str) -> None:
    """Mark a data type as fully synced to Drive."""
    existing = _read_flag(flag)
    _write_flag(flag, {
        "status":                "clean",
        "session_id":            existing.get("session_id", ""),
        "last_local_write":      existing.get("last_local_write", ""),
        "last_successful_upload": datetime.now().isoformat(),
        "pending_since":         "",
        "caller":                "",
    })


def check_flag(flag: str, caller: str = "", bypass: bool = False) -> None:
    """
    Raise DriveNotSyncedError if the flag is dirty, unless bypass=True.
    bypass=True logs a warning but does not raise.
    """
    state = _read_flag(flag)
    if state.get("status") == "clean":
        return

    msg = (
        f"Data type '{flag}' has unsynced local writes from session "
        f"'{state.get('session_id', '?')}' "
        f"(pending since {state.get('pending_since', '?')}, "
        f"written by {state.get('caller', '?')}).\n"
        f"  Run:  python3 drive_manager.py retry-buffer\n"
        f"  Or use --bypass_upload_check to proceed anyway (data may be stale)."
    )

    if bypass:
        _log(f"WARNING: bypassing dirty flag '{flag}' for caller '{caller}'")
        _log_bypass(caller, flag)
        return

    raise DriveNotSyncedError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# rclone wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _rclone_check() -> bool:
    """Return True if rclone can reach the remote."""
    try:
        r = subprocess.run(
            ["rclone", "lsd", f"{RCLONE_REMOTE}:"],
            capture_output=True, timeout=15
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _rclone_upload(local_path: Path, drive_dest: str, timeout: int = 120) -> bool:
    """
    Upload a single file to Drive.  drive_dest is the full remote path
    (e.g. "thesis_google_drive:vcf_ctp_data/outputs/session/embeds.parquet").
    Returns True on success.
    """
    r = subprocess.run(
        ["rclone", "copyto", str(local_path), drive_dest],
        capture_output=True, timeout=timeout
    )
    return r.returncode == 0


def _rclone_download(drive_src: str, local_path: Path,
                     timeout: int = 300) -> "tuple[bool, bool]":
    """
    Download a single file from Drive to local.
    Returns (ok, not_found):
      ok        — True if download succeeded
      not_found — True if the file simply doesn't exist on Drive yet
                  (vs a real connectivity/auth failure)
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rclone", "copyto", drive_src, str(local_path)],
        capture_output=True, text=True, timeout=timeout
    )
    if r.returncode == 0:
        return True, False
    # rclone exits non-zero for both "file not found" and network errors.
    # "object not found" or "no such file" in stderr → the file doesn't exist yet.
    err = (r.stderr or "").lower()
    not_found = any(phrase in err for phrase in (
        "object not found", "no such file", "not found", "no objects found",
        "couldn't find", "directory not found",
    ))
    return False, not_found


def _drive_path(relative: str) -> str:
    """Build a full Drive path from a relative path under DRIVE_ROOT."""
    return f"{DRIVE_ROOT}/{relative}"


# ─────────────────────────────────────────────────────────────────────────────
# Buffer (pending uploads)
# ─────────────────────────────────────────────────────────────────────────────

def _buffer_file(local_path: Path, drive_dest: str, caller: str,
                 size_bytes: int, error: str) -> None:
    """Copy a file to .buffer/pending/ and write a .meta.json sidecar."""
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    # Use a unique name: timestamp + original filename
    ts_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
    stem   = local_path.name
    buffered_path = PENDING_DIR / f"{ts_str}_{stem}"
    shutil.copy2(str(local_path), str(buffered_path))

    meta = {
        "original_local": str(local_path),
        "buffered_local":  str(buffered_path),
        "drive_dest":      drive_dest,
        "caller":          caller,
        "size_bytes":      size_bytes,
        "first_failure":   datetime.now().isoformat(),
        "last_failure":    datetime.now().isoformat(),
        "attempts":        1,
        "error":           error,
        "status":          "pending",
    }
    with open(str(buffered_path) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    _log(f"Buffered: {stem}  →  {buffered_path.name}  (will retry on next run)")


def flush_buffer(verbose: bool = True) -> tuple[int, int]:
    """
    Attempt to upload all pending buffered files.
    Returns (succeeded, remaining) counts.
    """
    if not PENDING_DIR.exists():
        return 0, 0

    meta_files = sorted(PENDING_DIR.glob("*.meta.json"))
    if not meta_files:
        return 0, 0

    if verbose:
        _log(f"Buffer flush: {len(meta_files)} pending file(s)")

    succeeded = 0
    remaining = 0

    for meta_path in meta_files:
        if _STOP:
            if verbose:
                _log("Buffer flush interrupted by Ctrl+C — remaining files left in buffer.")
            break

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("status") == "abandoned":
            remaining += 1
            continue

        buffered = Path(meta["buffered_local"])
        if not buffered.exists():
            # File was already cleaned up — remove orphan meta
            meta_path.unlink(missing_ok=True)
            continue

        attempts = meta.get("attempts", 1)
        if attempts >= MAX_ATTEMPTS:
            if meta.get("status") != "abandoned":
                meta["status"] = "abandoned"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                _log_failure(
                    meta["caller"], meta["drive_dest"], meta["size_bytes"],
                    f"ABANDONED after {attempts} attempts — manual intervention required"
                )
                _log(f"ABANDONED: {buffered.name} after {attempts} attempts — "
                     f"remove manually from {PENDING_DIR}")
            remaining += 1
            continue

        # Attempt upload
        try:
            ok = _rclone_upload(buffered, meta["drive_dest"])
        except Exception as e:
            ok = False
            error = str(e)
        else:
            error = "" if ok else "rclone non-zero exit"

        if ok:
            if verbose:
                _log(f"  Buffer upload OK: {buffered.name}")
            _log_upload_success(
                meta["caller"], buffered.name,
                meta["drive_dest"], meta["size_bytes"],
                attempts + 1
            )
            buffered.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            succeeded += 1
        else:
            meta["attempts"]     = attempts + 1
            meta["last_failure"] = datetime.now().isoformat()
            meta["error"]        = error
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            _log(f"  Buffer retry failed ({meta['attempts']}/{MAX_ATTEMPTS}): "
                 f"{buffered.name}")
            remaining += 1

    # One Drive round-trip to sync all log rows accumulated during this flush
    if succeeded > 0:
        _flush_upload_log()

    return succeeded, remaining


# ─────────────────────────────────────────────────────────────────────────────
# Core upload logic
# ─────────────────────────────────────────────────────────────────────────────

def write_file(local_path: Path, drive_relative_dest: str,
               caller: str = "", flag: str = "") -> bool:
    """
    Upload a local file to Drive.  If upload fails, buffer it for later retry.

    local_path           — path to the local file (must already exist)
    drive_relative_dest  — path relative to DRIVE_ROOT (e.g. "outputs/s1/embeds.parquet")
    caller               — __file__ of the calling script (for logging)
    flag                 — which dirty flag this write belongs to ("db"|"parquet"|"gallery"|"")

    Returns True if upload succeeded immediately, False if buffered.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"write_file: local file not found: {local_path}")

    caller_name = str(Path(caller).name) if caller else "unknown"
    size_bytes  = local_path.stat().st_size
    drive_dest  = _drive_path(drive_relative_dest)

    # Mark dirty before attempting upload
    if flag:
        mark_dirty(flag, caller=caller)

    # Flush any pending buffer first
    flush_buffer(verbose=False)

    # Attempt upload with exponential backoff
    last_error = ""
    for attempt in range(1, 4):
        try:
            ok = _rclone_upload(local_path, drive_dest)
        except subprocess.TimeoutExpired:
            ok = False
            last_error = "timeout"
        except Exception as e:
            ok = False
            last_error = str(e)
        else:
            last_error = "" if ok else "rclone non-zero exit"

        if ok:
            _log(f"Uploaded: {local_path.name}  →  {drive_dest}"
                 + (f"  (attempt {attempt})" if attempt > 1 else ""))
            _log_upload_success(caller_name, local_path.name,
                                drive_dest, size_bytes, attempt)
            if flag:
                mark_clean(flag)
            # Flush staged log rows to Drive — suppressed during batch uploads
            # (_cmd_upload_collar calls _flush_upload_log once at the end).
            if not getattr(write_file, "_batch_mode", False):
                _flush_upload_log()
            return True

        if attempt < 3:
            wait = BACKOFF_BASE ** (attempt * 2)
            _log(f"Upload failed (attempt {attempt}/3), retrying in {wait}s: "
                 f"{local_path.name}")
            time.sleep(wait)

    # All attempts failed — buffer the file
    _log(f"Upload failed after 3 attempts: {local_path.name} → buffered")
    _log_failure(caller_name, drive_dest, size_bytes, last_error)
    _buffer_file(local_path, drive_dest, caller_name, size_bytes, last_error)
    # flag stays dirty
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers (library API used by other scripts)
# ─────────────────────────────────────────────────────────────────────────────

def get_db_path(caller: str = "", bypass: bool = False) -> Path:
    """
    Return local path to calving_project.db.
    Checks the db dirty flag — raises DriveNotSyncedError if dirty and bypass=False.
    """
    check_flag("db", caller=caller, bypass=bypass)
    return LOCAL_DB_PATH


def get_gallery_path(modality: str, filename: str = "",
                     caller: str = "", bypass: bool = False) -> Path:
    """
    Return local path to a gallery .npy file.
    modality: "day" | "night"
    filename: specific filename (e.g. "gallery_pose_day.npy"); if empty, returns the directory.
    """
    check_flag("gallery", caller=caller, bypass=bypass)
    if filename:
        return LOCAL_GALLERY_DIR / filename
    return LOCAL_GALLERY_DIR / f"gallery_{modality}.npy"


def get_parquet_path(session_id: str, kind: str = "embeds",
                     caller: str = "", bypass: bool = False) -> Path:
    """
    Return local path to a parquet file for a session.
    kind: "embeds" | "kps"
    """
    check_flag("parquet", caller=caller, bypass=bypass)
    return LOCAL_OUTPUTS_DIR / session_id / f"{kind}.parquet"


def get_session_dir(session_id: str) -> Path:
    """Return (and create) the local output directory for a session."""
    d = LOCAL_OUTPUTS_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_kinetics_path(filename: str = "", caller: str = "") -> Path:
    """
    Return local path to a kinetics CSV.
    If filename is given, returns collar_data/<filename>.
    Otherwise returns the collar_data directory.
    """
    if filename:
        return LOCAL_COLLAR_DIR / filename
    return LOCAL_COLLAR_DIR


def get_behavior_path(filename: str = "", caller: str = "") -> Path:
    """Return local path to a behavior CSV."""
    if filename:
        return LOCAL_COLLAR_DIR / filename
    return LOCAL_COLLAR_DIR


def get_video_path(path: str) -> Path:
    """
    Pass-through for local video files.
    Validates the file exists locally and returns it unchanged.
    Videos are never uploaded to Drive.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")
    return p


def get_gallery_dir(caller: str = "", bypass: bool = False) -> Path:
    """Return local gallery directory path (after dirty-flag check)."""
    check_flag("gallery", caller=caller, bypass=bypass)
    LOCAL_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_GALLERY_DIR


# ─────────────────────────────────────────────────────────────────────────────
# DB sync helpers
# ─────────────────────────────────────────────────────────────────────────────

def pull_db(allow_stale: bool = False) -> Path:
    """
    Pull the canonical DB from Drive to local.  Blocking.

    Three outcomes:
      1. Pull succeeds            → return local path (normal case)
      2. File not on Drive yet    → first run; return local path (will be created fresh)
      3. Pull failed (network)    → use stale local copy if allow_stale=True, else raise

    Returns local DB path.
    """
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    drive_src = _drive_path(DRIVE_DB_PATH)

    _log(f"Pulling DB from Drive: {drive_src}")
    ok = False
    not_found = False
    try:
        ok, not_found = _rclone_download(drive_src, LOCAL_DB_PATH)
    except Exception as e:
        _log(f"DB pull exception: {e}")

    if ok:
        _log(f"DB pulled successfully: {LOCAL_DB_PATH}  "
             f"({LOCAL_DB_PATH.stat().st_size / 1e6:.1f} MB)")
        mark_clean("db")
        return LOCAL_DB_PATH

    if not_found:
        # First run — DB doesn't exist on Drive yet. A fresh one will be created
        # locally by init_db() and uploaded to Drive at end of session.
        _log("DB not found on Drive — first run, a new DB will be created locally.")
        mark_clean("db")
        return LOCAL_DB_PATH

    # Pull failed due to network / auth issue
    if LOCAL_DB_PATH.exists():
        msg = (f"WARNING: could not pull DB from Drive (Drive may be unavailable). "
               f"Using stale local copy: {LOCAL_DB_PATH}")
        _log(msg)
        _log_failure("pull_db", drive_src, 0, "pull failed — using stale local copy")
        if allow_stale:
            return LOCAL_DB_PATH
        raise DriveUnavailableError(
            "Cannot pull DB from Drive and allow_stale=False. "
            "Check your internet connection or run with allow_stale=True."
        )
    else:
        raise DriveUnavailableError(
            f"Cannot pull DB from Drive and no local copy exists at {LOCAL_DB_PATH}.\n"
            "Check your internet connection and rclone config."
        )


def sync_db(session_id: str = "", caller: str = "") -> bool:
    """
    Upload the local DB to Drive as a snapshot.
    Called at the end of a session or at checkpoint intervals.
    """
    if not LOCAL_DB_PATH.exists():
        _log("sync_db: no local DB to sync")
        return False
    return write_file(LOCAL_DB_PATH, DRIVE_DB_PATH, caller=caller, flag="db")


def pull_gallery(modality: str = "") -> None:
    """
    Pull gallery .npy files from Drive to local.
    modality: "day" | "night" | "" (pulls all gallery files)
    """
    LOCAL_GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    if modality:
        files = [
            f"gallery_{modality}.npy",
            f"gallery_pose_{modality}.npy",
            f"temp_gallery_pose_{modality}.npy",
        ]
    else:
        files = [
            "gallery_day.npy", "gallery_night.npy",
            "gallery_pose_day.npy", "gallery_pose_night.npy",
            "temp_gallery_pose_day.npy", "temp_gallery_pose_night.npy",
            "synthetic_id_counter.npy",
        ]

    for fname in files:
        drive_src   = _drive_path(f"{DRIVE_GALLERY_PREFIX}/{fname}")
        local_dest  = LOCAL_GALLERY_DIR / fname
        try:
            ok, _ = _rclone_download(drive_src, local_dest, timeout=30)
            if ok:
                _log(f"Pulled gallery: {fname}")
            else:
                _log(f"Gallery not found on Drive (first run?): {fname}")
        except Exception as e:
            _log(f"Gallery pull failed for {fname}: {e}")


def ensure_local(drive_relative: str, local_path: Path,
                 caller: str = "", bypass: bool = False) -> Path:
    """
    Ensure a file exists locally.  If missing, pull from Drive.
    Used for files that are needed by read operations.
    """
    if local_path.exists():
        return local_path

    _log(f"ensure_local: pulling missing file: {local_path.name}")
    drive_src = _drive_path(drive_relative)
    try:
        ok, _ = _rclone_download(drive_src, local_path)
    except Exception as e:
        ok = False
        _log(f"ensure_local: pull failed: {e}")

    if not ok:
        msg = (f"File not found locally and could not be pulled from Drive:\n"
               f"  local:  {local_path}\n"
               f"  drive:  {drive_src}\n"
               f"Check your internet connection.")
        if bypass:
            _log(f"WARNING: {msg}")
            return local_path
        raise DriveUnavailableError(msg)

    return local_path


# ─────────────────────────────────────────────────────────────────────────────
# Collar file discovery (Drive-based, time-window overlap)
# ─────────────────────────────────────────────────────────────────────────────

def find_collar_files(session_start_dt: "datetime",
                      session_end_dt:   "datetime",
                      kind: str = "kinetic") -> "list[Path]":
    """
    Discover collar CSV files on Drive whose time window overlaps the session.

    Canonical filename format (set by _cmd_upload_collar):
        kinetic_data_s<YYYY_MM_DD-HH_MM_SS>-e<YYYY_MM_DD-HH_MM_SS>__<ids>.csv
        behavior_data_s<YYYY_MM_DD-HH_MM_SS>-e<YYYY_MM_DD-HH_MM_SS>__<ids>.csv

    kind: "kinetic" | "behavior"

    Steps:
      1. List collar_data/ on Drive via rclone lsf
      2. Parse s/e timestamps from each matching filename
      3. Keep files where [file_start, file_end] overlaps [session_start, session_end]
      4. Pull matching files to local collar_data/ if not already present
      5. Return list of local Path objects (may be empty)
    """
    import re as _re

    prefix   = f"{kind}_data_"
    drive_dir = f"{DRIVE_ROOT}/{DRIVE_COLLAR_PREFIX}/"

    # List files on Drive
    try:
        r = subprocess.run(
            ["rclone", "lsf", drive_dir],
            capture_output=True, text=True, timeout=30
        )
    except Exception as e:
        _log(f"find_collar_files: rclone lsf failed: {e}")
        return []

    if r.returncode != 0:
        _log(f"find_collar_files: could not list {drive_dir}: {r.stderr.strip()}")
        return []

    all_files = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    candidates = [f for f in all_files if f.startswith(prefix) and f.endswith(".csv")]

    if not candidates:
        _log(f"find_collar_files: no {kind}_data_*.csv files found on Drive")
        return []

    # Parse timestamp pattern: s<YYYY_MM_DD-HH_MM_SS>-e<YYYY_MM_DD-HH_MM_SS>
    _ts_re  = _re.compile(
        r"s(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})-e(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})"
    )
    _ts_fmt = "%Y_%m_%d-%H_%M_%S"

    LOCAL_COLLAR_DIR.mkdir(parents=True, exist_ok=True)
    matched: list[Path] = []

    for fname in candidates:
        m = _ts_re.search(fname)
        if not m:
            _log(f"  find_collar_files: skipping unparseable filename: {fname}")
            continue
        try:
            file_start = datetime.strptime(m.group(1), _ts_fmt)
            file_end   = datetime.strptime(m.group(2), _ts_fmt)
        except ValueError:
            continue

        # Overlap check: [a_start, a_end] overlaps [b_start, b_end]
        # iff a_start <= b_end AND a_end >= b_start
        if file_start <= session_end_dt and file_end >= session_start_dt:
            local_dest = LOCAL_COLLAR_DIR / fname
            if not local_dest.exists():
                _log(f"  Pulling collar file: {fname}")
                ok, _ = _rclone_download(f"{drive_dir}{fname}", local_dest, timeout=60)
                if not ok:
                    _log(f"  WARNING: could not pull {fname} — skipping")
                    continue
            else:
                _log(f"  Collar file already local: {fname}")
            matched.append(local_dest)

    _log(f"find_collar_files({kind}): {len(matched)} file(s) match "
         f"[{session_start_dt.strftime('%Y-%m-%d %H:%M')} – "
         f"{session_end_dt.strftime('%Y-%m-%d %H:%M')}]")
    for p in matched:
        _log(f"    {p.name}")

    return matched


def load_collar_data(session_start_dt: "datetime",
                     session_end_dt:   "datetime",
                     kind: str = "kinetic") -> "pd.DataFrame | None":
    """
    Pull all overlapping collar CSVs from Drive and merge into a single DataFrame.
    Deduplicates rows by (AnimalId, datetime) after merging.

    kind: "kinetic" | "behavior"
    Returns a merged DataFrame, or None if no files found.
    """
    try:
        import pandas as _pd
    except ImportError:
        _log("load_collar_data: pandas not available")
        return None

    files = find_collar_files(session_start_dt, session_end_dt, kind)
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = _pd.read_csv(str(f), parse_dates=["datetime"])
            dfs.append(df)
        except Exception as e:
            _log(f"  load_collar_data: could not read {f.name}: {e}")

    if not dfs:
        return None

    merged = _pd.concat(dfs, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["AnimalId", "datetime"])
    after  = len(merged)
    if before != after:
        _log(f"  load_collar_data({kind}): deduplicated {before - after} overlapping rows")
    _log(f"  load_collar_data({kind}): {after} rows, "
         f"animals: {sorted(merged['AnimalId'].unique().tolist())}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# DriveManager class (convenience wrapper for scripts that prefer OOP style)
# ─────────────────────────────────────────────────────────────────────────────

class DriveManager:
    """
    Thin wrapper around module-level functions.
    Import and instantiate in any script:

        from drive_manager import DriveManager
        dm = DriveManager()
    """

    def __init__(self, bypass: bool = False, caller: str = ""):
        self.bypass = bypass
        self.caller = caller

    # ── path resolution ──────────────────────────────────────────────────────

    def get_db_path(self) -> Path:
        return get_db_path(caller=self.caller, bypass=self.bypass)

    def get_gallery_path(self, modality: str, filename: str = "") -> Path:
        return get_gallery_path(modality, filename,
                                caller=self.caller, bypass=self.bypass)

    def get_gallery_dir(self) -> Path:
        return get_gallery_dir(caller=self.caller, bypass=self.bypass)

    def get_parquet_path(self, session_id: str, kind: str = "embeds") -> Path:
        return get_parquet_path(session_id, kind,
                                caller=self.caller, bypass=self.bypass)

    def get_session_dir(self, session_id: str) -> Path:
        return get_session_dir(session_id)

    def get_kinetics_path(self, filename: str = "") -> Path:
        return get_kinetics_path(filename, caller=self.caller)

    def get_behavior_path(self, filename: str = "") -> Path:
        return get_behavior_path(filename, caller=self.caller)

    def get_video_path(self, path: str) -> Path:
        return get_video_path(path)

    # ── sync operations ──────────────────────────────────────────────────────

    def pull_db(self, allow_stale: bool = False) -> Path:
        return pull_db(allow_stale=allow_stale)

    def sync_db(self, session_id: str = "") -> bool:
        return sync_db(session_id=session_id, caller=self.caller)

    def pull_gallery(self, modality: str = "") -> None:
        return pull_gallery(modality)

    def write_file(self, local_path: Path, drive_relative_dest: str,
                   flag: str = "") -> bool:
        return write_file(local_path, drive_relative_dest,
                          caller=self.caller, flag=flag)

    # ── flag management ──────────────────────────────────────────────────────

    def mark_dirty(self, flag: str, session_id: str = "") -> None:
        mark_dirty(flag, session_id=session_id, caller=self.caller)

    def mark_clean(self, flag: str) -> None:
        mark_clean(flag)

    def check_flag(self, flag: str) -> None:
        check_flag(flag, caller=self.caller, bypass=self.bypass)

    # ── buffer ───────────────────────────────────────────────────────────────

    def flush_buffer(self) -> tuple[int, int]:
        return flush_buffer(verbose=True)

    # ── status ───────────────────────────────────────────────────────────────

    def status(self) -> None:
        _print_status()


# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_status() -> None:
    print("\n" + "─" * 60)
    print("  drive_manager — status")
    print("─" * 60)

    # rclone reachability
    _log("Checking Drive connection...")
    reachable = _rclone_check()
    print(f"  Drive reachable : {'YES ✓' if reachable else 'NO ✗  (check internet / rclone config)'}")
    print(f"  DRIVE_ROOT      : {DRIVE_ROOT}")
    print(f"  LOCAL_ROOT      : {LOCAL_ROOT}")

    # dirty flags
    print("\n  Sync flags:")
    for name, path in _FLAG_FILES.items():
        state = _read_flag(name)
        status = state.get("status", "clean")
        icon   = "✓" if status == "clean" else "⚠ DIRTY"
        extra  = ""
        if status == "dirty":
            extra = (f"  session={state.get('session_id','?')}  "
                     f"since={state.get('pending_since','?')[:19]}  "
                     f"caller={state.get('caller','?')}")
        print(f"    {name:10s}  {icon}{extra}")

    # buffer
    pending = list(PENDING_DIR.glob("*.meta.json")) if PENDING_DIR.exists() else []
    print(f"\n  Buffer pending  : {len(pending)} file(s)")
    for mp in pending[:10]:
        try:
            with open(mp) as f:
                m = json.load(f)
            print(f"    {Path(m['buffered_local']).name}  "
                  f"attempts={m.get('attempts',1)}/{MAX_ATTEMPTS}  "
                  f"status={m.get('status','?')}")
        except Exception:
            pass

    if reachable and pending:
        print("\n  Flushing buffer...")
        succ, rem = flush_buffer(verbose=True)
        print(f"  Buffer flush: {succ} succeeded, {rem} remaining")

    print()


def _build_collar_filename(csv_path: Path, kind: str = "kinetic_data") -> "str | None":
    """
    Scan a collar CSV and build a canonical filename:
        s<YYYY_MM_DD-HH_MM_SS>-e<YYYY_MM_DD-HH_MM_SS>__<id1>_<id2>_....csv

    Returns the new filename string, or None if the file should be skipped.

    Rules:
      - Skip files with fewer than 2 data lines (header + at least 2 rows).
      - Datetime column must be named 'datetime'.
      - Animal ID column must be named 'AnimalId'.
      - Start = earliest datetime across all rows; end = latest.
      - Animal IDs are sorted ascending and joined with '_'.
    """
    import csv as _csv

    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = _csv.reader(fh)
            rows = list(reader)
    except Exception as e:
        _log(f"  Cannot read {csv_path.name}: {e}")
        return None

    # Filter truly blank lines
    rows = [r for r in rows if any(cell.strip() for cell in r)]

    # Need header + at least 2 data rows
    if len(rows) < 3:
        _log(f"  SKIP {csv_path.name} — fewer than 2 data lines ({len(rows) - 1} found)")
        return None

    header = [h.strip() for h in rows[0]]

    # Locate required columns
    try:
        dt_col  = header.index("datetime")
        aid_col = header.index("AnimalId")
    except ValueError:
        _log(f"  SKIP {csv_path.name} — missing 'datetime' or 'AnimalId' column")
        return None

    datetimes = []
    animal_ids: set = set()

    for row in rows[1:]:
        if len(row) <= max(dt_col, aid_col):
            continue
        dt_raw  = row[dt_col].strip()
        aid_raw = row[aid_col].strip()
        if not dt_raw or not aid_raw:
            continue
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                datetimes.append(datetime.strptime(dt_raw, fmt))
                break
            except ValueError:
                continue
        try:
            animal_ids.add(str(int(float(aid_raw))))
        except ValueError:
            if aid_raw:
                animal_ids.add(aid_raw)

    if not datetimes:
        _log(f"  SKIP {csv_path.name} — no parseable datetime values")
        return None
    if not animal_ids:
        _log(f"  SKIP {csv_path.name} — no parseable AnimalId values")
        return None

    dt_fmt  = "%Y_%m_%d-%H_%M_%S"
    t_start = min(datetimes).strftime(dt_fmt)
    t_end   = max(datetimes).strftime(dt_fmt)
    ids_str = "_".join(
        sorted(animal_ids, key=lambda x: int(x) if x.isdigit() else x)
    )
    return f"{kind}_s{t_start}-e{t_end}__{ids_str}.csv"


def _cmd_upload_collar(dir_path: str, kind: str) -> None:
    """
    Batch-upload all .csv files from a directory as collar data (kinetics or behavior).

    For each file:
      1. Skip files with fewer than 2 data lines.
      2. Scan the file and derive a canonical filename from its content:
             s<YYYY_MM_DD-HH_MM_SS>-e<YYYY_MM_DD-HH_MM_SS>__<id1>_<id2>_....csv
      3. Copy renamed file to local collar_data/, then upload to Drive.
    """
    p = Path(dir_path).expanduser().resolve()
    if not p.exists():
        _log(f"ERROR: path not found: {p}")
        sys.exit(1)
    if not p.is_dir():
        _log(f"ERROR: expected a directory, got a file: {p}")
        _log(f"  Pass the directory containing your {kind} CSVs.")
        sys.exit(1)

    csv_files = sorted(p.glob("*.csv"))
    if not csv_files:
        _log(f"No .csv files found in: {p}")
        return

    _log(f"Scanning {len(csv_files)} CSV file(s) in {p} ...")
    LOCAL_COLLAR_DIR.mkdir(parents=True, exist_ok=True)

    uploaded = skipped = failed = 0

    # Suppress per-file Drive log flush — we do one flush at the end of the batch
    write_file._batch_mode = True
    interrupted = False
    try:
        for csv_path in csv_files:
            if _STOP:
                _log("Stopping batch — Ctrl+C received. Files not yet processed are untouched.")
                interrupted = True
                break

            _log(f"  Processing: {csv_path.name}")

            new_name = _build_collar_filename(csv_path, kind=f"{kind}_data")
            if new_name is None:
                skipped += 1
                continue

            local_dest = LOCAL_COLLAR_DIR / new_name
            shutil.copy2(str(csv_path), str(local_dest))
            _log(f"  Renamed  : {csv_path.name}  ->  {new_name}")

            drive_rel = f"{DRIVE_COLLAR_PREFIX}/{new_name}"
            ok = write_file(local_dest, drive_rel, caller="drive_manager_cli")
            if ok:
                _log(f"  Uploaded : {new_name}  ->  {DRIVE_ROOT}/{drive_rel}")
                uploaded += 1
            else:
                _log(f"  Buffered : {new_name} (upload failed — will retry)")
                failed += 1
    finally:
        write_file._batch_mode = False

    # Always flush staged log rows — even on interrupt, log what completed
    _flush_upload_log()

    print()
    status = "INTERRUPTED" if interrupted else "complete"
    _log(f"Batch {status} — {uploaded} uploaded, {skipped} skipped, {failed} buffered for retry"
         + (f"  ({len(csv_files) - uploaded - skipped - failed} not started)" if interrupted else ""))


def _cmd_upload_ledger(path: str) -> None:
    """Upload calving_ledger.csv to Drive."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        _log(f"ERROR: file not found: {p}")
        sys.exit(1)
    dest = p.name
    ok = write_file(p, dest, caller="drive_manager_cli")
    if ok:
        _log(f"Uploaded ledger: {p.name}  →  {DRIVE_ROOT}/{dest}")
    else:
        _log(f"Upload failed — {p.name} buffered for retry")


def _cmd_list_sessions() -> None:
    """List session folders on Drive."""
    _log(f"Listing sessions on Drive: {DRIVE_ROOT}/{DRIVE_OUTPUTS_PREFIX}/")
    r = subprocess.run(
        ["rclone", "lsd", f"{DRIVE_ROOT}/{DRIVE_OUTPUTS_PREFIX}/"],
        capture_output=True, text=True, timeout=30
    )
    if r.returncode != 0:
        _log(f"ERROR: {r.stderr.strip()}")
        sys.exit(1)
    if not r.stdout.strip():
        print("  (no sessions on Drive yet)")
    else:
        for line in r.stdout.strip().splitlines():
            print(f"  {line.strip()}")


def _cmd_pull_db() -> None:
    """Manually pull DB from Drive to local."""
    try:
        path = pull_db(allow_stale=False)
        _log(f"DB pulled to: {path}")
    except DriveUnavailableError as e:
        _log(f"ERROR: {e}")
        sys.exit(1)


def _cmd_clear_flag(flag_name: str) -> None:
    """Manually clear a dirty flag (emergency use)."""
    if flag_name not in _FLAG_FILES:
        _log(f"ERROR: unknown flag '{flag_name}'. "
             f"Valid flags: {list(_FLAG_FILES.keys())}")
        sys.exit(1)
    mark_clean(flag_name)
    _log(f"Flag '{flag_name}' cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="drive_manager — Google Drive sync manager for vcf-ctp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status                  Check Drive connection, dirty flags, and buffer
  retry-buffer            Attempt upload of all buffered files
  pull-db                 Pull DB from Drive to local
  list-sessions           List session folders on Drive
  upload-kinetics DIR     Batch-upload all kinetics CSVs from a directory
  upload-behavior DIR     Batch-upload all behavior CSVs from a directory
  upload-ledger   FILE    Upload calving_ledger.csv to Drive
  clear-flag      NAME    Manually clear a dirty flag (db|parquet|gallery)
        """
    )
    ap.add_argument("command", choices=[
        "status", "retry-buffer", "pull-db", "list-sessions",
        "upload-kinetics", "upload-behavior", "upload-ledger", "clear-flag"
    ])
    ap.add_argument("argument", nargs="?", default="",
                    help="File path or flag name depending on command")
    return ap.parse_args()


def main() -> None:
    args = _parse_cli()

    # Ensure local dirs exist
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    cmd = args.command
    arg = args.argument

    if cmd == "status":
        _print_status()

    elif cmd == "retry-buffer":
        _log("Flushing upload buffer...")
        succ, rem = flush_buffer(verbose=True)
        _log(f"Done: {succ} succeeded, {rem} remaining")

        # Re-check flags after flush
        for flag in _FLAG_FILES:
            state = _read_flag(flag)
            if state.get("status") == "dirty":
                # Check if any pending files for this flag are now gone
                pending = list(PENDING_DIR.glob("*.meta.json"))
                if not pending:
                    mark_clean(flag)
                    _log(f"Flag '{flag}' cleared (no more pending files)")

    elif cmd == "pull-db":
        _cmd_pull_db()

    elif cmd == "list-sessions":
        _cmd_list_sessions()

    elif cmd == "upload-kinetics":
        if not arg:
            _log("ERROR: provide a directory path:  upload-kinetics /path/to/collar_data/")
            sys.exit(1)
        _cmd_upload_collar(arg, "kinetics")

    elif cmd == "upload-behavior":
        if not arg:
            _log("ERROR: provide a directory path:  upload-behavior /path/to/collar_data/")
            sys.exit(1)
        _cmd_upload_collar(arg, "behavior")

    elif cmd == "upload-ledger":
        if not arg:
            _log("ERROR: provide a file path:  upload-ledger calving_ledger.csv")
            sys.exit(1)
        _cmd_upload_ledger(arg)

    elif cmd == "clear-flag":
        if not arg:
            _log("ERROR: provide a flag name: clear-flag db|parquet|gallery")
            sys.exit(1)
        _cmd_clear_flag(arg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Fallback: signal handler sets _STOP but if something blocks long enough
        # for a second Ctrl+C to arrive, catch it cleanly here.
        print("\n[drive_manager] Interrupted.", flush=True)
        # Flush any log rows that were staged before the interrupt
        try:
            _flush_upload_log()
        except Exception:
            pass
        sys.exit(130)   # standard exit code for SIGINT