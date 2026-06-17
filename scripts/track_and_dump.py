#!/usr/bin/env python3
"""
track_and_dump.py — detector + tracker (+ pose + appearance embedding)

Outputs:

No CSV or JSONL output. Arrays never stored in SQLite.
embed_parquet_row / kps_parquet_row columns are integer pointers into the parquets.

WSL-friendly, no GUI required.
"""

import argparse, json, re, signal, sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from time import time
from collections import defaultdict

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from ultralytics import YOLO

import torch
import torch.nn as nn
import torchvision.models as tv

# Drive I/O layer
from drive_manager import DriveManager, DriveUnavailableError

# Vision classifiers for crop tagging — optional (graceful fallback if absent)
_VISION_IMPORT_ERROR = None
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from vision_features.features.posture import extract_posture
    from vision_features.features.facing  import extract_facing
    from vision_features.schema import Posture, Facing, POSTURE_NAMES, FACING_NAMES
    _VISION_AVAILABLE = True
except Exception as _e:
    _VISION_AVAILABLE = False
    _VISION_IMPORT_ERROR = _e

EXPECTED_KP = 19
DRIVE_BATCH_LOG = "processing_log.csv"   # append-only log on Drive
VIDEO_EXTENSIONS = {".mp4", ".ts", ".avi", ".mov", ".mkv"}

_FNAME_RE = re.compile(r'^(?P<camera>.+?)_S(?P<start>\d{14})(?:_E(?P<end>\d{14}))?')

def parse_filename(video_path: str):
    """Extract (camera_id, start_dt, end_dt) from filename.
    Format: <camera>_S<YYYYMMDDHHmmss>_E<YYYYMMDDHHmmss>
    Returns (camera_id: str, start_dt: datetime|None, end_dt: datetime|None).
    """
    stem = Path(video_path).stem
    m = _FNAME_RE.search(stem)
    if not m:
        return stem, None, None
    camera_id = m.group("camera")
    try:
        start_dt = datetime.strptime(m.group("start"), "%Y%m%d%H%M%S")
    except (ValueError, TypeError):
        start_dt = None
    try:
        end_dt = datetime.strptime(m.group("end"), "%Y%m%d%H%M%S") if m.group("end") else None
    except (ValueError, TypeError):
        end_dt = None
    return camera_id, start_dt, end_dt

def log(msg: str) -> None:
    print(f"[track] {msg}", flush=True)


# ── Embedding backbone ────────────────────────────────────────────────────────
class Embedder128(nn.Module):
    def __init__(self, pretrained=True, out_dim=128):
        super().__init__()
        m = tv.mobilenet_v3_small(
            weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = m.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.proj     = nn.Linear(576, out_dim)

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        z = self.proj(f)
        return z / (z.norm(dim=1, keepdim=True) + 1e-8)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      required=True, help="Detector .pt")
    ap.add_argument("--source",     required=True,
                    help="Path to a single video file OR a directory of video files. "
                         "Supported extensions: .mp4 .ts .avi .mov .mkv. "
                         "Videos are local only — never uploaded to Drive.")
    # session_id and camera_id are now derived from the filename automatically
    ap.add_argument("--tracker",    default="bytetrack.yaml")
    ap.add_argument("--imgsz",      type=int,   default=960)
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--iou",        type=float, default=0.45)
    ap.add_argument("--save_crops", action="store_true")
    ap.add_argument("--crops_local", action="store_true",
                    help="Keep crops on local disk only — do not upload to Drive. "
                         "Has no effect unless --save_crops is also set.")
    ap.add_argument("--crop_tags", action="store_true",
                    help="Append posture (standing/lying) and facing (left/right/toward/away) "
                         "tags to crop filenames. Requires --pose_model and vision_features/. "
                         "Falls back to 'unk' tags if classifiers are unavailable.")
    ap.add_argument("--crop_every", type=int, default=1,
                    help="Save crop every N-th detection per track ID (default: 1)")
    ap.add_argument("--min_crop_wh", type=int, nargs=2, metavar=("W","H"), default=(0,0))
    ap.add_argument("--embed_size", type=int, default=128)
    ap.add_argument("--pose_model", default="", help="YOLO-Pose .pt (optional)")
    ap.add_argument("--pose_imgsz", type=int,   default=384)
    ap.add_argument("--pose_conf",  type=float, default=0.25)
    ap.add_argument("--pose_kp_conf_thresh", type=float, default=0.30,
                    help="Keypoint conf: >=thresh v=2, <thresh v=1 (default: 0.30)")
    ap.add_argument("--save_every", type=int, default=10,
                    help="Sample interval: keep only the most recent detection per "
                         "temp_id within each N-frame window (default: 10). "
                         "Higher = sparser data.")
    ap.add_argument("--flush_every", type=int, default=5,
                    help="Flush to SQLite + Parquet every M windows, i.e. every "
                         "save_every * flush_every frames (default: 5).")
    # ── reconcile.py integration ──────────────────────────────────────────────
    ap.add_argument("--kinetics", default="",
                    help="kinetic_data_*.csv path. If omitted, matching files are "
                         "discovered automatically from Drive using the session time window.")
    ap.add_argument("--gallery_dir", default="./reid_gallery",
                    help="Gallery directory for reconcile.py (default: ./reid_gallery)")
    ap.add_argument("--corr_threshold",   type=float, default=0.7)
    ap.add_argument("--cosine_threshold", type=float, default=0.75)
    ap.add_argument("--ema_alpha",        type=float, default=0.15)
    ap.add_argument("--bypass_upload_check", action="store_true",
                    help="Skip dirty-flag check when reading from Drive "
                         "(proceeds with potentially stale data — use with caution)")
    return ap.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_tensor_bchw(img_bgr, size=224):
    x    = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    x    = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return ((x - mean) / std).transpose(2, 0, 1)

def crops_from_bboxes(frame, bboxes, margin=0.1):
    H, W = frame.shape[:2]
    crops, used = [], []
    for (x1, y1, x2, y2) in bboxes:
        mx  = int(round((x2-x1)*margin));  my  = int(round((y2-y1)*margin))
        xx1 = max(0,     int(x1)-mx);      yy1 = max(0,     int(y1)-my)
        xx2 = min(W-1,   int(x2)+mx);      yy2 = min(H-1,   int(y2)+my)
        crops.append(frame[yy1:yy2, xx1:xx2].copy())
        used.append((xx1, yy1, xx2, yy2))
    return crops, used

def flat_kps_xyv(kps_xyv):
    out = []
    for (x, y, v) in kps_xyv:
        out.extend([float(round(x,3)), float(round(y,3)), int(v)])
    return out


# ── SQLite schema ─────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS video_sessions (
    session_id      TEXT PRIMARY KEY,
    video_path      TEXT,
    camera_id       TEXT,
    start_dt        TEXT,
    end_dt          TEXT,
    collar_csv_path TEXT,
    is_night        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS raw_tracks (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT    NOT NULL,
    frame_index        INTEGER NOT NULL,
    frame_time_sec     REAL,
    frame_datetime     TEXT,
    temp_id            INTEGER,
    det_conf           REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    cx REAL, cy REAL,
    w  REAL, h  REAL,
    kps_conf           REAL,
    embed_parquet_row  INTEGER,
    kps_parquet_row    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_raw_tracks_session_frame
    ON raw_tracks (session_id, frame_index);
"""

def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit()
    return conn

def register_session(conn, session_id, video_path, camera_id, start_dt, end_dt=""):
    # Delete any existing data for this session so a re-run is a clean overwrite.
    conn.execute("DELETE FROM raw_tracks WHERE session_id = ?", (session_id,))
    conn.execute(
        "INSERT OR REPLACE INTO video_sessions "
        "(session_id,video_path,camera_id,start_dt,end_dt) VALUES (?,?,?,?,?)",
        (session_id, video_path, camera_id, start_dt, end_dt)
    )
    conn.commit()


# ── Incremental Parquet writers ───────────────────────────────────────────────
# Open the file before the loop; flush one row-group per commit_every interval.
# Memory stays bounded to one batch at a time rather than the full run.

EMBED_SCHEMA = pa.schema([
    pa.field("session_id",  pa.string()),
    pa.field("frame_index", pa.int32()),
    pa.field("temp_id",     pa.int32()),
    pa.field("embed",       pa.list_(pa.float32(), 128)),
])

KPS_SCHEMA = pa.schema([
    pa.field("session_id",  pa.string()),
    pa.field("frame_index", pa.int32()),
    pa.field("temp_id",     pa.int32()),
    pa.field("kps",         pa.list_(pa.float32(), 57)),
    pa.field("kps_kconf",   pa.list_(pa.float32(), 19)),
])


class EmbedWriter:
    """Rolling multi-part Parquet writer.

    Produces embeds_part000.parquet, embeds_part001.parquet, …
    A new part file is opened whenever the current part would exceed
    `rows_per_part` rows.  Each rollover releases the Arrow memory pool
    so WSL doesn't accumulate a high-water mark across the whole session.
    """
    def __init__(self, outdir: Path, rows_per_part: int = 50_000):
        self.outdir        = outdir
        self.rows_per_part = rows_per_part
        self._part         = 0
        self._part_rows    = 0   # rows written to the current part
        self.total         = 0   # rows written across all parts
        self._writer       = None
        self._current_path: Path | None = None
        self._parts: list[Path] = []
        self._open_part()

    def _open_part(self) -> None:
        self._current_path = self.outdir / f"embeds_part{self._part:03d}.parquet"
        self._parts.append(self._current_path)
        self._writer    = pq.ParquetWriter(str(self._current_path),
                                           EMBED_SCHEMA, compression="snappy")
        self._part_rows = 0

    def _roll(self) -> None:
        self._writer.close()
        pa.default_cpu_memory_pool().release_unused()
        self._part += 1
        self._open_part()

    def flush(self, rows: list) -> None:
        if not rows:
            return
        # Split into sub-batches that respect the part boundary.
        start = 0
        while start < len(rows):
            remaining_in_part = self.rows_per_part - self._part_rows
            chunk = rows[start : start + remaining_in_part]
            emb_arr = np.stack([r["embed"] for r in chunk])
            tbl = pa.table({
                "session_id":  pa.array([r["session_id"]  for r in chunk], pa.string()),
                "frame_index": pa.array([r["frame_index"] for r in chunk], pa.int32()),
                "temp_id":     pa.array([r["temp_id"]     for r in chunk], pa.int32()),
                "embed": pa.FixedSizeListArray.from_arrays(
                    pa.array(emb_arr.ravel().tolist(), pa.float32()), 128),
            }, schema=EMBED_SCHEMA)
            self._writer.write_table(tbl)
            self._part_rows += len(chunk)
            self.total      += len(chunk)
            start           += len(chunk)
            if self._part_rows >= self.rows_per_part and start < len(rows):
                self._roll()

    def close(self) -> None:
        self._writer.close()
        pa.default_cpu_memory_pool().release_unused()
        total_mb = sum(p.stat().st_size for p in self._parts if p.exists()) / 1e6
        log(f"embeds  → {len(self._parts)} part(s), {self.total} rows, "
            f"{total_mb:.1f} MB total  [{self.outdir.name}]")


class KpsWriter:
    """Rolling multi-part Parquet writer — mirrors EmbedWriter logic."""
    def __init__(self, outdir: Path, rows_per_part: int = 50_000):
        self.outdir        = outdir
        self.rows_per_part = rows_per_part
        self._part         = 0
        self._part_rows    = 0
        self.total         = 0
        self._writer       = None
        self._current_path: Path | None = None
        self._parts: list[Path] = []
        self._open_part()

    def _open_part(self) -> None:
        self._current_path = self.outdir / f"kps_part{self._part:03d}.parquet"
        self._parts.append(self._current_path)
        self._writer    = pq.ParquetWriter(str(self._current_path),
                                           KPS_SCHEMA, compression="snappy")
        self._part_rows = 0

    def _roll(self) -> None:
        self._writer.close()
        pa.default_cpu_memory_pool().release_unused()
        self._part += 1
        self._open_part()

    def flush(self, rows: list) -> None:
        if not rows:
            return
        start = 0
        while start < len(rows):
            remaining_in_part = self.rows_per_part - self._part_rows
            chunk     = rows[start : start + remaining_in_part]
            kps_arr   = np.stack([r["kps"]       for r in chunk])
            kconf_arr = np.stack([r["kps_kconf"] for r in chunk])
            tbl = pa.table({
                "session_id":  pa.array([r["session_id"]  for r in chunk], pa.string()),
                "frame_index": pa.array([r["frame_index"] for r in chunk], pa.int32()),
                "temp_id":     pa.array([r["temp_id"]     for r in chunk], pa.int32()),
                "kps": pa.FixedSizeListArray.from_arrays(
                    pa.array(kps_arr.ravel().tolist(), pa.float32()), 57),
                "kps_kconf": pa.FixedSizeListArray.from_arrays(
                    pa.array(kconf_arr.ravel().tolist(), pa.float32()), 19),
            }, schema=KPS_SCHEMA)
            self._writer.write_table(tbl)
            self._part_rows += len(chunk)
            self.total      += len(chunk)
            start           += len(chunk)
            if self._part_rows >= self.rows_per_part and start < len(rows):
                self._roll()

    def close(self) -> None:
        self._writer.close()
        pa.default_cpu_memory_pool().release_unused()
        total_mb = sum(p.stat().st_size for p in self._parts if p.exists()) / 1e6
        log(f"kps     → {len(self._parts)} part(s), {self.total} rows, "
            f"{total_mb:.1f} MB total  [{self.outdir.name}]")



def _log_to_batch_log(dm, video_path: "Path", session_id: str,
                       status: str, frames: int, duration_s: float,
                       error: str = "") -> None:
    """Append one row to processing_log.csv on Drive (batch-mode, single flush)."""
    from drive_manager import LOCAL_UPLOAD_LOG, _flush_upload_log, BUFFER_DIR
    import drive_manager as _dm_mod

    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    local_log = BUFFER_DIR / "_batch_log_staging.csv"
    header_needed = not local_log.exists() or local_log.stat().st_size == 0
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fps_proc = f"{frames/max(duration_s,1e-6):.1f}" if frames else "0"
    row = (f"{ts},{session_id},{video_path.name},{status},"
           f"{frames},{duration_s:.1f},{fps_proc},{error}\n")
    with open(local_log, "a") as f:
        if header_needed:
            f.write("timestamp,session_id,filename,status,"
                    "frames,duration_s,fps_processed,error\n")
        f.write(row)


def _flush_batch_log(dm) -> None:
    """Push the local batch log staging file to Drive (one round-trip)."""
    import subprocess, drive_manager as _dm_mod
    from drive_manager import BUFFER_DIR, DRIVE_ROOT

    local_staging = BUFFER_DIR / "_batch_log_staging.csv"
    if not local_staging.exists() or local_staging.stat().st_size == 0:
        return

    drive_dest    = f"{DRIVE_ROOT}/{DRIVE_BATCH_LOG}"
    local_full    = BUFFER_DIR / "_batch_log_full.csv"
    header        = ("timestamp,session_id,filename,status,"
                     "frames,duration_s,fps_processed,error\n")
    try:
        subprocess.run(["rclone", "copyto", drive_dest, str(local_full)],
                       capture_output=True, timeout=30)
        if not local_full.exists() or local_full.stat().st_size == 0:
            with open(local_full, "w") as f:
                f.write(header)
        staged_lines = [l for l in local_staging.read_text().splitlines()
                        if l and not l.startswith("timestamp,")]
        if staged_lines:
            with open(local_full, "a") as f:
                f.write("\n".join(staged_lines) + "\n")
        subprocess.run(["rclone", "copyto", str(local_full), drive_dest],
                       capture_output=True, timeout=60)
        local_staging.write_text("")
        log(f"Batch log synced to Drive ({len(staged_lines)} new row(s))")
    except Exception as e:
        log(f"Batch log sync failed (non-fatal): {e}")


def process_video(video_path: "Path", args, dm, db_path: "Path",
                  det_model, embedder, pose_model, device: str,
                  stop_flag: "list") -> dict:
    """
    Process a single video file end-to-end:
      detect → track → embed → pose → flush → upload → reconcile

    stop_flag is a mutable 1-element list [False] shared with the SIGINT handler;
    set stop_flag[0] = True to request a graceful stop mid-video.

    Returns a result dict: {status, session_id, frames, duration_s, error}
    """
    result = {"status": "error", "session_id": "", "frames": 0,
              "duration_s": 0.0, "error": ""}

    # ── derive session metadata from filename ─────────────────────────────────
    camera_id, start_dt, end_dt = parse_filename(str(video_path))
    if start_dt is None:
        log(f"Warning: could not parse S<timestamp> from {video_path.name} "
            "— frame_datetime will be empty.")
    session_id = (f"{camera_id}_{start_dt.strftime('%Y%m%d%H%M%S')}"
                  if start_dt else f"{camera_id}_unknown")
    result["session_id"] = session_id

    outdir = dm.get_session_dir(session_id)

    # ── output dir safety check + clean ──────────────────────────────────────
    py_files = list(outdir.rglob("*.py"))
    if py_files:
        msg = (f"ERROR: .py files found in output dir for {session_id} "
               f"— refusing to clear it.")
        log(msg)
        result["error"] = msg
        return result

    if outdir.exists():
        import shutil
        for item in outdir.iterdir():
            (shutil.rmtree if item.is_dir() else item.unlink)(item if item.is_dir() else item)
        log(f"Cleared output directory: {outdir}")

    crops_dir = ensure_dir(outdir / "crops") if args.save_crops else None
    # Parquet writers use the session outdir; part files are named automatically.
    # embed_pq_path / kps_pq_path are gone — use embed_writer._parts / kps_writer._parts.

    dm.mark_dirty("parquet", session_id=session_id)
    dm.mark_dirty("db",      session_id=session_id)

    log("=" * 60)
    log(f"Processing: {video_path.name}")
    log(f"  session_id   : {session_id}")
    log(f"  camera_id    : {camera_id}")
    log(f"  start_dt     : {start_dt.isoformat() if start_dt else 'unknown'}")
    log(f"  end_dt       : {end_dt.isoformat()   if end_dt   else 'unknown'}")
    log(f"  outdir       : {outdir}")
    _save_every  = args.save_every
    _flush_every = args.flush_every
    log(f"  save_every   : {_save_every}  flush_every={_flush_every}  "
        f"(flush each {_save_every*_flush_every} frames)")
    if args.crop_tags:
        if _VISION_AVAILABLE:
            log("  crop_tags    : ON (posture+facing classifiers loaded)")
        else:
            log(f"  crop_tags    : ON but vision_features unavailable "
                f"({_VISION_IMPORT_ERROR}) — tags will be '_unk_unk'")

    conn = init_db(db_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Cannot open video: {video_path}"
        log(f"ERROR: {msg}")
        conn.close()
        result["error"] = msg
        return result

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    log(f"  video        : {W}x{H} @ {fps:.2f} fps  "
        f"frames~{total_frames or 'unknown'}")

    _epoch = start_dt
    register_session(conn, session_id, str(video_path),
                     camera_id,
                     start_dt.isoformat() if start_dt else "",
                     end_dt.isoformat()   if end_dt   else "")

    db_batch:   list = []
    embed_rows: list = []
    kps_rows:   list = []

    embed_writer = EmbedWriter(outdir)
    kps_writer   = KpsWriter(outdir)

    crop_occurrence    = defaultdict(int)
    warned_kp_mismatch = False
    embed_row_counter  = 0
    kps_row_counter    = 0
    windows_since_flush = 0
    frame_idx          = 0
    t_start            = time()

    _window_latest: dict = {}

    pbar = tqdm(total=total_frames or None,
                desc=f"Tracking {video_path.name}",
                unit="frame", dynamic_ncols=True)

    INSERT_SQL = """
        INSERT INTO raw_tracks
            (session_id, frame_index, frame_time_sec, frame_datetime,
             temp_id, det_conf, x1, y1, x2, y2, cx, cy, w, h,
             kps_conf, embed_parquet_row, kps_parquet_row)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    def _commit_window():
        nonlocal windows_since_flush, embed_row_counter, kps_row_counter
        for tid, det in _window_latest.items():
            ep_row = None
            if det["embed"] is not None:
                ep_row = embed_row_counter
                embed_rows.append({
                    "session_id":  session_id,
                    "frame_index": det["frame_index"],
                    "temp_id":     tid,
                    "embed":       det["embed"],
                })
                embed_row_counter += 1
            kp_row = None
            if det["kps_arr"] is not None:
                kp_row = kps_row_counter
                kps_rows.append({
                    "session_id":  session_id,
                    "frame_index": det["frame_index"],
                    "temp_id":     tid,
                    "kps":         det["kps_arr"],
                    "kps_kconf":   det["kconf_arr"],
                })
                kps_row_counter += 1
            db_batch.append((
                session_id, det["frame_index"], det["t_sec"], det["frame_datetime"],
                tid, det["conf"],
                det["x1"], det["y1"], det["x2"], det["y2"],
                det["cx"], det["cy"], det["w"], det["h"],
                det["kps_mean"], ep_row, kp_row,
            ))
        _window_latest.clear()
        windows_since_flush += 1
        if windows_since_flush >= _flush_every:
            if db_batch:
                conn.executemany(INSERT_SQL, db_batch)
                db_batch.clear()
            conn.commit()
            embed_writer.flush(embed_rows); embed_rows.clear()
            kps_writer.flush(kps_rows);     kps_rows.clear()
            pbar.set_postfix(frames=frame_idx,
                             fps=f"{frame_idx/max(time()-t_start,1e-6):.1f}")
            windows_since_flush = 0

    # ── main frame loop ───────────────────────────────────────────────────────
    while True:
        if stop_flag[0]:
            log(f"Stopping at frame {frame_idx} (graceful stop requested).")
            break
        ok, frame = cap.read()
        if not ok:
            break

        t_sec              = frame_idx / max(1e-6, fps)
        frame_datetime_str = ""
        if _epoch:
            frame_wall         = _epoch + timedelta(seconds=t_sec)
            frame_datetime_str = frame_wall.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        results = det_model.track(
            source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            tracker=args.tracker, persist=True, verbose=False,
        )

        if (not results or results[0].boxes is None
                or results[0].boxes.xyxy is None
                or results[0].boxes.id is None):
            frame_idx += 1; pbar.update(1)
            if frame_idx % _save_every == 0:
                _commit_window()
            continue

        r     = results[0]
        xyxy  = r.boxes.xyxy.cpu().numpy()
        confs = (r.boxes.conf.cpu().numpy() if r.boxes.conf is not None
                 else np.zeros(len(xyxy), dtype=np.float32))
        tids  = r.boxes.id.cpu().numpy().astype(int)
        crops, used_boxes = crops_from_bboxes(frame, xyxy, margin=0.10)

        embed_vecs = [None] * len(crops)
        if crops:
            X = torch.from_numpy(
                np.stack([to_tensor_bchw(c) for c in crops])
            ).to(device)
            with torch.no_grad():
                Z = embedder(X).cpu().numpy()
            embed_vecs = list(Z)

        kps_flat_list  = [None] * len(crops)
        kps_mean_list  = [None] * len(crops)
        kps_kconf_list = [None] * len(crops)

        if pose_model and crops:
            for i, pres in enumerate(pose_model.predict(
                    source=crops, imgsz=args.pose_imgsz,
                    conf=args.pose_conf, verbose=False, stream=False)):
                if (pres.keypoints is None or pres.keypoints.data is None
                        or not len(pres.keypoints.data)):
                    continue
                k  = pres.keypoints
                xy = k.xy[0].cpu().numpy()
                sc = k.conf[0].cpu().numpy() if k.conf is not None else None
                Kn = xy.shape[0]
                if EXPECTED_KP and Kn != EXPECTED_KP and not warned_kp_mismatch:
                    log(f"Warning: pose model returned {Kn} kps, expected {EXPECTED_KP}")
                    warned_kp_mismatch = True
                vis = (np.where(sc >= args.pose_kp_conf_thresh, 2, 1).astype(int)
                       if sc is not None else np.ones(Kn, dtype=int) * 2)
                cx1, cy1, cx2, cy2 = used_boxes[i]
                sx = max(1, cx2-cx1) / max(1, pres.orig_img.shape[1])
                sy = max(1, cy2-cy1) / max(1, pres.orig_img.shape[0])
                xy_full      = np.zeros((Kn,2), np.float32)
                xy_full[:,0] = cx1 + xy[:,0] * sx
                xy_full[:,1] = cy1 + xy[:,1] * sy
                kps_flat_list[i]  = flat_kps_xyv(
                    np.concatenate([xy_full, vis.reshape(-1,1)], axis=1))
                kps_mean_list[i]  = float(np.nanmean(sc)) if sc is not None else 0.0
                kps_kconf_list[i] = sc.tolist() if sc is not None else [0.0]*Kn

        for j, (box, tid, conf_j) in enumerate(zip(xyxy, tids, confs)):
            x1,y1,x2,y2 = box.tolist()
            cx_j = (x1+x2)/2.0; cy_j = (y1+y2)/2.0
            w_j  = x2-x1;       h_j  = y2-y1
            kps_mean = None; kps_arr = None; kconf_arr = None
            if kps_flat_list[j] is not None:
                kps_mean  = kps_mean_list[j]
                kps_arr   = np.resize(np.array(kps_flat_list[j],  np.float32), (57,))
                kconf_arr = np.resize(np.array(kps_kconf_list[j], np.float32), (19,))
            _window_latest[int(tid)] = {
                "frame_index":    frame_idx,
                "t_sec":          round(t_sec, 3),
                "frame_datetime": frame_datetime_str,
                "conf":           float(conf_j),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "cx": float(cx_j), "cy": float(cy_j),
                "w":  float(w_j),  "h":  float(h_j),
                "kps_mean":  kps_mean,
                "embed":     embed_vecs[j].astype(np.float32) if embed_vecs[j] is not None else None,
                "kps_arr":   kps_arr,
                "kconf_arr": kconf_arr,
            }

            if crops_dir is not None:
                tid_i = int(tid)
                crop_occurrence[tid_i] += 1
                if crop_occurrence[tid_i] % max(1, args.crop_every) == 0:
                    cx1,cy1,cx2,cy2 = used_boxes[j]
                    cw,ch = int(cx2-cx1), int(cy2-cy1)
                    mw,mh = args.min_crop_wh
                    if cw >= int(mw) and ch >= int(mh):
                        tag = ""
                        if args.crop_tags and _VISION_AVAILABLE and kps_arr is not None:
                            try:
                                _kps  = kps_arr.reshape(1, 19, 3)
                                _kc   = kconf_arr.reshape(1, 19)
                                _bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                                _det  = np.array([float(conf_j)], dtype=np.float32)
                                p_out = extract_posture(_kps, _kc, _bbox, _det)
                                f_out = extract_facing(_kps, _kc, _bbox,
                                                       posture=p_out["posture"])
                                posture_val = int(p_out["posture"][0])
                                facing_val  = int(f_out["facing"][0])
                                p_name = POSTURE_NAMES.get(Posture(posture_val), "unk")
                                f_name = FACING_NAMES.get(Facing(facing_val),   "unk")
                                p_tag = p_name if p_name != "uncertain" else "unk"
                                f_tag = f_name if f_name != "uncertain" else "unk"
                                tag = f"_{p_tag}_{f_tag}"
                            except Exception:
                                tag = "_unk_unk"
                        elif args.crop_tags:
                            tag = "_unk_unk"
                        cv2.imwrite(
                            str(crops_dir/f"{camera_id}_id{tid_i:04d}_f{frame_idx:06d}{tag}.jpg"),
                            frame[cy1:cy2, cx1:cx2])

        frame_idx += 1; pbar.update(1)
        if frame_idx % _save_every == 0:
            _commit_window()

    # ── flush & close ─────────────────────────────────────────────────────────
    pbar.close()
    if _window_latest:
        _commit_window()
    if db_batch:
        conn.executemany(INSERT_SQL, db_batch)
    if _epoch:
        conn.execute("UPDATE video_sessions SET end_dt=? WHERE session_id=?",
                     ((_epoch + timedelta(seconds=frame_idx/max(1e-6,fps))).isoformat(),
                      session_id))
    conn.commit()
    conn.close()
    cap.release()

    embed_writer.flush(embed_rows); embed_rows.clear()
    kps_writer.flush(kps_rows);     kps_rows.clear()
    embed_writer.close()
    kps_writer.close()

    t_total = time() - t_start
    log(f"Done. {frame_idx} frames in {t_total:.1f}s "
        f"({frame_idx/max(t_total,1e-6):.1f} fps processed)")
    if crops_dir:
        log(f"crops → {crops_dir}")

    # ── Upload outputs to Drive ───────────────────────────────────────────────
    log("Uploading session outputs to Drive...")
    dm.sync_db(session_id=session_id)
    for pq_path in sorted(outdir.glob("embeds_part*.parquet")) + \
                   sorted(outdir.glob("kps_part*.parquet")):
        if pq_path.exists():
            dm.write_file(pq_path, f"outputs/{session_id}/{pq_path.name}",
                          flag="parquet")

    if crops_dir and crops_dir.exists() and not args.crops_local:
        crop_files = sorted(crops_dir.glob("*.jpg"))
        if crop_files:
            log(f"Uploading {len(crop_files)} crop(s) to Drive...")
            from drive_manager import write_file as _wf, _flush_upload_log
            _wf._batch_mode = True
            try:
                for cf in crop_files:
                    _wf(cf, f"outputs/{session_id}/crops/{cf.name}",
                        caller=__file__)
            finally:
                _wf._batch_mode = False
            _flush_upload_log()
    elif crops_dir and args.crops_local:
        log(f"crops_local=True — crops kept locally only: {crops_dir}")

    # ── run reconcile.py ──────────────────────────────────────────────────────
    log("=" * 60)
    log("Starting reconcile.py ...")
    log("=" * 60)
    try:
        import importlib.util
        import argparse as _ap
        reconcile_path = Path(__file__).parent / "reconcile.py"
        if not reconcile_path.exists():
            raise FileNotFoundError(f"reconcile.py not found at {reconcile_path}")
        spec  = importlib.util.spec_from_file_location("reconcile", reconcile_path)
        r_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r_mod)
        r_args = _ap.Namespace(
            db                 = str(db_path),
            session            = session_id,
            kinetics           = args.kinetics,
            gallery_dir        = args.gallery_dir,
            embed_parquet      = str(outdir),   # directory — reconcile globs *_part*.parquet
            corr_threshold     = args.corr_threshold,
            min_active_bins    = 3,
            min_temp_id_frames = 0.10,
            activity_pct       = 0.25,
            bin_minutes        = 15,
            ema_alpha          = args.ema_alpha,
            min_embeds_gallery = 10,
            cosine_threshold   = args.cosine_threshold,
            cosine_min_embeds  = 5,
            dry_run            = False,
            verbose            = False,
        )
        r_mod.run(r_args)
    except Exception as exc:
        log(f"reconcile.py failed: {exc}")
        log("Tracking output is intact — run reconcile.py manually to retry.")

    result.update(status="ok" if not stop_flag[0] else "interrupted",
                  frames=frame_idx, duration_s=t_total)
    return result


def main():
    args = parse_args()

    # ── Drive setup ───────────────────────────────────────────────────────────
    dm = DriveManager(bypass=args.bypass_upload_check, caller=__file__)

    # ── Resolve source → list of video files ─────────────────────────────────
    source = Path(args.source).expanduser().resolve()
    if source.is_dir():
        video_files = sorted(
            f for f in source.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not video_files:
            print(f"[track] ERROR: no video files found in {source}")
            raise SystemExit(1)
        log(f"Batch mode: {len(video_files)} video(s) found in {source}")
        for vf in video_files:
            log(f"  {vf.name}")
    else:
        try:
            video_files = [dm.get_video_path(str(source))]
        except FileNotFoundError as e:
            print(f"[track] ERROR: {e}")
            raise SystemExit(1)

    # ── Pull canonical DB from Drive (once for the whole batch) ──────────────
    try:
        db_path = dm.pull_db(allow_stale=False)
    except DriveUnavailableError as e:
        print(f"[track] ERROR pulling DB from Drive: {e}")
        raise SystemExit(1)

    # ── Load models once ──────────────────────────────────────────────────────
    log(f"Loading models...")
    log(f"  detector     : {args.model}")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = YOLO(args.model); det_model.fuse()
    embedder  = Embedder128(pretrained=True, out_dim=args.embed_size).to(device).eval()
    pose_model = None
    if args.pose_model:
        log(f"  pose         : {args.pose_model}")
        pose_model = YOLO(args.pose_model); pose_model.fuse()
    log(f"  device       : {device}")

    # ── Graceful stop shared across all videos ────────────────────────────────
    stop_flag = [False]
    def _handle_sigint(sig, frame):
        if not stop_flag[0]:
            stop_flag[0] = True
            print("\n[track] Ctrl+C received — finishing current video then stopping...",
                  flush=True)
    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Process each video ────────────────────────────────────────────────────
    results_summary = []
    for i, video_path in enumerate(video_files):
        if stop_flag[0]:
            log(f"Batch stopped before {video_path.name} (Ctrl+C).")
            results_summary.append({
                "status": "skipped", "session_id": "", "frames": 0,
                "duration_s": 0.0, "error": "batch interrupted"
            })
            _log_to_batch_log(dm, video_path, "", "skipped", 0, 0.0,
                              "batch interrupted by Ctrl+C")
            continue

        log(f"\n{'='*60}")
        log(f"Video {i+1}/{len(video_files)}: {video_path.name}")
        log(f"{'='*60}")

        try:
            result = process_video(video_path, args, dm, db_path,
                                   det_model, embedder, pose_model,
                                   device, stop_flag)
        except Exception as exc:
            import traceback
            result = {"status": "error", "session_id": "", "frames": 0,
                      "duration_s": 0.0, "error": str(exc)}
            log(f"ERROR processing {video_path.name}: {exc}")
            traceback.print_exc()

        results_summary.append(result)
        _log_to_batch_log(dm, video_path,
                          result["session_id"], result["status"],
                          result["frames"],     result["duration_s"],
                          result.get("error", ""))

    # ── Flush batch log to Drive ──────────────────────────────────────────────
    _flush_batch_log(dm)

    # ── Print batch summary ───────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"Batch complete — {len(video_files)} video(s)")
    ok       = sum(1 for r in results_summary if r["status"] == "ok")
    stopped  = sum(1 for r in results_summary if r["status"] == "interrupted")
    skipped  = sum(1 for r in results_summary if r["status"] == "skipped")
    errors   = sum(1 for r in results_summary if r["status"] == "error")
    log(f"  ok={ok}  interrupted={stopped}  skipped={skipped}  errors={errors}")
    for vf, r in zip(video_files, results_summary):
        flag = {"ok":"✓","interrupted":"⚠","skipped":"–","error":"✗"}.get(r["status"],"?")
        log(f"  {flag} {vf.name}  [{r['status']}]"
            + (f"  {r['frames']} frames  {r['duration_s']:.0f}s" if r["frames"] else "")
            + (f"  ERROR: {r['error']}" if r.get("error") else ""))
    log(f"{'='*60}")


if __name__ == "__main__":
    main()



# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
#
# Paths
# DET=~/thesis_workspace/vcf-ctp/models/cow_detector/best.pt
# POSE=~/thesis_workspace/vcf-ctp/models/cow_pose/best.pt
#
# ── Single video ─────────────────────────────────────────────────────────────
# python3 track_and_dump.py \
#   --model      "$DET" \
#   --source     ~/thesis_workspace/raw_data/calving/refet_33_S20241221070000_E20241221080000.ts \
#   --pose_model "$POSE" \
#   --imgsz 960 --conf 0.30 --iou 0.60 \
#   --pose_imgsz 384 --pose_conf 0.25 \
#   --save_every 10 --flush_every 5
#
# ── Batch — all videos in a directory ────────────────────────────────────────
# python3 track_and_dump.py \
#   --model      "$DET" \
#   --source     ~/thesis_workspace/raw_data/calving/ \
#   --pose_model "$POSE" \
#   --imgsz 960 --conf 0.30 --iou 0.60 \
#   --pose_imgsz 384 --pose_conf 0.25 \
#   --save_every 10 --flush_every 5
#
# ── With crops ───────────────────────────────────────────────────────────────
# python3 track_and_dump.py ... \
#   --save_crops --crop_every 300 --min_crop_wh 100 100 --crop_tags
#   # add --crops_local to skip uploading crops to Drive
#
# ── Manual kinetics override (skip auto-discovery) ───────────────────────────
# python3 track_and_dump.py ... \
#   --kinetics ~/thesis_workspace/vcf-ctp/data/collar_data/kinetic_data_s...__6366_7507_7513.csv
#
# ── Re-run reconcile manually on an existing session ─────────────────────────
# SESSION=refet_33_20241221070000   # auto-derived from filename _S<timestamp>
# python3 reconcile.py \
#   --session       "$SESSION" \
#   --db            ~/thesis_workspace/vcf-ctp/data/calving_project.db \
#   --embed_parquet ~/thesis_workspace/vcf-ctp/data/outputs/$SESSION/embeds.parquet
#   # add --dry_run to test without writing
#   # add --kinetics /path/to/file.csv to override auto-discovery
#
# Notes:
#   Models are loaded once and reused across all videos in a batch.
#   session_id is auto-derived from each filename's _S<timestamp> token.
#   Kinetics/behavior CSVs are auto-discovered from Drive per video time window.
#   A processing_log.csv is maintained on Drive with one row per processed video.
#   Ctrl+C finishes the current video, logs it as 'interrupted', then stops.
#   Upload collar CSVs to Drive once before processing any sessions:
#     python3 drive_manager.py upload-kinetics ~/thesis_workspace/raw_data/CollarData/
#     python3 drive_manager.py upload-behavior ~/thesis_workspace/raw_data/CollarData/