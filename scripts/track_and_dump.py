#!/usr/bin/env python3
"""
track_and_dump.py — detector + tracker (+ pose + appearance embedding)

Outputs:
  <outdir>/calving_project.db   — SQLite: video_sessions + raw_tracks (scalar columns)
  <outdir>/embeds.parquet       — embed[128] per detection (float32, Snappy)
  <outdir>/kps.parquet          — kps[19x3 flat=57] + kps_kconf[19] per detection
  <outdir>/crops/               — optional JPEG crops

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

EXPECTED_KP = 19

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
                    help="Video path (local file — videos are never uploaded to Drive)")
    ap.add_argument("--outdir",     required=True,
                    help="Output folder (ignored if drive_manager resolves session dir)")
    # session_id and camera_id are now derived from the filename automatically
    ap.add_argument("--tracker",    default="bytetrack.yaml")
    ap.add_argument("--imgsz",      type=int,   default=960)
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--iou",        type=float, default=0.45)
    ap.add_argument("--save_crops", action="store_true")
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
    ap.add_argument("--kinetics", required=True,
                    help="kinetic_data_*.csv — passed to reconcile.py after tracking finishes.")
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
    def __init__(self, path: Path):
        self.path    = path
        self._writer = pq.ParquetWriter(str(path), EMBED_SCHEMA, compression="snappy")
        self.total   = 0

    def flush(self, rows: list) -> None:
        if not rows:
            return
        emb_arr = np.stack([r["embed"] for r in rows])
        tbl = pa.table({
            "session_id":  pa.array([r["session_id"]  for r in rows], pa.string()),
            "frame_index": pa.array([r["frame_index"] for r in rows], pa.int32()),
            "temp_id":     pa.array([r["temp_id"]     for r in rows], pa.int32()),
            "embed": pa.FixedSizeListArray.from_arrays(
                pa.array(emb_arr.ravel().tolist(), pa.float32()), 128),
        }, schema=EMBED_SCHEMA)
        self._writer.write_table(tbl)
        self.total += len(rows)

    def close(self) -> None:
        self._writer.close()
        size = self.path.stat().st_size / 1e6 if self.path.exists() else 0
        log(f"embeds.parquet  -> {self.path}  ({self.total} rows, {size:.1f} MB)")


class KpsWriter:
    def __init__(self, path: Path):
        self.path    = path
        self._writer = pq.ParquetWriter(str(path), KPS_SCHEMA, compression="snappy")
        self.total   = 0

    def flush(self, rows: list) -> None:
        if not rows:
            return
        kps_arr   = np.stack([r["kps"]       for r in rows])
        kconf_arr = np.stack([r["kps_kconf"] for r in rows])
        tbl = pa.table({
            "session_id":  pa.array([r["session_id"]  for r in rows], pa.string()),
            "frame_index": pa.array([r["frame_index"] for r in rows], pa.int32()),
            "temp_id":     pa.array([r["temp_id"]     for r in rows], pa.int32()),
            "kps": pa.FixedSizeListArray.from_arrays(
                pa.array(kps_arr.ravel().tolist(), pa.float32()), 57),
            "kps_kconf": pa.FixedSizeListArray.from_arrays(
                pa.array(kconf_arr.ravel().tolist(), pa.float32()), 19),
        }, schema=KPS_SCHEMA)
        self._writer.write_table(tbl)
        self.total += len(rows)

    def close(self) -> None:
        self._writer.close()
        size = self.path.stat().st_size / 1e6 if self.path.exists() else 0
        log(f"kps.parquet     -> {self.path}  ({self.total} rows, {size:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Drive setup ───────────────────────────────────────────────────────────
    dm = DriveManager(bypass=args.bypass_upload_check, caller=__file__)

    # Validate video path (local only — pass-through)
    try:
        video_path = dm.get_video_path(args.source)
    except FileNotFoundError as e:
        print(f"[track] ERROR: {e}")
        raise SystemExit(1)

    # Pull canonical DB from Drive before writing anything
    try:
        db_path = dm.pull_db(allow_stale=False)
    except DriveUnavailableError as e:
        print(f"[track] ERROR pulling DB from Drive: {e}")
        raise SystemExit(1)

    # derive camera_id, start_dt, end_dt from filename
    camera_id, start_dt, end_dt = parse_filename(args.source)
    if start_dt is None:
        log("Warning: could not parse S<timestamp> from filename — frame_datetime will be empty.")
    session_id = f"{camera_id}_{start_dt.strftime('%Y%m%d%H%M%S') if start_dt else 'unknown'}"

    # Resolve output dir through drive_manager (local write buffer)
    outdir = dm.get_session_dir(session_id)

    # ── output dir safety check + clean ──────────────────────────────────────
    py_files = list(outdir.rglob("*.py"))
    if py_files:
        print(f"[track] ERROR: .py files found in output directory — refusing to clear it.")
        print(f"[track]   outdir : {outdir}")
        for f in py_files:
            print(f"[track]   {f.relative_to(outdir)}")
        print(f"[track] Move or remove these files manually, then re-run.")
        raise SystemExit(1)

    if outdir.exists():
        import shutil
        for item in outdir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        log(f"Cleared output directory: {outdir}")
    # ─────────────────────────────────────────────────────────────────────────

    crops_dir     = ensure_dir(outdir / "crops") if args.save_crops else None
    embed_pq_path = outdir / "embeds.parquet"
    kps_pq_path   = outdir / "kps.parquet"

    # Mark parquet files as dirty (local writes about to start)
    dm.mark_dirty("parquet", session_id=session_id)
    dm.mark_dirty("db",      session_id=session_id)

    log("Starting tracking")
    log(f"  model        : {args.model}")
    log(f"  source       : {args.source}")
    log(f"  camera_id    : {camera_id}")
    log(f"  start_dt     : {start_dt.isoformat() if start_dt else 'unknown'}")
    log(f"  end_dt       : {end_dt.isoformat() if end_dt else 'unknown'}")
    log(f"  session_id   : {session_id}")
    log(f"  outdir       : {outdir}")
    log(f"  db           : {db_path}")
    _save_every     = args.save_every                    # N-frame sampling window
    _flush_every    = args.flush_every                   # flush every M windows
    _flush_interval = _save_every * _flush_every         # absolute frame count between flushes
    log(f"  save_every   : {_save_every} frames/window  "
        f"flush_every={_flush_every} windows  "
        f"(flush each {_flush_interval} frames)")
    if args.pose_model:
        log(f"  pose_model   : {args.pose_model}")

    conn = init_db(db_path)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = YOLO(args.model); det_model.fuse()
    embedder  = Embedder128(pretrained=True, out_dim=args.embed_size).to(device).eval()
    pose_model = None
    if args.pose_model:
        pose_model = YOLO(args.pose_model); pose_model.fuse()

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    log(f"Video: {W}x{H} @ {fps:.2f} fps  frames~{total_frames or 'unknown'}")

    _epoch = start_dt  # already parsed from filename above
    register_session(conn, session_id, str(args.source),
                     camera_id,
                     start_dt.isoformat() if start_dt else "",
                     end_dt.isoformat()   if end_dt   else "")

    db_batch:   list = []
    embed_rows: list = []
    kps_rows:   list = []

    embed_writer = EmbedWriter(embed_pq_path)
    kps_writer   = KpsWriter(kps_pq_path)

    # ── graceful stop on Ctrl+C ───────────────────────────────────────────────
    _stop = False
    def _handle_sigint(sig, frame):
        nonlocal _stop
        if not _stop:
            _stop = True
            print("\n[track] Ctrl+C received — finishing current frame then saving...",
                  flush=True)
    signal.signal(signal.SIGINT, _handle_sigint)
    crop_occurrence       = defaultdict(int)
    warned_kp_mismatch    = False
    embed_row_counter     = 0
    kps_row_counter       = 0
    windows_since_flush   = 0   # counts completed N-frame windows since last flush
    frame_idx             = 0
    t_start               = time()

    # Holds the most recent detection per temp_id within the current N-frame window.
    # Overwritten on each new detection for that temp_id; committed at window boundary.
    _window_latest: dict = {}

    pbar = tqdm(total=total_frames or None,
                desc=f"Tracking {Path(args.source).name}",
                unit="frame", dynamic_ncols=True)

    INSERT_SQL = """
        INSERT INTO raw_tracks
            (session_id, frame_index, frame_time_sec, frame_datetime,
             temp_id, det_conf, x1, y1, x2, y2, cx, cy, w, h,
             kps_conf, embed_parquet_row, kps_parquet_row)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    def _commit_window():
        """Called at every N-frame window boundary.
        Moves the latest-per-tid rows into the db/embed/kps batches,
        then flushes to disk every M windows."""
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

    while True:
        if _stop:
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

        # ── embeddings ────────────────────────────────────────────────────────
        embed_vecs = [None] * len(crops)
        if crops:
            X = torch.from_numpy(
                np.stack([to_tensor_bchw(c) for c in crops])
            ).to(device)
            with torch.no_grad():
                Z = embedder(X).cpu().numpy()
            embed_vecs = list(Z)

        # ── pose ──────────────────────────────────────────────────────────────
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

                xy_full       = np.zeros((Kn,2), np.float32)
                xy_full[:,0]  = cx1 + xy[:,0] * sx
                xy_full[:,1]  = cy1 + xy[:,1] * sy

                xy_norm       = np.zeros_like(xy_full)
                xy_norm[:,0]  = xy_full[:,0] / W
                xy_norm[:,1]  = xy_full[:,1] / H

                kps_flat_list[i]  = flat_kps_xyv(
                    np.concatenate([xy_full, vis.reshape(-1,1)], axis=1))
                kps_mean_list[i]  = float(np.nanmean(sc)) if sc is not None else 0.0
                kps_kconf_list[i] = sc.tolist() if sc is not None else [0.0]*Kn

        # ── update window-latest per temp_id ─────────────────────────────────
        for j, (box, tid, conf_j) in enumerate(zip(xyxy, tids, confs)):
            x1,y1,x2,y2 = box.tolist()
            cx_j = (x1+x2)/2.0;  cy_j = (y1+y2)/2.0
            w_j  = x2-x1;        h_j  = y2-y1

            kps_mean = None
            kps_arr  = None
            kconf_arr = None
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

            # optional crop (unchanged — runs every detection, not gated by window)
            if crops_dir is not None:
                tid_i = int(tid)
                crop_occurrence[tid_i] += 1
                if crop_occurrence[tid_i] % max(1, args.crop_every) == 0:
                    cx1,cy1,cx2,cy2 = used_boxes[j]
                    cw,ch = int(cx2-cx1), int(cy2-cy1)
                    mw,mh = args.min_crop_wh
                    if cw >= int(mw) and ch >= int(mh):
                        cv2.imwrite(
                            str(crops_dir/f"{camera_id}_id{tid_i:04d}_f{frame_idx:06d}.jpg"),
                            frame[cy1:cy2, cx1:cx2])

        frame_idx += 1; pbar.update(1)
        # commit window at every N-th frame boundary
        if frame_idx % _save_every == 0:
            _commit_window()

    # ── flush & close ─────────────────────────────────────────────────────────
    pbar.close()
    # commit any partial window that didn't land on a boundary
    if _window_latest:
        _commit_window()
    if db_batch:
        conn.executemany(INSERT_SQL, db_batch)
    if _epoch:
        # Update end_dt with frame-accurate value (overrides filename-parsed end_dt)
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
    log(f"DB  → {db_path}")
    if crops_dir:
        log(f"crops → {crops_dir}")

    # ── Upload outputs to Drive ───────────────────────────────────────────────
    log("=" * 60)
    log("Uploading session outputs to Drive...")
    log("=" * 60)

    # DB snapshot
    dm.sync_db(session_id=session_id)

    # Parquet files
    for pq_path in [embed_pq_path, kps_pq_path]:
        if pq_path.exists():
            drive_rel = f"outputs/{session_id}/{pq_path.name}"
            dm.write_file(pq_path, drive_rel, flag="parquet")

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
            embed_parquet      = str(embed_pq_path),
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


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Example
# ─────────────────────────────────────────────────────────────────────────────
# DET=/home/anton/thesis_workspace/vcf-ctp/models/cow_detector/best.pt
# POSE=/home/anton/thesis_workspace/vcf-ctp/models/cow_pose/best.pt
# CALV=/home/anton/thesis_workspace/raw_data/calving/6558
# VID_NAME=refet_33_S20241221070000_E20241221080000_6558.mp4
# VID=$CALV/$VID_NAME
# OUT=/home/anton/thesis_workspace/outputs/tracks/refet33_2024-12-21
# COLLAR=/home/anton/thesis_workspace/raw_data/CollarData
# KIN=$COLLAR/kinetic_data_6558_7509_7774.csv
# BIH=$COLLAR/behavior_data_6558_7509_7774.csv
# DB=$OUT/calving_project.db
#
# python3 track_and_dump.py \
#   --model      "$DET" \
#   --source     "$VID" \
#   --outdir     "$OUT" \
#   --kinetics   "$KIN" \
#   --gallery_dir /home/anton/thesis_workspace/reid_gallery \
#   --imgsz 960 --conf 0.30 --iou 0.60 \
#   --pose_model "$POSE" --pose_imgsz 384 --pose_conf 0.25 \
#   --save_every 10    # snapshot latest detection per cow every N frames (default: 10)
#   --flush_every 5    # flush to SQLite + Parquet every M windows, i.e. N*M frames (default: 5)
#   --save_crops --crop_every 100 --min_crop_wh 100 100
#
# session_id is auto-derived from filename: refet_33_20241221070000
# Re-running the same file overwrites the previous session cleanly.
#
# reconcile.py runs automatically when tracking finishes (or on Ctrl+C).
# To run reconcile manually on an existing session:
#   python3 reconcile.py \
#     --db       "$OUT/calving_project.db" \
#     --session  "refet_33_20241221070000" \
#     --kinetics "$KIN" \
#     --gallery_dir /home/anton/thesis_workspace/reid_gallery
#
# Outputs in $OUT/:
#   calving_project.db   — SQLite (video_sessions + raw_tracks)
#   embeds.parquet       — embed[128], one row per cow per window
#   kps.parquet          — kps[57] + kps_kconf[19], one row per cow per window
#   crops/               — optional JPEGs