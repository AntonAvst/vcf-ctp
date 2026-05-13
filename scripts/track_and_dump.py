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

import argparse, json, re, sqlite3
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

EXPECTED_KP = 19

_FNAME_RE = re.compile(r'_S(\d{14})')

def epoch_from_filename(video_path: str):
    m = _FNAME_RE.search(Path(video_path).stem)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None

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
    ap.add_argument("--source",     required=True, help="Video path")
    ap.add_argument("--outdir",     required=True, help="Output folder")
    ap.add_argument("--session_id", default="",
                    help="Unique session id. Defaults to video filename stem.")
    ap.add_argument("--tracker",    default="bytetrack.yaml")
    ap.add_argument("--camera_id",  default="cam0")
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
    ap.add_argument("--commit_every", type=int, default=50,
                    help="Commit SQLite transaction every N frames (default: 50). "
                         "Lower = more crash-safe, slightly more I/O.")
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

def register_session(conn, session_id, video_path, camera_id, start_dt):
    conn.execute(
        "INSERT OR IGNORE INTO video_sessions (session_id,video_path,camera_id,start_dt) VALUES (?,?,?,?)",
        (session_id, video_path, camera_id, start_dt)
    )
    conn.commit()


# ── Parquet writers ───────────────────────────────────────────────────────────
def write_embed_parquet(path: Path, rows: list) -> None:
    if not rows:
        return
    emb_arr = np.stack([r["embed"] for r in rows])
    tbl = pa.table({
        "session_id":  pa.array([r["session_id"]  for r in rows], pa.string()),
        "frame_index": pa.array([r["frame_index"] for r in rows], pa.int32()),
        "temp_id":     pa.array([r["temp_id"]     for r in rows], pa.int32()),
        "embed": pa.FixedSizeListArray.from_arrays(
            pa.array(emb_arr.ravel().tolist(), pa.float32()), 128),
    })
    pq.write_table(tbl, str(path), compression="snappy")
    log(f"embeds.parquet  → {path}  ({len(rows)} rows, {path.stat().st_size/1e6:.1f} MB)")

def write_kps_parquet(path: Path, rows: list) -> None:
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
    })
    pq.write_table(tbl, str(path), compression="snappy")
    log(f"kps.parquet     → {path}  ({len(rows)} rows, {path.stat().st_size/1e6:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    outdir        = ensure_dir(Path(args.outdir))
    crops_dir     = ensure_dir(outdir / "crops") if args.save_crops else None
    session_id    = args.session_id or Path(args.source).stem
    db_path       = outdir / "calving_project.db"
    embed_pq_path = outdir / "embeds.parquet"
    kps_pq_path   = outdir / "kps.parquet"

    log("Starting tracking")
    log(f"  model        : {args.model}")
    log(f"  source       : {args.source}")
    log(f"  session_id   : {session_id}")
    log(f"  outdir       : {outdir}")
    log(f"  db           : {db_path}")
    log(f"  commit_every : {args.commit_every} frames")
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

    _epoch = epoch_from_filename(args.source)
    log(f"Epoch from filename: {_epoch.isoformat()}" if _epoch
        else "Warning: could not parse epoch — frame_datetime will be empty.")
    register_session(conn, session_id, str(args.source),
                     args.camera_id, _epoch.isoformat() if _epoch else "")

    db_batch:   list = []
    embed_rows: list = []
    kps_rows:   list = []
    crop_occurrence       = defaultdict(int)
    warned_kp_mismatch    = False
    embed_row_counter     = 0
    kps_row_counter       = 0
    last_commit_frame     = 0
    frame_idx             = 0
    t_start               = time()

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

    def _maybe_commit():
        nonlocal last_commit_frame
        if frame_idx - last_commit_frame >= args.commit_every:
            if db_batch:
                conn.executemany(INSERT_SQL, db_batch)
                db_batch.clear()
            conn.commit()
            pbar.set_postfix(frames=frame_idx,
                             fps=f"{frame_idx/max(time()-t_start,1e-6):.1f}")
            last_commit_frame = frame_idx

    while True:
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
            frame_idx += 1; pbar.update(1); _maybe_commit(); continue

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

        # ── accumulate rows ───────────────────────────────────────────────────
        for j, (box, tid, conf_j) in enumerate(zip(xyxy, tids, confs)):
            x1,y1,x2,y2 = box.tolist()
            cx_j = (x1+x2)/2.0;  cy_j = (y1+y2)/2.0
            w_j  = x2-x1;        h_j  = y2-y1

            # embed
            ep_row = None
            if embed_vecs[j] is not None:
                ep_row = embed_row_counter
                embed_rows.append({
                    "session_id":  session_id,
                    "frame_index": frame_idx,
                    "temp_id":     int(tid),
                    "embed":       embed_vecs[j].astype(np.float32),
                })
                embed_row_counter += 1

            # kps
            kp_row   = None
            kps_mean = None
            if kps_flat_list[j] is not None:
                kp_row   = kps_row_counter
                kps_mean = kps_mean_list[j]
                kps_arr   = np.resize(np.array(kps_flat_list[j],  np.float32), (57,))
                kconf_arr = np.resize(np.array(kps_kconf_list[j], np.float32), (19,))
                kps_rows.append({
                    "session_id":  session_id,
                    "frame_index": frame_idx,
                    "temp_id":     int(tid),
                    "kps":         kps_arr,
                    "kps_kconf":   kconf_arr,
                })
                kps_row_counter += 1

            db_batch.append((
                session_id, frame_idx, round(t_sec,3), frame_datetime_str,
                int(tid), float(conf_j),
                float(x1), float(y1), float(x2), float(y2),
                float(cx_j), float(cy_j), float(w_j), float(h_j),
                kps_mean, ep_row, kp_row,
            ))

            # optional crop
            if crops_dir is not None:
                tid_i = int(tid)
                crop_occurrence[tid_i] += 1
                if crop_occurrence[tid_i] % max(1, args.crop_every) == 0:
                    cx1,cy1,cx2,cy2 = used_boxes[j]
                    cw,ch = int(cx2-cx1), int(cy2-cy1)
                    mw,mh = args.min_crop_wh
                    if cw >= int(mw) and ch >= int(mh):
                        cv2.imwrite(
                            str(crops_dir/f"{args.camera_id}_id{tid_i:04d}_f{frame_idx:06d}.jpg"),
                            frame[cy1:cy2, cx1:cx2])

        frame_idx += 1; pbar.update(1); _maybe_commit()

    # ── flush & close ─────────────────────────────────────────────────────────
    pbar.close()
    if db_batch:
        conn.executemany(INSERT_SQL, db_batch)
    if _epoch:
        conn.execute("UPDATE video_sessions SET end_dt=? WHERE session_id=?",
                     ((_epoch + timedelta(seconds=frame_idx/max(1e-6,fps))).isoformat(),
                      session_id))
    conn.commit()
    conn.close()
    cap.release()

    write_embed_parquet(embed_pq_path, embed_rows)
    write_kps_parquet(kps_pq_path, kps_rows)

    t_total = time() - t_start
    log(f"Done. {frame_idx} frames in {t_total:.1f}s "
        f"({frame_idx/max(t_total,1e-6):.1f} fps processed)")
    log(f"DB  → {db_path}")
    if crops_dir:
        log(f"crops → {crops_dir}")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Example
# ─────────────────────────────────────────────────────────────────────────────
# DET=/home/anton/thesis_workspace/vcf-ctp/models/cow_detector/best.pt
# POSE=/home/anton/thesis_workspace/vcf-ctp/models/cow_pose/best.pt
# VID=/home/anton/thesis_workspace/raw_data/calving/6558/refet_33_S20241221070000_E20241221080000_6558.mp4
# OUT=/home/anton/thesis_workspace/outputs/tracks/refet33_2024-12-21
#
# python3 track_and_dump.py \
#   --model      "$DET" \
#   --source     "$VID" \
#   --outdir     "$OUT" \
#   --session_id "refet33_20241221" \
#   --camera_id  "refet_33" \
#   --imgsz 960 --conf 0.30 --iou 0.60 \
#   --save_crops \
#   --pose_model "$POSE" --pose_imgsz 384 --pose_conf 0.25 \
#   --crop_every 100 --min_crop_wh 100 100 \
#   --commit_every 50
#
# Outputs in $OUT/:
#   calving_project.db   — SQLite (video_sessions + raw_tracks)
#   embeds.parquet       — embed[128] per detection
#   kps.parquet          — kps[57] + kps_kconf[19] per detection
#   crops/               — optional JPEGs