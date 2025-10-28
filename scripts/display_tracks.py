#!/usr/bin/env python3
import argparse, csv, json, os, sys, subprocess
from pathlib import Path
from time import time

import cv2
import numpy as np

def log(msg): print(f"[viewer] {msg}", flush=True)

# -------- Keyboard (non-blocking, terminal) ----------
class KB:
    def __enter__(self):
        import termios, tty
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, exc_type, exc, tb):
        import termios
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def getch(self):
        import select
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

# -------- Colors / drawing ----------
def id_color(tid: int) -> tuple:
    rng = np.random.default_rng(tid * 123457)
    c = rng.integers(64, 255, size=3, dtype=np.uint8).tolist()
    return int(c[0]), int(c[1]), int(c[2])

def draw_box(img, x1, y1, x2, y2, tid, conf=None):
    color = id_color(int(tid))
    x1, y1, x2, y2 = [int(round(float(v))) for v in (x1,y1,x2,y2)]
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    label = f"id {int(tid)}" + (f" {conf:.2f}" if conf is not None else "")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y0), (x1+tw+6, y0+th+6), color, -1)
    cv2.putText(img, label, (x1+3, y0+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)

# -------- Pose skeleton (18 KP layout) ----------
# Indices:
# 0 nose, 1 forehead, 2 withers, 3 spine_mid, 4 sacrum, 5 tail_base, 6 tail_tip,
# 7 shoulder_L, 8 shoulder_R, 9 elbow_L, 10 elbow_R, 11 fetlock_fore_L, 12 fetlock_fore_R,
# 13 hock_L, 14 hock_R, 15 fetlock_hind_L, 16 fetlock_hind_R, 17 udder_center
BASE_EDGES = [
    (0,1), (1,2), (2,3), (3,4), (4,5), (5,6),      # head → spine → tail
    (2,7), (7,9), (9,11),                          # left fore chain
    (2,8), (8,10), (10,12),                        # right fore chain (standard)
    (4,13), (13,15),                               # left hind chain (standard)
    (4,14), (14,16),                               # right hind chain (standard)
    (4,17),                                        # sacrum → udder_center
]
# Your custom tweaks to emphasize:
CUSTOM_EDGES = [
    (2,10), (10,12),  # 2 -> 10 -> 12
    (4,17),           # (already present, kept)
    (14,15),          # right hock -> left hind fetlock (as requested)
    (13,16),          # left hock -> right hind fetlock (as requested)
]
# Merge (avoid duplicates)
EDGE_SET = set(tuple(sorted(e)) for e in BASE_EDGES)
for e in CUSTOM_EDGES:
    EDGE_SET.add(tuple(sorted(e)))
EDGES = [(a,b) for (a,b) in EDGE_SET]

def draw_pose(img, kps_xyv, color, kp_radius=3, sk_thickness=2, kp_thresh=0.0):
    """
    kps_xyv: list/array of shape (K,3) with (x,y,v), where v in {0,1,2} (0=not labeled/ignored, 1/2=visible)
    Draw circles for v>0 and lines only if both endpoints v>0.
    """
    K = len(kps_xyv)
    # points
    for i in range(K):
        x,y,v = kps_xyv[i]
        if v is None: continue
        if float(v) > kp_thresh:
            cv2.circle(img, (int(round(x)), int(round(y))), kp_radius, color, -1, lineType=cv2.LINE_AA)

    # bones
    for (i,j) in EDGES:
        if i < K and j < K:
            xi, yi, vi = kps_xyv[i]
            xj, yj, vj = kps_xyv[j]
            if float(vi) > kp_thresh and float(vj) > kp_thresh:
                cv2.line(img,
                         (int(round(xi)), int(round(yi))),
                         (int(round(xj)), int(round(yj))),
                         color, sk_thickness, lineType=cv2.LINE_AA)

# -------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_fps", type=float, default=0.0)
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument("--sink", choices=["ffplay","cv2","mp4"], default="ffplay",
                    help="ffplay (no GUI deps), cv2 (needs Qt/X11), or mp4 (save to file)")
    ap.add_argument("--outmp4", default="annotated.mp4",
                    help="Output MP4 path if --sink mp4")
    ap.add_argument("--limit", type=int, default=0, help="Optional max frames to run")
    # pose drawing
    ap.add_argument("--draw_pose", action="store_true", help="Draw keypoints/skeleton if present in tracks")
    ap.add_argument("--kp_radius", type=int, default=3)
    ap.add_argument("--sk_thickness", type=int, default=2)
    ap.add_argument("--kp_thresh", type=float, default=0.0, help="Draw keypoints with v>kp_thresh")
    return ap.parse_args()

# -------- Track readers ----------
def sniff_csv_header_and_delim(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        except csv.Error:
            dialect = csv.get_dialect("excel")
        reader = csv.reader(f, dialect)
        header = next(reader)
    return header, dialect

def parse_kps_fields(row, col, W, H):
    """
    Return list of (x,y,v) in full-frame pixels if found, else None.
    Supports either:
      - kps: JSON list [x1,y1,v1,...] in full-frame pixels
      - kps_norm: JSON list [x1n,y1n,v1,...] (x,y) normalized to [0,1] by full frame size
    """
    def as_xyv_list(flat):
        if not flat or len(flat) % 3 != 0:
            return None
        out = []
        for i in range(0, len(flat), 3):
            x = float(flat[i]); y = float(flat[i+1]); v = float(flat[i+2])
            out.append((x,y,v))
        return out

    # prefer absolute pixels if available
    if "kps" in col:
        raw = row[col["kps"]]
        if raw and raw.strip():
            try:
                flat = json.loads(raw)
                return as_xyv_list(flat)
            except Exception:
                pass

    # fall back to normalized
    if "kps_norm" in col:
        raw = row[col["kps_norm"]]
        if raw and raw.strip():
            try:
                flat = json.loads(raw)
                if flat and len(flat) % 3 == 0:
                    out = []
                    for i in range(0, len(flat), 3):
                        xn = float(flat[i]); yn = float(flat[i+1]); v = float(flat[i+2])
                        out.append((xn*W, yn*H, v))
                    return out
            except Exception:
                pass
    return None

def stream_csv(path, start_frame=0, W=0, H=0, want_pose=False):
    header, dialect = sniff_csv_header_and_delim(path)
    col = {k:i for i,k in enumerate(header)}
    req = ["frame_index","temp_id","det_conf","x1","y1","x2","y2"]
    for k in req:
        if k not in col:
            raise ValueError(f"CSV missing column: {k}")
    # pose columns are optional
    cur_fi = None; bucket=[]
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, dialect)
        _ = next(reader)  # header
        for row in reader:
            fi = int(float(row[col["frame_index"]]))
            if fi < start_frame:
                continue
            if cur_fi is None:
                cur_fi = fi
            if fi != cur_fi:
                yield cur_fi, bucket
                bucket = []; cur_fi = fi

            d = {
                "temp_id": int(float(row[col["temp_id"]])) if row[col["temp_id"]] != "" else -1,
                "conf": float(row[col["det_conf"]]) if row[col["det_conf"]] != "" else 0.0,
                "x1": float(row[col["x1"]]), "y1": float(row[col["y1"]]),
                "x2": float(row[col["x2"]]), "y2": float(row[col["y2"]]),
            }
            if want_pose:
                kps = parse_kps_fields(row, col, W, H)
                if kps is not None:
                    d["kps"] = kps
            bucket.append(d)
    if bucket:
        yield cur_fi, bucket

def stream_jsonl(path, start_frame=0, W=0, H=0, want_pose=False):
    cur_fi=None; bucket=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            fi = int(o.get("frame_index",0))
            if fi < start_frame: continue
            if cur_fi is None: cur_fi = fi
            if fi != cur_fi:
                yield cur_fi, bucket
                bucket=[]; cur_fi=fi
            d = {
                "temp_id": int(o.get("temp_id",-1)),
                "conf": float(o.get("det_conf",0.0)),
                "x1": float(o.get("x1",0)), "y1": float(o.get("y1",0)),
                "x2": float(o.get("x2",0)), "y2": float(o.get("y2",0)),
            }
            if want_pose:
                # prefer pixel kps
                kps = None
                if "kps" in o and isinstance(o["kps"], list) and len(o["kps"])%3==0:
                    flat = o["kps"]; kps=[]
                    for i in range(0,len(flat),3):
                        kps.append((float(flat[i]), float(flat[i+1]), float(flat[i+2])))
                elif "kps_norm" in o and isinstance(o["kps_norm"], list) and len(o["kps_norm"])%3==0:
                    flat = o["kps_norm"]; kps=[]
                    for i in range(0,len(flat),3):
                        kps.append((float(flat[i])*W, float(flat[i+1])*H, float(flat[i+2])))
                if kps is not None:
                    d["kps"] = kps
            bucket.append(d)
    if bucket: yield cur_fi, bucket

# -------- Main ----------
def main():
    args = parse_args()
    vid = Path(args.video); trk = Path(args.tracks)
    if not vid.exists(): raise FileNotFoundError(f"Video not found: {vid}")
    if not trk.exists(): raise FileNotFoundError(f"Tracks not found: {trk}")

    log(f"Opening video: {vid}")
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {vid}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"Video info: {W}x{H} @ {fps:.2f} fps")

    log(f"Opening tracks: {trk.name}")
    is_csv = trk.suffix.lower()==".csv"
    row_stream = (stream_csv(str(trk), args.start, W, H, args.draw_pose)
                  if is_csv else
                  stream_jsonl(str(trk), args.start, W, H, args.draw_pose))

    # prime stream
    try:
        next_fi, next_rows = next(row_stream)
        log(f"First tracks frame_index: {next_fi}  rows: {len(next_rows)}")
    except StopIteration:
        log("Tracks contain no rows >= start frame. Exiting.")
        return

    # start pos
    if args.start>0: cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # sinks
    writer = None; ffplay_proc=None
    if args.sink=="mp4":
        log(f"Writing annotated MP4 to: {args.outmp4}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.outmp4, fourcc, fps, (W,H))
    elif args.sink=="ffplay":
        disp_fps = fps if args.max_fps<=0 else min(fps, args.max_fps)
        cmd = ["ffplay","-loglevel","error","-fflags","nobuffer",
               "-f","rawvideo","-pixel_format","bgr24","-video_size",f"{W}x{H}",
               "-framerate",f"{disp_fps}","-"]
        try:
            ffplay_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            log("ffplay started.")
        except FileNotFoundError:
            log("ffplay not found. Falling back to MP4 writer.")
            args.sink="mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.outmp4, fourcc, fps, (W,H))

    paused=False; prev_t=time(); n_frames=0
    log("Press 'p' to pause, 'q' to quit.")

    # terminal keyboard
    with KB() as kb:
        while True:
            ch = kb.getch()
            if ch=='q': log("Quit requested."); break
            if ch=='p': paused=not paused; log(f"{'Paused' if paused else 'Resumed'}.")

            if paused:
                if args.sink=="cv2": cv2.waitKey(1)
                continue

            ok, frame = cap.read()
            if not ok:
                log("End of video."); break

            # step the track stream to current frame
            while next_fi < frame_idx:
                try: next_fi, next_rows = next(row_stream)
                except StopIteration:
                    next_rows=[]; next_fi=frame_idx; break

            if next_fi == frame_idx:
                dets = next_rows
                try: next_fi, next_rows = next(row_stream)
                except StopIteration:
                    next_rows=[]; next_fi=10**12
            else:
                dets = []

            # draw
            for d in dets:
                draw_box(frame, d["x1"], d["y1"], d["x2"], d["y2"], d["temp_id"], d["conf"])
                if args.draw_pose and ("kps" in d) and d["kps"]:
                    draw_pose(frame, d["kps"],
                              color=id_color(int(d["temp_id"])),
                              kp_radius=args.kp_radius,
                              sk_thickness=args.sk_thickness,
                              kp_thresh=args.kp_thresh)

            if args.show_fps:
                now=time(); fps_now = 1.0/max(1e-6, now-prev_t); prev_t=now
                cv2.putText(frame, f"{fps_now:5.1f} FPS", (12,28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, f"{fps_now:5.1f} FPS", (12,28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

            # sink
            if args.sink=="ffplay":
                try:
                    ffplay_proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, AttributeError):
                    log("ffplay closed. Stopping."); break
            elif args.sink=="cv2":
                cv2.imshow("tracks", frame)
                if args.max_fps>0: cv2.waitKey(int(1000/args.max_fps))
            else:  # mp4
                writer.write(frame)

            frame_idx += 1; n_frames += 1
            if args.limit>0 and n_frames>=args.limit:
                log("Limit reached."); break

    cap.release()
    if args.sink=="cv2":
        cv2.destroyAllWindows()
    elif args.sink=="ffplay":
        if ffplay_proc and ffplay_proc.stdin:
            ffplay_proc.stdin.close()
        if ffplay_proc: ffplay_proc.terminate()
    else:
        if writer: writer.release()
    log("Done.")

if __name__=="__main__":
    main()


# python display_tracks.py \
#   --video "/home/anton/thesis_workspace/raw_data/calving/6558/refet_33_S20241221070000_E20241221080000_6558.mp4" \
#   --tracks "/home/anton/thesis_workspace/outputs/tracks/refet33_2024-12-21_pose/tracks.csv" \
#   --draw_pose --show_fps --sink ffplay
