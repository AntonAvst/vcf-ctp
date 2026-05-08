#!/usr/bin/env python3
import argparse, csv, json, os, sys, subprocess
from pathlib import Path
from time import time

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))

import cv2
import numpy as np

try:
    import match_identity as mi
    _MI_AVAILABLE = True
    print(f"[DEBUG] match_identity import ok", flush=True)
except Exception as e:
    _MI_AVAILABLE = False
    print(f"[DEBUG] match_identity import FAILED!!!! :( exception - {e}) ", flush=True)


def log(msg):
    print(f"[viewer] {msg}", flush=True)


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


def draw_box(img, x1, y1, x2, y2, tid, conf=None, animal_id=None):
    color = id_color(int(tid))
    x1, y1, x2, y2 = [int(round(float(v))) for v in (x1, y1, x2, y2)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if animal_id is not None:
        label = f"cow {animal_id} (t{int(tid)})"
    else:
        label = f"id {int(tid)}" + (f" {conf:.2f}" if conf is not None else "")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y0 + th + 6), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 3, y0 + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


# -------- Pose skeleton (19 KP layout) ----------
# Indices:
# 0 nose
# 1 forehead
# 2 withers
# 3 spine_mid
# 4 sacrum
# 5 tail_base
# 6 tail_tip
# 7 shoulder_L
# 8 elbow_L
# 9 fetlock_fore_L
# 10 shoulder_R
# 11 elbow_R
# 12 fetlock_fore_R
# 13 hock_R
# 14 hock_L
# 15 fetlock_hind_L
# 16 fetlock_hind_R
# 17 udder_center
# 18 neck
#
# (If your YAML uses a slightly different order, circles will still be drawn
# from the coordinates; these edges just define how the skeleton is connected.)
BASE_EDGES = [
    (0, 1),
    (1, 18),  # forehead -> neck
    (18, 2),  # neck -> withers
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),  # head → spine → tail
    # fore limbs
    (2, 7),
    (7, 8),
    (8, 9),   # left fore chain
    (2, 10),
    (10, 11),
    (11, 12), # right fore chain
    # hind limbs (note the R/L order in this dataset)
    (4, 14),
    (14, 15),  # left hind chain
    (4, 13),
    (13, 16),  # right hind chain
    (4, 17),   # sacrum → udder_center
]

# Extra custom edges you wanted to emphasize
CUSTOM_EDGES = [
    (2, 10),
    (10, 12),  # withers-ish to right fore chain emphasis
    (4, 17),
    (14, 15),
    (13, 16),
]

EDGE_SET = set(tuple(sorted(e)) for e in BASE_EDGES)
for e in CUSTOM_EDGES:
    EDGE_SET.add(tuple(sorted(e)))
EDGES = [(a, b) for (a, b) in EDGE_SET]


def draw_pose(
    img,
    kps_xyv,
    color,
    kps_conf=None,
    kp_radius=3,
    sk_thickness=2,
    kp_thresh=0.0,
    kp_conf_thresh=0.30,
    hide_lowconf=False,
    show_index=False,
    index_scale=0.45,
    index_thickness=1,
    index_offset=6,
):
    """
    kps_xyv: list/array of shape (K,3) with (x,y,v), where v in {0,1,2}
             (0=not labeled/ignored, 1/2=labeled; typically 2=visible, 1=occluded)
    Draw circles for v > kp_thresh and lines only if both endpoints v > kp_thresh.
    Optionally draw a numeric index next to each visible keypoint.
    """
    K = len(kps_xyv)

    # Decide which points to draw
    draw_mask = [False] * K
    for i in range(K):
        x, y, v = kps_xyv[i]
        if v is None:
            continue
        if float(v) <= kp_thresh:
            continue
        if hide_lowconf and kps_conf is not None and i < len(kps_conf):
            try:
                if float(kps_conf[i]) < float(kp_conf_thresh):
                    continue
            except Exception:
                pass
        draw_mask[i] = True

    # draw keypoints
    for i in range(K):
        if not draw_mask[i]:
            continue
        x, y, _v = kps_xyv[i]
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(img, (xi, yi), kp_radius, color, -1, lineType=cv2.LINE_AA)

        if show_index:
            idx_txt = str(i)
            # outline for readability
            cv2.putText(
                img,
                idx_txt,
                (xi + index_offset, yi - index_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                index_scale,
                (0, 0, 0),
                index_thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                idx_txt,
                (xi + index_offset, yi - index_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                index_scale,
                (255, 255, 255),
                index_thickness,
                cv2.LINE_AA,
            )

    # draw bones
    for (i, j) in EDGES:
        if i < K and j < K and draw_mask[i] and draw_mask[j]:
            xi, yi, _vi = kps_xyv[i]
            xj, yj, _vj = kps_xyv[j]
            cv2.line(
                img,
                (int(round(xi)), int(round(yi))),
                (int(round(xj)), int(round(yj))),
                color,
                sk_thickness,
                lineType=cv2.LINE_AA,
            )


# -------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--kinetics", default="",
        help="Path to kinetic_data_*.csv. If provided, identity matching runs "
             "before playback and labels are shown as AnimalId instead of temp_id.")
    ap.add_argument("--corr_threshold", type=float, default=0.7,
        help="Min Pearson r to accept a temp_id->AnimalId match (default: 0.7)")
    ap.add_argument("--min_active_bins", type=int, default=1,
        help="Min active kinetics bins required for a match (default: 1 for live mode)")
    ap.add_argument("--bin_minutes", type=int, default=15,
        help="Kinetics bin width in minutes (default: 15)")
    ap.add_argument("--min_temp_id_frames", type=float, default=0.10,
        help="Min fraction of frames a temp_id must appear to be considered real (default: 0.10)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_fps", type=float, default=0.0)
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument(
        "--sink",
        choices=["ffplay", "cv2", "mp4"],
        default="ffplay",
        help="ffplay (no GUI deps), cv2 (needs Qt/X11), or mp4 (save to file)",
    )
    ap.add_argument(
        "--outmp4",
        default="annotated.mp4",
        help="Output MP4 path if --sink mp4",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max frames to run",
    )

    # pose drawing
    ap.add_argument(
        "--draw_pose",
        action="store_true",
        help="Draw keypoints/skeleton if present in tracks",
    )
    ap.add_argument("--kp_radius", type=int, default=3)
    ap.add_argument("--sk_thickness", type=int, default=2)
    ap.add_argument(
        "--kp_thresh",
        type=float,
        default=0.0,
        help="Base threshold on v (0/1/2); draw keypoints with v > kp_thresh",
    )

    # new: hide occluded / low visibility joints (v == 1)
    ap.add_argument(
        "--hide_occluded",
        action="store_true",
        help="If set, draw only joints with v >= 2 (visible only).",
    )

    # new: optionally hide low-confidence keypoints (uses per-keypoint conf if available)
    ap.add_argument(
        "--kp_conf_thresh",
        type=float,
        default=0.30,
        help="If per-keypoint confidences are available, hide keypoints with conf < this threshold (unless --show_lowconf).",
    )
    ap.add_argument(
        "--show_lowconf",
        action="store_true",
        help="If set, draw low-confidence keypoints too (conf < --kp_conf_thresh).",
    )

    # keypoint index overlay
    ap.add_argument(
        "--kp_index",
        action="store_true",
        help="Overlay keypoint indices (0..K-1) next to points",
    )
    ap.add_argument(
        "--kp_index_scale",
        type=float,
        default=0.45,
        help="Font scale for keypoint indices",
    )
    ap.add_argument(
        "--kp_index_thickness",
        type=int,
        default=1,
        help="Font thickness for keypoint indices",
    )
    ap.add_argument(
        "--kp_index_offset",
        type=int,
        default=6,
        help="Pixel offset of index text from the keypoint",
    )

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
    Return (kps_xyv, kps_conf) where:
      - kps_xyv is list of (x,y,v) in full-frame pixels
      - kps_conf is list of per-keypoint confidences (floats) or None
    If keypoints are not found, returns (None, None).
    Supports either:
      - kps: JSON list [x1,y1,v1,...] in full-frame pixels
      - kps_norm: JSON list [x1n,y1n,v1,...] (x,y) normalized to [0,1] by full frame size
    """

    def as_xyv_list(flat):
        if not flat or len(flat) % 3 != 0:
            return None
        out = []
        for i in range(0, len(flat), 3):
            x = float(flat[i])
            y = float(flat[i + 1])
            v = float(flat[i + 2])
            out.append((x, y, v))
        return out

    kps_xyv = None
    kps_conf = None

    # per-keypoint confidence list (if present)
    # track_and_dump writes this as 'kps_kconf'
    if "kps_kconf" in col:
        rawc = row[col["kps_kconf"]]
        if rawc and rawc.strip():
            try:
                kps_conf = json.loads(rawc)
            except Exception:
                kps_conf = None

    # prefer absolute pixels if available
    if "kps" in col:
        raw = row[col["kps"]]
        if raw and raw.strip():
            try:
                flat = json.loads(raw)
                kps_xyv = as_xyv_list(flat)
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
                        xn = float(flat[i])
                        yn = float(flat[i + 1])
                        v = float(flat[i + 2])
                        out.append((xn * W, yn * H, v))
                    kps_xyv = out
            except Exception:
                pass

    return kps_xyv, kps_conf


def stream_csv(path, start_frame=0, W=0, H=0, want_pose=False):
    header, dialect = sniff_csv_header_and_delim(path)
    col = {k: i for i, k in enumerate(header)}
    req = ["frame_index", "temp_id", "det_conf", "x1", "y1", "x2", "y2"]
    for k in req:
        if k not in col:
            raise ValueError(f"CSV missing column: {k}")

    cur_fi = None
    bucket = []

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
                bucket = []
                cur_fi = fi

            d = {
                "temp_id": int(float(row[col["temp_id"]]))
                if row[col["temp_id"]] != ""
                else -1,
                "conf": float(row[col["det_conf"]])
                if row[col["det_conf"]] != ""
                else 0.0,
                "x1": float(row[col["x1"]]),
                "y1": float(row[col["y1"]]),
                "x2": float(row[col["x2"]]),
                "y2": float(row[col["y2"]]),
            }
            if want_pose:
                kps, kconf = parse_kps_fields(row, col, W, H)
                if kps is not None:
                    d["kps"] = kps
                if kconf is not None:
                    d["kps_conf"] = kconf
            bucket.append(d)

    if bucket:
        yield cur_fi, bucket


def stream_jsonl(path, start_frame=0, W=0, H=0, want_pose=False):
    cur_fi = None
    bucket = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            fi = int(o.get("frame_index", 0))
            if fi < start_frame:
                continue
            if cur_fi is None:
                cur_fi = fi
            if fi != cur_fi:
                yield cur_fi, bucket
                bucket = []
                cur_fi = fi

            d = {
                "temp_id": int(o.get("temp_id", -1)),
                "conf": float(o.get("det_conf", 0.0)),
                "x1": float(o.get("x1", 0)),
                "y1": float(o.get("y1", 0)),
                "x2": float(o.get("x2", 0)),
                "y2": float(o.get("y2", 0)),
            }

            if want_pose:
                kps = None
                if (
                    "kps" in o
                    and isinstance(o["kps"], list)
                    and len(o["kps"]) % 3 == 0
                ):
                    flat = o["kps"]
                    kps = []
                    for i in range(0, len(flat), 3):
                        kps.append(
                            (
                                float(flat[i]),
                                float(flat[i + 1]),
                                float(flat[i + 2]),
                            )
                        )
                elif (
                    "kps_norm" in o
                    and isinstance(o["kps_norm"], list)
                    and len(o["kps_norm"]) % 3 == 0
                ):
                    flat = o["kps_norm"]
                    kps = []
                    for i in range(0, len(flat), 3):
                        kps.append(
                            (
                                float(flat[i]) * W,
                                float(flat[i + 1]) * H,
                                float(flat[i + 2]),
                            )
                        )

                if kps is not None:
                    d["kps"] = kps

                # optional per-kp confidences (emitted by track_and_dump.py)
                if "kps_kconf" in o and isinstance(o["kps_kconf"], list):
                    try:
                        d["kps_conf"] = [float(v) for v in o["kps_kconf"]]
                    except Exception:
                        pass

            bucket.append(d)

    if bucket:
        yield cur_fi, bucket



# -------- Match score table overlay ----------
def draw_score_table(img, scores_df, assignment, margin=10):
    """
    Draw a semi-transparent score table (temp_id x AnimalId) in the
    bottom-left corner of the frame.
    Green cell = confirmed assignment, yellow = high score but unassigned,
    gray = low/no score.
    """
    if scores_df is None or scores_df.empty:
        return

    import pandas as pd
    pivot = scores_df.pivot_table(
        index="temp_id", columns="AnimalId", values="correlation", aggfunc="first"
    )

    tids     = list(pivot.index)
    aids     = list(pivot.columns)
    n_rows   = len(tids) + 1   # +1 header
    n_cols   = len(aids) + 1   # +1 row header

    cell_w, cell_h = 90, 22
    table_w = n_cols * cell_w
    table_h = n_rows * cell_h

    H, W = img.shape[:2]
    x0 = margin
    y0 = H - table_h - margin

    # semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4),
                  (x0 + table_w + 4, y0 + table_h + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness  = 1

    def draw_cell(row, col, text, bg, fg=(255, 255, 255)):
        cx = x0 + col * cell_w
        cy = y0 + row * cell_h
        cv2.rectangle(img, (cx + 1, cy + 1),
                      (cx + cell_w - 1, cy + cell_h - 1), bg, -1)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = cx + (cell_w - tw) // 2
        ty = cy + (cell_h + th) // 2 - 1
        cv2.putText(img, text, (tx, ty), font, font_scale, fg, thickness, cv2.LINE_AA)

    # header row
    draw_cell(0, 0, "tid\aid", (40, 40, 40))
    for ci, aid in enumerate(aids):
        draw_cell(0, ci + 1, str(aid), (40, 40, 40))

    # data rows
    for ri, tid in enumerate(tids):
        draw_cell(ri + 1, 0, f"t{tid}", (40, 40, 40))
        for ci, aid in enumerate(aids):
            val = pivot.loc[tid, aid] if (tid in pivot.index and aid in pivot.columns) else float("nan")
            assigned = assignment.get(tid) == aid
            if assigned:
                bg = (0, 120, 0)      # green — confirmed match
            elif not pd.isna(val) and val >= 0.5:
                bg = (0, 100, 150)    # blue — strong but unassigned
            else:
                bg = (60, 60, 60)     # gray — weak / no data
            txt = f"{val:.2f}" if not pd.isna(val) else "—"
            draw_cell(ri + 1, ci + 1, txt, bg)

    # legend line
    ly = y0 - 8
    cv2.putText(img, "green=matched  blue=strong  gray=weak",
                (x0, ly), font, 0.32, (180, 180, 180), 1, cv2.LINE_AA)


# -------- Main ----------
def main():
    args = parse_args()
    vid = Path(args.video)
    trk = Path(args.tracks)
    if not vid.exists():
        raise FileNotFoundError(f"Video not found: {vid}")
    if not trk.exists():
        raise FileNotFoundError(f"Tracks not found: {trk}")

    # ---- live identity matching state ----
    # Scores and assignment are recomputed each time a new kinetics interval boundary
    # is crossed during playback, creating the illusion of live matching.
    assignment    = {}     # {temp_id -> AnimalId}  — updated each interval
    scores_df     = None   # full score DataFrame    — updated each interval
    _last_bin     = None   # last kinetics bin boundary we computed at
    _tracks_df    = None   # full tracks dataframe cached for incremental scoring
    _kinetics_df  = None   # kinetics dataframe cached

    _live_matching = False
    if args.kinetics:
        if not _MI_AVAILABLE:
            log("WARNING: match_identity.py not on PYTHONPATH — skipping identity matching.")
        else:
            kin_path = Path(args.kinetics)
            if not kin_path.exists():
                log(f"WARNING: kinetics file not found: {kin_path} — skipping.")
            else:
                import pandas as _pd
                log(f"Loading tracks and kinetics for live matching...")
                _tracks_df   = _pd.read_csv(str(trk), parse_dates=["frame_datetime"])
                _kinetics_df = _pd.read_csv(str(kin_path), parse_dates=["datetime"])
                _live_matching = True
                log(f"Live matching ready. Scores will update every {args.bin_minutes}-min interval.")
                log(f"  corr_threshold={args.corr_threshold}  min_active_bins={args.min_active_bins}")

    log(f"Opening video: {vid}")
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vid}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"Video info: {W}x{H} @ {fps:.2f} fps")

    log(f"Opening tracks: {trk.name}")
    is_csv = trk.suffix.lower() == ".csv"
    row_stream = (
        stream_csv(str(trk), args.start, W, H, args.draw_pose)
        if is_csv
        else stream_jsonl(str(trk), args.start, W, H, args.draw_pose)
    )

    # prime stream
    try:
        next_fi, next_rows = next(row_stream)
        log(f"First tracks frame_index: {next_fi}  rows: {len(next_rows)}")
    except StopIteration:
        log("Tracks contain no rows >= start frame. Exiting.")
        return

    # start position in video
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # sinks
    writer = None
    ffplay_proc = None
    if args.sink == "mp4":
        log(f"Writing annotated MP4 to: {args.outmp4}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.outmp4, fourcc, fps, (W, H))
    elif args.sink == "ffplay":
        disp_fps = fps if args.max_fps <= 0 else min(fps, args.max_fps)
        cmd = [
            "ffplay",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            f"{disp_fps}",
            "-",
        ]
        try:
            ffplay_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            log("ffplay started.")
        except FileNotFoundError:
            log("ffplay not found. Falling back to MP4 writer.")
            args.sink = "mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.outmp4, fourcc, fps, (W, H))

    paused = False
    show_table = True   # toggle with 't'
    prev_t = time()
    n_frames = 0
    log("Press 'p' to pause, 't' to toggle score table, 'q' to quit.")

    # decide effective kp threshold (v is 0/1/2)
    kp_thresh_effective = args.kp_thresh
    if args.hide_occluded:
        # draw only v >= 2
        kp_thresh_effective = max(kp_thresh_effective, 0.5)

    with KB() as kb:
        while True:
            ch = kb.getch()
            if ch == "q":
                log("Quit requested.")
                break
            if ch == "p":
                paused = not paused
                log("Paused." if paused else "Resumed.")
            if ch == "t":
                show_table = not show_table
                log("Score table: " + ("visible" if show_table else "hidden"))

            if paused:
                if args.sink == "cv2":
                    cv2.waitKey(1)
                continue

            ok, frame = cap.read()
            if not ok:
                log("End of video.")
                break

            # advance track stream to current frame
            while next_fi < frame_idx:
                try:
                    next_fi, next_rows = next(row_stream)
                except StopIteration:
                    next_rows = []
                    next_fi = frame_idx
                    break

            if next_fi == frame_idx:
                dets = next_rows
                try:
                    next_fi, next_rows = next(row_stream)
                except StopIteration:
                    next_rows = []
                    next_fi = 10**12
            else:
                dets = []

            # draw detections + pose
            for d in dets:
                draw_box(
                    frame,
                    d["x1"],
                    d["y1"],
                    d["x2"],
                    d["y2"],
                    d["temp_id"],
                    d["conf"],
                    animal_id=assignment.get(int(d["temp_id"])),
                )
                if args.draw_pose and ("kps" in d) and d["kps"]:
                    draw_pose(
                        frame,
                        d["kps"],
                        color=id_color(int(d["temp_id"])),
                        kps_conf=d.get("kps_conf"),
                        kp_radius=args.kp_radius,
                        sk_thickness=args.sk_thickness,
                        kp_thresh=kp_thresh_effective,
                        kp_conf_thresh=args.kp_conf_thresh,
                        hide_lowconf=(not args.show_lowconf),
                        show_index=args.kp_index,
                        index_scale=args.kp_index_scale,
                        index_thickness=args.kp_index_thickness,
                        index_offset=args.kp_index_offset,
                    )

            # ---- live matching: recompute on each new kinetics interval ----
            if _live_matching and next_fi is not None:
                # get the frame_datetime for this frame from the tracks df
                _frame_rows = _tracks_df[_tracks_df["frame_index"] == frame_idx]
                if not _frame_rows.empty:
                    import pandas as _pd
                    _now_dt = _frame_rows["frame_datetime"].iloc[0]
                    # compute which kinetics bin we are currently in
                    _cur_bin = _now_dt.floor(f"{args.bin_minutes}min")
                    if _cur_bin != _last_bin:
                        # crossed a new interval boundary — recompute
                        try:
                            assignment, scores_df = mi.score_up_to(
                                tracks_df=_tracks_df,
                                kinetics_df=_kinetics_df,
                                up_to_datetime=_now_dt,
                                bin_minutes=args.bin_minutes,
                                activity_pct=0.25,
                                min_active_bins=args.min_active_bins,
                                min_temp_id_frames=args.min_temp_id_frames,
                            )
                            _last_bin = _cur_bin
                            # print update to terminal
                            n_bins_done = int((_now_dt - _tracks_df["frame_datetime"].min())
                                              .total_seconds() / 60 / args.bin_minutes)
                            log(f"[{_now_dt.strftime('%H:%M:%S')}] "
                                f"Interval {n_bins_done} — "
                                + (", ".join(f"t{t}->{a}" for t,a in sorted(assignment.items()))
                                   if assignment else "no confident matches yet"))
                        except Exception as _exc:
                            log(f"Live match error: {_exc}")

            # identity match score table overlay
            if show_table and scores_df is not None:
                draw_score_table(frame, scores_df, assignment)

            # FPS overlay
            if args.show_fps:
                now = time()
                fps_now = 1.0 / max(1e-6, now - prev_t)
                prev_t = now
                cv2.putText(
                    frame,
                    f"{fps_now:5.1f} FPS",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"{fps_now:5.1f} FPS",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # sink
            if args.sink == "ffplay":
                try:
                    ffplay_proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, AttributeError):
                    log("ffplay closed. Stopping.")
                    break
            elif args.sink == "cv2":
                cv2.imshow("tracks", frame)
                if args.max_fps > 0:
                    cv2.waitKey(int(1000 / args.max_fps))
            else:
                writer.write(frame)

            frame_idx += 1
            n_frames += 1
            if args.limit > 0 and n_frames >= args.limit:
                log("Limit reached.")
                break

    cap.release()
    if args.sink == "cv2":
        cv2.destroyAllWindows()
    elif args.sink == "ffplay":
        if ffplay_proc and ffplay_proc.stdin:
            ffplay_proc.stdin.close()
        if ffplay_proc:
            ffplay_proc.terminate()
    else:
        if writer:
            writer.release()
    log("Done.")


if __name__ == "__main__":
    main()


# Example usage:
# python3 display_tracks.py \
#   --video "/home/anton/thesis_workspace/raw_data/calving/6558/refet_33_S20241221070000_E20241221080000_6558.mp4" \
#   --tracks "/home/anton/thesis_workspace/outputs/tracks/refet33_2024-12-21_pose/tracks.csv" \
#   --draw_pose --kp_index --show_fps --sink ffplay --hide_occluded --kp_conf_thresh 0.35 \
#   --kinetics /home/anton/thesis_workspace/raw_data/CollarData/kinetic_data_6558_7509_7774.csv --corr_threshold 0.7 --min_active_bins 3