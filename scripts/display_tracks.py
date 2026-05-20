#!/usr/bin/env python3
import argparse, json, os, sqlite3, sys, subprocess
from pathlib import Path
from time import time

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))

import cv2
import numpy as np

def log(msg):
    print(f"[viewer] {msg}", flush=True)


# -------- Tkinter control panel (runs in a background thread) ----------
class TkControls:
    """
    A small always-on-top Tkinter window with buttons for all playback controls.
    Runs in its own daemon thread so the main OpenCV loop is never blocked.
    State is shared via plain Python attributes (reads/writes are GIL-safe for booleans/ints).
    """
    FF_SPEEDS = [1, 2, 4, 8]

    def __init__(self):
        self.paused     = False
        self.show_table = True
        self.ff_idx     = 0
        self.ff_speed   = 1
        self.quit       = False
        self._root      = None

        import threading
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

        # give Tk a moment to appear before the main loop starts
        import time as _t
        _t.sleep(0.3)

    def _run(self):
        import tkinter as tk

        root = tk.Tk()
        self._root = root
        root.title("Playback Controls")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        BG     = "#1e1e2e"
        BTN_BG = "#313244"
        BTN_ACT= "#45475a"
        FG     = "#cdd6f4"
        ACC    = "#89b4fa"
        WARN   = "#f38ba8"
        FONT   = ("Helvetica", 13, "bold")
        SFONT  = ("Helvetica", 10)

        root.configure(bg=BG)

        tk.Label(root, text="▶  Playback Controls", bg=BG, fg=ACC,
                 font=("Helvetica", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(12, 4), padx=16)

        def make_btn(parent, text, cmd, row, col, fg=FG, colspan=1):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=BTN_BG, fg=fg, activebackground=BTN_ACT,
                          activeforeground=FG, font=FONT,
                          relief="flat", bd=0, padx=14, pady=8,
                          cursor="hand2")
            b.grid(row=row, column=col, columnspan=colspan,
                   padx=6, pady=4, sticky="ew")
            return b

        self._pause_btn = make_btn(root, "⏸  Pause", self._toggle_pause, 1, 0, colspan=2)

        self._ff_label = tk.StringVar(value="Speed: 1×")
        tk.Label(root, textvariable=self._ff_label, bg=BG, fg=FG,
                 font=SFONT).grid(row=2, column=0, columnspan=2, pady=(6, 0))
        self._ff_btn = make_btn(root, "⏩  Fast Forward", self._cycle_ff, 3, 0, colspan=2)

        self._table_btn = make_btn(root, "📊  Hide Score Table", self._toggle_table, 4, 0, colspan=2)

        make_btn(root, "⏹  Quit", self._do_quit, 5, 0, colspan=2, fg=WARN)

        root.bind("<p>", lambda e: self._toggle_pause())
        root.bind("<f>", lambda e: self._cycle_ff())
        root.bind("<t>", lambda e: self._toggle_table())
        root.bind("<q>", lambda e: self._do_quit())
        root.protocol("WM_DELETE_WINDOW", self._do_quit)

        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        root.mainloop()

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self._pause_btn.config(text="▶  Resume")
            log("Paused.")
        else:
            self._pause_btn.config(text="⏸  Pause")
            log("Resumed.")

    def _cycle_ff(self):
        self.ff_idx = (self.ff_idx + 1) % len(self.FF_SPEEDS)
        self.ff_speed = self.FF_SPEEDS[self.ff_idx]
        self._ff_label.set(f"Speed: {self.ff_speed}×")
        accent = "#a6e3a1" if self.ff_speed > 1 else "#cdd6f4"
        self._ff_btn.config(fg=accent)
        log(f"Fast-forward: {self.ff_speed}x")

    def _toggle_table(self):
        self.show_table = not self.show_table
        label = "📊  Show Score Table" if not self.show_table else "📊  Hide Score Table"
        self._table_btn.config(text=label)
        log("Score table: " + ("visible" if self.show_table else "hidden"))

    def _do_quit(self):
        self.quit = True
        log("Quit requested.")
        if self._root:
            try:
                self._root.destroy()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._root:
            try:
                self._root.destroy()
            except Exception:
                pass


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
    ap.add_argument("--video",      required=True)
    ap.add_argument("--db",         required=True,
                    help="Path to calving_project.db (SQLite)")
    ap.add_argument("--session_id", required=True,
                    help="session_id to display (must exist in video_sessions table)")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_fps", type=float, default=0.0)
    ap.add_argument("--show_fps", action="store_true")
    ap.add_argument(
        "--sink",
        choices=["ffplay", "cv2", "mp4"],
        default="cv2",
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


# -------- Track reader — SQLite ----------
def stream_sqlite(db_path: str, session_id: str,
                  start_frame: int = 0, W: int = 0, H: int = 0,
                  want_pose: bool = False,
                  kps_parquet_path: str = ""):
    """
    Generator that yields (frame_index, [detections]) in frame_index order,
    matching the contract of the old stream_csv / stream_jsonl.
    Optionally joins kps.parquet for pose data when want_pose=True.
    """
    import pandas as _pd

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # load kps parquet once if needed
    kps_df = None
    if want_pose and kps_parquet_path and Path(kps_parquet_path).exists():
        kps_df = _pd.read_parquet(kps_parquet_path)
        kps_df = kps_df[kps_df["session_id"] == session_id].set_index(
            ["frame_index", "temp_id"])

    rows = conn.execute("""
        SELECT frame_index, temp_id, det_conf,
               x1, y1, x2, y2,
               kps_conf, kps_parquet_row
        FROM   raw_tracks
        WHERE  session_id = ?
          AND  frame_index >= ?
        ORDER  BY frame_index, temp_id
    """, (session_id, start_frame)).fetchall()
    conn.close()

    cur_fi  = None
    bucket  = []

    for row in rows:
        fi = row["frame_index"]
        if cur_fi is None:
            cur_fi = fi
        if fi != cur_fi:
            yield cur_fi, bucket
            bucket  = []
            cur_fi  = fi

        d = {
            "temp_id": row["temp_id"] if row["temp_id"] is not None else -1,
            "conf":    row["det_conf"] or 0.0,
            "x1":      row["x1"], "y1": row["y1"],
            "x2":      row["x2"], "y2": row["y2"],
        }

        if want_pose and kps_df is not None:
            try:
                krow = kps_df.loc[(fi, row["temp_id"])]
                flat = list(krow["kps"])            # 57 floats [x,y,v, ...]
                if len(flat) % 3 == 0:
                    kps_xyv = [(flat[i], flat[i+1], flat[i+2])
                               for i in range(0, len(flat), 3)]
                    d["kps"]      = kps_xyv
                    d["kps_conf"] = list(krow["kps_kconf"])  # 19 floats
            except (KeyError, TypeError):
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
    vid  = Path(args.video)
    db   = Path(args.db)
    if not vid.exists():
        raise FileNotFoundError(f"Video not found: {vid}")
    if not db.exists():
        raise FileNotFoundError(f"DB not found: {db}")

    # derive kps parquet path (same folder as the db)
    kps_pq_path = str(db.parent / "kps.parquet")

    # ---- identity assignment (loaded from DB) ----
    assignment = {}
    scores_df  = None

    # ---- pre-load resolved assignments from DB ----
    # Pulls from manual_assignments + reid_registry so labels show animal IDs
    # immediately without waiting for live kinetic matching to fire.
    try:
        _conn = sqlite3.connect(str(db))

        # manual assignments for this session
        _manual = _conn.execute(
            "SELECT temp_id, real_id FROM manual_assignments WHERE session_id = ?",
            (args.session_id,)
        ).fetchall()
        for tid, aid in _manual:
            assignment[int(tid)] = int(aid)

        # kinetic/cosine assignments stored in reid_registry.known_temp_ids
        _reid = _conn.execute(
            "SELECT real_id, known_temp_ids FROM reid_registry "
            "WHERE known_temp_ids IS NOT NULL"
        ).fetchall()
        for real_id, known_json in _reid:
            try:
                known = json.loads(known_json)
                for entry in known:
                    if entry.get("session_id") == args.session_id:
                        tid = int(entry["temp_id"])
                        if tid not in assignment:   # manual takes priority
                            assignment[tid] = int(real_id)
            except Exception:
                pass

        _conn.close()
        if assignment:
            pairs = ', '.join('t%d->%d' % (t, a) for t, a in sorted(assignment.items()))
            log(f"Pre-loaded {len(assignment)} assignment(s) from DB: {pairs}")
    except Exception as e:
        log(f"WARNING: could not pre-load assignments from DB: {e}")

    log(f"Opening video: {vid}")
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vid}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"Video info: {W}x{H} @ {fps:.2f} fps")

    log(f"Opening tracks: session '{args.session_id}' from {db.name}")
    row_stream = stream_sqlite(
        str(db), args.session_id,
        start_frame=args.start, W=W, H=H,
        want_pose=args.draw_pose,
        kps_parquet_path=kps_pq_path,
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

    prev_t = time()
    n_frames = 0
    log("Use the Tkinter control panel to pause, fast-forward, toggle score table, or quit.")

    # decide effective kp threshold (v is 0/1/2)
    kp_thresh_effective = args.kp_thresh
    if args.hide_occluded:
        # draw only v >= 2
        kp_thresh_effective = max(kp_thresh_effective, 0.5)

    with TkControls() as ctrl:
        while True:
            if ctrl.quit:
                break

            paused    = ctrl.paused
            show_table= ctrl.show_table
            ff_speed  = ctrl.ff_speed

            if paused:
                if args.sink == "cv2":
                    cv2.waitKey(30)
                else:
                    import time as _t; _t.sleep(0.03)
                continue

            ok, frame = cap.read()
            if not ok:
                log("End of video.")
                break

            # fast-forward: skip (ff_speed - 1) frames between rendered frames
            if ff_speed > 1:
                skip = ff_speed - 1
                for _ in range(skip):
                    ok_skip = cap.grab()  # grab without decode — much faster
                    if not ok_skip:
                        break
                    frame_idx += 1
                    n_frames += 1
                    # advance track stream past skipped frames
                    while next_fi < frame_idx:
                        try:
                            next_fi, next_rows = next(row_stream)
                        except StopIteration:
                            next_rows = []
                            next_fi = frame_idx
                            break
                    if args.limit > 0 and n_frames >= args.limit:
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

            # fast-forward speed overlay
            if ff_speed > 1:
                spd_label = f">> {ff_speed}x"
                (sw, sh), _ = cv2.getTextSize(spd_label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                sx = frame.shape[1] - sw - 14
                sy = 38
                cv2.putText(frame, spd_label, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, spd_label, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 2, cv2.LINE_AA)

            # sink
            if args.sink == "ffplay":
                try:
                    ffplay_proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, AttributeError):
                    log("ffplay closed. Stopping.")
                    break
            elif args.sink == "cv2":
                cv2.imshow("tracks", frame)
                effective_fps = args.max_fps if args.max_fps > 0 else fps
                wait_ms = max(1, int(1000 / (effective_fps * ff_speed)))
                cv2.waitKey(wait_ms)
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
#   --video      "$VID" \
#   --db         "$DB" \
#   --session_id refet33_20241221 \
#   --draw_pose --kp_index --show_fps --sink ffplay --hide_occluded --kp_conf_thresh 0.35
#
# Identity labels (cow XXXX) are loaded automatically from the DB.
# Run reconcile.py or assign_identity.py first to populate assignments.