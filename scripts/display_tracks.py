#!/usr/bin/env python3
import argparse, json, os, signal as _sig, sqlite3, sys, subprocess
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

        # FIX: force an initial render pass so the window isn't blank in WSL/WSLg
        root.update()
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


# -------- Sensor / Behaviour Visualisation Window ----------
class SensorWindow:
    """
    A separate Tkinter window that shows kinetic + behaviour sensor data
    for the matched animals, driven live by the video playhead.

    Layout
    ------
    Top bar  : one button per matched AnimalId — clicking selects that animal.
               A clock label shows the current video wall-clock time.
    Bottom   : matplotlib canvas with 7 subplots sharing a time axis.
               Signals: f_1_2, f_2_3, v, ΔKinX, ΔKinY, ΔKinZ, ΔKinR.
               The plot shows a rolling window of LOOKBACK_HOURS ending at
               the current playhead. A vertical red cursor marks "now".
               As the video plays, the window scrolls forward automatically.

    Thread safety
    -------------
    The main video loop calls push_time(dt) which atomically stores a
    datetime object. The Tk thread polls it every POLL_MS milliseconds via
    root.after() and redraws only when the time has moved by at least
    REDRAW_THRESH_S seconds.  No matplotlib calls are ever made from the
    main thread.
    """

    # Catppuccin-inspired dark palette
    _BG       = "#1e1e2e"
    _SURFACE  = "#313244"
    _OVERLAY  = "#45475a"
    _FG       = "#cdd6f4"
    _ACC      = "#89b4fa"   # blue  — idle button
    _ACC_ON   = "#a6e3a1"   # green — selected / active button
    _WARN     = "#f38ba8"
    _MUTED    = "#6c7086"
    _PLOT_BG  = "#181825"
    _CURSOR   = "#f38ba8"   # red vertical "now" line

    # live-update knobs
    POLL_MS         = 200    # how often Tk polls for a new timestamp (ms)
    REDRAW_THRESH_S = 15     # only redraw if playhead moved ≥ this many seconds
    LOOKBACK_HOURS  = 2.0    # rolling window width shown behind the cursor

    # colours for the 7 subplots
    _LINE_COLORS = ["#89b4fa", "#cba6f7", "#a6e3a1",
                    "#fab387", "#f9e2af", "#89dceb", "#f38ba8"]

    _SIGNAL_META = [
        # (column_in_merged_df,  y-axis label,  panel title)
        ("f_1_2",  "f₁₂",   "Behaviour · f₁₂"),
        ("f_2_3",  "f₂₃",   "Behaviour · f₂₃"),
        ("v",      "v",     "Behaviour · v"),
        ("dKinX",  "ΔKinX", "Kinetics · ΔX (interval)"),
        ("dKinY",  "ΔKinY", "Kinetics · ΔY (interval)"),
        ("dKinZ",  "ΔKinZ", "Kinetics · ΔZ (interval)"),
        ("dKinR",  "ΔKinR", "Kinetics · ΔR (interval)"),
    ]

    def __init__(self, kinetic_csv: str, behavior_csv: str, animal_ids: list):
        import threading

        self._animal_ids    = sorted(int(a) for a in animal_ids)
        self._selected_id   = None
        self._btn_widgets   = {}      # AnimalId → tk.Button
        self._root          = None
        self._canvas_widget = None
        self._fig           = None
        self._axes          = None

        # shared playhead state — written by main thread, read by Tk thread
        self._current_dt    = None    # datetime | None
        self._last_drawn_dt = None    # datetime of last completed redraw
        self._lock          = __import__("threading").Lock()

        # ---- load & pre-process data once ----
        self._merged = self._load_data(kinetic_csv, behavior_csv)

        t = threading.Thread(target=self._run, daemon=True)
        t.start()

        import time as _t
        _t.sleep(0.6)   # give Tk a moment to appear

    # ------------------------------------------------------------------
    # Public API — called from the main video loop (any thread)
    # ------------------------------------------------------------------
    def push_time(self, dt):
        """Update the playhead to datetime dt.  GIL-safe single assignment."""
        with self._lock:
            self._current_dt = dt

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_data(kin_path: str, beh_path: str):
        """
        Read kinetic + behaviour CSVs, compute delta columns for kinetics,
        return {AnimalId: DataFrame indexed by datetime, sorted ascending}.
        """
        import pandas as pd

        frames = {}

        # --- kinetics ---
        if kin_path and Path(kin_path).exists():
            kdf = pd.read_csv(kin_path, parse_dates=["datetime"])
            kdf = kdf.sort_values(["AnimalId", "datetime"]).reset_index(drop=True)
            for col in ["KineticsCountX", "KineticsCountY",
                        "KineticsCountZ", "KineticsCountR"]:
                short = col.replace("KineticsCount", "Kin")
                kdf[f"d{short}"] = kdf.groupby("AnimalId")[col].diff()
            for aid, grp in kdf.groupby("AnimalId"):
                frames[int(aid)] = grp.set_index("datetime").sort_index()

        # --- behaviour ---
        if beh_path and Path(beh_path).exists():
            bdf = pd.read_csv(beh_path, parse_dates=["datetime"])
            bdf = bdf.sort_values(["AnimalId", "datetime"]).reset_index(drop=True)
            for aid, grp in bdf.groupby("AnimalId"):
                aid = int(aid)
                sub = grp.set_index("datetime")[["f_1_2", "f_2_3", "v"]].sort_index()
                if aid in frames:
                    frames[aid] = frames[aid].join(sub, how="outer").sort_index()
                else:
                    frames[aid] = sub

        return frames

    # ------------------------------------------------------------------
    # Tkinter + matplotlib UI  (runs in its own thread)
    # ------------------------------------------------------------------
    def _run(self):
        import tkinter as tk

        # Use the TkAgg backend but construct the Figure directly — never call
        # plt.subplots() from a non-main thread, which triggers the UserWarning
        # and can crash on some platforms.
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.dates as mdates

        root = tk.Tk()
        self._root = root
        root.title("Sensor & Behaviour — Live")
        root.configure(bg=self._BG)
        root.geometry("1280x820")

        # ── Top bar ────────────────────────────────────────────────────
        bar = tk.Frame(root, bg=self._BG, pady=6)
        bar.pack(side="top", fill="x", padx=10)

        tk.Label(bar, text="Animal ID:", bg=self._BG, fg=self._MUTED,
                 font=("Helvetica", 11)).pack(side="left", padx=(4, 10))

        for aid in self._animal_ids:
            btn = tk.Button(
                bar,
                text=str(aid),
                bg=self._ACC,
                fg=self._BG,
                activebackground=self._ACC_ON,
                activeforeground=self._BG,
                font=("Helvetica", 12, "bold"),
                relief="flat", bd=0, padx=16, pady=6,
                cursor="hand2",
                command=lambda a=aid: self._select_animal(a),
            )
            btn.pack(side="left", padx=5)
            self._btn_widgets[aid] = btn

        # status / clock
        self._status_var = tk.StringVar(value="← click an animal")
        tk.Label(bar, textvariable=self._status_var,
                 bg=self._BG, fg=self._MUTED,
                 font=("Helvetica", 10, "italic")).pack(side="left", padx=14)

        self._clock_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._clock_var,
                 bg=self._BG, fg=self._ACC,
                 font=("Helvetica", 11, "bold")).pack(side="right", padx=14)

        # ── Matplotlib figure — built with Figure(), not plt.subplots() ─
        n   = len(self._SIGNAL_META)
        fig = Figure(figsize=(13, 9.5), facecolor=self._PLOT_BG)
        fig.subplots_adjust(hspace=0.06, top=0.97, bottom=0.06,
                            left=0.07, right=0.97)

        axes = []
        shared_ax = None
        for i in range(n):
            if shared_ax is None:
                ax = fig.add_subplot(n, 1, i + 1)
                shared_ax = ax
            else:
                ax = fig.add_subplot(n, 1, i + 1, sharex=shared_ax)
            axes.append(ax)
        axes = axes  # plain list

        self._fig  = fig
        self._axes = axes

        for ax, (_, ylabel, title) in zip(axes, self._SIGNAL_META):
            ax.set_facecolor(self._PLOT_BG)
            ax.tick_params(colors=self._MUTED, labelsize=7)
            ax.spines[:].set_color(self._OVERLAY)
            ax.set_ylabel(ylabel, fontsize=8, color=self._MUTED)
            ax.set_title(title, fontsize=8, color=self._MUTED, loc="left", pad=2)
            ax.yaxis.grid(True, color=self._OVERLAY, linewidth=0.5, linestyle="--")
            ax.set_axisbelow(True)
            # hide x-tick labels on all but the bottom panel
            if ax is not axes[-1]:
                ax.tick_params(labelbottom=False)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        axes[-1].tick_params(axis="x", labelsize=7, colors=self._MUTED)

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        cw = canvas.get_tk_widget()
        cw.configure(bg=self._PLOT_BG)
        cw.pack(side="bottom", fill="both", expand=True, padx=6, pady=(0, 6))
        self._canvas_widget = canvas

        # store mdates for use in _live_redraw
        self._mdates = mdates

        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # FIX: force an initial render pass so the window isn't blank in WSL/WSLg
        root.update()
        root.after(self.POLL_MS, self._poll)
        root.mainloop()

    # ------------------------------------------------------------------
    # Polling — called from Tk's event loop every POLL_MS ms
    # ------------------------------------------------------------------
    def _poll(self):
        """Check if playhead has moved enough to warrant a redraw."""
        if self._root is None:
            return

        with self._lock:
            current = self._current_dt

        # update clock label regardless
        if current is not None:
            self._clock_var.set(current.strftime("▶  %Y-%m-%d  %H:%M:%S"))
        else:
            self._clock_var.set("")

        # decide whether to redraw
        if current is not None and self._selected_id is not None:
            if self._last_drawn_dt is None:
                self._live_redraw(current)
            else:
                delta = abs((current - self._last_drawn_dt).total_seconds())
                if delta >= self.REDRAW_THRESH_S:
                    self._live_redraw(current)

        try:
            self._root.after(self.POLL_MS, self._poll)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Live redraw — always called from Tk thread via _poll
    # ------------------------------------------------------------------
    def _live_redraw(self, now_dt):
        """
        Redraw all 7 panels for self._selected_id clipped to
        [now_dt - LOOKBACK, now_dt].  Vertical dashed cursor marks now_dt.
        Uses ax.cla() for clearing — compatible with all matplotlib versions.
        """
        from datetime import timedelta

        aid = self._selected_id
        df  = self._merged.get(aid)

        lookback  = timedelta(hours=self.LOOKBACK_HOURS)
        win_start = now_dt - lookback
        win_end   = now_dt
        right_pad = lookback * 0.04

        for i, (ax, color, (col, ylabel, title)) in enumerate(
                zip(self._axes, self._LINE_COLORS, self._SIGNAL_META)):

            ax.cla()

            # ── restore axis cosmetics lost by cla() ──────────────────
            ax.set_facecolor(self._PLOT_BG)
            ax.tick_params(colors=self._MUTED, labelsize=7)
            ax.spines[:].set_color(self._OVERLAY)
            ax.set_ylabel(ylabel, fontsize=8, color=self._MUTED)
            ax.set_title(title, fontsize=8, color=self._MUTED, loc="left", pad=2)
            ax.yaxis.grid(True, color=self._OVERLAY, linewidth=0.5, linestyle="--")
            ax.set_axisbelow(True)
            # hide x-tick labels on all but the bottom panel
            if i < len(self._axes) - 1:
                ax.tick_params(labelbottom=False)

            # ── plot data up to now_dt ─────────────────────────────────
            if df is not None and col in df.columns:
                series = df[col].loc[:win_end].dropna()
                if not series.empty:
                    ax.plot(series.index, series.values,
                            color=color, linewidth=1.1, alpha=0.92)
                    ax.fill_between(series.index, series.values,
                                    alpha=0.10, color=color)

            # ── cursor line ────────────────────────────────────────────
            ax.axvline(x=now_dt, color=self._CURSOR,
                       linewidth=1.4, linestyle="--", alpha=0.85, zorder=10)

            # ── x window ──────────────────────────────────────────────
            ax.set_xlim(win_start, win_end + right_pad)

            # ── y auto-scale to the visible window only ────────────────
            if df is not None and col in df.columns:
                vis = df[col].loc[win_start:win_end].dropna()
                if not vis.empty:
                    lo, hi = vis.min(), vis.max()
                    margin = max((hi - lo) * 0.12, abs(hi) * 0.05, 1e-6)
                    ax.set_ylim(lo - margin, hi + margin)

        # ── x-axis formatting on bottom panel (restored after cla) ────
        self._axes[-1].xaxis.set_major_formatter(
            self._mdates.DateFormatter("%m-%d\n%H:%M"))
        self._axes[-1].tick_params(axis="x", labelsize=7, colors=self._MUTED)

        self._fig.canvas.draw_idle()
        self._last_drawn_dt = now_dt

    # ------------------------------------------------------------------
    def _select_animal(self, aid: int):
        self._selected_id   = aid
        self._last_drawn_dt = None   # force immediate redraw on next poll
        self._status_var.set(f"Showing Animal ID: {aid}")

        for a, btn in self._btn_widgets.items():
            if a == aid:
                btn.config(bg=self._ACC_ON, fg=self._BG, relief="sunken")
            else:
                btn.config(bg=self._ACC,    fg=self._BG, relief="flat")

    def _on_close(self):
        try:
            if self._root:
                self._root.destroy()
        except Exception:
            pass

    def destroy(self):
        self._on_close()


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
        img, label, (x1 + 3, y0 + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
    )


# -------- Pose skeleton (19 KP layout) ----------
# Indices:
# 0 nose          1 forehead      2 withers       3 spine_mid
# 4 sacrum        5 tail_base     6 tail_tip       7 shoulder_L
# 8 elbow_L       9 fetlock_fore_L  10 shoulder_R  11 elbow_R
# 12 fetlock_fore_R  13 hock_R    14 hock_L        15 fetlock_hind_L
# 16 fetlock_hind_R  17 udder_center  18 neck
BASE_EDGES = [
    (0, 1),
    (1, 18),  # forehead -> neck
    (18, 2),  # neck -> withers
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),  # head → spine → tail
    # fore limbs
    (2, 7), (7, 8), (8, 9),
    (2, 10), (10, 11), (11, 12),
    # hind limbs
    (4, 14), (14, 15),
    (4, 13), (13, 16),
    (4, 17),
]
CUSTOM_EDGES = [
    (2, 10), (10, 12),
    (4, 17), (14, 15), (13, 16),
]
EDGE_SET = set(tuple(sorted(e)) for e in BASE_EDGES)
for e in CUSTOM_EDGES:
    EDGE_SET.add(tuple(sorted(e)))
EDGES = [(a, b) for (a, b) in EDGE_SET]


def draw_pose(
    img, kps_xyv, color, kps_conf=None,
    kp_radius=3, sk_thickness=2, kp_thresh=0.0, kp_conf_thresh=0.30,
    hide_lowconf=False, show_index=False,
    index_scale=0.45, index_thickness=1, index_offset=6,
):
    K = len(kps_xyv)
    draw_mask = [False] * K
    for i in range(K):
        x, y, v = kps_xyv[i]
        if v is None: continue
        if float(v) <= kp_thresh: continue
        if hide_lowconf and kps_conf is not None and i < len(kps_conf):
            try:
                if float(kps_conf[i]) < float(kp_conf_thresh): continue
            except Exception: pass
        draw_mask[i] = True

    for i in range(K):
        if not draw_mask[i]: continue
        x, y, _v = kps_xyv[i]
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(img, (xi, yi), kp_radius, color, -1, lineType=cv2.LINE_AA)
        if show_index:
            idx_txt = str(i)
            cv2.putText(img, idx_txt, (xi + index_offset, yi - index_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, index_scale, (0, 0, 0),
                        index_thickness + 2, cv2.LINE_AA)
            cv2.putText(img, idx_txt, (xi + index_offset, yi - index_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, index_scale, (255, 255, 255),
                        index_thickness, cv2.LINE_AA)

    for (i, j) in EDGES:
        if i < K and j < K and draw_mask[i] and draw_mask[j]:
            xi, yi, _vi = kps_xyv[i]
            xj, yj, _vj = kps_xyv[j]
            cv2.line(img,
                     (int(round(xi)), int(round(yi))),
                     (int(round(xj)), int(round(yj))),
                     color, sk_thickness, lineType=cv2.LINE_AA)


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
    ap.add_argument(
        "--sink", choices=["ffplay", "cv2", "mp4"], default="cv2",
        help="ffplay (no GUI deps), cv2 (needs Qt/X11), or mp4 (save to file)",
    )
    ap.add_argument("--outmp4", default="annotated.mp4",
                    help="Output MP4 path if --sink mp4")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional max frames to run")

    # pose drawing
    ap.add_argument("--draw_pose", action="store_true")
    ap.add_argument("--kp_radius", type=int, default=3)
    ap.add_argument("--sk_thickness", type=int, default=2)
    ap.add_argument("--kp_thresh", type=float, default=0.0)
    ap.add_argument("--hide_occluded", action="store_true")
    ap.add_argument("--kp_conf_thresh", type=float, default=0.30)
    ap.add_argument("--show_lowconf", action="store_true")
    ap.add_argument("--kp_index", action="store_true")
    ap.add_argument("--kp_index_scale", type=float, default=0.45)
    ap.add_argument("--kp_index_thickness", type=int, default=1)
    ap.add_argument("--kp_index_offset", type=int, default=6)

    # sensor window
    ap.add_argument("--kinetic_csv", default="",
                    help="Path to kinetic_data CSV. Enables the live sensor window.")
    ap.add_argument("--behavior_csv", default="",
                    help="Path to behavior_data CSV. Merged with kinetics in sensor window.")
    ap.add_argument("--sensor_lookback", type=float, default=2.0,
                    help="Hours of data shown behind the playhead cursor (default 2.0)")

    return ap.parse_args()


# -------- Track reader — SQLite ----------
def stream_sqlite(db_path: str, session_id: str,
                  start_frame: int = 0, W: int = 0, H: int = 0,
                  want_pose: bool = False,
                  kps_parquet_path: str = ""):
    """
    Generator that yields (frame_index, frame_datetime, [detections]).
    frame_datetime is a Python datetime if stored in raw_tracks, else None.
    """
    import pandas as _pd
    from datetime import datetime as _dt

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    kps_df = None
    if want_pose and kps_parquet_path and Path(kps_parquet_path).exists():
        kps_df = _pd.read_parquet(kps_parquet_path)
        kps_df = kps_df[kps_df["session_id"] == session_id].set_index(
            ["frame_index", "temp_id"])

    # try to fetch frame_datetime if the column exists
    try:
        rows = conn.execute("""
            SELECT frame_index, frame_datetime, temp_id, det_conf,
                   x1, y1, x2, y2, kps_conf, kps_parquet_row
            FROM   raw_tracks
            WHERE  session_id = ?
              AND  frame_index >= ?
            ORDER  BY frame_index, temp_id
        """, (session_id, start_frame)).fetchall()
        has_dt_col = True
    except sqlite3.OperationalError:
        rows = conn.execute("""
            SELECT frame_index, temp_id, det_conf,
                   x1, y1, x2, y2, kps_conf, kps_parquet_row
            FROM   raw_tracks
            WHERE  session_id = ?
              AND  frame_index >= ?
            ORDER  BY frame_index, temp_id
        """, (session_id, start_frame)).fetchall()
        has_dt_col = False
    conn.close()

    cur_fi  = None
    cur_fdt = None
    bucket  = []

    for row in rows:
        fi = row["frame_index"]
        fdt = None
        if has_dt_col:
            raw_fdt = row["frame_datetime"]
            if raw_fdt:
                try:
                    # stored as ISO string or unix timestamp
                    if isinstance(raw_fdt, (int, float)):
                        fdt = _dt.utcfromtimestamp(raw_fdt)
                    else:
                        fdt = _dt.fromisoformat(str(raw_fdt))
                except Exception:
                    fdt = None

        if cur_fi is None:
            cur_fi  = fi
            cur_fdt = fdt
        if fi != cur_fi:
            yield cur_fi, cur_fdt, bucket
            bucket  = []
            cur_fi  = fi
            cur_fdt = fdt

        d = {
            "temp_id": row["temp_id"] if row["temp_id"] is not None else -1,
            "conf":    row["det_conf"] or 0.0,
            "x1": row["x1"], "y1": row["y1"],
            "x2": row["x2"], "y2": row["y2"],
        }

        if want_pose and kps_df is not None:
            try:
                krow = kps_df.loc[(fi, row["temp_id"])]
                flat = list(krow["kps"])
                if len(flat) % 3 == 0:
                    kps_xyv = [(flat[i], flat[i+1], flat[i+2])
                               for i in range(0, len(flat), 3)]
                    d["kps"]      = kps_xyv
                    d["kps_conf"] = list(krow["kps_kconf"])
            except (KeyError, TypeError):
                pass

        bucket.append(d)

    if bucket:
        yield cur_fi, cur_fdt, bucket


# -------- Match score table overlay ----------
def draw_score_table(img, scores_df, assignment, margin=10):
    if scores_df is None or scores_df.empty:
        return

    import pandas as pd
    pivot = scores_df.pivot_table(
        index="temp_id", columns="AnimalId", values="correlation", aggfunc="first"
    )

    tids   = list(pivot.index)
    aids   = list(pivot.columns)
    n_rows = len(tids) + 1
    n_cols = len(aids) + 1

    cell_w, cell_h = 90, 22
    table_w = n_cols * cell_w
    table_h = n_rows * cell_h

    H, W = img.shape[:2]
    x0 = margin
    y0 = H - table_h - margin

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

    draw_cell(0, 0, "tid\\aid", (40, 40, 40))
    for ci, aid in enumerate(aids):
        draw_cell(0, ci + 1, str(aid), (40, 40, 40))

    for ri, tid in enumerate(tids):
        draw_cell(ri + 1, 0, f"t{tid}", (40, 40, 40))
        for ci, aid in enumerate(aids):
            val = pivot.loc[tid, aid] if (tid in pivot.index and aid in pivot.columns) else float("nan")
            assigned = assignment.get(tid) == aid
            if assigned:
                bg = (0, 120, 0)
            elif not pd.isna(val) and val >= 0.5:
                bg = (0, 100, 150)
            else:
                bg = (60, 60, 60)
            txt = f"{val:.2f}" if not pd.isna(val) else "—"
            draw_cell(ri + 1, ci + 1, txt, bg)

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

    kps_pq_path = str(db.parent / "kps.parquet")

    # ---- identity assignment (loaded from DB) ----
    assignment  = {}
    scores_df   = None
    sensor_win  = None

    try:
        _conn = sqlite3.connect(str(db))

        _manual = _conn.execute(
            "SELECT temp_id, real_id FROM manual_assignments WHERE session_id = ?",
            (args.session_id,)
        ).fetchall()
        for tid, aid in _manual:
            assignment[int(tid)] = int(aid)

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
                        if tid not in assignment:
                            assignment[tid] = int(real_id)
            except Exception:
                pass

        _conn.close()
        if assignment:
            pairs = ', '.join('t%d->%d' % (t, a) for t, a in sorted(assignment.items()))
            log(f"Pre-loaded {len(assignment)} assignment(s) from DB: {pairs}")
    except Exception as e:
        log(f"WARNING: could not pre-load assignments from DB: {e}")

    # ---- sensor window (optional) ----
    if args.kinetic_csv or args.behavior_csv:
        matched_aids = sorted(set(assignment.values())) if assignment else []
        if not matched_aids:
            try:
                import pandas as _pd2
                _src = args.kinetic_csv or args.behavior_csv
                matched_aids = sorted(
                    _pd2.read_csv(_src, usecols=["AnimalId"])["AnimalId"].unique().tolist()
                )
            except Exception as _e:
                log(f"WARNING: could not read AnimalIds from sensor CSV: {_e}")
        if matched_aids:
            log(f"Opening sensor window for AnimalIds: {matched_aids}")
            sensor_win = SensorWindow(
                kinetic_csv=args.kinetic_csv,
                behavior_csv=args.behavior_csv,
                animal_ids=matched_aids,
            )
            sensor_win.LOOKBACK_HOURS = args.sensor_lookback
        else:
            log("No matched AnimalIds found — sensor window not opened.")

    # ---- session start time (for frame_datetime fallback) ----
    # Try to load start_dt from video_sessions; also parse from filename token.
    session_start_dt = None
    try:
        import re as _re
        from datetime import datetime as _dt2
        _conn2 = sqlite3.connect(str(db))
        _row = _conn2.execute(
            "SELECT start_dt FROM video_sessions WHERE session_id = ?",
            (args.session_id,)
        ).fetchone()
        _conn2.close()
        if _row and _row[0]:
            session_start_dt = _dt2.fromisoformat(str(_row[0]))
            log(f"Session start_dt from DB: {session_start_dt}")
    except Exception:
        pass
    if session_start_dt is None:
        # fall back: parse _S<YYYYMMDDHHmmss> token from filename
        import re as _re2
        from datetime import datetime as _dt3
        m = _re2.search(r'_S(\d{14})', vid.name)
        if m:
            try:
                session_start_dt = _dt3.strptime(m.group(1), "%Y%m%d%H%M%S")
                log(f"Session start_dt from filename: {session_start_dt}")
            except ValueError:
                pass

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

    # prime stream  (now yields 3-tuple)
    try:
        next_fi, next_fdt, next_rows = next(row_stream)
        log(f"First tracks frame_index: {next_fi}  rows: {len(next_rows)}")
    except StopIteration:
        log("Tracks contain no rows >= start frame. Exiting.")
        return

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
            "ffplay", "-loglevel", "error",
            "-an",
            "-fflags", "nobuffer",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{W}x{H}", "-framerate", f"{disp_fps:.2f}", "-",
        ]
        try:
            ffplay_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            log("ffplay started.")
        except FileNotFoundError:
            log("ffplay not found. Falling back to MP4 writer.")
            args.sink = "mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.outmp4, fourcc, fps, (W, H))

    # FIX: SIGINT handler — sets a flag instead of relying on KeyboardInterrupt
    # delivered to a thread that may be blocked on a pipe write.
    _interrupted = [False]

    def _handle_sigint(signum, frame):
        _interrupted[0] = True
        log("Interrupted — stopping.")
        import os as _os
        _os._exit(0)

    _sig.signal(_sig.SIGINT, _handle_sigint)

    n_frames = 0
    stream_done = False   # set True once row_stream is exhausted
    log("Use the Tkinter control panel to pause, fast-forward, toggle score table, or quit.")
    log("Press Ctrl+C in this terminal to stop at any time.")

    kp_thresh_effective = args.kp_thresh
    if args.hide_occluded:
        kp_thresh_effective = max(kp_thresh_effective, 0.5)

    # current_fdt tracks the wall-clock datetime of the frame being rendered
    current_fdt = None

    with TkControls() as ctrl:
        while True:
            # FIX: check both the Tk quit button and the SIGINT flag
            if ctrl.quit or _interrupted[0]:
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
                for _ in range(ff_speed - 1):
                    ok_skip = cap.grab()
                    if not ok_skip:
                        break
                    frame_idx += 1
                    n_frames  += 1
                    # advance track stream past skipped frames
                    if not stream_done:
                        while next_fi < frame_idx:
                            try:
                                next_fi, next_fdt, next_rows = next(row_stream)
                            except StopIteration:
                                stream_done = True
                                next_rows = []
                                next_fi   = frame_idx
                                break
                    if args.limit > 0 and n_frames >= args.limit:
                        break

            # advance track stream to current frame
            if not stream_done:
                while next_fi < frame_idx:
                    try:
                        next_fi, next_fdt, next_rows = next(row_stream)
                    except StopIteration:
                        stream_done = True
                        next_rows = []
                        next_fi   = frame_idx
                        break

            if next_fi == frame_idx:
                dets = next_rows
                # capture the datetime from this frame's track rows
                if next_fdt is not None:
                    current_fdt = next_fdt
                if not stream_done:
                    try:
                        next_fi, next_fdt, next_rows = next(row_stream)
                    except StopIteration:
                        stream_done = True
                        next_rows = []
                        next_fi   = 10**12
            else:
                dets = []

            # ---- compute wall-clock datetime for this frame ----
            if current_fdt is None and session_start_dt is not None:
                from datetime import timedelta as _td
                current_fdt = session_start_dt + _td(seconds=frame_idx / fps)
            elif session_start_dt is not None:
                # always recompute from frame position so FF scrubs correctly
                from datetime import timedelta as _td
                current_fdt = session_start_dt + _td(seconds=frame_idx / fps)

            # push to sensor window (thread-safe)
            if sensor_win is not None and current_fdt is not None:
                sensor_win.push_time(current_fdt)

            # draw detections + pose
            for d in dets:
                draw_box(
                    frame, d["x1"], d["y1"], d["x2"], d["y2"],
                    d["temp_id"], d["conf"],
                    animal_id=assignment.get(int(d["temp_id"])),
                )
                if args.draw_pose and ("kps" in d) and d["kps"]:
                    draw_pose(
                        frame, d["kps"],
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

            if show_table and scores_df is not None:
                draw_score_table(frame, scores_df, assignment)

            if ff_speed > 1:
                spd_label = f">> {ff_speed}x"
                (sw, sh), _ = cv2.getTextSize(spd_label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                sx = frame.shape[1] - sw - 14
                sy = 38
                cv2.putText(frame, spd_label, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, spd_label, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 2, cv2.LINE_AA)

            # FIX: wrap ffplay write in broader except and add flush;
            # BrokenPipeError alone wasn't catching all WSL pipe errors.
            if args.sink == "ffplay":
                try:
                    ffplay_proc.stdin.write(frame.tobytes())
                    ffplay_proc.stdin.flush()
                except (BrokenPipeError, AttributeError, OSError):
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
            n_frames  += 1
            if args.limit > 0 and n_frames >= args.limit:
                log("Limit reached.")
                break

    cap.release()
    if args.sink == "cv2":
        cv2.destroyAllWindows()
    elif args.sink == "ffplay":
        if ffplay_proc and ffplay_proc.stdin:
            try:
                ffplay_proc.stdin.close()
            except OSError:
                pass
        if ffplay_proc:
            ffplay_proc.terminate()
    else:
        if writer:
            writer.release()
    if sensor_win is not None:
        sensor_win.destroy()
    log("Done.")


if __name__ == "__main__":
    main()


# Example usage:
# python3 display_tracks.py \
#   --video      "$VID" \
#   --db         "$DB" \
#   --session_id refet33_20241221 \
#   --draw_pose --kp_index --sink ffplay --hide_occluded --kp_conf_thresh 0.35 \
#   --kinetic_csv  "$KIN" \
#   --behavior_csv "$BIH"