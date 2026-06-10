#!/usr/bin/env python3
"""
display_tracks.py — browser-based track visualiser (WSLg-compatible)

Replaces the X11/SDL2 GUI with a Flask web server that streams annotated
video as MJPEG and renders sensor charts as PNG via matplotlib.
No X11, Wayland, or SDL2 required — runs in any Windows browser over localhost.

Usage:  same CLI as before.
Opens:  http://localhost:5000  (auto-launched in Windows browser)
Stop:   Ctrl+C in terminal, or click Quit in browser.
"""

import argparse, json, os, signal as _sig, sqlite3, subprocess, threading, time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

# Vision feature classifiers — live per-frame labels for all temp_ids
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from vision_features.features.posture import extract_posture
    from vision_features.features.facing  import extract_facing
    from vision_features.schema import Posture, Facing, POSTURE_NAMES, FACING_NAMES
    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

# Drive I/O layer
from drive_manager import DriveManager, DriveNotSyncedError, DriveUnavailableError


# ── optional deps checked at runtime ─────────────────────────────────────────
try:
    from flask import Flask, Response, request, jsonify
except ImportError:
    print("[viewer] Flask not found.  Run:  pip install flask --break-system-packages")
    raise

def log(msg):
    print(f"[viewer] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared playback state  (video thread writes, Flask threads read)
# ═══════════════════════════════════════════════════════════════════════════════
class _State:
    FF_SPEEDS = [1, 2, 4, 8]

    def __init__(self):
        self._lock    = threading.Lock()
        self.paused   = False
        self.ff_idx   = 0
        self.ff_speed = 1
        self.quit     = False
        self.show_table = True
        self.current_dt = None      # datetime | None — updated by video thread

    def toggle_pause(self):
        with self._lock:
            self.paused = not self.paused
            return self.paused

    def cycle_ff(self):
        with self._lock:
            self.ff_idx   = (self.ff_idx + 1) % len(self.FF_SPEEDS)
            self.ff_speed = self.FF_SPEEDS[self.ff_idx]
            return self.ff_speed

    def do_quit(self):
        with self._lock:
            self.quit = True

    def set_dt(self, dt):
        with self._lock:
            self.current_dt = dt

    def snapshot(self):
        with self._lock:
            return dict(paused=self.paused, ff_speed=self.ff_speed,
                        quit=self.quit,
                        current_dt=self.current_dt.isoformat()
                        if self.current_dt else None)

STATE = _State()

# Latest annotated JPEG (video thread writes, MJPEG generator reads)
_latest_jpeg      = None
_latest_jpeg_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════
def id_color(tid):
    rng = np.random.default_rng(int(tid) * 123457)
    c = rng.integers(64, 255, size=3, dtype=np.uint8).tolist()
    return int(c[0]), int(c[1]), int(c[2])

def classify_frame_features(d: dict) -> dict | None:
    """
    Run posture + facing classifiers on a single detection dict.
    d must have "kps" (list of 19 (x,y,v) tuples) and "kps_conf" (list[19]).
    Returns {"posture": str|None, "facing": str|None} or None if unavailable.
    """
    if not _VISION_AVAILABLE:
        return None
    kps_raw  = d.get("kps")
    kps_conf = d.get("kps_conf")
    if not kps_raw or not kps_conf:
        return None
    try:
        # Reshape to (1, 19, 3) and (1, 19) for the vectorised extractors
        kps_arr  = np.array(kps_raw,  dtype=np.float32).reshape(1, 19, 3)
        kc_arr   = np.array(kps_conf, dtype=np.float32).reshape(1, 19)
        bbox_arr = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]]],
                            dtype=np.float32)
        det_arr  = np.array([d.get("conf", 1.0)], dtype=np.float32)

        p_out = extract_posture(kps_arr, kc_arr, bbox_arr, det_arr)
        f_out = extract_facing(kps_arr, kc_arr, bbox_arr,
                               posture=p_out["posture"])

        posture_label = int(p_out["posture"][0])
        facing_label  = int(f_out["facing"][0])

        # Return None for uncertain — draw_box omits None parts from the label
        posture_str = (POSTURE_NAMES[Posture(posture_label)]
                       if posture_label != int(Posture.UNCERTAIN) else None)
        facing_str  = (FACING_NAMES[Facing(facing_label)]
                       if facing_label != int(Facing.UNCERTAIN) else None)

        # Never return the string "uncertain" — treat it as None
        if posture_str == "uncertain": posture_str = None
        if facing_str  == "uncertain": facing_str  = None

        # Only return a dict if at least one label is meaningful
        if posture_str is None and facing_str is None:
            return None
        return {"posture": posture_str, "facing": facing_str}
    except Exception:
        return None


def draw_box(img, x1, y1, x2, y2, tid, conf=None, animal_id=None, features=None):
    """
    Draw bounding box with a two-line label stack above it:
        Line 1 (top)    — vision features: posture · facing  (only when available)
        Line 2 (bottom) — identity: "cow 6366 (t2)"  or  "id 2"
    conf is accepted for API compatibility but never displayed.

    features: dict with optional keys:
        "posture" : str  e.g. "standing" | "lying"
        "facing"  : str  e.g. "left" | "right" | "toward" | "away"
    """
    color = id_color(tid)
    x1,y1,x2,y2 = [int(round(float(v))) for v in (x1,y1,x2,y2)]
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    id_scale  = 0.55
    ft_scale  = 0.48
    pad       = 4

    # ── identity label ────────────────────────────────────────────────────────
    id_lbl = (f"cow {animal_id} (t{int(tid)})" if animal_id is not None
              else f"id {int(tid)}")
    (id_w, id_h), _ = cv2.getTextSize(id_lbl, font, id_scale, 2)
    id_pill_h = id_h + pad * 2
    id_y0     = max(0, y1 - id_pill_h)
    cv2.rectangle(img, (x1, id_y0), (x1 + id_w + pad*2, id_y0 + id_pill_h), color, -1)
    cv2.putText(img, id_lbl, (x1 + pad, id_y0 + id_h + pad - 1),
                font, id_scale, (0,0,0), 2, cv2.LINE_AA)

    # ── features label (posture · facing) ────────────────────────────────────
    feat_parts = []
    if features:
        if features.get("posture"): feat_parts.append(features["posture"])
        if features.get("facing"):  feat_parts.append(features["facing"])
    if feat_parts:
        ft_lbl = "  ·  ".join(feat_parts)
        (ft_w, ft_h), _ = cv2.getTextSize(ft_lbl, font, ft_scale, 1)
        ft_pill_h = ft_h + pad * 2
        ft_y0     = max(0, id_y0 - ft_pill_h)
        # slightly darker version of the track colour
        dark = tuple(max(0, int(c * 0.60)) for c in color)
        cv2.rectangle(img, (x1, ft_y0), (x1 + ft_w + pad*2, ft_y0 + ft_pill_h), dark, -1)
        cv2.putText(img, ft_lbl, (x1 + pad, ft_y0 + ft_h + pad - 1),
                    font, ft_scale, (255,255,255), 1, cv2.LINE_AA)

BASE_EDGES = [(0,1),(1,18),(18,2),(2,3),(3,4),(4,5),(5,6),
              (2,7),(7,8),(8,9),(2,10),(10,11),(11,12),
              (4,14),(14,15),(4,13),(13,16),(4,17)]
CUSTOM_EDGES = [(2,10),(10,12),(4,17),(14,15),(13,16)]
EDGE_SET = set(tuple(sorted(e)) for e in BASE_EDGES)
for e in CUSTOM_EDGES: EDGE_SET.add(tuple(sorted(e)))
EDGES = list(EDGE_SET)

def draw_pose(img, kps_xyv, color, kps_conf=None, kp_radius=3, sk_thickness=2,
              kp_thresh=0.0, kp_conf_thresh=0.30, hide_lowconf=False,
              show_index=False, index_scale=0.45, index_thickness=1, index_offset=6):
    K = len(kps_xyv)
    mask = [False]*K
    for i in range(K):
        x,y,v = kps_xyv[i]
        if v is None or float(v)<=kp_thresh: continue
        if hide_lowconf and kps_conf and i<len(kps_conf):
            try:
                if float(kps_conf[i])<kp_conf_thresh: continue
            except: pass
        mask[i]=True
    for i in range(K):
        if not mask[i]: continue
        x,y,_ = kps_xyv[i]
        xi,yi = int(round(x)),int(round(y))
        cv2.circle(img,(xi,yi),kp_radius,color,-1,lineType=cv2.LINE_AA)
        if show_index:
            s = str(i)
            cv2.putText(img,s,(xi+index_offset,yi-index_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,index_scale,(0,0,0),index_thickness+2,cv2.LINE_AA)
            cv2.putText(img,s,(xi+index_offset,yi-index_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,index_scale,(255,255,255),index_thickness,cv2.LINE_AA)
    for (i,j) in EDGES:
        if i<K and j<K and mask[i] and mask[j]:
            xi,yi,_ = kps_xyv[i]; xj,yj,_ = kps_xyv[j]
            cv2.line(img,(int(round(xi)),int(round(yi))),(int(round(xj)),int(round(yj))),
                     color,sk_thickness,lineType=cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite track stream  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def stream_sqlite(db_path, session_id, start_frame=0, want_pose=False, kps_parquet_path=""):
    import pandas as pd
    from datetime import datetime as _dt

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    kps_df = None
    if want_pose and kps_parquet_path and Path(kps_parquet_path).exists():
        kps_df = pd.read_parquet(kps_parquet_path)
        kps_df = kps_df[kps_df["session_id"]==session_id].set_index(["frame_index","temp_id"])
    try:
        rows = conn.execute("""
            SELECT frame_index,frame_datetime,temp_id,det_conf,x1,y1,x2,y2,kps_conf,kps_parquet_row
            FROM raw_tracks WHERE session_id=? AND frame_index>=?
            ORDER BY frame_index,temp_id""", (session_id,start_frame)).fetchall()
        has_dt=True
    except sqlite3.OperationalError:
        rows = conn.execute("""
            SELECT frame_index,temp_id,det_conf,x1,y1,x2,y2,kps_conf,kps_parquet_row
            FROM raw_tracks WHERE session_id=? AND frame_index>=?
            ORDER BY frame_index,temp_id""", (session_id,start_frame)).fetchall()
        has_dt=False
    conn.close()
    cur_fi=cur_fdt=None; bucket=[]
    for row in rows:
        fi=row["frame_index"]; fdt=None
        if has_dt:
            r=row["frame_datetime"]
            if r:
                try: fdt=(_dt.utcfromtimestamp(r) if isinstance(r,(int,float)) else _dt.fromisoformat(str(r)))
                except: pass
        if cur_fi is None: cur_fi,cur_fdt=fi,fdt
        if fi!=cur_fi:
            yield cur_fi,cur_fdt,bucket; bucket=[]; cur_fi,cur_fdt=fi,fdt
        d={"temp_id":row["temp_id"] if row["temp_id"] is not None else -1,
           "conf":row["det_conf"] or 0.0,
           "x1":row["x1"],"y1":row["y1"],"x2":row["x2"],"y2":row["y2"]}
        if want_pose and kps_df is not None:
            try:
                krow=kps_df.loc[(fi,row["temp_id"])]
                flat=list(krow["kps"])
                if len(flat)%3==0:
                    d["kps"]=[(flat[i],flat[i+1],flat[i+2]) for i in range(0,len(flat),3)]
                    d["kps_conf"]=list(krow["kps_kconf"])
            except: pass
        bucket.append(d)
    if bucket: yield cur_fi,cur_fdt,bucket


# ═══════════════════════════════════════════════════════════════════════════════
# Vision features  (loaded once from resolved_cow_timeline)
# ═══════════════════════════════════════════════════════════════════════════════
def load_vision_features(db_path: str, session_id: str) -> dict:
    """
    Load posture + facing features from resolved_cow_timeline for this session.

    Returns nested dict:
        { real_id (int) -> { window_start_dt (datetime) -> {"posture": str, "facing": str} } }

    Returns empty dict if the vision columns don't exist yet (pre-migration DB).
    """
    from datetime import datetime as _dt
    result = {}
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT real_id, window_start_dt, facing_dominant, lying_fraction
            FROM   resolved_cow_timeline
            WHERE  session_id = ?
              AND  (facing_dominant IS NOT NULL OR lying_fraction IS NOT NULL)
        """, (session_id,)).fetchall()
        conn.close()
        for real_id, win_dt_str, facing, lying_frac in rows:
            if real_id is None:
                continue
            try:
                win_dt = _dt.fromisoformat(str(win_dt_str))
            except Exception:
                continue
            posture = None
            if lying_frac is not None:
                posture = "lying" if float(lying_frac) >= 0.5 else "standing"
            # Sanitise: treat "uncertain" and empty string as None
            clean_facing = facing if (facing and facing != "uncertain") else None
            result.setdefault(int(real_id), {})[win_dt] = {
                "posture": posture,
                "facing":  clean_facing,
            }
        log(f"Vision features loaded: {sum(len(v) for v in result.values())} windows "
            f"across {len(result)} cows")
    except sqlite3.OperationalError:
        log("Vision feature columns not yet in DB — labels will be omitted")
    return result


def get_features_for(vision_index: dict, real_id: int, frame_dt) -> dict | None:
    """
    Look up vision features for a cow at a given frame datetime.
    Finds the most recent window that started at or before frame_dt.
    Returns None if no data available.
    """
    if real_id is None or real_id not in vision_index or frame_dt is None:
        return None
    windows = vision_index[real_id]
    best = None
    for win_dt in windows:
        if win_dt <= frame_dt:
            if best is None or win_dt > best:
                best = win_dt
    return windows[best] if best is not None else None



# ═══════════════════════════════════════════════════════════════════════════════
# Sensor data  (loaded once into DataFrames)
# ═══════════════════════════════════════════════════════════════════════════════
def load_sensor_data(kin_path, beh_path):
    import pandas as pd
    frames={}
    if kin_path and Path(kin_path).exists():
        kdf=pd.read_csv(kin_path,parse_dates=["datetime"])
        kdf=kdf.sort_values(["AnimalId","datetime"]).reset_index(drop=True)
        for col in ["KineticsCountX","KineticsCountY","KineticsCountZ","KineticsCountR"]:
            kdf[f"d{col.replace('KineticsCount','Kin')}"]=kdf.groupby("AnimalId")[col].diff()
        for aid,grp in kdf.groupby("AnimalId"):
            frames[int(aid)]=grp.set_index("datetime").sort_index()
    if beh_path and Path(beh_path).exists():
        bdf=pd.read_csv(beh_path,parse_dates=["datetime"])
        bdf=bdf.sort_values(["AnimalId","datetime"]).reset_index(drop=True)
        for aid,grp in bdf.groupby("AnimalId"):
            aid=int(aid)
            sub=grp.set_index("datetime")[["f_1_2","f_2_3","v"]].sort_index()
            frames[aid]=(frames[aid].join(sub,how="outer").sort_index() if aid in frames else sub)
    return frames

SENSOR_DATA   = {}   # aid → DataFrame
SENSOR_AIDS   = []   # sorted list of animal ids
_SENSOR_LOCK  = threading.Lock()   # guards matplotlib figure generation


# ═══════════════════════════════════════════════════════════════════════════════
# Sensor chart PNG generator
# ═══════════════════════════════════════════════════════════════════════════════
_SIGNAL_META = [
    ("f_1_2",  "f₁₂",  "Behaviour · f₁₂"),
    ("f_2_3",  "f₂₃",  "Behaviour · f₂₃"),
    ("v",      "v",    "Behaviour · v"),
    ("dKinX",  "ΔKinX","Kinetics · ΔX"),
    ("dKinY",  "ΔKinY","Kinetics · ΔY"),
    ("dKinZ",  "ΔKinZ","Kinetics · ΔZ"),
    ("dKinR",  "ΔKinR","Kinetics · ΔR"),
]
_LINE_COLORS = ["#89b4fa","#cba6f7","#a6e3a1","#fab387","#f9e2af","#89dceb","#f38ba8"]
_PLOT_BG  = "#181825"
_BG       = "#1e1e2e"
_MUTED    = "#6c7086"
_OVERLAY  = "#45475a"
_CURSOR   = "#f38ba8"

def render_sensor_png(aid: int, now_dt, lookback_hours: float) -> bytes:
    """Generate a 7-panel sensor chart and return it as PNG bytes."""
    import io, matplotlib
    matplotlib.use("Agg")   # non-interactive backend — no display needed
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates

    df = SENSOR_DATA.get(int(aid))
    lookback  = timedelta(hours=lookback_hours)
    win_start = now_dt - lookback
    win_end   = now_dt
    right_pad = lookback * 0.04

    n = len(_SIGNAL_META)
    with _SENSOR_LOCK:
        fig = Figure(figsize=(12, 8), facecolor=_PLOT_BG)
        fig.subplots_adjust(hspace=0.06, top=0.97, bottom=0.07, left=0.07, right=0.97)
        shared_ax = None
        axes = []
        for i in range(n):
            ax = fig.add_subplot(n, 1, i+1, sharex=shared_ax)
            if shared_ax is None: shared_ax = ax
            axes.append(ax)

        for i,(ax,color,(col,ylabel,title)) in enumerate(zip(axes,_LINE_COLORS,_SIGNAL_META)):
            ax.set_facecolor(_PLOT_BG)
            ax.tick_params(colors=_MUTED, labelsize=7)
            ax.spines[:].set_color(_OVERLAY)
            ax.set_ylabel(ylabel, fontsize=8, color=_MUTED)
            ax.set_title(title, fontsize=8, color=_MUTED, loc="left", pad=2)
            ax.yaxis.grid(True, color=_OVERLAY, linewidth=0.5, linestyle="--")
            ax.set_axisbelow(True)
            if i < n-1: ax.tick_params(labelbottom=False)

            if df is not None and col in df.columns:
                series = df[col].loc[:win_end].dropna()
                if not series.empty:
                    ax.plot(series.index, series.values, color=color, linewidth=1.1, alpha=0.92)
                    ax.fill_between(series.index, series.values, alpha=0.10, color=color)

            ax.axvline(x=now_dt, color=_CURSOR, linewidth=1.4, linestyle="--", alpha=0.85, zorder=10)
            ax.set_xlim(win_start, win_end+right_pad)

            if df is not None and col in df.columns:
                vis = df[col].loc[win_start:win_end].dropna()
                if not vis.empty:
                    lo,hi = vis.min(),vis.max()
                    m = max((hi-lo)*0.12, abs(hi)*0.05, 1e-6)
                    ax.set_ylim(lo-m, hi+m)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        axes[-1].tick_params(axis="x", labelsize=7, colors=_MUTED)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=_PLOT_BG, dpi=110)
        fig.clf()

    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════════
# Flask app  (runs in main thread via app.run)
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder=None)
app.logger.disabled = True
import logging; logging.getLogger("werkzeug").setLevel(logging.ERROR)

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Display Tracks</title>
<style>
* { box-sizing: border-box; margin:0; padding:0; }
body { background:#1e1e2e; color:#cdd6f4; font-family:Helvetica,sans-serif;
       display:flex; flex-direction:column; height:100vh; overflow:hidden; }
#top  { display:flex; flex:1; overflow:hidden; }
#vid  { flex:0 0 auto; padding:8px; display:flex; align-items:flex-start; }
#vid img { border:2px solid #45475a; max-height:calc(100vh - 60px); width:auto; }
#sensor { flex:1; padding:8px; overflow:hidden; display:flex; flex-direction:column; }
#animalRow { display:flex; gap:8px; margin-bottom:8px; flex-wrap:wrap; align-items:center; }
#sensorImg { flex:1; overflow:hidden; }
#sensorImg img { width:100%; height:100%; object-fit:contain; }
#bar { background:#313244; padding:8px 14px; display:flex; gap:10px;
       align-items:center; height:52px; flex-shrink:0; }
button { background:#45475a; color:#cdd6f4; border:none; padding:8px 18px;
         font-size:13px; cursor:pointer; border-radius:4px; }
button:hover { background:#585b70; }
button.active { background:#a6e3a1; color:#1e1e2e; }
.abtn { background:#89b4fa; color:#1e1e2e; font-weight:bold; border-radius:4px;
        padding:6px 14px; cursor:pointer; border:none; }
.abtn.sel { background:#a6e3a1; }
#clock { color:#89b4fa; font-weight:bold; margin-left:auto; font-size:13px; }
#ffLabel { color:#cdd6f4; font-size:13px; }
#qbtn   { color:#f38ba8; }
#status { font-size:11px; color:#6c7086; }
</style>
</head>
<body>
<div id="top">
  <div id="vid">
    <img id="vstream" src="/video" alt="video stream">
  </div>
  <div id="sensor">
    <div id="animalRow">
      <span style="color:#6c7086;font-size:12px;">Animal ID:</span>
      <!-- buttons injected by JS -->
    </div>
    <div id="sensorImg">
      <img id="simg" src="" alt="select an animal" style="display:none">
      <span id="noSensor" style="color:#6c7086;font-size:13px;">← select an animal to see sensor data</span>
    </div>
  </div>
</div>
<div id="bar">
  <button id="pauseBtn" onclick="togglePause()">⏸ Pause</button>
  <button onclick="cycleFF()">⏩ Fast Forward</button>
  <span id="ffLabel">Speed: 1×</span>
  <button onclick="toggleTable()">📊 Toggle Score Table</button>
  <button id="qbtn" onclick="doQuit()">⏹ Quit</button>
  <span id="clock"></span>
  <span id="status"></span>
</div>
<script>
let selectedAid = null;
let sensorTimer = null;
const LOOKBACK = 2.0;

// ── populate animal buttons ──────────────────────────────────────────────────
fetch('/api/animals').then(r=>r.json()).then(aids=>{
  const row = document.getElementById('animalRow');
  aids.forEach(aid=>{
    const b = document.createElement('button');
    b.className='abtn'; b.textContent=aid; b.id='abtn_'+aid;
    b.onclick=()=>selectAnimal(aid);
    row.appendChild(b);
  });
});

function selectAnimal(aid){
  selectedAid = aid;
  document.querySelectorAll('.abtn').forEach(b=>b.classList.remove('sel'));
  const btn = document.getElementById('abtn_'+aid);
  if(btn) btn.classList.add('sel');
  document.getElementById('noSensor').style.display='none';
  document.getElementById('simg').style.display='block';
  refreshSensor();
}

function refreshSensor(){
  if(!selectedAid) return;
  fetch('/api/status').then(r=>r.json()).then(s=>{
    if(!s.current_dt) return;
    const url = `/api/sensor_chart?aid=${selectedAid}&lookback=${LOOKBACK}&_t=${Date.now()}`;
    document.getElementById('simg').src = url;
  });
}

// ── status polling ───────────────────────────────────────────────────────────
let paused=false;
setInterval(()=>{
  fetch('/api/status').then(r=>r.json()).then(s=>{
    paused = s.paused;
    document.getElementById('pauseBtn').textContent = s.paused ? '▶ Resume' : '⏸ Pause';
    document.getElementById('pauseBtn').classList.toggle('active', s.paused);
    document.getElementById('ffLabel').textContent = `Speed: ${s.ff_speed}×`;
    if(s.current_dt) document.getElementById('clock').textContent = '▶  '+s.current_dt.replace('T',' ').slice(0,19);
    if(s.quit){ document.getElementById('status').textContent='[stopped]'; }
  }).catch(()=>{});
}, 500);

// sensor refresh every 3s
setInterval(refreshSensor, 3000);

// ── controls ─────────────────────────────────────────────────────────────────
function togglePause(){ fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'pause'})}); }
function cycleFF()    { fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'ff'})}); }
function toggleTable(){ fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'table'})}); }
function doQuit()     { fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'quit'})}); }
</script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML_PAGE

@app.route("/video")
def video():
    def generate():
        last = None
        while True:
            with _latest_jpeg_lock:
                jpeg = _latest_jpeg
            if jpeg is None or jpeg is last:
                time.sleep(0.02)
                continue
            last = jpeg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.02)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/animals")
def api_animals():
    return jsonify(SENSOR_AIDS)

@app.route("/api/status")
def api_status():
    return jsonify(STATE.snapshot())

@app.route("/api/control", methods=["POST"])
def api_control():
    action = request.json.get("action","")
    if action=="pause":   STATE.toggle_pause()
    elif action=="ff":    STATE.cycle_ff()
    elif action=="table": STATE.show_table = not STATE.show_table
    elif action=="quit":  STATE.do_quit()
    return jsonify({"ok":True})

@app.route("/api/sensor_chart")
def api_sensor_chart():
    try:
        aid   = int(request.args.get("aid",-1))
        lb    = float(request.args.get("lookback",2.0))
        now   = STATE.snapshot()["current_dt"]
        if now is None:
            return Response(b"", mimetype="image/png")
        from datetime import datetime as _dt
        now_dt = _dt.fromisoformat(now)
        png = render_sensor_png(aid, now_dt, lb)
        return Response(png, mimetype="image/png",
                        headers={"Cache-Control":"no-store"})
    except Exception as e:
        log(f"sensor_chart error: {e}")
        return Response(b"", mimetype="image/png")


# ═══════════════════════════════════════════════════════════════════════════════
# Video loop  (background thread)
# ═══════════════════════════════════════════════════════════════════════════════
def _video_loop(args, assignment, scores_df, session_start_dt, vision_index=None):
    global _latest_jpeg

    vid = Path(args.video)
    db  = Path(args.db)
    kps_pq = str(db.parent/"kps.parquet")

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        log(f"Cannot open video: {vid}"); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"Video: {W}x{H} @ {fps:.2f} fps")

    row_stream = stream_sqlite(str(db), args.session_id,
                               start_frame=args.start,
                               want_pose=args.draw_pose,
                               kps_parquet_path=kps_pq)
    try:
        next_fi, next_fdt, next_rows = next(row_stream)
        log(f"First track frame_index: {next_fi}  rows: {len(next_rows)}")
    except StopIteration:
        log("No tracks found."); cap.release(); return

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_idx   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_fdt = None
    stream_done = False
    n_frames    = 0

    kp_thresh_eff = args.kp_thresh
    if args.hide_occluded: kp_thresh_eff = max(kp_thresh_eff, 0.5)

    # target ~25 fps output to browser (don't saturate the MJPEG stream)
    _frame_interval = 1.0 / min(fps, 25.0)

    while True:
        if STATE.quit: break

        if STATE.paused:
            time.sleep(0.05)
            continue

        ff_speed = STATE.ff_speed

        ok, frame = cap.read()
        if not ok:
            log("End of video."); break

        if ff_speed > 1:
            for _ in range(ff_speed-1):
                if not cap.grab(): break
                frame_idx+=1; n_frames+=1
                if not stream_done:
                    while next_fi < frame_idx:
                        try: next_fi,next_fdt,next_rows = next(row_stream)
                        except StopIteration:
                            stream_done=True; next_rows=[]; next_fi=frame_idx; break
                if args.limit>0 and n_frames>=args.limit: break

        if not stream_done:
            while next_fi < frame_idx:
                try: next_fi,next_fdt,next_rows = next(row_stream)
                except StopIteration:
                    stream_done=True; next_rows=[]; next_fi=frame_idx; break

        if next_fi==frame_idx:
            dets=next_rows
            if next_fdt is not None: current_fdt=next_fdt
            if not stream_done:
                try: next_fi,next_fdt,next_rows = next(row_stream)
                except StopIteration:
                    stream_done=True; next_rows=[]; next_fi=10**12
        else:
            dets=[]

        if session_start_dt is not None:
            current_fdt = session_start_dt + timedelta(seconds=frame_idx/fps)
        if current_fdt: STATE.set_dt(current_fdt)

        for d in dets:
            _aid = assignment.get(int(d["temp_id"]))

            # Live classification (all temp_ids, requires --draw_pose / kps loaded)
            _feats = classify_frame_features(d) if d.get("kps") else None

            # Fall back to DB lookup for resolved cows when live classify unavailable
            if _feats is None and _aid is not None:
                _feats = get_features_for(vision_index or {}, _aid, current_fdt)

            draw_box(frame, d["x1"], d["y1"], d["x2"], d["y2"],
                     d["temp_id"], d["conf"],
                     animal_id=_aid, features=_feats)
            if args.draw_pose and d.get("kps"):
                draw_pose(frame,d["kps"],
                          color=id_color(int(d["temp_id"])),
                          kps_conf=d.get("kps_conf"),
                          kp_radius=args.kp_radius,
                          sk_thickness=args.sk_thickness,
                          kp_thresh=kp_thresh_eff,
                          kp_conf_thresh=args.kp_conf_thresh,
                          hide_lowconf=(not args.show_lowconf),
                          show_index=args.kp_index,
                          index_scale=args.kp_index_scale,
                          index_thickness=args.kp_index_thickness,
                          index_offset=args.kp_index_offset)

        if ff_speed>1:
            lbl=f">> {ff_speed}x"
            (sw,sh),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,1.1,3)
            sx,sy=frame.shape[1]-sw-14,38
            cv2.putText(frame,lbl,(sx,sy),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(frame,lbl,(sx,sy),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,220,255),2,cv2.LINE_AA)

        # encode to JPEG and publish
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok2:
            with _latest_jpeg_lock:
                _latest_jpeg = buf.tobytes()

        frame_idx+=1; n_frames+=1
        if args.limit>0 and n_frames>=args.limit:
            log("Limit reached."); break

        time.sleep(_frame_interval / max(ff_speed,1))

    cap.release()
    log("Video loop done.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True)
    ap.add_argument("--db",         required=True)
    ap.add_argument("--session_id", required=True)
    ap.add_argument("--start",      type=int,   default=0)
    ap.add_argument("--max_fps",    type=float, default=0.0)
    ap.add_argument("--sink",       default="web",
                    choices=["web","mp4"],
                    help="web = browser MJPEG (default), mp4 = save to file")
    ap.add_argument("--outmp4",     default="annotated.mp4")
    ap.add_argument("--limit",      type=int,   default=0)
    ap.add_argument("--port",       type=int,   default=5000)
    ap.add_argument("--draw_pose",  action="store_true")
    ap.add_argument("--kp_radius",  type=int,   default=3)
    ap.add_argument("--sk_thickness",type=int,  default=2)
    ap.add_argument("--kp_thresh",  type=float, default=0.0)
    ap.add_argument("--hide_occluded",  action="store_true")
    ap.add_argument("--kp_conf_thresh", type=float, default=0.30)
    ap.add_argument("--show_lowconf",   action="store_true")
    ap.add_argument("--kp_index",       action="store_true")
    ap.add_argument("--kp_index_scale",     type=float, default=0.45)
    ap.add_argument("--kp_index_thickness", type=int,   default=1)
    ap.add_argument("--kp_index_offset",    type=int,   default=6)
    ap.add_argument("--kinetic_csv",  default="")
    ap.add_argument("--behavior_csv", default="")
    ap.add_argument("--sensor_lookback", type=float, default=2.0)
    ap.add_argument("--bypass_upload_check", action="store_true",
                    help="Skip dirty-flag check when reading from Drive "
                         "(proceeds with potentially stale data — use with caution)")
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    global SENSOR_DATA, SENSOR_AIDS
    args = parse_args()

    vid = Path(args.video)
    if not vid.exists(): raise FileNotFoundError(f"Video not found: {vid}")

    # ── Drive: pull canonical DB before reading ───────────────────────────────
    dm = DriveManager(bypass=args.bypass_upload_check, caller=__file__)
    try:
        dm.check_flag("db")
    except DriveNotSyncedError as e:
        log(f"ERROR: {e}")
        log("Use --bypass_upload_check to proceed with current Drive state.")
        return
    try:
        db = dm.pull_db(allow_stale=args.bypass_upload_check)
    except DriveUnavailableError as e:
        log(f"ERROR pulling DB: {e}")
        return

    # ── identity assignment ──────────────────────────────────────────────────
    assignment = {}
    try:
        conn = sqlite3.connect(str(db))
        for tid,aid in conn.execute(
                "SELECT temp_id,real_id FROM manual_assignments WHERE session_id=?",
                (args.session_id,)).fetchall():
            assignment[int(tid)]=int(aid)
        for real_id,kj in conn.execute(
                "SELECT real_id,known_temp_ids FROM reid_registry WHERE known_temp_ids IS NOT NULL"
                ).fetchall():
            try:
                for e in json.loads(kj):
                    if e.get("session_id")==args.session_id:
                        tid=int(e["temp_id"])
                        if tid not in assignment: assignment[tid]=int(real_id)
            except: pass
        conn.close()
        if assignment:
            log("Assignments: "+", ".join(f"t{t}->{a}" for t,a in sorted(assignment.items())))
    except Exception as e:
        log(f"WARNING: assignments not loaded: {e}")

    # ── session start time ───────────────────────────────────────────────────
    session_start_dt = None
    try:
        from datetime import datetime as _dt2
        c2=sqlite3.connect(str(db))
        row=c2.execute("SELECT start_dt FROM video_sessions WHERE session_id=?",
                       (args.session_id,)).fetchone()
        c2.close()
        if row and row[0]:
            session_start_dt=_dt2.fromisoformat(str(row[0]))
            log(f"Session start_dt: {session_start_dt}")
    except: pass
    if session_start_dt is None:
        import re; from datetime import datetime as _dt3
        m=re.search(r'_S(\d{14})',vid.name)
        if m:
            try: session_start_dt=_dt3.strptime(m.group(1),"%Y%m%d%H%M%S"); log(f"start_dt from filename: {session_start_dt}")
            except: pass

    # ── sensor data ──────────────────────────────────────────────────────────
    if args.kinetic_csv or args.behavior_csv:
        log("Loading sensor data...")
        SENSOR_DATA = load_sensor_data(args.kinetic_csv, args.behavior_csv)
        if SENSOR_DATA:
            SENSOR_AIDS = sorted(SENSOR_DATA.keys())
            log(f"Sensor data loaded for AnimalIds: {SENSOR_AIDS}")
        else:
            log("No sensor data loaded.")

    # ── vision features ──────────────────────────────────────────────────────
    vision_index = load_vision_features(str(db), args.session_id)

    # ── SIGINT ───────────────────────────────────────────────────────────────
    def _sigint(s,f):
        log("Interrupted.")
        os._exit(0)
    _sig.signal(_sig.SIGINT, _sigint)

    # ── video thread ─────────────────────────────────────────────────────────
    vt = threading.Thread(
        target=_video_loop,
        args=(args, assignment, None, session_start_dt, vision_index),
        daemon=True)
    vt.start()

    # ── open Windows browser ─────────────────────────────────────────────────
    url = f"http://localhost:{args.port}"
    log(f"Starting web server on {url}")
    log(f"Opening browser...")
    try:
        subprocess.Popen(
            ["cmd.exe", "/c", "start", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        try:
            subprocess.Popen(
                ["/mnt/c/Windows/System32/cmd.exe", "/c", "start", url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            log(f"Could not auto-open browser. Open manually: {url}")

    # ── Flask (blocking, main thread) ────────────────────────────────────────
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()


# Example usage (--sink flag is now optional, defaults to web):
# python3 display_tracks.py \
#   --video      "$VID" \
#   --db         "$DB" \
#   --session_id refet33_20241221 \
#   --draw_pose --kp_index --hide_occluded --kp_conf_thresh 0.35 \
#   --kinetic_csv  "$KIN" \
#   --behavior_csv "$BIH"