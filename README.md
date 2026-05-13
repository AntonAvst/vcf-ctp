# vcf-ctp
**V**ision–**C**ollar **F**usion for **C**alving-**T**ype **P**rediction in Dairy Cows

# Full Architecture Reference

## Project
**Thesis:** Predictive Modelling of Calving Outcomes in Dairy Cows Using Multi-Modal Sensor and Vision Data  
**Author:** Anton Avstreikh, University of Haifa  
**Advisor:** Prof. Ilan Shimshoni  
**Output:** 4-class calving type prediction — Unassisted · Assisted · Twin · Veterinarian-assisted

---

## Scripts

### `track_and_dump.py`
Inference-only. Runs YOLO detection + ByteTrack/BoT-SORT tracking + YOLOv8-Pose (19 kp) + Embedder128 (MobileNetV3 → 128D L2-norm).

**CLI flags:** `--model`, `--source`, `--outdir`, `--session_id`, `--camera_id`, `--tracker`, `--imgsz`, `--conf`, `--iou`, `--save_crops`, `--crop_every`, `--min_crop_wh`, `--embed_size`, `--pose_model`, `--pose_imgsz`, `--pose_conf`, `--pose_kp_conf_thresh`, `--commit_every`

**Outputs written to `--outdir`:**
- `calving_project.db` — SQLite: `video_sessions` row + `raw_tracks` rows (scalar columns only)
- `embeds.parquet` — embed[128] per detection (float32, Snappy compressed)
- `kps.parquet` — kps[57 = 19×3 flat] + kps_kconf[19] per detection (float32, Snappy compressed)
- `crops/` — optional JPEG crops

**Key rules:**
- No CSV, no JSONL. Arrays never stored in SQLite.
- `embed_parquet_row` and `kps_parquet_row` in `raw_tracks` are integer pointers into the parquets.
- Embeddings are always computed (not gated by a flag).
- SQLite commits every `--commit_every` frames (default 50) — same crash-safety as old CSV flush. WAL mode enabled.
- `session_id` defaults to video filename stem if not supplied.
- Wall-clock start time parsed from filename token `_S<YYYYMMDDHHmmss>`. `end_dt` stamped after loop finishes.

### `display_tracks.py`
Visualization tool. Overlays tracks + pose skeleton on video. Uses SQLite as track source (no CSV).

**CLI flags:** `--video`, `--db`, `--session_id`, `--kinetics`, `--corr_threshold`, `--min_active_bins`, `--bin_minutes`, `--min_temp_id_frames`, `--start`, `--max_fps`, `--show_fps`, `--sink` (ffplay/cv2/mp4), `--outmp4`, `--limit`, `--draw_pose`, `--kp_radius`, `--sk_thickness`, `--kp_thresh`, `--hide_occluded`, `--kp_conf_thresh`, `--show_lowconf`, `--kp_index`, `--kp_index_scale`, `--kp_index_thickness`, `--kp_index_offset`

**Track source:** `stream_sqlite(db, session_id, ...)` — queries `raw_tracks` ordered by `frame_index`, joins `kps.parquet` (auto-located as `db.parent/kps.parquet`) for pose data. Same `(frame_index, [dets])` generator contract as old CSV stream.

**Live identity matching:** imports `match_identity` module. Loads `raw_tracks` from SQLite via `pd.read_sql`. Recomputes Pearson r scores at each new kinetics interval boundary during playback. Draws score table overlay (temp_id × AnimalId correlation matrix).

**Sinks:** ffplay / cv2 / mp4. TkControls panel for pause/FF/table toggle/quit.

### `match_identity.py`
Kinetic matching: bbox centroid speed ↔ collar ΔKineticsCountR, Pearson correlation per 15-min bin, greedy assignment.

**Key functions:**
- `run(tracks_path, kinetics_path, ...)` → `(assignment dict, scores DataFrame)`
- `score_up_to(tracks_df, kinetics_df, up_to_datetime, ...)` → `(assignment, scores)` — used by `display_tracks.py` for live playback matching

### `reconcile.py`
Post-processing pipeline. Runs after `track_and_dump.py`. Reads `raw_tracks` + collar CSVs, resolves `real_id`, writes `resolved_cow_timeline`.

**CLI flags:** `--db`, `--session`, `--kinetics`, `--tracks` (tracks CSV path, for kinetic matching), `--gallery_dir`, `--embed_parquet`, `--corr_threshold`, `--min_active_bins`, `--min_temp_id_frames`, `--activity_pct`, `--bin_minutes`, `--ema_alpha`, `--min_embeds_gallery`, `--cosine_threshold`, `--cosine_min_embeds`, `--dry_run`, `--verbose`

**Steps (in order):**

**A. Kinetic matcher** — bbox centroid speed ↔ ΔKineticsCountR · Pearson r · greedy assignment → `{temp_id → AnimalId}`. Inlined from `match_identity.py` logic (no import dependency).

**B. Gallery builder** — for each kinetically-confirmed pair: mean-pool embeds → EMA-blend into `gallery_day.npy` or `gallery_night.npy` (routed by `video_sessions.is_night`). Updates `reid_registry` in SQLite. α = `--ema_alpha` (default 0.15).

**C. Cosine resolver** — for temp_ids not resolved by kinetics: query the correct gallery (day vs night) via cosine similarity. Heals within-session tracker switches. Cosine-confirmed sessions get conservative α/2 gallery update. Falls back gracefully when gallery is empty.

**D. Sensor sequencer** — forward-fill behavior (~90s) and kinetics (~15min) onto the video time grid. Auto-discovers `behavior_data_*.csv` alongside the kinetics file. Produces one row per `(real_id, 15-min bin)`.

**E. Write resolved_cow_timeline** — inserts sensor-grid rows into SQLite. Vision feature columns (`spine_angle`, etc.) are NULL until pose extractor is added.

---

## Sensor Data

### `behavior_data_*.csv`
- Columns: AnimalId, datetime, f_1_2, f_2_3, v
- Interval: ~90 seconds
- Proprietary collar-derived behavioral features

### `kinetic_data_*.csv`
- Columns: AnimalId, datetime, KineticsCountX, KineticsCountY, KineticsCountZ, KineticsCountR
- Interval: ~15 minutes
- Cumulative accelerometer counts; deltas computed for matching and features

---

## Three-Stage Pipeline

### Stage 1 — Inference (`track_and_dump.py`, per video)
**Inputs:** Raw MP4, YOLO .pt models  
**Process:** Detect → Track (temp_id) → Pose (kps) → Embed (128D)  
**Outputs written:**
- `raw_tracks` — append-only, one row per detection (scalars only)
- `embeds.parquet` — embed[128] per detection
- `kps.parquet` — kps[57] + kps_kconf[19] per detection
- `video_sessions` — one row per video registered

**Key rule:** No identity resolution at inference time. Embeddings always computed.

### Stage 2 — Post-processing (`reconcile.py`, once per video after Stage 1)
**Steps A–E** as described above.  
**Output written:** `resolved_cow_timeline` — central join table

**Key rule:** Kinetic matching is always the primary confirmation signal. Cosine resolver is secondary (heals switches, handles cross-video). Both run post-hoc, never during inference.

### Stage 3 — Model (dataset builder + CNN-LSTM/GRU)
**Inputs:** `resolved_cow_timeline` + `calving_ledger` + `cow_registry`  
**Process:** Build labeled multi-hour windows per calving event → train temporal model  
**Output:** 4-class probability distribution + risk score + prediction horizon

---

## Databases (7 total)

### `cow_registry` — static lookup
- real_id (PK), breed, parity, pen_id, collar_id, baseline_window
- Used for: sensor normalization baseline, static feature injection

### `video_sessions` — one row per video file
- session_id (PK), video_path, camera_id, start_dt, end_dt, collar_csv_path
- **is_night** (bool) — auto-detected: sample N frames, check if mean per-channel variance is below threshold (IR/grayscale frames have near-zero R/G/B channel divergence)
- Written by `track_and_dump.py` at run start; `end_dt` stamped after loop finishes

### `collar_signals` — raw sensor time-series
- AnimalId (FK → cow_registry), datetime, signal_type ('behavior' | 'kinetic')
- f_1_2, f_2_3, v (behavior) + kin_X, kin_Y, kin_Z, kin_R (kinetics)
- Ingested separately from video pipeline

### `raw_tracks` — append-only inference output
- session_id (FK → video_sessions), frame_index, frame_datetime, temp_id
- bbox (x1/y1/x2/y2), det_conf, cx, cy, w, h
- kps_conf (scalar mean confidence across keypoints)
- **embed_parquet_row** (INTEGER) — pointer into `embeds.parquet`
- **kps_parquet_row** (INTEGER) — pointer into `kps.parquet`
- Index on (session_id, frame_index)

Arrays are NOT stored in SQLite — they live in companion Parquet files:

**`embeds.parquet`** — columns: session_id, frame_index, temp_id, embed (FixedSizeList[128 × float32])  
**`kps.parquet`** — columns: session_id, frame_index, temp_id, kps (FixedSizeList[57 × float32]), kps_kconf (FixedSizeList[19 × float32])

### `reid_registry` — one row per confirmed real identity
- real_id (PK = AnimalId from collar)
- **gallery_embed_day[128]** — EMA mean embedding from daytime sessions (RGB)
- **gallery_embed_night[128]** — EMA mean embedding from night/IR sessions (grayscale)
- gallery_n_day, gallery_n_night — session counts contributing to each gallery
- gallery_confidence_day, gallery_confidence_night — quality score
- last_updated_day_dt, last_updated_night_dt
- known_temp_ids (JSON list of {session_id, temp_id} objects)
- first_seen_dt, match_method ('kinetic' | 'cosine_day' | 'cosine_night')
- Updated by `reconcile.py` after each video

**EMA update rule:**
```
gallery_embed_new = α × mean(session_embeds) + (1 − α) × gallery_embed_old
α = 0.15  (default; slow drift, old sightings fade gradually)
```
- Day sessions (is_night=False) → update gallery_embed_day only
- Night sessions (is_night=True) → update gallery_embed_night only
- Galleries NEVER cross-contaminate across modalities
- Only kinetic-confirmed sessions trigger a full α update
- Cosine-only confirmed sessions trigger a conservative α/2 update

Gallery vectors also persisted as `.npy` files:  
`gallery_day.npy`, `gallery_night.npy` — dict `{real_id (int): np.ndarray(128,)}`

### `resolved_cow_timeline` — CENTRAL JOIN TABLE
- real_id (FK → reid_registry, nullable), window_start_dt, session_id
- modality_mask (bits: 1=sensor_ok | 2=vision_ok | 4=reid_ok)
- **Sensor cols** (forward-filled): Δf_12, Δf_23, Δv, Δkin_X, Δkin_Y, Δkin_Z, Δkin_R
- **Vision cols** (derived from kps by reconcile.py, currently NULL — pose extractor pending):
  - spine_angle (kp2→kp3→kp4)
  - pelvic_tilt (kp7 ↔ kp10)
  - tail_elevation (kp5/kp6 vs kp4)
  - limb_symmetry (L/R hock distance ratio)
  - head_drop (kp0/kp1 vs kp2)
  - lying_flag (bbox aspect ratio heuristic)
  - restlessness (variance of spine_angle over window)
  - kps_coverage (mean conf across 19 kp — reliability indicator)
  - embed_mean[128] (mean pool of embed rows in window, stored as JSON)
- Raw kps arrays are NOT copied here — they stay in `kps.parquet`

### `calving_ledger` — ground truth
- event_id (PK), real_id (FK → reid_registry), calving_dt
- outcome: Unassisted | Assisted | Twin | Veterinarian-assisted

---

## Pose Keypoint Index Map (19 kp)
```
0  nose          1  forehead      2  withers       3  spine_mid
4  sacrum        5  tail_base     6  tail_tip       7  shoulder_L
8  elbow_L       9  fetlock_fore_L  10 shoulder_R  11 elbow_R
12 fetlock_fore_R  13 hock_R      14 hock_L        15 fetlock_hind_L
16 fetlock_hind_R  17 udder_center  18 neck
```

kps stored flat in parquet as [x0,y0,v0, x1,y1,v1, ... x18,y18,v18] = 57 values.

---

## Key Design Decisions

1. **No ReID during inference** — temp_ids are saved raw; all identity resolution is post-hoc in `reconcile.py`
2. **Wall-clock as universal join key** — `frame_datetime` links video to collar data; filename `_S<YYYYMMDDHHmmss>` is the source
3. **real_id is nullable** — pipeline doesn't block on unresolved identities; `modality_mask` signals quality
4. **Two-stage identity resolution** — kinetic matching (primary, offline) confirms AnimalId; cosine resolver (secondary) heals temp_id switches and enables cross-video continuity
5. **Arrays never in SQLite** — `raw_tracks` holds scalars only; embed[128] and kps[57] live in Parquet files pointed to by integer row indices
6. **Pose raw data stays in `kps.parquet`** — `reconcile.py` extracts scalar features into `resolved_cow_timeline`; raw kps recomputable any time
7. **Sensor temporal mismatch handled by forward-fill** — behavior ~90s, kinetics ~15min → upsampled into video time grid in `resolved_cow_timeline`
8. **kps_coverage column** — tells model how reliable vision features are per window (partial occlusion awareness)
9. **`display_tracks.py` is the integration testbed** — use it to visually validate identity assignments before committing to feature extraction
10. **Dual day/night gallery** — `reid_registry` stores separate `gallery_embed_day` and `gallery_embed_night` vectors per cow. Night sessions (IR/grayscale, auto-detected via per-channel variance in `video_sessions.is_night`) never update or query the day gallery and vice versa. Early night sessions with no gallery fall back to kinetics-only. Galleries build independently as more sessions of each type are kinetically confirmed.
11. **`--commit_every` controls crash safety** — SQLite commits in the same periodic cadence as the old CSV flush. Default 50 frames. Lower = more crash-safe, marginally more I/O.

---

## Milestone Schedule (from proposal)
- Pose Estimation: March 29, 2026 ✓
- Sensor Pipeline: April 12, 2026
- Re-Identification Module: May 3, 2026 ← current focus
- Vision Feature Extraction: May 24, 2026
- System Integration: June 14, 2026
- Temporal Prediction Model: July 26, 2026
- Empirical Evaluation: August 30, 2026
- Research Paper: September 30, 2026
- Thesis Document: October 30, 2026

---

## Storage Architecture

**Database engine:** SQLite for all 7 structured tables.  
**Array storage:** Parquet files (pyarrow) for embed[128], kps[57], kps_kconf[19]. One `embeds.parquet` and one `kps.parquet` per session, co-located with the DB in the session output folder.  
**Gallery vectors:** `gallery_day.npy` / `gallery_night.npy` — dict `{real_id: ndarray(128,)}`. One pair shared across all sessions.  
**Cloud backup:** University OneDrive (50GB) via `rclone` mount in WSL.  
**Raw video:** Local disk only — not synced to OneDrive. Re-processable from source.

### Storage budget (per hour of video processed)
| Artifact | Size/hour |
|---|---|
| embeds.parquet (embed[128] float32, Snappy) | ~180 MB |
| kps.parquet (kps[57] + kps_kconf[19], Snappy) | ~80 MB |
| resolved_cow_timeline (scalar features) | ~20 MB |
| SQLite raw_tracks (scalars only) | ~15 MB |
| gallery .npy (total across all sessions, not per hour) | ~1 MB |
| **Total processed output per hour** | **~295 MB** |

50 GB OneDrive → ~170 hours of processed output.  
Raw video (1–3 GB/hour) stays on local disk only.

### Directory layout
```
~/thesis_workspace/                          # local WSL
  raw_data/
    videos/                                  # raw MP4s — LOCAL ONLY, never synced
    collar_data/                             # collar CSVs

~/onedrive_mount/thesis_data/                # rclone → university OneDrive
  models/
    cow_detector/best.pt
    cow_pose/best.pt
  outputs/
    session_001/
      calving_project.db                     # SQLite (video_sessions + raw_tracks)
      embeds.parquet                         # embed[128] for this session
      kps.parquet                            # kps[57] + kps_kconf[19] for this session
      crops/                                 # optional JPEGs
    session_002/
      calving_project.db
      embeds.parquet
      kps.parquet
  reid_gallery/
    gallery_day.npy                          # gallery_embed_day per cow
    gallery_night.npy                        # gallery_embed_night per cow
  collar_data/                               # collar CSVs backed up here
```

### rclone setup (WSL)
```bash
sudo apt install rclone
rclone config                                # interactive wizard — OneDrive, sign in
rclone mount "university_onedrive:thesis_data" ~/onedrive_mount \
  --vfs-cache-mode writes &
```
Scripts write directly to `~/onedrive_mount/` — syncs automatically.
