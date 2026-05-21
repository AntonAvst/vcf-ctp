# vcf-ctp
**V**ision–**C**ollar **F**usion for **C**alving-**T**ype **P**rediction in Dairy Cows

## Project
**Thesis:** Predictive Modelling of Calving Outcomes in Dairy Cows Using Multi-Modal Sensor and Vision Data  
**Author:** Anton Avstreikh, University of Haifa  
**Advisors:** Prof. Ilan Shimshoni  
**Output:** 4-class calving type prediction — Unassisted · Assisted · Twin · Veterinarian-assisted

---

## Existing Code

### `track_and_dump.py`
- Runs YOLO detection + ByteTrack/BoT-SORT tracking + YOLOv8-Pose (19 kp) + Embedder128 (MobileNetV3 → 128D L2-norm)
- Outputs to `<outdir>/`: `calving_project.db` (SQLite), `embeds.parquet`, `kps.parquet`, optional `crops/`
- Arrays (embed[128], kps[19×3], kps_kconf[19]) stored in Parquet — **never in SQLite**. SQLite stores integer row pointers (`embed_parquet_row`, `kps_parquet_row`)
- Parses wall-clock start time from filename token `_S<YYYYMMDDHHmmss>`
- **Calls `reconcile.py` automatically** after tracking finishes (pass `--kinetics`)
- Key CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--model` | required | Detector `.pt` |
| `--source` | required | Video path |
| `--outdir` | required | Output folder |
| `--session_id` | filename stem | Unique session key |
| `--tracker` | `bytetrack.yaml` | Tracker config |
| `--camera_id` | `cam0` | Camera label |
| `--imgsz` | 960 | Detector image size |
| `--conf` | 0.25 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--pose_model` | *(optional)* | YOLOv8-Pose `.pt` |
| `--pose_imgsz` | 384 | Pose crop size |
| `--pose_conf` | 0.25 | Pose confidence threshold |
| `--pose_kp_conf_thresh` | 0.30 | Per-keypoint conf: ≥ → vis=2, < → vis=1 |
| `--embed_size` | 128 | Embedding dimension |
| `--save_crops` | off | Save JPEG crops |
| `--crop_every` | 1 | Save crop every N-th detection |
| `--min_crop_wh` | 0 0 | Min crop width/height |
| `--commit_every` | 50 | SQLite commit interval (frames) |
| `--kinetics` | required | `kinetic_data_*.csv` — forwarded to reconcile.py |
| `--gallery_dir` | `./reid_gallery` | Gallery directory — forwarded to reconcile.py |
| `--corr_threshold` | 0.7 | Kinetic match Pearson r threshold |
| `--cosine_threshold` | 0.75 | Cosine similarity threshold |
| `--ema_alpha` | 0.15 | Gallery EMA decay |

### `reconcile.py`
- Post-processing pipeline: kinetic match → gallery update → cosine resolve → timeline write
- Called automatically by `track_and_dump.py`, or run manually per session
- Steps in order: A (kinetic match) → A.5 (merge manual overrides) → B (gallery builder) → C (cosine resolver) → A.6 (resolve duplicate assignments) → D (sensor sequencer) → E (write to DB)
- Reads tracks from SQLite (`raw_tracks`); optionally loads embeds from `--embed_parquet`
- Writes to `resolved_cow_timeline` and updates `reid_registry` + gallery `.npy` files
- Key CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--db` | required | Path to `calving_project.db` |
| `--session` | required | `session_id` to process |
| `--kinetics` | required | `kinetic_data_*.csv` |
| `--gallery_dir` | `./reid_gallery` | Directory with `gallery_day.npy` / `gallery_night.npy` |
| `--embed_parquet` | *(optional)* | Parquet file with embed[128] per detection |
| `--corr_threshold` | 0.7 | Min Pearson r for kinetic match |
| `--min_active_bins` | 3 | Min active kinetics bins required |
| `--min_temp_id_frames` | 0.10 | Min fraction of frames a temp_id must appear in |
| `--activity_pct` | 0.25 | Percentile threshold for "active" kinetics bins |
| `--bin_minutes` | 15 | Kinetics bin width in minutes |
| `--ema_alpha` | 0.15 | EMA decay — full α for kinetic-confirmed, α/2 for cosine-only |
| `--min_embeds_gallery` | 10 | Min embed rows per temp_id to contribute to gallery |
| `--cosine_threshold` | 0.75 | Min cosine similarity for cross-video / switch-healing |
| `--cosine_min_embeds` | 5 | Min embed rows for a temp_id to be queried via cosine |
| `--dry_run` | off | Run all steps, write nothing |
| `--verbose` | off | Extra diagnostic output |

### `display_tracks.py`
- Visualization tool: overlays tracks + pose skeleton on video
- Reads directly from SQLite (`raw_tracks`, `video_sessions`) — no CSV required
- Imports `match_identity` for live kinetic matching (Pearson r + Hungarian)
- Draws score table overlay: temp_id × AnimalId correlation matrix
- Supports ffplay / cv2 / mp4 sinks; TkControls panel for pause / fast-forward (1×/2×/4×/8×) / quit

### `match_identity.py`
- Kinetic matching: bbox centroid speed ↔ collar ΔKineticsCountR
- Pearson correlation per 15-min bin, Hungarian assignment
- `score_up_to(tracks_df, kinetics_df, up_to_datetime, bin_minutes, ...)` → assignment dict + scores_df (used by `display_tracks.py` for live overlays)
- Standalone CLI: `python3 match_identity.py --tracks tracks.csv --kinetics ... --output ...`

### `assign_identity.py`
- Manual identity assignment tool — use when kinetics data is unavailable or doesn't cover the video window
- Writes assignments to a `manual_assignments` table in SQLite; reconcile.py merges these in Step A.5, with manual overriding kinetic on conflict
- Can optionally call `reconcile.py` directly after writing (`--run_reconcile`)
- Key operations:

```bash
# Assign temp_ids manually after watching display_tracks.py
python3 assign_identity.py --db ... --session ... \
    --assign 2:7507  1:6366  71:7513 \
    --note "manual — no kinetics coverage for Dec 21" \
    --run_reconcile

# List current assignments for a session
python3 assign_identity.py --db ... --session ... --list

# Remove a specific temp_id assignment
python3 assign_identity.py --db ... --session ... --remove 71
```

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
**Inputs:** Raw MP4, YOLO `.pt` models  
**Process:** Detect → Track (temp_id) → Pose (kps) → Embed (128D) → call reconcile.py  
**Outputs written:**
- `raw_tracks` — append-only, one row per detection (scalar columns); arrays in Parquet
- `video_sessions` — one row per video file registered

**Key rule:** No identity resolution at inference time. Just save everything.

### Stage 2 — Post-processing (`reconcile.py`, once per video after Stage 1)
**Steps in order:**

| Step | Name | Description |
|---|---|---|
| **0** | Collar candidate selector | Given `session.start_dt` / `session.end_dt`, filter to AnimalIds with ΔKineticsCountR above threshold. Cows not in the pen are automatically excluded — no manual roster needed. |
| **A** | Kinetic matcher | bbox centroid speed ↔ ΔR · Pearson r · Hungarian → temp_id ↔ AnimalId |
| **A.5** | Manual merge | Load `manual_assignments` from DB; manual overrides kinetic on conflict |
| **B** | Gallery builder | Group embeds by confirmed AnimalId → EMA mean → separate day/night galleries. Routes on `video_sessions.is_night`. Never mixes modalities. |
| **C** | Cosine resolver | Query `gallery_embed_day` or `gallery_embed_night` (never cross-query). Heals temp_id switches and enables cross-video continuity. Falls back to kinetics-only if gallery is empty. |
| **A.6** | Duplicate resolver | Merge any temp_ids assigned to the same AnimalId (tracker switches); remap loser embed rows to winner and rebuild gallery. |
| **D** | Sensor sequencer | Δf_12/f_23/v/kinR → forward-fill to video time grid |
| **E** | Write timeline | Insert rows into `resolved_cow_timeline` |

**Output written:** `resolved_cow_timeline` — central join table

**Key rule:** Kinetic matching is always the primary confirmation signal. Cosine resolver is secondary (heals switches, handles cross-video). Both run post-hoc, never during inference.

### Stage 3 — Model (dataset builder + CNN-LSTM/GRU) *(planned)*
**Inputs:** resolved_cow_timeline + calving_ledger + cow_registry  
**Process:** Build labeled multi-hour windows per calving event → train temporal model  
**Output:** 4-class probability distribution + risk score + prediction horizon

---

## Databases (7 total)

### `cow_registry` — static lookup
- real_id (PK), breed, parity, pen_id, collar_id, baseline_window
- Used for: sensor normalization baseline, static feature injection

### `video_sessions` — one row per video file
- session_id (PK), video_path, camera_id, start_dt, end_dt, collar_csv_path
- **is_night** (bool) — auto-detected by reconcile.py from frame_datetime hour distribution; also detectable via per-channel variance (IR/grayscale frames have near-zero R/G/B channel divergence)

### `collar_signals` — raw sensor time-series
- AnimalId (FK → cow_registry), datetime, signal_type ('behavior'|'kinetic')
- f_1_2, f_2_3, v (behavior) + kin_X, kin_Y, kin_Z, kin_R (kinetics)
- Ingested separately from video pipeline

### `raw_tracks` — append-only inference output
- session_id (FK → video_sessions), frame_index, frame_datetime, temp_id
- bbox (x1/y1/x2/y2), cx, cy, det_conf
- embed (JSON), kps (JSON flat list[57]), kps_norm (JSON), kps_conf (REAL), kps_kconf (JSON)
- **Note:** `embed_parquet_row` and `kps_parquet_row` are integer pointers into the session Parquet files — the canonical array store. JSON columns are a convenience fallback.

### `manual_assignments` — operator overrides
- session_id, temp_id, real_id, note, created_dt
- Written by `assign_identity.py`; consumed by reconcile.py Step A.5
- Manual assignments override kinetic matches on conflict

### `reid_registry` — one row per confirmed real identity
- real_id (PK = AnimalId from collar)
- **gallery_embed_day[128]** — EMA mean embedding from daytime sessions (RGB)
- **gallery_embed_night[128]** — EMA mean embedding from night/IR sessions (grayscale)
- gallery_n_day, gallery_n_night — session counts per gallery
- gallery_conf_day, gallery_conf_night — quality score
- last_updated_day_dt, last_updated_night_dt
- known_temp_ids (JSON list of {session_id, temp_id})
- first_seen_dt, match_method ('kinetic' | 'cosine_day' | 'cosine_night')
- Updated by reconcile.py after each video

**EMA update rule:**
```
gallery_embed_new = α × mean(session_embeds) + (1 − α) × gallery_embed_old
α = 0.1–0.2  (slow drift; old sightings fade gradually)
```
- Day sessions (is_night=False) → update gallery_embed_day only
- Night sessions (is_night=True) → update gallery_embed_night only
- Galleries **never** cross-contaminate across modalities
- Kinetic-confirmed sessions → full α update
- Cosine-only confirmed sessions → α/2 update (self-referential — weight conservatively)

### `resolved_cow_timeline` — central join table
- real_id (FK → reid_registry, nullable), window_start_dt, session_id
- modality_mask (bitmask: 1=sensor_ok | 2=vision_ok | 4=reid_ok)
- **Sensor cols** (forward-filled): d_f12, d_f23, d_v, d_kin_x, d_kin_y, d_kin_z, d_kin_r
- **Vision cols** (derived from kps by reconcile.py — currently NULL, pending pose extractor):
  - spine_angle (kp2→kp3→kp4)
  - pelvic_tilt (kp7 ↔ kp10)
  - tail_elevation (kp5/kp6 vs kp4)
  - limb_symmetry (L/R hock distance ratio)
  - head_drop (kp0/kp1 vs kp2)
  - lying_flag (bbox aspect ratio heuristic)
  - restlessness (variance of spine_angle over window)
  - kps_coverage (mean conf across 19 kp — reliability indicator)
  - embed_mean (JSON list[128], mean-pooled over window)
- Raw kps arrays stay in `raw_tracks` / Parquet — not copied here

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

---

## Typical Workflow

```bash
# 1. Run inference + reconcile (single command)
python3 track_and_dump.py \
    --model   models/cow_detector/best.pt \
    --source  raw_data/videos/refet33_S20241221070000_E20241221080000.mp4 \
    --outdir  outputs/refet33_20241221 \
    --session_id refet33_20241221 \
    --pose_model models/cow_pose/best.pt \
    --kinetics   raw_data/collar_data/kinetic_data_6366_7507_7513.csv \
    --gallery_dir ./reid_gallery

# 2. Validate visually
python3 display_tracks.py \
    --video      raw_data/videos/refet33_S20241221070000_E20241221080000.mp4 \
    --db         outputs/refet33_20241221/calving_project.db \
    --session_id refet33_20241221 \
    --kinetics   raw_data/collar_data/kinetic_data_6366_7507_7513.csv \
    --draw_pose --show_fps --sink ffplay

# 3a. If kinetics coverage is good — reconcile runs automatically (Step 1 above)

# 3b. If kinetics unavailable — assign manually after watching display_tracks.py
python3 assign_identity.py \
    --db      outputs/refet33_20241221/calving_project.db \
    --session refet33_20241221 \
    --assign  2:7507  1:6366  71:7513 \
    --note    "manual — no kinetics for this date" \
    --run_reconcile

# 4. Re-run reconcile standalone (e.g. after tuning thresholds)
python3 reconcile.py \
    --db        outputs/refet33_20241221/calving_project.db \
    --session   refet33_20241221 \
    --kinetics  raw_data/collar_data/kinetic_data_6366_7507_7513.csv \
    --gallery_dir ./reid_gallery \
    --corr_threshold 0.7 \
    --cosine_threshold 0.75 \
    --ema_alpha 0.15
    # add --dry_run to test without writing
```

---

## Key Design Decisions

1. **No ReID during inference** — temp_ids saved raw; all identity resolution is post-hoc in reconcile.py
2. **Wall-clock as universal join key** — frame_datetime links video to collar data; filename `_S<timestamp>` is the source
3. **Arrays in Parquet, never SQLite blobs** — embed[128], kps[19×3], kps_kconf[19] live in per-session Parquet files; SQLite holds integer row pointers
4. **real_id is nullable** — pipeline doesn't block on unresolved identities; modality_mask signals quality
5. **Two-stage identity resolution** — kinetic matching (primary) confirms AnimalId; cosine resolver (secondary) heals temp_id switches and enables cross-video continuity
6. **Manual assignment as first-class fallback** — `assign_identity.py` writes to `manual_assignments`; reconcile merges these with kinetic results, manual taking priority
7. **Duplicate resolver (Step A.6)** — when two temp_ids are assigned the same AnimalId (tracker switch), the loser's embed rows are remapped to the winner before gallery rebuild
8. **Pose raw data stays in raw_tracks / Parquet** — reconcile.py extracts scalar features into resolved_cow_timeline; raw kps recomputable any time
9. **Sensor temporal mismatch handled by forward-fill** — behavior ~90s, kinetics ~15min → upsampled into video time grid in resolved_cow_timeline
10. **kps_coverage column** — tells the model how reliable vision features are per window (partial occlusion awareness)
11. **display_tracks.py is the integration testbed** — use it to visually validate identity assignments before committing to feature extraction
12. **Dual day/night gallery** — `reid_registry` stores separate `gallery_embed_day` and `gallery_embed_night` vectors per cow; modalities never cross-contaminate

---

## Storage Architecture

### Decision: SQLite + Parquet + University OneDrive (50GB)

**Database engine:** SQLite for all structured tables. Zero infrastructure, full SQL, pandas-native.  
**Array storage:** Parquet files (pyarrow) for embed[128], kps[19×3], kps_kconf[19] — one file pair per session. SQLite stores integer row pointers, never raw blobs.  
**Cloud backup:** University OneDrive (50GB) via `rclone` mount in WSL.  
**Raw video:** Local disk only — not synced. Re-processable from source.

### Storage budget (per hour of video processed)
| Artifact | Size/hour |
|---|---|
| Parquet embeds (embed[128] float32, Snappy) | ~180 MB |
| Parquet keypoints (kps[19×3] + kps_kconf) | ~80 MB |
| resolved_cow_timeline (scalar features) | ~20 MB |
| SQLite DB (all structured tables) | negligible |
| reid_gallery .npy (total, not per hour) | ~1 MB |
| **Total processed output per hour** | **~280 MB** |

50 GB OneDrive → comfortably holds ~175 hours of processed output.

### Directory layout
```
~/thesis_workspace/                         # local WSL
  raw_data/
    videos/                                 # raw MP4s — LOCAL ONLY, never synced
    collar_data/                            # collar CSVs — small, sync these

~/onedrive_mount/thesis_data/               # rclone → university OneDrive
  calving_project.db                        # SQLite — all structured tables
  models/
    cow_detector/best.pt
    cow_pose/best.pt
  outputs/
    <session_id>/
      embeds.parquet                        # embed[128] per detection
      kps.parquet                           # kps[19×3] + kps_kconf per detection
  reid_gallery/
    gallery_day.npy                         # gallery_embed_day[128] per cow
    gallery_night.npy                       # gallery_embed_night[128] per cow
  collar_data/                              # collar CSVs backed up here
```

### rclone setup (WSL)
```bash
sudo apt install rclone
rclone config                               # interactive wizard — choose OneDrive, sign in
rclone mount "university_onedrive:thesis_data" ~/onedrive_mount \
    --vfs-cache-mode writes &
```
Scripts write directly to `~/onedrive_mount/` — syncs automatically.


---

## Milestone Schedule
| Milestone | Target | Status |
|---|---|---|
| Pose Estimation | March 29, 2026 | ✓ Done |
| Sensor Pipeline | April 12, 2026 | ✓ Done |
| Re-Identification Module | May 3, 2026 | ✓ Done |
| Vision Feature Extraction | May 24, 2026 | In progress |
| Database pipline + cloud storage | June 1, 2026 | — |
| Working Pipeline | June 14, 2026 | — |
| Temporal Prediction Model | July 26, 2026 | — |
| Empirical Evaluation | August 30, 2026 | — |
| Research Paper | September 30, 2026 | — |
| Thesis Document | October 30, 2026 | — |

---
