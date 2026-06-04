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
| `--save_every` | 24 | Flush tracks (SQLite + Parquet) every N frames. Overrides `--commit_every`. Lower = more crash-safe, slightly more I/O. |
| `--commit_every` | 50 | SQLite commit interval (frames) — legacy, use `--save_every` instead |
| `--kinetics` | required | `kinetic_data_*.csv` — forwarded to reconcile.py |
| `--gallery_dir` | `./reid_gallery` | Gallery directory — forwarded to reconcile.py |
| `--corr_threshold` | 0.7 | Kinetic match Pearson r threshold |
| `--cosine_threshold` | 0.75 | Cosine similarity threshold |
| `--ema_alpha` | 0.15 | Gallery EMA decay |

### `reconcile.py`
- Post-processing pipeline: kinetic match → gallery update → cosine resolve → vision features → timeline write
- Called automatically by `track_and_dump.py`, or run manually per session
- Steps in order: A (kinetic match) → A.5 (merge manual overrides) → B (gallery builder) → C (cosine resolver) → A.6 (resolve duplicate assignments) → D (sensor sequencer) → B-vision (vision feature extractor) → E (write to DB)
- Reads tracks from SQLite (`raw_tracks`); optionally loads embeds from `--embed_parquet`
- Writes to `resolved_cow_timeline` and updates `reid_registry` + gallery `.npy` files
- Imports `vision_features` package for Step B-vision — the `vision_features/` folder must be in the same directory
- Key CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--db` | required | Path to `calving_project.db` |
| `--session` | required | `session_id` to process |
| `--kinetics` | required | `kinetic_data_*.csv` |
| `--gallery_dir` | `./reid_gallery` | Directory with gallery `.npy` files |
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

### `vision_features/` *(new)*
- Python package implementing Stage 2-B: vision feature extraction from raw keypoints
- Must be placed in the same directory as `reconcile.py` (i.e. `scripts/vision_features/`)
- Called by `reconcile.py` after Step D; never called directly
- Structure:

```
vision_features/
├── __init__.py          # public API: run_vision_features, migrate_timeline_schema
├── schema.py            # KP enum, Posture/Facing enums, 8 gallery slots, column names
├── extractor.py         # orchestrator — the only file reconcile.py touches
├── features/
│   ├── __init__.py
│   ├── posture.py       # standing / lying classifier (bbox aspect ratio + kps_coverage)
│   └── facing.py        # left / right / toward / away classifier (nose→sacrum vector)
└── gallery/
    ├── __init__.py
    └── pose_conditioned.py  # 8-slot pose-conditioned ReID gallery (load/update/query/save)
```

**Per-frame classifiers:**

| Classifier | Signal | Output |
|---|---|---|
| Posture | Bbox aspect ratio (AR < 0.85 → standing, AR > 1.10 → lying, band = uncertain) + kps_coverage gate | `posture` ∈ {standing, lying, uncertain}, `posture_conf` |
| Facing | Nose→sacrum body-axis vector projected onto frame axes; lateral dominance ratio separates left/right from toward/away | `facing` ∈ {left, right, toward, away, uncertain}, `facing_conf` |

**Window aggregates written to `resolved_cow_timeline`:**

| Column | Type | Description |
|---|---|---|
| `lying_fraction` | float [0,1] | Fraction of confident frames classified as lying |
| `posture_transitions` | int | Number of standing↔lying switches in window |
| `facing_dominant` | str | Modal facing direction across confident frames |
| `facing_entropy` | float [0,1] | Spread of facing distribution (0 = always same direction) |

**Pose-conditioned gallery:**
- Extends `reid_gallery/` with two new files: `gallery_pose_day.npy` and `gallery_pose_night.npy`
- Each stores shape `(N_cows, 8, 128)` — 8 slots = {standing, lying} × {left, right, toward, away}
- Only confident frames (both posture_conf and facing_conf ≥ threshold) contribute to a slot
- Slot query uses a 3-level fallback chain: exact slot → same posture any facing → all populated slots
- EMA update rule same as flat gallery (full α for kinetic-confirmed, α/2 for cosine-only)

**Adding a new feature:**
1. Create `vision_features/features/my_feature.py` with `extract_my_feature()` and `aggregate_my_feature()`
2. Add output columns to `schema.py` (`TIMELINE_VISION_COLS` and `TIMELINE_ALTER_SQLS`)
3. Call both functions in `extractor.py` (two marked locations)

### `display_tracks.py`
- Browser-based visualizer — replaces X11/SDL2/ffplay/Tkinter; no display server required (WSL-compatible)
- Streams annotated video as MJPEG via Flask; sensor charts rendered server-side as PNG via matplotlib
- Opens `http://localhost:5000` automatically in the Windows browser (via `cmd.exe /c start`)
- Reads directly from SQLite (`raw_tracks`, `video_sessions`) — no CSV required
- Imports `match_identity` for live kinetic matching; draws score table overlay (temp_id × AnimalId)
- Runs live posture + facing classification per frame if `vision_features/` is present (optional import)
- Browser controls: pause / fast-forward (1×/2×/4×/8×) / quit; sensor chart auto-refreshes every 3 s
- Requires: `pip install flask`

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
| **C** | Cosine resolver | 3-tier query per unresolved temp_id: (1) flat real_id gallery, (2) pose-conditioned real_id gallery, (3) TempPoseGallery — cross-session same-camera temp matching. If a match to a real_id is found → assign it. If a match to another temp_id is found → inherit its synthetic id or mint a new one. Backpropagates: if a synthetic id gets resolved to a real_id, all prior timeline rows are updated. Falls back to kinetics-only if all galleries are empty. |
| **A.6** | Duplicate resolver | Merge any temp_ids assigned to the same AnimalId (tracker switches); remap loser embed rows to winner and rebuild gallery. |
| **D** | Sensor sequencer | Δf_12/f_23/v/kinR → forward-fill to video time grid |
| **B-vision** | Vision feature extractor | Load kps from Parquet → classify posture + facing per frame → aggregate to window scalars → write to timeline → update pose-conditioned gallery |
| **E** | Write timeline | Insert rows into `resolved_cow_timeline` |

**Output written:** `resolved_cow_timeline` — central join table

**Key rule:** Kinetic matching is always the primary confirmation signal. Cosine resolver is secondary (heals switches, handles cross-video). Both run post-hoc, never during inference.

### Stage 3 — Model (dataset builder + CNN-LSTM/GRU) *(planned)*
**Inputs:** resolved_cow_timeline + calving_ledger + cow_registry  
**Process:** Build labeled multi-hour windows per calving event → train temporal model  
**Output:** 4-class probability distribution + risk score + prediction horizon

---

## Databases (8 total)

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
- **gallery_embed_day[128]** — flat EMA mean embedding from daytime sessions (RGB); fallback vector
- **gallery_embed_night[128]** — flat EMA mean embedding from night/IR sessions (grayscale); fallback vector
- gallery_n_day, gallery_n_night — session counts per gallery
- gallery_conf_day, gallery_conf_night — quality score
- last_updated_day_dt, last_updated_night_dt
- known_temp_ids (JSON list of {session_id, temp_id})
- first_seen_dt, match_method ('kinetic' | 'cosine_day' | 'cosine_night')
- Updated by reconcile.py after each video

**Per-cow gallery — two layers:**

The primary gallery is the **pose-conditioned 8-slot gallery** stored in `gallery_pose_{day|night}.npy` (outside SQLite). Each cow gets up to 8 independent 128D vectors — one per (posture × facing) combination:

```
slot 0: standing × left       slot 4: lying × left
slot 1: standing × right      slot 5: lying × right
slot 2: standing × toward     slot 6: lying × toward
slot 3: standing × away       slot 7: lying × away
```

Only frames where both `posture_conf` and `facing_conf` meet the threshold contribute to a slot. Unpopulated slots are absent (not zero) — the cosine resolver's 3-level fallback handles sparsity:
1. Query exact slot (e.g. `standing_left`)
2. Fall back to same posture, any facing (e.g. all `standing_*` populated slots, take max cosine)
3. Fall back to all populated slots across both postures (equivalent to flat gallery)

The **flat gallery** (`gallery_embed_day/night` in SQLite + `gallery_{day|night}.npy`) is the mean-pool of all session embeddings regardless of pose. It is the Level 3 fallback and is also used during early sessions before pose slots are populated.

**EMA update rule (per slot, and for flat):**
```
gallery_embed_new = α × mean(session_embeds_in_slot) + (1 − α) × gallery_embed_old
α = 0.1–0.2  (slow drift; old sightings fade gradually)
```
- Day sessions (is_night=False) → update day galleries only; night → night only
- Galleries **never** cross-contaminate across modalities
- Kinetic-confirmed sessions → full α update
- Cosine-only confirmed sessions → α/2 update (self-referential — weight conservatively)

### `resolved_cow_timeline` — central join table
- real_id (FK → reid_registry, nullable), window_start_dt, session_id
- modality_mask (bitmask: 1=sensor_ok | 2=vision_ok | 4=reid_ok)
- **Sensor cols** (forward-filled): d_f12, d_f23, d_v, d_kin_x, d_kin_y, d_kin_z, d_kin_r
- **Vision cols** (derived from kps by `vision_features/` — currently populated for posture/facing; remaining cols NULL pending future extractors):
  - spine_angle (kp2→kp3→kp4) — *pending*
  - pelvic_tilt (kp7 ↔ kp10) — *pending*
  - tail_elevation (kp5/kp6 vs kp4) — *pending*
  - limb_symmetry (L/R hock distance ratio) — *pending*
  - head_drop (kp0/kp1 vs kp2) — *pending*
  - lying_flag (bbox aspect ratio heuristic) — *pending (superseded by lying_fraction)*
  - restlessness (variance of spine_angle over window) — *pending*
  - kps_coverage (mean conf across 19 kp — reliability indicator) — *pending*
  - embed_mean (JSON list[128], mean-pooled over window) — *pending*
  - **lying_fraction** (float [0,1] — fraction of confident frames lying) — ✓ active
  - **posture_transitions** (int — standing↔lying switches in window) — ✓ active
  - **facing_dominant** (str — modal facing direction) — ✓ active
  - **facing_entropy** (float [0,1] — spread of facing distribution) — ✓ active
- Raw kps arrays stay in `raw_tracks` / Parquet — not copied here

### `calving_ledger` — ground truth
- event_id (PK), real_id (FK → reid_registry), calving_dt
- outcome: Unassisted | Assisted | Twin | Veterinarian-assisted

### `temp_id_merges` — tracker-switch merge log
- session_id, winner_tid, loser_tid, real_id, winner_reason, loser_reason, merged_dt
- Written by reconcile.py Step A.6 when two temp_ids are assigned the same AnimalId (tracker switch)
- `winner_tid` survives; `loser_tid` embed rows are remapped to winner before gallery rebuild
- winner_reason / loser_reason: `'manual'` | `'more_frames'` | `'kinetic'`
- UNIQUE on (session_id, loser_tid) — a loser can only be merged once per session

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
    --model      models/cow_detector/best.pt \
    --source     raw_data/videos/refet33_S20241221070000_E20241221080000.mp4 \
    --outdir     outputs/refet33_20241221 \
    --session_id refet33_20241221 \
    --pose_model models/cow_pose/best.pt \
    --kinetics   raw_data/collar_data/kinetic_data_6366_7507_7513.csv \
    --gallery_dir ./reid_gallery \
    --save_every 24

# 2. Validate visually (opens http://localhost:5000 in browser)
python3 display_tracks.py \
    --video      raw_data/videos/refet33_S20241221070000_E20241221080000.mp4 \
    --db         outputs/refet33_20241221/calving_project.db \
    --session_id refet33_20241221 \
    --kinetics   raw_data/collar_data/kinetic_data_6366_7507_7513.csv \
    --draw_pose --show_fps

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
    --db          outputs/refet33_20241221/calving_project.db \
    --session     refet33_20241221 \
    --kinetics    raw_data/collar_data/kinetic_data_6366_7507_7513.csv \
    --gallery_dir ./reid_gallery \
    --embed_parquet outputs/refet33_20241221/embeds.parquet \
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
5. **Two-stage identity resolution** — kinetic matching (primary) confirms AnimalId; cosine resolver (secondary) heals temp_id switches and enables cross-video continuity. Cosine resolver uses a 3-tier query: flat gallery → pose-conditioned gallery → TempPoseGallery (cross-session temp matching)
6. **Manual assignment as first-class fallback** — `assign_identity.py` writes to `manual_assignments`; reconcile merges these with kinetic results, manual taking priority
7. **Duplicate resolver (Step A.6)** — when two temp_ids are assigned the same AnimalId (tracker switch), the loser's embed rows are remapped to the winner before gallery rebuild
8. **Pose raw data stays in raw_tracks / Parquet** — reconcile.py extracts scalar features into resolved_cow_timeline; raw kps recomputable any time
9. **Sensor temporal mismatch handled by forward-fill** — behavior ~90s, kinetics ~15min → upsampled into video time grid in resolved_cow_timeline
10. **kps_coverage column** — tells the model how reliable vision features are per window (partial occlusion awareness)
11. **display_tracks.py is the integration testbed** — use it to visually validate identity assignments before committing to feature extraction
12. **Dual day/night gallery, two layers per cow** — each cow has a flat fallback vector (`gallery_embed_{day|night}` in SQLite) and up to 8 pose-conditioned vectors (`gallery_pose_{day|night}.npy`). The 8-slot gallery is the primary query target; the flat vector is the Level 3 fallback when all pose slots are empty. Modalities never cross-contaminate.
13. **Pluggable vision feature extractor** — `vision_features/` is a self-contained package; adding a new feature means one new file in `features/` and two lines in `extractor.py`; no other files change
14. **Pose-conditioned 8-slot gallery** — 8 slots per cow per modality = {standing, lying} × {left, right, toward, away}. Only high-confidence classified frames contribute to each slot. Cosine resolver queries exact slot first, then degrades gracefully to posture-level then flat. Galleries build up progressively — early sessions use flat fallback; discrimination improves automatically as slots populate.
15. **Synthetic IDs for unresolved cows** — when a temp_id matches another temp_id cross-session but no real_id is known yet, reconcile.py mints a negative synthetic id (from `synthetic_id_counter.npy`). The synthetic id propagates consistently until a future kinetic match resolves it to a real AnimalId, at which point all prior timeline rows are backpropagated automatically

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
| reid_gallery .npy flat (total, not per hour) | ~1 MB |
| reid_gallery .npy pose-conditioned (total, not per hour) | ~8 MB |
| **Total processed output per hour** | **~280 MB** |

50 GB OneDrive → comfortably holds ~175 hours of processed output.

### Directory layout
```
~/thesis_workspace/                         # local WSL
  raw_data/
    videos/                                 # raw MP4s — LOCAL ONLY, never synced
    collar_data/                            # collar CSVs — small, sync these
  scripts/
    reconcile.py
    track_and_dump.py
    display_tracks.py
    match_identity.py
    assign_identity.py
    vision_features/                        # vision feature extraction package
      __init__.py
      schema.py
      extractor.py
      features/
        __init__.py
        posture.py
        facing.py
      gallery/
        __init__.py
        pose_conditioned.py

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
    gallery_day.npy                         # flat gallery_embed_day[128] per cow
    gallery_night.npy                       # flat gallery_embed_night[128] per cow
    gallery_pose_day.npy                    # 8-slot pose-conditioned (N_cows, 8, 128)
    gallery_pose_night.npy                  # 8-slot pose-conditioned (N_cows, 8, 128)
    temp_gallery_pose_day.npy               # TempPoseGallery — cross-session temp matching (day)
    temp_gallery_pose_night.npy             # TempPoseGallery — cross-session temp matching (night)
    synthetic_id_counter.npy               # monotonic counter for minting synthetic IDs
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
| Vision Feature Extraction | May 29, 2026 | ✓ Done |
| Database pipeline + cloud storage | June 5, 2026 | — |
| Working Pipeline | June 14, 2026 | — |
| Temporal Prediction Model | July 26, 2026 | — |
| Model Retraining and Fine-tuning | August 30, 2026 | — |
| Empirical Evaluation | September 30, 2026 | — |
| Research Paper | October 30, 2026 | — |
| Thesis Document | November 30, 2026 | — |

---