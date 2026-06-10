# vcf-ctp
**V**ision–**C**ollar **F**usion for **C**alving-**T**ype **P**rediction in Dairy Cows

## Project
**Thesis:** Predictive Modelling of Calving Outcomes in Dairy Cows Using Multi-Modal Sensor and Vision Data  
**Author:** Anton Avstreikh, University of Haifa  
**Advisors:** Prof. Ilan Shimshoni  
**Output:** 4-class calving type prediction — Unassisted · Assisted · Twin · Veterinarian-assisted

---

## Existing Code

### `drive_manager.py`
- Central Google Drive I/O abstraction layer — all scripts read/write through this module
- Drive (`thesis_google_drive:vcf_ctp_data`) is the single source of truth for all data
- Local (`~/thesis_workspace/vcf-ctp/data/`) is a write buffer for the current session only
- Dirty flags (`db`, `parquet`, `gallery`) block downstream read scripts when local writes are unsynced
- Upload failures are buffered locally (`.buffer/pending/`) and retried automatically
- All uploads logged locally to `upload_log_staging.csv`; flushed to `upload_log.csv` on Drive in a single round-trip at end of batch — never once per file
- Graceful Ctrl+C: finishes the current file, flushes staged log rows, then exits cleanly
- Key path constants (edit once at top of file):

| Constant | Value |
|---|---|
| `RCLONE_REMOTE` | `thesis_google_drive` |
| `DRIVE_ROOT` | `thesis_google_drive:vcf_ctp_data` |
| `LOCAL_ROOT` | `~/thesis_workspace/vcf-ctp/data/` |
| `BUFFER_DIR` | `~/thesis_workspace/vcf-ctp/.buffer/` |

**Library API (imported by other scripts):**

```python
from drive_manager import DriveManager
dm = DriveManager(caller=__file__, bypass=False)

dm.pull_db()                                       # pull canonical DB from Drive (blocking)
dm.sync_db(session_id)                             # push local DB snapshot to Drive
dm.pull_gallery(modality)                          # pull gallery .npy files from Drive
dm.get_db_path()                                   # → local path (checks dirty flag)
dm.get_gallery_dir()                               # → local reid_gallery/ path
dm.get_parquet_path(session_id, "embeds")          # → local path (checks dirty flag)
dm.get_session_dir(session_id)                     # → local outputs/<session_id>/
dm.get_kinetics_path(filename)                     # → local collar_data/<filename>
dm.get_video_path(path)                            # pass-through — validates local file exists
dm.write_file(local_path, drive_rel_dest)          # upload with retry + buffer on failure
dm.find_collar_files(start_dt, end_dt, "kinetic")  # → list of local paths overlapping session
dm.load_collar_data(start_dt, end_dt, "kinetic")   # → merged DataFrame of all matching CSVs
dm.mark_dirty(flag, session_id)                    # set flag before writing
dm.mark_clean(flag)                                # set flag after successful upload
dm.check_flag(flag)                                # raises DriveNotSyncedError if dirty
dm.flush_buffer()                                  # retry all pending buffered uploads
```

**CLI commands:**

```bash
python3 drive_manager.py status                          # Drive connection, dirty flags, buffer
python3 drive_manager.py retry-buffer                    # upload all buffered files
python3 drive_manager.py pull-db                         # pull DB from Drive to local
python3 drive_manager.py list-sessions                   # list session folders on Drive
python3 drive_manager.py upload-kinetics /path/to/dir/   # batch-upload kinetics CSVs
python3 drive_manager.py upload-behavior /path/to/dir/   # batch-upload behavior CSVs
python3 drive_manager.py clear-flag      db|parquet|gallery   # emergency manual flag clear
```

**Collar CSV batch upload** (`upload-kinetics` / `upload-behavior`):
- Accepts a directory path — processes all `.csv` files found
- Skips files with fewer than 2 data lines
- Scans each file and renames it on upload:
  `kinetic_data_s<YYYY_MM_DD-HH_MM_SS>-e<YYYY_MM_DD-HH_MM_SS>__<id1>_<id2>_....csv`
  `behavior_data_s<YYYY_MM_DD-HH_MM_SS>-e<YYYY_MM_DD-HH_MM_SS>__<id1>_<id2>_....csv`
- Start/end timestamps derived from the earliest and latest `datetime` values in the file
- Animal IDs sorted ascending, joined with `_`

**Collar auto-discovery** (`find_collar_files` / `load_collar_data`):
- Called automatically by `reconcile.py` and `track_and_dump.py` when `--kinetics` is not provided
- Lists `collar_data/` on Drive, parses `s/e` timestamps from filenames, keeps files whose window overlaps the session window
- Pulls matching files to local `collar_data/` (skips if already present)
- Merges all matching files into a single DataFrame, deduplicating rows by `(AnimalId, datetime)`

**Dirty flags — what they block:**

| Flag | Set dirty by | Blocks |
|---|---|---|
| `db` | `track_and_dump.py` on first DB write | `reconcile.py`, `display_tracks.py`, `assign_identity.py` |
| `parquet` | `track_and_dump.py` on first parquet write | `reconcile.py` |
| `gallery` | `reconcile.py` on gallery update | `reconcile.py` (next session), `display_tracks.py` |

All blocked scripts accept `--bypass_upload_check` to skip the flag check and proceed with whatever is currently on Drive (logs a warning, flag stays dirty).

**Failure buffer:**
- Failed uploads are copied to `.buffer/pending/` with a `.meta.json` sidecar (dest, caller, size, attempt count)
- Max 5 attempts across sessions; after that the file is marked `abandoned` and requires manual intervention
- Retry backoff: 2 s → 8 s → 32 s within a session; cross-session retries triggered by any new upload or `retry-buffer`
- `upload_failures.log` and `bypass_warnings.log` live in `.buffer/` — local only, never uploaded

---

### `track_and_dump.py`
- Runs YOLO detection + ByteTrack/BoT-SORT tracking + YOLOv8-Pose (19 kp) + Embedder128 (MobileNetV3 → 128D L2-norm)
- **Batch mode:** `--source` accepts either a single video file or a directory — processes all video files in the directory sequentially
- Supported video extensions: `.mp4`, `.ts`, `.avi`, `.mov`, `.mkv`
- Models loaded **once** at startup and reused across all videos in a batch
- DB pulled from Drive **once** at startup and reused across all videos
- Outputs to Drive via `drive_manager` per session: `calving_project.db` (SQLite), `embeds.parquet`, `kps.parquet`, optional `crops/`
- Arrays (embed[128], kps[19×3], kps_kconf[19]) stored in Parquet — **never in SQLite**
- Parses wall-clock start time from filename token `_S<YYYYMMDDHHmmss>`
- Collar CSVs auto-discovered from Drive by matching session time window — no `--kinetics` needed
- **Calls `reconcile.py` automatically** after each video finishes
- Appends one row per processed video to `processing_log.csv` on Drive (flushed once at end of batch)
- Ctrl+C finishes the current video cleanly, logs it as `interrupted`, then stops before the next video

**`processing_log.csv` columns:** `timestamp, session_id, filename, status, frames, duration_s, fps_processed, error`  
**Status values:** `ok` · `interrupted` (Ctrl+C mid-video) · `skipped` (Ctrl+C before start) · `error`

- Key CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--model` | required | Detector `.pt` |
| `--source` | required | Single video file or directory of videos (local only) |
| `--tracker` | `bytetrack.yaml` | Tracker config |
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
| `--crop_tags` | off | Append `_<posture>_<facing>` to crop filenames (requires `--pose_model` and `vision_features/`) |
| `--crops_local` | off | Keep crops on local disk only — do not upload to Drive |
| `--save_every` | 10 | Sampling window: keep latest detection per temp_id per N frames |
| `--flush_every` | 5 | Flush to SQLite + Parquet every M windows (i.e. every N×M frames) |
| `--kinetics` | *(auto)* | Manual kinetics CSV override; if omitted, auto-discovered from Drive |
| `--corr_threshold` | 0.7 | Kinetic match Pearson r threshold |
| `--cosine_threshold` | 0.75 | Cosine similarity threshold |
| `--ema_alpha` | 0.15 | Gallery EMA decay |
| `--bypass_upload_check` | off | Skip dirty-flag check (proceeds with stale Drive data) |

---

### `reconcile.py`
- Post-processing pipeline: kinetic match → gallery update → cosine resolve → vision features → timeline write
- Called automatically by `track_and_dump.py`, or run manually per session
- Pulls canonical DB and gallery files from Drive via `drive_manager` before processing
- Checks `db`, `parquet`, and `gallery` dirty flags at entry — aborts if any are dirty (unless `--bypass_upload_check`)
- Collar CSVs (kinetics + behavior) auto-discovered from Drive by matching session time window — no `--kinetics` needed
- Syncs DB and updated gallery `.npy` files back to Drive after completion
- Steps in order: A (kinetic match) → A.5 (merge manual overrides) → B (gallery builder) → C (cosine resolver) → A.6 (resolve duplicate assignments) → D (sensor sequencer) → B-vision (vision feature extractor) → E (write to DB)
- Imports `vision_features` package for Step B-vision — the `vision_features/` folder must be in the same directory
- Key CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--db` | required | Path to `calving_project.db` |
| `--session` | required | `session_id` to process |
| `--kinetics` | *(auto)* | Manual kinetics CSV override; if omitted, auto-discovered from Drive |
| `--gallery_dir` | `./reid_gallery` | Gallery directory (overridden by drive_manager) |
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
| `--bypass_upload_check` | off | Skip dirty-flag check |

---

### `vision_features/`
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

**Also used by `track_and_dump.py`** for `--crop_tags`: classifiers are imported at startup (optional; graceful fallback to `_unk_unk` if unavailable) and run per-detection at crop-save time to append `_<posture>_<facing>` to crop filenames.

**Adding a new feature:**
1. Create `vision_features/features/my_feature.py` with `extract_my_feature()` and `aggregate_my_feature()`
2. Add output columns to `schema.py` (`TIMELINE_VISION_COLS` and `TIMELINE_ALTER_SQLS`)
3. Call both functions in `extractor.py` (two marked locations)

---

### `display_tracks.py`
- Browser-based visualizer — replaces X11/SDL2/ffplay/Tkinter; no display server required (WSL-compatible)
- Streams annotated video as MJPEG via Flask; sensor charts rendered server-side as PNG via matplotlib
- Opens `http://localhost:5000` automatically in the Windows browser (via `cmd.exe /c start`)
- Pulls canonical DB from Drive via `drive_manager` at startup; checks `db` dirty flag
- Imports `match_identity` for live kinetic matching; draws score table overlay (temp_id × AnimalId)
- Runs live posture + facing classification per frame if `vision_features/` is present (optional import)
- Browser controls: pause / fast-forward (1×/2×/4×/8×) / quit; sensor chart auto-refreshes every 3 s
- Accepts `--bypass_upload_check` to proceed when DB flag is dirty
- Requires: `pip install flask`

---

### `match_identity.py`
- Kinetic matching: bbox centroid speed ↔ collar ΔKineticsCountR
- Pearson correlation per 15-min bin, Hungarian assignment
- Does not use drive_manager — never touches the DB; reads CSVs passed directly as arguments
- `score_up_to(tracks_df, kinetics_df, up_to_datetime, bin_minutes, ...)` → assignment dict + scores_df (used by `display_tracks.py` for live overlays)
- Standalone CLI: `python3 match_identity.py --tracks tracks.csv --kinetics ... --output ...`

---

### `assign_identity.py`
- Manual identity assignment tool — use when kinetics data is unavailable or doesn't cover the video window
- Pulls canonical DB from Drive via `drive_manager` at startup; checks `db` dirty flag
- Syncs DB back to Drive after writing assignments
- Writes assignments to a `manual_assignments` table in SQLite; reconcile.py merges these in Step A.5, with manual overriding kinetic on conflict
- Can optionally call `reconcile.py` directly after writing (`--run_reconcile`)
- Accepts `--bypass_upload_check` to proceed when DB flag is dirty
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

### `calving_ledger_ingest.py`
- Ingests a farm's calving ledger file (.xlsx or .csv) into `calving_ledger` + `calving_features`
- Pulls canonical DB from Drive via `drive_manager` at startup; syncs back after writing
- Farm-specific parsing is fully isolated in adapter classes — the orchestrator never contains farm logic
- Idempotent: skips already-ingested events by default (keyed on `real_id + calving_dt`); `--overwrite` to replace
- `--dry_run` parses and validates without touching the DB — works with no Drive connection
- Auto-migrates schema: any extra column returned by an adapter triggers `ALTER TABLE ADD COLUMN`

**Adapter structure:**
```
scripts/
  calving_ledger_ingest.py      ← orchestrator
  adapters/
    __init__.py
    base_adapter.py             ← abstract base: load_and_parse() → canonical DataFrame
    gazit_adapter.py            ← Gazit farm parsing logic
```

**`BaseAdapter` contract** — every adapter implements one method:
```python
def parse(self, df_raw: pd.DataFrame, source_file: str) -> pd.DataFrame:
    # returns canonical DataFrame with real_id, calving_dt, outcome,
    # parity, n_calves (required) + any subset of nullable feature cols
```

**Gazit-specific parsing rules** (in `gazit_adapter.py`, nowhere else):

| Rule | Value |
|---|---|
| Date format | Excel serial + time fraction → datetime |
| `n_calves >= 2` | → `Twin` (overrides all other rules) |
| `קשה` (hard) | → `Assisted` |
| `קלה` (easy) | → `Unassisted` |
| `ללא התערבות` (no intervention) | → `Unassisted` |
| "הפוכה" in notes | → `is_breech = True` |

**Adding a new farm:**
1. Create `adapters/my_farm_adapter.py`, subclass `BaseAdapter`, set `FARM_NAME`
2. Implement `parse()` with that farm's column names and outcome vocabulary
3. Add one entry to `ADAPTERS` dict in `calving_ledger_ingest.py`

**CLI flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--file` | required | Path to ledger file (.xlsx or .csv) |
| `--farm` | *(auto from filename)* | Farm identifier — e.g. `gazit` |
| `--dry_run` | off | Parse + validate only, no DB writes, no Drive needed |
| `--overwrite` | off | Replace existing rows (default: skip) |
| `--bypass_upload_check` | off | Skip dirty-flag check |

---

## Sensor Data

### `behavior_data_*.csv`
- Columns: AnimalId, datetime, f_1_2, f_2_3, v
- Interval: ~90 seconds
- Proprietary collar-derived behavioral features
- Upload with: `python3 drive_manager.py upload-behavior /path/to/dir/`
- Auto-discovered per session from Drive by time-window overlap

### `kinetic_data_*.csv`
- Columns: AnimalId, datetime, KineticsCountX, KineticsCountY, KineticsCountZ, KineticsCountR
- Interval: ~15 minutes
- Cumulative accelerometer counts; deltas computed for matching and features
- Upload with: `python3 drive_manager.py upload-kinetics /path/to/dir/`
- Auto-discovered per session from Drive by time-window overlap

---

## Three-Stage Pipeline

### Stage 1 — Inference (`track_and_dump.py`, per video or batch)
**Inputs:** Raw video(s), YOLO `.pt` models  
**Process:** Detect → Track (temp_id) → Pose (kps) → Embed (128D) → call reconcile.py  
**Outputs written (to Drive via drive_manager):**
- `raw_tracks` — append-only, one row per detection (scalar columns); arrays in Parquet
- `video_sessions` — one row per video file registered
- `processing_log.csv` — one row per processed video (batch audit trail)

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

## Databases (9 total)

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
- event_id (PK), real_id (FK → reid_registry), calving_dt, outcome
- outcome enum: `Unassisted` | `Assisted` | `Twin` | `Veterinarian`
- source_farm, source_file, ingested_at — audit trail
- Written by `calving_ledger_ingest.py`; consumed by Stage 3 dataset builder

### `calving_features` — per-calving model input features (1:1 with calving_ledger)
- event_id (PK, FK → calving_ledger)
- **Direct from ledger:** parity, gestation_days, days_in_milk, dry_days, milk_at_dryoff, dry_off_scc, n_calves
- **Derived from calving_dt:** calving_hour, calving_month, calving_season
- **Derived from notes:** is_breech
- **Cross-event (prior calving for same cow):** days_since_last_calving, prior_outcome, prior_n_calves
- notes_raw — cleaned free-text
- All feature columns are nullable — missing data is a first-class value, not an error
- Schema is additive: new farm-specific columns trigger `ALTER TABLE ADD COLUMN` automatically

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
# 0. First-time setup — upload collar CSVs to Drive once
python3 drive_manager.py upload-kinetics ~/thesis_workspace/raw_data/CollarData/
python3 drive_manager.py upload-behavior ~/thesis_workspace/raw_data/CollarData/

# 1a. Single video — inference + reconcile (collar CSVs auto-discovered)
python3 track_and_dump.py \
    --model      models/cow_detector/best.pt \
    --source     ~/thesis_workspace/raw_data/calving/refet_33_S20241221070000_E20241221080000.ts \
    --pose_model models/cow_pose/best.pt \
    --imgsz 960 --conf 0.30 --iou 0.60 \
    --save_every 10 --flush_every 5

# 1b. Batch — all videos in a directory (models loaded once, runs sequentially)
python3 track_and_dump.py \
    --model      models/cow_detector/best.pt \
    --source     ~/thesis_workspace/raw_data/calving/ \
    --pose_model models/cow_pose/best.pt \
    --imgsz 960 --conf 0.30 --iou 0.60 \
    --save_every 10 --flush_every 5

# 2. Validate visually (opens http://localhost:5000 in browser)
python3 display_tracks.py \
    --video      ~/thesis_workspace/raw_data/calving/refet_33_S20241221070000_E20241221080000.ts \
    --db         ~/thesis_workspace/vcf-ctp/data/calving_project.db \
    --session_id refet_33_20241221070000 \
    --draw_pose --show_fps

# 3. If kinetics unavailable — assign manually after watching display_tracks.py
python3 assign_identity.py \
    --db      ~/thesis_workspace/vcf-ctp/data/calving_project.db \
    --session refet_33_20241221070000 \
    --assign  2:7507  1:6366  71:7513 \
    --note    "manual — no kinetics for this date" \
    --run_reconcile

# 4. Re-run reconcile standalone (e.g. after tuning thresholds)
python3 reconcile.py \
    --db          ~/thesis_workspace/vcf-ctp/data/calving_project.db \
    --session     refet_33_20241221070000 \
    --embed_parquet ~/thesis_workspace/vcf-ctp/data/outputs/refet_33_20241221070000/embeds.parquet \
    --corr_threshold 0.7 --cosine_threshold 0.75 --ema_alpha 0.15
    # add --dry_run to test without writing
    # add --kinetics /path/to/file.csv to override auto-discovery

# 5. Ingest calving ledger (once per farm file — idempotent)
python3 calving_ledger_ingest.py \
    --file  ~/thesis_workspace/raw_data/gazit_calving_events_202505_202605.xlsx \
    --farm  gazit
    # add --dry_run to validate without writing
    # add --overwrite to re-ingest an already-loaded file

# 6. Check sync status / recover from failures
python3 drive_manager.py status
python3 drive_manager.py retry-buffer
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
15. **Synthetic IDs for unresolved cows** — when a temp_id matches another temp_id cross-session but no real_id is known yet, reconcile.py mints a negative synthetic id (from `synthetic_id_counter.npy`). The synthetic id propagates consistently until a future kinetic match resolves it to a real AnimalId, at which point all prior timeline rows are backpropagated automatically.
16. **Drive as single source of truth** — all structured data lives on Google Drive (`thesis_google_drive:vcf_ctp_data`). Local disk is a write buffer only. drive_manager.py enforces this via dirty flags that block read scripts from running against stale Drive data.
17. **Collar CSVs auto-discovered by time window** — reconcile.py and track_and_dump.py call `load_collar_data()` which lists Drive's `collar_data/`, parses `s/e` timestamps from canonical filenames, and merges all overlapping files. No `--kinetics` argument needed in normal operation.
18. **Batch processing with single model load** — track_and_dump.py accepts a directory as `--source`. Models and DB are loaded once; `process_video()` is called per file. A `processing_log.csv` on Drive records every video processed, its outcome, and performance metrics.
19. **Pluggable calving ledger adapters** — each farm's ledger format is parsed by a dedicated `BaseAdapter` subclass. Farm-specific column names, date formats, language of outcome values, and derivation rules are fully isolated in the adapter. The orchestrator (`calving_ledger_ingest.py`) only calls `parse()`. Adding a new farm = one new file in `adapters/`, zero changes elsewhere.
20. **Calving features split from ground truth** — `calving_ledger` holds only the label (outcome + identity + datetime). `calving_features` holds all model input features keyed on `event_id`. The two tables evolve independently: new farms can introduce new feature columns via auto-migration without touching existing rows or code.

---

## Storage Architecture

### Decision: SQLite + Parquet + Google Drive

**Database engine:** SQLite for all structured tables. Zero infrastructure, full SQL, pandas-native.  
**Array storage:** Parquet files (pyarrow) for embed[128], kps[19×3], kps_kconf[19] — one file pair per session. SQLite stores integer row pointers, never raw blobs.  
**Cloud storage:** Google Drive via rclone (`thesis_google_drive` remote). drive_manager.py handles all sync.  
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

### Directory layout
```
~/thesis_workspace/vcf-ctp/
  scripts/
    drive_manager.py                        # Drive I/O layer — import in all scripts
    reconcile.py
    track_and_dump.py
    display_tracks.py
    match_identity.py
    assign_identity.py
    calving_ledger_ingest.py                # ledger ingestion orchestrator
    adapters/
      __init__.py
      base_adapter.py                       # abstract base — farm adapter contract
      gazit_adapter.py                      # Gazit farm parsing logic
    vision_features/
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
  data/                                     # LOCAL write buffer (drive_manager managed)
    calving_project.db                      # pulled from Drive before each batch
    outputs/<session_id>/
      embeds.parquet
      kps.parquet
      crops/                                # optional — local only if --crops_local
    reid_gallery/                           # pulled from Drive before reconcile
    collar_data/                            # local copies of auto-discovered collar CSVs
  .buffer/                                  # LOCAL only — never uploaded
    db_sync_status.json                     # dirty flag for DB
    parquet_sync_status.json                # dirty flag for parquet files
    gallery_sync_status.json                # dirty flag for gallery files
    upload_log_staging.csv                  # append-only; flushed to Drive at end of batch
    _batch_log_staging.csv                  # processing_log rows staged locally
    upload_failures.log                     # failure log
    bypass_warnings.log                     # log of --bypass_upload_check usages
    pending/                                # buffered failed uploads
      <timestamp>_<filename>               # copy of file to retry
      <timestamp>_<filename>.meta.json     # dest, caller, attempt count, error

thesis_google_drive:vcf_ctp_data/           # Google Drive (source of truth)
  calving_project.db
  outputs/<session_id>/
    embeds.parquet
    kps.parquet
    crops/                                  # uploaded unless --crops_local
  reid_gallery/
    gallery_day.npy
    gallery_night.npy
    gallery_pose_day.npy
    gallery_pose_night.npy
    temp_gallery_pose_day.npy
    temp_gallery_pose_night.npy
    synthetic_id_counter.npy
  collar_data/
    kinetic_data_s<start>-e<end>__<ids>.csv
    behavior_data_s<start>-e<end>__<ids>.csv
  models/
    cow_detector/best.pt
    cow_pose/best.pt
  upload_log.csv                            # permanent audit trail of all uploads
  processing_log.csv                        # one row per processed video (batch log)

raw_data/videos/                            # LOCAL ONLY — never synced
```


---

## ToDo

- [ ] **Collar data — handle overlapping files on upload** (`drive_manager.py` → `_cmd_upload_collar`):
  When two CSV files cover the same animals and overlapping time windows (e.g. an export was re-run with a wider date range), the current upload creates separate files on Drive with different names but partially duplicate data. Need to detect overlap on upload and decide a strategy — options include: merge rows and re-export as a single file with the combined time range, skip files whose entire window is already covered by an existing Drive file, or flag for manual review and skip. Overlapping files are identified by matching animal ID sets and intersecting `[s_timestamp, e_timestamp]` ranges parsed from the canonical filename.

---

## Milestone Schedule
| Milestone | Target | Status |
|---|---|---|
| Pose Estimation | March 29, 2026 | ✓ Done |
| Sensor Pipeline | April 12, 2026 | ✓ Done |
| Re-Identification Module | May 3, 2026 | ✓ Done |
| Vision Feature Extraction | May 29, 2026 | ✓ Done |
| Database pipeline + cloud storage | June 5, 2026 | ✓ Done |
| Calving Ledger Ingestion | June 14, 2026 | ✓ Done |
| Working Pipeline | June 20, 2026 | — |
| Temporal Prediction Model | July 26, 2026 | — |
| Model Retraining and Fine-tuning | August 30, 2026 | — |
| Empirical Evaluation | September 30, 2026 | — |
| Research Paper | October 30, 2026 | — |
| Thesis Document | November 30, 2026 | — |