# vcf-ctp
**V**ision–**C**ollar **F**usion for **C**alving-**T**ype **P**rediction in Dairy Cows

## Project
**Thesis:** Predictive Modelling of Calving Outcomes in Dairy Cows Using Multi-Modal Sensor and Vision Data  
**Author:** Anton Avstreikh, University of Haifa  
**Advisors:** Prof. Ilan Shimshoni  
**Output:** 4-class calving type prediction — Unassisted · Assisted · Twin · Veterinarian-assisted

---

## Existing Code

### Configuration — `config.py` + `config/`
Two YAML files hold every setting; scripts keep no defaults of their own.

| File | Holds | Examples |
|---|---|---|
| `config/params.yaml` | Everything that changes what the pipeline **computes** | model paths, detection/pose thresholds, kinetic match, gallery contribution ledger, robust mean / ID-switch check, cosine threshold, co-occurrence, trust order, posture/facing thresholds, cross-camera tiers, cameras + zone polygons |
| `config/cli.yaml` | How scripts **run** and what QA output they produce | `save_crops`, `save_masks`, `crop_tags`, display overlays, metrics output, Drive/local storage |
| `vision_features/schema.py` | **Fixed** structural constants — not config | `EMBED_DIM`=128, `N_KP`=19, KP map, 8 gallery slots, ImageNet mean/std, `BACK_KPS` |

- `config.py` loads both into frozen, typed objects. **Strict:** a missing key, an unknown key (typo) or a wrong type is an error naming the key; cross-field rules are checked too (`standing_max_ar < lying_min_ar`, `manual` first in `source_priority`, topology cameras must be listed under `cameras`, ASCII camera codes, values in [0,1] where required).
- Validate after editing: `python3 config.py` (or `python3 config.py my_params.yaml`)
- Threshold sweep without touching the main file: any script `--params config/params_strict.yaml`. Only params can be swapped; `cli.yaml` is fixed at `config/cli.yaml` because `drive_manager` reads `storage:` at import.
- Values are read when a function runs, never at import, so `--params` always takes effect.
- Relative paths resolve against `scripts/`; `detection.tracker` as a bare name resolves to `config/` (where `bytetrack.yaml` / `botsort.yaml` live).
- **Command-line flags that remain:** inputs (`--source`, `--video`, `--session_id`, `--session`), operations (`assign_identity --assign/--remove/--list/--note/--run_reconcile`, `drive_manager <command>`, ledger `--file/--farm/--overwrite`, `gallery_ledger <command>`, `match_identity --tracks/--kinetics/--output/--scores_json`), one-shot safety switches (`--dry_run`, `--bypass_upload_check` — deliberately never settable in a file), and `--params`.
- **Reproducibility:** every session stores the exact params file it ran with on Drive — `outputs/<session_id>/params_track.yaml` (written by `track_and_dump.py`) and `params_reconcile.yaml` (rewritten on every reconcile run, since reconcile can be re-run with different params later). `processing_log.csv` has a `params_hash` column (short sha256 of the file); older logs are upgraded in place on the next flush.

---

### `drive_manager.py`
- Central Google Drive I/O abstraction layer — all scripts read/write through this module
- Drive (`thesis_google_drive:vcf_ctp_data`) is the single source of truth for all data
- Local (`~/thesis_workspace/vcf-ctp/data/`) is a write buffer for the current session only
- Dirty flags (`db`, `parquet`, `gallery`) block downstream read scripts when local writes are unsynced
- Upload failures are buffered locally (`.buffer/pending/`) and retried automatically
- All uploads logged locally to `upload_log_staging.csv`; flushed to `upload_log.csv` on Drive in a single round-trip at end of batch — never once per file
- Graceful Ctrl+C: finishes the current file, flushes staged log rows, then exits cleanly
- Storage settings live in `config/cli.yaml` → `storage:` (module constants are derived from it at import):

| Key | Value | Becomes |
|---|---|---|
| `rclone_remote` | `thesis_google_drive` | `RCLONE_REMOTE` |
| `drive_folder` | `vcf_ctp_data` | `DRIVE_ROOT` = `thesis_google_drive:vcf_ctp_data` |
| `local_root` | `~/thesis_workspace/vcf-ctp/data` | `LOCAL_ROOT` |
| `buffer_dir` | `~/thesis_workspace/vcf-ctp/.buffer` | `BUFFER_DIR` |
| `max_attempts` | 5 | `MAX_ATTEMPTS` |
| `backoff_base` | 2 | `BACKOFF_BASE` |

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
dm.remove_stale_remote_files(drive_rel_dir, keep, patterns)  # delete matching Drive files not in keep
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
- Called automatically by `reconcile.py` and `track_and_dump.py` (there is no manual kinetics-path flag)
- Lists `collar_data/` on Drive, parses `s/e` timestamps from filenames, keeps files whose window overlaps the session window
- Pulls matching files to local `collar_data/` (skips if already present)
- Merges all matching files into a single DataFrame, deduplicating rows by `(AnimalId, datetime)`

**Dirty flags — what they block:**

| Flag | Set dirty by | Blocks |
|---|---|---|
| `db` | `track_and_dump.py` on first DB write | `reconcile.py`, `display_tracks.py`, `assign_identity.py` |
| `parquet` | `track_and_dump.py` on first parquet write | `reconcile.py` |
| `gallery` | `reconcile.py` on gallery update | `reconcile.py` (next session) |

All blocked scripts accept `--bypass_upload_check` to skip the flag check and proceed with whatever is currently on Drive (logs a warning, flag stays dirty).

**Pull failure classification (`_rclone_download`):**
- Distinguishes "file doesn't exist on Drive yet" (first run — proceeds, creates local) from a genuine connectivity/auth failure (`allow_stale` governs whether to proceed with a stale local copy or raise `DriveUnavailableError`), by pattern-matching rclone's `stderr`
- Known "not found" phrasings include both directory-style (`"directory not found"`) and file-style (`"source doesn't exist"`) wording — different rclone versions phrase the same missing-source condition differently depending on whether the local destination already exists
- Any failure that doesn't match a known "not found" phrase is logged with the **full raw rclone stderr** via `_log()` — never silently discarded — so a genuine new failure mode (quota, permissions, version incompatibility, etc.) is diagnosable from the log immediately rather than surfacing only as a generic "Drive unavailable"

**Failure buffer:**
- Failed uploads are copied to `.buffer/pending/` with a `.meta.json` sidecar (dest, caller, size, attempt count)
- Max 5 attempts across sessions; after that the file is marked `abandoned` and requires manual intervention
- Retry backoff: 2 s → 8 s → 32 s within a session; cross-session retries triggered by any new upload or `retry-buffer`
- `upload_failures.log` and `bypass_warnings.log` live in `.buffer/` — local only, never uploaded

---

### `track_and_dump.py`
- Runs YOLO detection + ByteTrack/BoT-SORT tracking + YOLOv8-Pose (19 kp) + Embedder128 (MobileNetV3 → 128D L2-norm)
- **Detector accepts either a plain detection model or a YOLO-seg (instance segmentation) model** — same `models.detector` setting in params.yaml, auto-detected via `det_model.task` at load time (logged at startup: `det. task : detect|segment`)
  - **Segmentation model loaded** → `r.masks.xy` polygons are extracted per detection (same index order as `r.boxes`) and used to build **background-removed crops for the embedder only** (`masked_crops_from_polys()`), fixing background contamination in ReID embeddings. Pose-model input and saved crop JPEGs (`save_crops`) deliberately keep using raw bbox crops — unaffected, so the existing pose model's input distribution doesn't shift.
  - **Plain detection model loaded** → behavior unchanged from before segmentation support; embedder uses raw bbox crops.
  - Tracker association (ByteTrack/BoT-SORT) stays **box-IoU only** regardless of model type — mask-IoU association is a separate, not-yet-implemented upgrade (see ToDo).
- **Batch mode:** `--source` accepts either a single video file or a directory — processes all video files in the directory sequentially
- Supported video extensions: `.mp4`, `.ts`, `.avi`, `.mov`, `.mkv`
- Models loaded **once** at startup and reused across all videos in a batch
- DB pulled from Drive **once** at startup and reused across all videos
- Outputs to Drive via `drive_manager` per session: `calving_project.db` (SQLite), `embeds.parquet`, `kps.parquet`, optional `crops/`
- Arrays (embed[128], kps[19×3], kps_kconf[19]) stored in Parquet — **never in SQLite**
- **Camera mode (day colour / night IR) detected per frame from the pixels** — the cameras switch themselves to grayscale when light drops, possibly mid-video, so day/night is read from the video, not the clock. Each frame's metric is the fraction of (strided) pixels with `max(|R−G|,|G−B|,|R−B|) > chroma_tol`: ~0 in IR mode, clearly above 0 in colour mode even when the scene is washed out (mean HSV saturation was rejected — it's unstable on the dark pixels that dominate night frames). A frame is night if the fraction is ≤ `color_pixel_frac`. A new mode is only recorded after it holds for `debounce_seconds` (filter-swap / auto-exposure flicker is ignored; the first frame sets the starting mode). Stored per row in `raw_tracks.is_night`; `video_sessions.is_night` (majority) and `night_frame_fraction` are summaries. Cost ≈ 0.3 ms per 1080p frame. End-of-video log: % night, each switch with frame + wall time, and the metric's p1/p5/median/p95/p99 for threshold tuning.
- **QA-only mask persistence (`save_masks` in cli.yaml):** when enabled with a segmentation model, mask polygons are additionally written to `outputs/<session_id>/qa_masks/masks_part*.parquet` (own schema, own filename prefix) — purely for visualization in `display_tracks.py` (`draw_masks`). Structurally isolated from the canonical outputs: separate subdirectory, separate file prefix, **never uploaded to Drive** (the end-of-video upload loop globs `embeds_part*`/`kps_part*` directly in `outdir`, non-recursive — `qa_masks/` is invisible to it by directory and by filename pattern), never touches `raw_tracks` or any other pipeline stage. No-op with a log line if the loaded model has no segmentation head.
- Parses wall-clock start time from filename token `_S<YYYYMMDDHHmmss>`
- Collar CSVs auto-discovered from Drive by matching session time window
- **Calls `reconcile.py` automatically** after each video finishes
- Appends one row per processed video to `processing_log.csv` on Drive (flushed once at end of batch)
- Ctrl+C finishes the current video cleanly, logs it as `interrupted`, then stops before the next video

**`processing_log.csv` columns:** `timestamp, session_id, filename, status, frames, duration_s, fps_processed, error, params_hash`  
**Status values:** `ok` · `interrupted` (Ctrl+C mid-video) · `skipped` (Ctrl+C before start) · `error`

- CLI: `--source` (file or directory), `--bypass_upload_check`, `--params`. Everything else is config:

| Setting | File → key | Default |
|---|---|---|
| Detector / pose model | params `models.detector` / `models.pose` (`null` = no pose) | `../models/cow_seg/run14/weights/best.pt` / `../models/cow_pose/best.pt` |
| Detector size / conf / NMS IoU | params `detection.imgsz` / `conf` / `iou` | 960 / 0.25 / 0.45 |
| Tracker config | params `detection.tracker` | `bytetrack.yaml` (in `config/`) |
| Pose size / conf / per-kp vis split | params `pose.imgsz` / `conf` / `kp_conf_thresh` | 384 / 0.25 / 0.30 |
| Embedder crop size / bbox margin | params `embedder.input_size` / `crop_margin` | 224 / 0.10 |
| Sampling window / flush cadence / parquet part size | params `sampling.save_every` / `flush_every` / `parquet_rows_per_part` | 10 / 5 / 50000 |
| Crops: save / local-only / tags / every N / min size | cli `track_and_dump.save_crops` / `crops_local` / `crop_tags` / `crop_every` / `min_crop_wh` | off / off / off / 1 / [0,0] |
| QA mask polygons | cli `track_and_dump.save_masks` | off |

Reconcile thresholds are no longer passed by `track_and_dump.py` — reconcile reads the same loaded params (previously six reconcile values were hardcoded at this call site and ignored reconcile's own flags).

---

### `reconcile.py`
- Post-processing pipeline: kinetic match → gallery update → cosine resolve → vision features → timeline write
- Called automatically by `track_and_dump.py`, or run manually per session
- Pulls canonical DB and gallery files from Drive via `drive_manager` before processing
- Checks `db`, `parquet`, and `gallery` dirty flags at entry — aborts if any are dirty (unless `--bypass_upload_check`)
- Collar CSVs (kinetics + behavior) auto-discovered from Drive by matching session time window — no `--kinetics` needed
- Syncs DB and updated gallery `.npy` files back to Drive after completion
- Steps in order: A (kinetic match) → A.5 (merge manual overrides) → B (gallery builder) → C (cosine resolver) → A.6 (resolve duplicate assignments) → D (sensor sequencer) → B-vision (vision feature extractor) → E (write to DB)
- Imports `vision_features` package for the Vision Feature Extractor step — the `vision_features/` folder must be in the same directory
- CLI: `--session`, `--dry_run`, `--bypass_upload_check`, `--params`. `run(session, dry_run, bypass_upload_check)` is the entry point for `track_and_dump.py` and `assign_identity.py`.
- Thresholds (all `params.yaml`):

| Step | Keys | Defaults |
|---|---|---|
| Tracklet Stitching | `stitching.enabled` / `max_gap_s` / `max_dist` / `dist_per_s` / `end_window_s` / `max_size_ratio` / `ambiguity_margin` | true / 8.5 / 0.5 / 0.05 / 1.0 / 1.5 / 0.15 |
| Noise Filter | `noise_filter.min_duration_s` / `min_detections` | 2.0 / 25 |
| Kinetic Matcher | `kinetic_match.corr_threshold` / `min_active_bins` / `min_temp_id_frames` / `activity_pct` | 0.7 / 3 / 0.10 / 0.25 |
| Shared time grid | `timeline.bin_minutes` (matcher, Sensor Sequencer, vision windows) | 15 |
| Camera mode (read per row from `raw_tracks.is_night`, set by track_and_dump.py) | `camera_mode.chroma_tol` / `color_pixel_frac` / `debounce_seconds` / `sample_stride` | 6 / 0.05 / 2.0 / 8 |
| Legacy is_night fallback — only rows with NULL `is_night` (tracked before per-frame detection) | `session.night_start_hour` / `night_end_hour` / `night_fraction` | 20 / 6 / 0.5 |
| Gallery commit (flat + pose, real ids) | `gallery.source_weights` (manual / kinetic / boundary_rendezvous / cosine) / `min_embeds` (effective frames) | 1.0 / 1.0 / 0.5 / 0.5 / 10 |
| Gallery contribution ledger | `gallery.contribution.half_life_days` / `drift_gate_cosine` / `frames_weight_power` / `min_weight` / `legacy_seed_frames` | 180 / 0.5 / 0.5 / 0.02 / 200 (gate provisional) |
| Synthetic ids + TempPoseGallery (still EMA) | `gallery.ema_alpha` / `synthetic_merge_alpha` | 0.15 / 0.5 |
| Appearance quality (every step that pools embeddings) | `appearance_quality.min_px` / `full_px` / `edge_margin_px` / `edge_weight` | 24 / 96 / 4 / 0.3 (size provisional) |
| Robust session mean (every pooled embedding) | `appearance_quality.robust_mean.enabled` / `min_frames_for_trim` / `trim_mad_k` / `max_trim_frac` | true / 20 / 3.0 / 0.3 |
| ID-switch flag (Gallery commit) | `appearance_quality.id_switch.enabled` / `min_frames` / `min_cluster_frac` / `min_gap` | true / 20 / 0.25 / 0.15 (provisional) |
| Cosine Resolver | `cosine_resolver.threshold` (also Boundary Rendezvous tier 2) / `min_embeds` | 0.75 / 5 |
| Cosine ids → gallery (margin gate) | `gallery.cosine_gate.enabled` / `threshold_margin` / `second_best_margin` | true / 0.05 / 0.10 (provisional) |
| Cosine + Duplicate Resolver | `cooccurrence.iou_same_animal` / `frame_tolerance_min` / `gap_multiplier` | 0.3 / 20 / 2.0 |
| Duplicate Resolver, Boundary Rendezvous | `identity.source_priority` / `protected_sources` | manual > kinetic > boundary_rendezvous > cosine / manual, kinetic |

---

### `vision_features/`
- Python package implementing Stage 2-B: vision feature extraction from raw keypoints
- Must be placed in the same directory as `reconcile.py` (i.e. `scripts/vision_features/`)
- Called by `reconcile.py` after Sensor Sequencer; never called directly
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
| Posture | Bbox aspect ratio (AR ≤ `standing_max_ar` 0.85 → standing, ≥ `lying_min_ar` 1.10 → lying, band = uncertain) + kps_coverage gate — `params.yaml vision.posture` | `posture` ∈ {standing, lying, uncertain}, `posture_conf` |
| Facing | Spine axis from `schema.BACK_KPS` (withers→tail base); angle zones `toward_center`/`away_center` ± `depth_half_width` — `params.yaml vision.facing` | `facing` ∈ {left, right, toward, away, uncertain}, `facing_conf` |

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
- Only confident frames (both posture_conf and facing_conf ≥ `vision.min_slot_conf`, 0.30) contribute to a slot; window aggregates count frames ≥ `vision.aggregate_min_conf` (0.30)
- A slot vector is the appearance-quality-weighted mean of its frames and needs ≥ `vision.min_slot_frames` (3) effective frames — one confident frame no longer makes a slot
- Slot query uses a 3-level fallback chain: exact slot → same posture any facing → all populated slots
- Updated by the extractor once per run through the gallery contribution ledger (`gallery_ledger.py`), with the same `source_weights` and drift gate as the flat gallery — see `reid_registry` below. Synthetic ids and the TempPoseGallery still use a plain EMA (`gallery.ema_alpha`)

**Also used by `track_and_dump.py`** for `crop_tags` (cli.yaml): classifiers are imported at startup (optional; graceful fallback to `_unk_unk` if unavailable) and run per-detection at crop-save time to append `_<posture>_<facing>` to crop filenames.

**Adding a new feature:**
1. Create `vision_features/features/my_feature.py` with `extract_my_feature()` and `aggregate_my_feature()`
2. Add output columns to `schema.py` (`TIMELINE_VISION_COLS` and `TIMELINE_ALTER_SQLS`)
3. Call both functions in `extractor.py` (two marked locations)
4. Put its thresholds under `vision:` in `params.yaml` and add the matching dataclass fields in `config.py` (read them with `config.params()` inside the function, never at import)

---

### `cross_camera.py`
- Boundary Rendezvous — triggered cross-camera reconcile
- Not a standalone CLI; called by `reconcile.py` once per session, after Cosine Resolver and before Duplicate Resolver
- Must be in the same directory as `reconcile.py` — `from cross_camera import run_boundary_rendezvous, CrossCameraConfig`; reconcile builds the config with `CrossCameraConfig.from_params(P)`
- `camera_topology` in `params.yaml` — a static, hand-annotated config of which camera pairs share a physical boundary and the rough pixel polygon marking that zone in each camera's own frame. Currently: `רפת 26`↔`רפת 38` and `רפת 33`↔`רפת 38` (38 is the hub; 26 and 33 don't overlap directly). No calibration/homography — polygons are rough regions, not precise boundaries.
- **Staging:** for each temp_id, computes cumulative dwell time inside any zone shared with another camera (point-in-polygon on bbox centroid). Past `boundary_dwell_seconds` (and a `boundary_dwell_min_frames` noise floor), writes a row to `boundary_observations` — camera, zone, time range, flat embedding, current resolution state.
- **Triggering:** that write immediately checks `boundary_observations` for a pending row from the zone's *other* camera with an overlapping time range. Found → resolve now. Not found → the row waits, matchable by any future session on any camera, in any order.
- **Matching, when multiple candidates are in the zone on both sides:** scores every candidate pair, takes the globally best non-conflicting set of pairings (not a naive one-at-a-time waterfall), using a 4-tier hierarchy — mirrors reconcile's own trust order, with two new tiers appended at the bottom:
  1. Both sides already independently kinetically confirmed to the same AnimalId
  2. Flat-embedding cosine similarity ≥ `cosine_threshold` (same mechanism Cosine Resolver uses)
  3. Motion-magnitude correlation ≥ `motion_corr_threshold` — both tracks are read from `raw_tracks` (every camera's tracks share the DB), over the dwell overlap padded by `motion_window_pad_seconds`. Each becomes a speed curve in body-heights per second (1-second bins), and the two are compared with Pearson r, allowing a small time shift for clock offset. Speed only, never direction, since 33 faces opposite to 26/38. No score (tier skipped, not a mismatch) when there are too few shared bins or either cow barely moved, because two near-still curves are just detection jitter. Only computed for pairs tiers 1–2 didn't match. Default scorer is `make_motion_corr_fn`; `run_boundary_rendezvous(motion_corr_fn=...)` can override it.
  4. Only one cow in the zone on each side, gated by a `single_cow_sanity_floor` cosine score (rejects a stray false-positive detection)
- **Resolving a pairing:** zone-consistent (expected — both tracks plausibly near the shared boundary) → corroboration, no conflict. Zone-inconsistent (same AnimalId claimed somewhere that isn't physically plausible) → real conflict: auto-resolved by confidence if the gap ≥ `conflict_confidence_gap`, otherwise logged to `cross_camera_conflicts` for manual review instead of guessing.
- **Priority:** follows `identity.source_priority` in `params.yaml` — manual > kinetic > boundary_rendezvous > cosine. Boundary Rendezvous can set an unassigned temp_id or replace a Cosine Resolver id (verified shared-zone + overlapping-time evidence beats appearance alone), but never overwrites a manual or kinetic id: `reconcile.py` passes those temp_ids as `protected_tids`, and a disagreeing boundary match is logged as `KEPT` instead.
- **Gallery enrichment:** no special handling. When two cameras see the same cow at the boundary, each session updates the pose-conditioned gallery normally through `_update_pose_gallery`. Different posture/facing slots land in different rows of the cow's `(8,128)` array, so both views are kept. Same-slot overlap is two ordinary ledger contributions, one per session — see Key Design Decision 22 for why a "keep only the better view" rule was considered and dropped.
- **Known gap, deferred:** a pairing's result isn't checked against an independent identity claim already on record elsewhere (e.g. a non-adjacent camera's own kinetic claim on the same AnimalId at an overlapping time) — a real conflict that can't currently arise, by construction, from the pairing logic itself. Explicitly punted; needs its own pass before relying on this for anything adjacent-camera-sensitive.
- Key config — `params.yaml` `cross_camera:` (tier-3 keys under `cross_camera.motion:`; table shows the `CrossCameraConfig` field names):

| Flag | Default | Purpose |
|---|---|---|
| `boundary_dwell_seconds` | 3.0 | Min cumulative dwell in a shared zone to stage a candidate |
| `boundary_dwell_min_frames` | 5 | Min detections contributing to that dwell — rejects single-frame noise |
| `conflict_confidence_gap` | 0.10 | Min confidence difference to auto-resolve a conflict silently; closer calls go to review |
| `motion_corr_threshold` | 0.85 | Tier-3 minimum r between the two speed curves |
| `motion_bin_seconds` | 1.0 | Tier-3 speed bin width |
| `motion_window_pad_seconds` | 30.0 | Tier-3 window: dwell overlap ± this |
| `motion_min_samples` | 8 | Tier-3 minimum shared speed bins to score at all |
| `motion_min_std` | 0.05 | Tier-3 minimum speed variation (body-heights/s); below it the cow barely moved and r is noise |
| `motion_max_lag_bins` | 2 | Tier-3 tolerated clock offset between cameras, in bins |
| `single_cow_sanity_floor` | 0.40 | Tier-4 minimum cosine score even with only one candidate on each side |
| `cosine_threshold` | 0.75 | Not a separate key: `cosine_resolver.threshold`, the same value the Cosine Resolver uses (before the config migration this was a separate 0.75 that ignored reconcile's flag) |

---

### `display_tracks.py`
- Browser-based visualizer — replaces X11/SDL2/ffplay/Tkinter; no display server required (WSL-compatible)
- Streams annotated video as MJPEG via Flask; sensor charts rendered server-side as PNG via matplotlib
- Opens `http://localhost:5000` automatically in the Windows browser (via `cmd.exe /c start`)
- Pulls canonical DB from Drive via `drive_manager` at startup; checks `db` dirty flag
- Labels each raw temp_id with its manual assignment, else its id from the last reconcile run (`session_assignments`)
- Runs live posture + facing classification per frame if `vision_features/` is present (optional import)
- **`draw_masks` (cli.yaml; QA/visualization only):** overlays segmentation mask polygons as a translucent fill + outline, drawn beneath the bbox/label pills so text stays legible. Reads `outputs/<session_id>/qa_masks/masks_part*.parquet` — the file `track_and_dump.py` produces with `save_masks: true`. No-op with a log warning if that file doesn't exist for the session (re-run `track_and_dump.py` with `save_masks: true` first). Toggleable live in-browser ("🎭 Mask" button), same generic toggle mechanism as Bbox/Pose/Direction — flips `STATE.show_mask` via `POST /api/control {action: "toggle_mask"}`.
- Browser controls: pause / fast-forward (1×/2×/4×/8×) / quit; sensor chart auto-refreshes every 3 s; feature toggles: Bbox, ID, Keypoints, Pose, Direction, Mask
- CLI: `--video`, `--session_id` (override), `--bypass_upload_check`, `--params`. All overlay options (`draw_pose`, `draw_masks`, `kp_index`, `kp_conf_thresh`, `hide_occluded`, `port`, `start`, `limit`, …) are in `cli.yaml` → `display_tracks:`. Its `kp_conf_thresh` is a display filter only, independent of `params pose.kp_conf_thresh`.
- Requires: `pip install flask`

---

### `kinetic_match.py` — the one Kinetic Matcher implementation
- Kinetic matching: bbox centroid displacement ↔ collar ΔKineticsCountR
- Stable temp_ids only (≥ `min_temp_id_frames` of the session's frames); Pearson r per pair over the animal's active `timeline.bin_minutes` bins (≥ `max(min_active_bins, 2)` of them); greedy one-to-one assignment by r above `corr_threshold`
- `match(tracks, kinetics)` → `KineticMatchResult` (assignment, per-temp_id r, full score table, stable temp_ids, bins, warnings). No logging — callers decide what to print.
- Used by reconcile.py's Kinetic Matcher step and by `match_identity.py`, so both give identical results for the same data and params

### `match_identity.py`
- Thin standalone CLI over `kinetic_match.match()` for exported CSVs: prints the correlation matrix, score table and assignments, writes `tracks.csv` + an `animal_id` column
- Does not use drive_manager — never touches the DB; reads CSVs passed directly as arguments
- `run(tracks_path, kinetics_path)` → (assignment, scores) for library use
- Standalone CLI: `python3 match_identity.py --tracks tracks.csv --kinetics ... --output ... [--scores_json ...] [--params ...]`

---

### `assign_identity.py`
- Manual identity assignment tool — use when kinetics data is unavailable or doesn't cover the video window
- Pulls canonical DB from Drive via `drive_manager` at startup; checks `db` dirty flag
- Syncs DB back to Drive after writing assignments
- Writes assignments to a `manual_assignments` table in SQLite; reconcile.py merges these in Manual Merge, with manual overriding kinetic on conflict
- Warns when a temp_id already has a different id from the last reconcile run (any non-manual source, read from `session_assignments`)
- Can optionally call `reconcile.py` directly after writing (`--run_reconcile`), with the thresholds from `params.yaml` (or `--params`) — no threshold flags of its own
- Accepts `--bypass_upload_check` to proceed when DB flag is dirty
- Key operations:

```bash
# Assign temp_ids manually after watching display_tracks.py
python3 assign_identity.py --session ... \
    --assign 2:7507  1:6366  71:7513 \
    --note "manual — no kinetics coverage for Dec 21" \
    --run_reconcile

# List current assignments for a session
python3 assign_identity.py --session ... --list

# Remove a specific temp_id assignment
python3 assign_identity.py --session ... --remove 71
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

| Name | Description | Legacy ID |
|---|---|---|
| **Collar Candidate Selector** | Given `session.start_dt` / `session.end_dt`, filter to AnimalIds with ΔKineticsCountR above threshold. Cows not in the pen are automatically excluded — no manual roster needed. | *(was Step 0)* |
| **Tracklet Stitching** | Join a cow's track fragments across short tracker dropouts (`tracklet_stitch.py`), by position, box size and timing only — no appearance, so detector-independent. A link A → B needs B to start after A's last frame (never on screen together) within `max_gap_s`, A's end and B's start (median box centre over `end_window_s`) within `max_dist + dist_per_s × gap` box sizes, and similar box sizes. Greedy one-to-one by distance → chains; ambiguous links (a rival within `ambiguity_margin`) and links joining two different manual assignments are skipped. Each chain is renamed to its first temp_id; links are recorded in `tracklet_stitches`. Raw temp_ids read back from storage are renamed the same way (embeds, the extractor via `merge_map`, cross_camera's motion tier via the table, and `session_assignments` has a row for every raw fragment so display_tracks shows the id). Calibrated by cutting gaps into ~700 real long tracks with all other tracks left in place: 90% of ≤3 s and 83% of 5–8 s dropouts rejoined, 1.4–3.4% wrong links. Velocity extrapolation was tested and gave no gain. On June 16 it joined 716 fragments on cam26 (3,032 → 2,316 temp_ids) and 101 on cam33_6. | — |
| **Noise Filter** | Drop noise-level temp_ids before any identity step (after stitching, so fragments get joined first): lasting < `noise_filter.min_duration_s` or with < `min_detections` rows. Manually assigned temp_ids are exempt. Dropped temp_ids leave tracks, embeds, and the extractor's pose galleries (TempPoseGallery would otherwise turn each into a match target for later sessions); rows stay in `raw_tracks`. On the June 16 sessions this removed 14–68% of temp_ids but 0.4–6% of detections. | — |
| **Kinetic Matcher** | bbox centroid displacement ↔ ΔR · Pearson r · greedy one-to-one → temp_id ↔ AnimalId (`kinetic_match.py`, shared with `match_identity.py`) | *(was Step A)* |
| **Manual Merge** | Load `manual_assignments` from DB; manual overrides kinetic on conflict | *(was Step A.5)* |
| **Appearance Quality** | Weight per detection for every step that pools embeddings (`appearance_quality.py`): 0 below `min_px` box size (sqrt(w·h)), rising linearly to 1 at `full_px`; × `edge_weight` when the box touches the frame border (frame size from `video_sessions.frame_w/frame_h`; skipped when unknown). Weight-0 frames are dropped from the embeddings; every `min_embeds` / `min_slot_frames` threshold counts effective frames (sum of weights). A track's mean embedding is also outlier-trimmed (`appearance_quality.robust_mean`): frames whose cosine to the preliminary mean falls more than `trim_mad_k` robust SDs (1.4826 × MAD) below the median are dropped, at most `max_trim_frac` of the effective frames, only for tracks with ≥ `min_frames_for_trim`. Never used for positions or kinetics. | — |
| **Gallery preview** | In memory only: stored flat galleries plus this session's manual/kinetic ids blended in, so the Cosine Resolver can match a fragment to a cow confirmed earlier in the same session. Nothing saved. | *(was Step B)* |
| **Cosine Resolver** | 3-tier query per unresolved temp_id: (1) flat real_id gallery, (2) pose-conditioned real_id gallery, (3) TempPoseGallery — cross-session, **cross-camera** temp matching (cows move freely between cameras, so this is not restricted to one). If a match to a real_id is found → assign it. If a match to another temp_id is found → inherit the id `session_assignments` has for it, or mint a new synthetic id (the other session's temp_id is recorded as a link row). A temp_id's embeds are split by camera mode and each mode's mean is only compared with that mode's galleries; real-id steps (1–2) run in every mode the temp_id has ≥ `min_embeds` in (most embeds first) before any temp-gallery match (3). Its matches reach the galleries in the Gallery commit (below), in every mode with enough frames. Backpropagates: if a synthetic id gets resolved to a real_id, all prior timeline rows are updated. Falls back to kinetics-only if all galleries are empty. | *(was Step C)* |
| **Boundary Rendezvous** | Cows crossing between cameras 26/33/38 are linked via known shared-zone polygons (`camera_topology` in `cross_camera.py`). A session's track dwelling in a shared zone past a dwell threshold is staged in `boundary_observations`; a matching observation from the zone's other camera triggers an immediate cross-camera resolve, scored by a tiered hierarchy — both-sides-kinetic → cosine → motion-magnitude correlation → single-cow-in-zone sanity check — mirroring reconcile's own kinetic-over-appearance trust order, with two new, lower-priority tiers for cases reconcile never had to handle before. Order-agnostic: works regardless of which camera's session is processed first, or how long apart. Matches on the flat embedding, not the pose-conditioned one, so it resolves in time for Sensor Sequencer to include the result. | *(was Step C.5)* |
| **Duplicate Resolver** | When several temp_ids claim the same AnimalId, rank them by the source of their id (manual > kinetic > boundary_rendezvous > cosine), then frame count. Fragments never on screen together in different spots are the same animal split by the tracker: merged into the winner, embed rows remapped. A temp_id on screen at the same time as the winner in a different spot is a different animal: it loses the id and stays unassigned. A run replaces the session's `temp_id_merges` rows from earlier runs. | *(was Step A.6)* |
| **Gallery commit** | The run's **single** flat-gallery update, once every id is final (merged fragments pooled under their winner). Per temp_id and camera mode: robust, quality-weighted mean → recorded as this session's row in the gallery contribution ledger (`gallery_ledger.py`, table `gallery_contributions`); the cow's vector is rebuilt as the weighted mean of all its active rows (see `reid_registry`). The session's rows from earlier runs are replaced as a whole, so a re-run applies nothing twice and a cow it no longer contributes to is rebuilt without it. A contribution whose cosine with the cow's current vector is below `gallery.contribution.drift_gate_cosine` is held for review instead. Synthetic ids and source weight 0 never contribute. A cosine id contributes only if it clears the margin gate (`gallery.cosine_gate`): cosine to the cow ≥ `cosine_resolver.threshold` + `threshold_margin`, and ≥ every other cow + `second_best_margin`; the pose slots follow that decision. The id itself is assigned either way. Each contributing track is also checked for a two-group split of its frames (likely tracker ID switch) and flagged in `id_switch_flags` — review only, nothing is changed. Writes reid_registry. | — |
| **Identity record** | Writes `session_assignments` (identity_record.py): one row per raw temp_id of the session with its final id and source, plus link rows for other sessions' temp_ids tied to a synthetic id this run. Replaces everything the session wrote before. Raises each cow's `reid_registry.match_method` to the best source seen (never lowers it). | — |
| **Sensor Sequencer** | Δf_12/f_23/v/kinR → forward-fill to video time grid | *(was Step D)* |
| **Vision Feature Extractor** | Load kps from Parquet → classify posture + facing per frame → aggregate to window scalars → write to timeline → update pose-conditioned gallery | *(was Step B-vision)* |
| **Timeline Writer** | Insert rows into `resolved_cow_timeline` | *(was Step E)* |

**Output written:** `resolved_cow_timeline` — central join table

**Key rule:** Kinetic matching is always the primary confirmation signal. Cosine resolver is secondary (heals switches, handles cross-video). Both run post-hoc, never during inference.

### Stage 3 — Model (dataset builder + CNN-LSTM/GRU) *(planned)*
**Inputs:** resolved_cow_timeline + calving_ledger + cow_registry  
**Process:** Build labeled multi-hour windows per calving event → train temporal model  
**Output:** 4-class probability distribution + risk score + prediction horizon

---

## Databases (14 total)

### `cow_registry` — static lookup
- real_id (PK), breed, parity, pen_id, collar_id, baseline_window
- Used for: sensor normalization baseline, static feature injection

### `video_sessions` — one row per video file
- session_id (PK), video_path, camera_id, start_dt, end_dt, collar_csv_path
- **is_night** (bool) — majority camera mode over the video, written by track_and_dump.py. **Summary only** — reconcile routes on the per-row `raw_tracks.is_night`.
- **night_frame_fraction** (REAL) — fraction of frames in grayscale/IR mode (0 < x < 1 ⇒ the camera switched during the video)
- **frame_w, frame_h** (INTEGER) — video frame size in px, written by track_and_dump.py; used by the appearance-quality edge check (NULL for sessions tracked before it existed → check skipped)

### `collar_signals` — raw sensor time-series
- AnimalId (FK → cow_registry), datetime, signal_type ('behavior'|'kinetic')
- f_1_2, f_2_3, v (behavior) + kin_X, kin_Y, kin_Z, kin_R (kinetics)
- Ingested separately from video pipeline

### `raw_tracks` — append-only inference output
- session_id (FK → video_sessions), frame_index, frame_datetime, temp_id
- bbox (x1/y1/x2/y2), cx, cy, det_conf
- embed (JSON), kps (JSON flat list[57]), kps_norm (JSON), kps_conf (REAL), kps_kconf (JSON)
- **is_night** (INTEGER) — camera mode at this frame: 1 = grayscale/IR, 0 = colour, NULL = tracked before per-frame detection existed (reconcile then applies the legacy clock fallback and warns). Added to older DBs automatically by `init_db()` in both scripts.
- **Note:** `embed_parquet_row` and `kps_parquet_row` are integer pointers into the session Parquet files — the canonical array store. JSON columns are a convenience fallback.

### `manual_assignments` — operator overrides
- session_id, temp_id, real_id, note, created_dt
- Written by `assign_identity.py`; consumed by reconcile.py's Manual Merge step
- Manual assignments override kinetic matches on conflict

### `reid_registry` — one row per confirmed real identity
- real_id (PK = AnimalId from collar)
- **gallery_embed_day[128]** — flat gallery vector from daytime frames (RGB), rebuilt from `gallery_contributions`; fallback vector
- **gallery_embed_night[128]** — same for night/IR frames (grayscale)
- gallery_n_day, gallery_n_night — active ledger contributions behind each vector
- gallery_conf_day, gallery_conf_night — quality score
- last_updated_day_dt, last_updated_night_dt
- known_temp_ids — **deprecated, no longer written**; replaced by `session_assignments`. Copied into it once (as `legacy` rows) when that table is first created
- first_seen_dt
- match_method — the **best** source that has ever identified the cow, by `identity.source_priority`: `manual` | `kinetic` | `boundary_rendezvous` | `cosine`; raised, never lowered. `synthetic` for synthetic ids; rows from before may hold `cosine_day` / `cosine_night` (rank as cosine)
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

**Update rule — contribution ledger (per slot, and for flat; real ids):**

Every session's contribution to a (cow, modality, flat-or-slot) key is stored as its own row in `gallery_contributions`; the gallery vector is recomputed from those rows on every change, never accumulated as a running EMA:
```
weight(row) = source_weight × 0.5^(age_days / half_life_days) × n_effective_frames^frames_weight_power
vector      = L2( Σ weight × row.vector / Σ weight )     over active rows, weight ≥ min_weight
```
- Old contributions fade by elapsed days (`half_life_days` 180); stronger evidence counts more (source weight, frames^0.5). Exponential decay keeps two rows' weight ratio constant, so a stored vector stays correct between commits.
- **Re-runs replace, never stack:** a run sets aside all of its session's earlier rows for that gallery, records its new ones, and rebuilds any cow it no longer contributes to (re-assigned, below `min_embeds`, slot lost). `--dry_run` previews exactly what a real run would do.
- **Drift gate:** each row stores `drift` = cosine(cow's vector from the other rows, this row). Below `drift_gate_cosine` the row is `held_for_review` and doesn't count until approved. The gate is below `cosine_resolver.threshold`, so it guards kinetic / manual / boundary ids that contradict the cow's appearance — a cosine id has already matched this gallery.
- **Exact removal:** `python3 gallery_ledger.py reject ID` (or `--session S --real_id R` for one session's whole contribution to one cow) takes it out and rebuilds; a human approve/reject survives re-runs of that session. `remove` deletes the row, but the next re-run writes it again.
- **Pre-ledger galleries:** the first commit to a key whose gallery holds a vector but has no rows seeds that vector as one `legacy` row (`legacy_seed_frames` 200), so deploying the ledger didn't reset every cow to one session.
- Synthetic ids and TempPoseGallery are scratch space and keep the plain EMA (`gallery.ema_alpha`).
- Routing is **per frame** (`raw_tracks.is_night`): day frames update day galleries only, night frames night galleries only. A session that spans the camera's colour↔IR switch updates both — that is how the night gallery learns cows first confirmed in daylight (and vice versa)
- Galleries **never** cross-contaminate across modalities
- reconcile uploads **both** modalities' gallery files after every run (a mixed session, a backprop fold or a temp-gallery purge can touch either)
- One flat commit per run, after every identity step (Gallery commit), and one pose commit in the Vision Feature Extractor; `gallery.source_weights[source]` — manual/kinetic 1.0, boundary_rendezvous 0.5 (inferred across cameras), cosine 0.5 (self-referential — weight conservatively); synthetic ids never
- Session means are appearance-quality-weighted and outlier-trimmed (`appearance_quality.py`); `min_embeds` counts effective frames

### `resolved_cow_timeline` — central join table
- reconcile **replaces** a session's rows on every run (delete + insert in one transaction) — re-runs such as `assign_identity.py --run_reconcile` no longer append a second copy
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

### `boundary_observations` — cross-camera rendezvous staging
- id (PK), camera_id, zone_pair_id, session_id, temp_id
- enter_time, exit_time, frames_in_zone, flat_embedding (JSON)
- resolved_animal_id (nullable), resolved_confidence (nullable)
- status: `pending` | `matched` | `no_match_found`
- match_kind, match_animal_id, partner_obs_id — the pairing's outcome, recorded on **both** rows when matched
- Re-running a session deletes only its own `pending` rows (nobody used them) and stages afresh. A crossing already `matched` or `no_match_found` is not re-staged — the partner side is no longer pending — and a matched one's recorded outcome is re-applied to this session's assignment (manual/kinetic ids still win). So the session that ran *first* also picks up the cross-camera id when re-run. Rows matched before outcomes were recorded can't be re-applied (logged).
- Written and consumed entirely by `cross_camera.py` (Boundary Rendezvous) — a row is a candidate crossing waiting for its counterpart from the zone's other camera, however long that takes
- A `pending` row is only ever finalized as `no_match_found` once the partner camera has an actually-processed session fully covering its time range with nothing matching — never by a timeout

### `cross_camera_conflicts` — cross-camera identity disputes
- id (PK), real_id, session_a, session_b, camera_a, camera_b
- overlap_window_start, overlap_window_end, zone_check_passed
- confidence_a, confidence_b, status: `resolved_by_confidence` | `needs_review`
- Written by `cross_camera.py` only when two sessions on different cameras independently claim the same real_id for a pairing that isn't zone-consistent (a genuine conflict, not ordinary corroboration)
- `needs_review` rows need a human — the confidence gap was too close to call automatically

### `gallery_contributions` — gallery contribution ledger
- One row per (session_id, real_id, modality, gallery `'flat'|'pose'`, slot_idx) — slot_idx −1 for flat, 0–7 for pose slots; UNIQUE on those five
- source (`manual` | `kinetic` | `boundary_rendezvous` | `cosine` | `legacy`), source_weight (snapshot), n_effective_frames, vector (JSON list[128]), drift, created_dt
- status: `active` | `held_for_review` | `rejected`; reviewed_dt set by a human decision (carried over to re-runs)
- session_id `__legacy__` = a pre-ledger gallery vector seeded on first use
- Written by reconcile.py (flat) and the Vision Feature Extractor (pose) via `gallery_ledger.SessionCommit`; reviewed with the `gallery_ledger.py` CLI

### `id_switch_flags` — likely tracker ID switches (review queue)
- session_id, temp_id, real_id, modality, n_effective_frames, trim_frac, split_gap, split_cluster_frac, created_dt, reviewed; UNIQUE(session_id, temp_id, modality)
- A track's (outlier-trimmed) frames split into two groups by spherical 2-means; flagged when the smaller group holds ≥ `id_switch.min_cluster_frac` of the frames and (within-group − between-group cosine) ≥ `min_gap`
- Never acts on anything; a re-run replaces the session's flags. Only tracks that reach the Gallery commit are checked

### `tracklet_stitches` — tracklet stitching log
- One row per stitched fragment: session_id, loser_tid (raw temp_id that was renamed), root_tid (the chain's first temp_id, which it now carries), prev_tid (fragment it was linked to directly), gap_s, dist (box sizes), size_ratio, created_dt
- UNIQUE(session_id, loser_tid); reconcile replaces a session's rows on every run
- Read by cross_camera.py's motion tier to load every fragment of a track

### `session_assignments` — which id each temp_id got (identity_record.py)
- session_id, temp_id (**raw**, as in raw_tracks), final_temp_id (after stitching + Duplicate Resolver; NULL on link rows), real_id (AnimalId or negative synthetic id), source (`manual` | `kinetic` | `boundary_rendezvous` | `cosine` | `legacy`), written_by, created_dt; PK (session_id, temp_id, written_by)
- One row per raw temp_id of an identified track, so every fragment of a stitched or merged track has one
- **Link rows:** when the Cosine Resolver mints a synthetic id for a track of this session and an unresolved track of another session, the other session's temp_id is recorded with session_id = that session, written_by = this one
- A reconcile run replaces every row it wrote before (written_by = its session) — own rows and links — so nothing goes stale. A lookup prefers the session's own row, else the newest link
- Read by display_tracks.py, assign_identity.py and the Cosine Resolver's temp-gallery step; synthetic → real backprop rewrites real_id here
- Created on first use and backfilled from `reid_registry.known_temp_ids`; a (session, temp_id) that known_temp_ids gave two different ids is skipped and its session named in the log — re-run it

### `temp_id_merges` — duplicate-claim log
- session_id, winner_tid, loser_tid, real_id, winner_reason, loser_reason, merged_dt, action
- Written by reconcile.py's Duplicate Resolver step when two temp_ids are assigned the same AnimalId
- winner_reason / loser_reason: the **source** of each temp_id's id — `'manual'` | `'kinetic'` | `'boundary_rendezvous'` | `'cosine'`
- action:
  - `'merged'` — same animal split by the tracker; `loser_tid` embed/track rows are remapped to `winner_tid`
  - `'unassigned_cooccurring'` — on screen at the same time as the winner in a different spot, so a different animal; `loser_tid` lost the id and stays unassigned (its rows are not pooled)
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

# 1a. Single video — inference + reconcile (collar CSVs auto-discovered;
#     models + thresholds from config/params.yaml, QA output from config/cli.yaml)
python3 track_and_dump.py \
    --source ~/thesis_workspace/raw_data/calving/refet_33_S20241221070000_E20241221080000.ts

# 1b. Batch — all videos in a directory (models loaded once, runs sequentially)
python3 track_and_dump.py --source ~/thesis_workspace/raw_data/calving/

# 2. Validate visually (opens http://localhost:5000 in browser;
#    overlays: config/cli.yaml → display_tracks:)
python3 display_tracks.py \
    --video ~/thesis_workspace/raw_data/calving/refet_33_S20241221070000_E20241221080000.ts

# 3. If kinetics unavailable — assign manually after watching display_tracks.py
python3 assign_identity.py \
    --session cam33_6_S20260616100001_E20260616110000 \
    --assign  2:7507  1:6366  71:7513 \
    --note    "manual — no kinetics for this date" \
    --run_reconcile

# 4. Re-run reconcile standalone (e.g. after tuning thresholds in params.yaml)
python3 reconcile.py --session cam33_6_S20260616100001_E20260616110000
    # add --dry_run to test without writing
    # add --params config/params_strict.yaml to try other thresholds

# 5. Ingest calving ledger (once per farm file — idempotent)
python3 calving_ledger_ingest.py \
    --file  ~/thesis_workspace/raw_data/gazit_calving_events_202505_202605.xlsx \
    --farm  gazit
    # add --dry_run to validate without writing
    # add --overwrite to re-ingest an already-loaded file

# 6. Validate config after editing
python3 config.py

# 6b. Review gallery contributions held by the drift gate, and ID-switch flags
python3 gallery_ledger.py list-pending                     # [--real_id 7507]
python3 gallery_ledger.py approve 42                        # or reject 42 / remove 42
python3 gallery_ledger.py reject --session cam33_6_S20260616100001_E20260616110000 --real_id 7507
python3 gallery_ledger.py show 7507 --modality day --gallery flat   # [--gallery pose --slot 3]
python3 gallery_ledger.py list-id-switch-flags              # [--all]
python3 gallery_ledger.py review-id-switch-flag 5

# 7. Check sync status / recover from failures
python3 drive_manager.py status
python3 drive_manager.py retry-buffer
```

---

## Key Design Decisions

1. **No ReID during inference** — temp_ids saved raw; all identity resolution is post-hoc in reconcile.py
2. **Wall-clock as universal join key** — frame_datetime links video to collar data; filename `_S<timestamp>` is the source
3. **Arrays in Parquet, never SQLite blobs** — embed[128], kps[19×3], kps_kconf[19] live in per-session Parquet files; SQLite holds integer row pointers
4. **real_id is nullable** — pipeline doesn't block on unresolved identities; modality_mask signals quality
5. **Two-stage identity resolution** — kinetic matching (primary) confirms AnimalId; cosine resolver (secondary) heals temp_id switches and enables cross-video, cross-camera continuity. Cosine resolver uses a 3-tier query: flat gallery → pose-conditioned gallery → TempPoseGallery (cross-session, cross-camera temp matching)
6. **Manual assignment as first-class fallback** — `assign_identity.py` writes to `manual_assignments`; reconcile merges these with kinetic results, manual taking priority
7. **Duplicate Resolver ranks by provenance** — every step records which source set each temp_id's id (`provenance` in `reconcile.run()`), and duplicate claims are ranked manual > kinetic > boundary_rendezvous > cosine, the same trust order as the rest of the pipeline. Ranking only changes the outcome when the claimants were on screen together in different spots (a real conflict — the loser is unassigned); otherwise they're one animal and get merged whichever wins, with the loser's embed rows remapped before gallery rebuild. Uses the same co-occurrence rule as the Cosine Resolver (frame tolerance + IoU ≥ 0.3 means a handoff, not a conflict).
8. **Pose raw data stays in raw_tracks / Parquet** — reconcile.py extracts scalar features into resolved_cow_timeline; raw kps recomputable any time
9. **Sensor temporal mismatch handled by forward-fill** — behavior ~90s, kinetics ~15min → upsampled into video time grid in resolved_cow_timeline
10. **kps_coverage column** — tells the model how reliable vision features are per window (partial occlusion awareness)
11. **display_tracks.py is the integration testbed** — use it to visually validate identity assignments before committing to feature extraction
12. **Dual day/night gallery, two layers per cow** — each cow has a flat fallback vector (`gallery_embed_{day|night}` in SQLite) and up to 8 pose-conditioned vectors (`gallery_pose_{day|night}.npy`). The 8-slot gallery is the primary query target; the flat vector is the Level 3 fallback when all pose slots are empty. Modalities never cross-contaminate. Modality is the camera's own mode per frame (see 25), not the session.
13. **Pluggable vision feature extractor** — `vision_features/` is a self-contained package; adding a new feature means one new file in `features/` and two lines in `extractor.py`; no other files change
14. **Pose-conditioned 8-slot gallery** — 8 slots per cow per modality = {standing, lying} × {left, right, toward, away}. Only high-confidence classified frames contribute to each slot. Cosine resolver queries exact slot first, then degrades gracefully to posture-level then flat. Galleries build up progressively — early sessions use flat fallback; discrimination improves automatically as slots populate.
15. **Synthetic IDs for unresolved cows** — when a temp_id matches another temp_id cross-session but no real_id is known yet, reconcile.py mints a negative synthetic id (from `synthetic_id_counter.npy`). The synthetic id propagates consistently until a future kinetic match resolves it to a real AnimalId, at which point all prior timeline rows are backpropagated automatically.
16. **Drive as single source of truth** — all structured data lives on Google Drive (`thesis_google_drive:vcf_ctp_data`). Local disk is a write buffer only. drive_manager.py enforces this via dirty flags that block read scripts from running against stale Drive data.
17. **Collar CSVs auto-discovered by time window** — reconcile.py and track_and_dump.py call `load_collar_data()` which lists Drive's `collar_data/`, parses `s/e` timestamps from canonical filenames, and merges all overlapping files. No `--kinetics` argument needed in normal operation.
18. **Batch processing with single model load** — track_and_dump.py accepts a directory as `--source`. Models and DB are loaded once; `process_video()` is called per file. A `processing_log.csv` on Drive records every video processed, its outcome, and performance metrics.
19. **Pluggable calving ledger adapters** — each farm's ledger format is parsed by a dedicated `BaseAdapter` subclass. Farm-specific column names, date formats, language of outcome values, and derivation rules are fully isolated in the adapter. The orchestrator (`calving_ledger_ingest.py`) only calls `parse()`. Adding a new farm = one new file in `adapters/`, zero changes elsewhere.
20. **Calving features split from ground truth** — `calving_ledger` holds only the label (outcome + identity + datetime). `calving_features` holds all model input features keyed on `event_id`. The two tables evolve independently: new farms can introduce new feature columns via auto-migration without touching existing rows or code.
21. **Boundary rendezvous instead of a batch cross-camera pass** — processing is strictly serial but not time-ordered (a batch might process camera 33's whole day before camera 26 even starts, and a camera's feed can be entirely missing for a period). Rather than waiting for all three cameras to finish a shared time window, `cross_camera.py` stages qualifying boundary-zone tracks in `boundary_observations` and triggers a match the moment a partner observation shows up, however much later that is. A pending observation is only finalized as unmatched once the partner camera has an *actually-processed* session that fully covers it and still found nothing — proof of absence, not a timeout guess.
22. **No "keep the better view" rule for same-slot boundary overlap** — considered and dropped. The idea: when two cameras see the same cow in the same posture/facing slot at the same moment, keep only the higher-quality view instead of blending both into the gallery. It doesn't fit how the gallery is built. `build_slot_embeds` averages every confident frame of a cow in a slot across the **whole session**, and `PoseGallery.update()` runs once per (cow, slot) per session — so the shared-zone moment is only a few seconds inside an average of minutes to hours of frames. Skipping the "worse" update would discard that camera's entire session of frames for the slot, not just the overlap. A frame-level version (drop only the overlapping frames) is possible, but serial processing means it only works one way: the later session can hold back its own frames, and while the contribution ledger (26) can now remove an earlier session's contribution exactly, it does so per slot per session, not per overlapping frame. The effect being corrected — a few seconds counted twice — is too small to justify that.
23. **session_id = filename stem, camera name swapped for an ASCII code** — built in exactly one place, `make_session_id()` in `track_and_dump.py` (which `display_tracks.py` imports). `רפת 33_6_S20260616100001_E20260616110000.mp4` → `cam33_6_S20260616100001_E20260616110000`. One id per file, and the same id every time that file is processed, so re-running a video replaces its own rows and two files can never collide. ASCII with no spaces, so `--session $SES` works without quoting and nothing displays right-to-left. Camera names and codes live in `params.yaml` → `cameras` (one list, shared with reconcile and cross_camera). An unrecognized camera still gets a unique id — unsafe characters are replaced and a short hash of the original name is appended — plus a warning to add it to the map. Replaced the old `<camera_id>_<start time>` format, which dropped the segment number, needed shell quoting, and fell back to `<camera>_unknown` for any filename without a timestamp — giving every such video from one camera the same id, so processing the second deleted the first one's tracks. session_id is an opaque key: nothing parses it, use `video_sessions.camera_id` / `start_dt` for camera and time.
24. **Two config files, strict, no code defaults** — `params.yaml` (what is computed) and `cli.yaml` (how it runs / QA output), loaded by `config.py` into frozen typed objects; unknown or missing keys are errors. One value per concept: e.g. a single `cosine_resolver.threshold` used by both the Cosine Resolver and Boundary Rendezvous, a single `cameras` list. Structural constants that stored data depends on stay fixed in `schema.py`. Safety switches (`--dry_run`, `--bypass_upload_check`) stay one-shot flags so they can't be left on in a file. Each session keeps a copy of the params it ran with, and `processing_log.csv` records the hash.
25. **Day/night = the camera's mode, per frame, from the pixels** — replaced the clock rule (`session.night_*`), which labelled a whole session at once and drifts with the season (sunset moves by ~3 h between June and December). The cameras switch to grayscale themselves, so the pixels are ground truth: track_and_dump.py labels every frame (colour-pixel fraction + debounce) and stores it per row in `raw_tracks.is_night`. Everything downstream routes per row — Gallery Builder, Cosine Resolver, the pose-gallery update, backprop — so a video that switches mid-way is split correctly instead of being shoved into one modality. Per-row rather than per-session also means a dusk-crossing track whose identity is confirmed in daylight populates that cow's night gallery for free. A re-run purges the session's entries from **both** temp galleries first, so a session whose labels changed doesn't leave stale entries in the other modality's file. The clock rule survives only as a fallback for rows tracked before this existed.
26. **Galleries are rebuilt from a contribution ledger, not accumulated by EMA** — the EMA could neither be re-run safely (each run applied its update again) nor corrected (a wrong match stayed blended in forever). Each session's contribution is a row in `gallery_contributions`; the vector is a decayed, evidence-weighted mean of the rows, recomputed on change. A re-run replaces all of its session's rows (including contributions it no longer makes), a per-cow drift gate holds sudden jumps for review, human decisions survive re-runs, and removal is exact. Scoped to real ids: synthetic ids and the TempPoseGallery are short-lived scratch space and keep the EMA.
27. **Robust track mean + ID-switch flag, review only** — a track's mean embedding drops outlier frames (robust z-score on cosine to the mean, capped) everywhere embeddings are pooled. At the Gallery commit the same frames are tested for a clean two-group split, the signature of a tracker ID switch, and flagged in `id_switch_flags`. Flagged tracks are not split or unassigned automatically: the test can't yet tell a switch from one cow whose two sides look different, so a human decides.
28. **Identity record per session, replaced on every run** — `known_temp_ids` was an append-only JSON list per cow, so a re-run that changed a temp_id's id left the old one behind, and readers got whichever entry they scanned first. `session_assignments` is a table keyed by (session, raw temp_id, writer) that each run rewrites wholesale, including its links into other sessions. `match_method` is deliberately the opposite — a monotonic summary ("best evidence ever seen"); the current truth for any session is in the table.
29. **Cosine ids are gated before they teach the gallery, not before they are assigned** — an appearance match that is barely above the threshold, or nearly as close to another cow, would otherwise feed the gallery that produced it and pull it toward the wrong cow. The gate sits at the Gallery commit, so it covers flat, pose-slot and temp-gallery-inherited matches alike, and it scores against each cow's vector without this session's earlier runs (a re-run can't vouch for itself). Kept as a gate rather than `source_weights.cosine: 0` so clear cosine matches still enrich galleries across sessions; setting the weight to 0 remains a config-only switch.

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
  models/                                   # local model weights (params.yaml models:)
  scripts/
    config.py                               # loader for config/*.yaml — import in all scripts
    config/
      params.yaml                           # algorithm / pipeline settings
      cli.yaml                              # run behaviour, QA output, storage
      bytetrack.yaml                        # ultralytics tracker configs
      botsort.yaml
    drive_manager.py                        # Drive I/O layer — import in all scripts
    reconcile.py
    kinetic_match.py                        # Kinetic Matcher — shared by reconcile.py and match_identity.py
    tracklet_stitch.py                      # Tracklet Stitching — called by reconcile.py
    appearance_quality.py                   # per-frame appearance weight + robust mean / ID-switch check
    gallery_ledger.py                       # gallery contribution ledger + review CLI — reconcile, extractor
    identity_record.py                      # session_assignments + match_method — reconcile, display_tracks, assign_identity
    cross_camera.py                         # Boundary Rendezvous — called by reconcile.py, not standalone
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
      crops/                                # optional — local only if crops_local
      qa_masks/                             # optional — only if save_masks; QA/visualization
        masks_part*.parquet                 #   only, NEVER uploaded to Drive, read by display_tracks.py draw_masks
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
    params_track.yaml                       # params used by track_and_dump.py
    params_reconcile.yaml                   # params used by the latest reconcile run
    crops/                                  # uploaded unless crops_local
    # qa_masks/ is intentionally NEVER here — local-only QA artifact, see track_and_dump.py
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

---

## ToDo

- [x] resolve session id logic (video name, time, etc...) — see Key Design Decision 23 / `make_session_id`
- [x] change day-night logic to grayscale vs rgb — per-frame camera-mode detection in `track_and_dump.py`, stored in `raw_tracks.is_night`, routed per row everywhere downstream; see Key Design Decision 25
- [x] Gallery Builder runs twice when the Duplicate Resolver merges anything — replaced by a single Gallery commit per run with per-source weights (`gallery.source_weights`). Also: the timeline, `temp_id_merges` and pending `boundary_observations` are now replaced on re-run instead of duplicated, and boundary match outcomes are recorded so a re-run re-applies them.
  - [x] Contribution ledger: re-running a session no longer re-applies its flat- or pose-gallery update — `gallery_ledger.py`, see Key Design Decision 26.
- [x] Per-frame appearance quality weight — `appearance_quality.py`; box size ramp + frame-edge penalty, effective-frame thresholds, weighted pose slots and boundary embeddings. `video_sessions.frame_w/frame_h` added (new sessions only).
  - [ ] Calibrate `appearance_quality.min_px` / `full_px` once embeddings are available: per-frame agreement with its track's mean vs box size, per camera. 24 / 96 are provisional (June 16 boxes: median ~40 px).
- [x] ~~The Cosine Resolver's soft update overwrites `reid_registry.match_method`~~ — `match_method` is now upgrade-only (best source ever seen), and `assign_identity.py` reads `session_assignments` instead of it (Key Design Decision 28)
- [x] Identity record — `session_assignments` replaces append-only `known_temp_ids`; display_tracks.py, the Cosine Resolver's temp step and assign_identity.py read it (also fixes assign_identity's `→ None`: it never selected `real_id`)
- [x] Gate cosine matches before they update a gallery — `gallery.cosine_gate` (Key Design Decision 29)
  - [ ] Calibrate `threshold_margin` / `second_best_margin`: on kinetically confirmed tracks, compare the margin a cosine query would have had for right vs. wrong cows
- [x] Carry a provenance tag through Manual Merge→Gallery Builder→Duplicate Resolver — `provenance` dict + `identity.source_priority` in `params.yaml`
- [x] drop camera_id restriction in TempPoseGallery
- [x] Delete stale Drive-side parquet parts before re-uploading on reprocess — `track_and_dump.py` removes leftover `embeds_part*`/`kps_part*` parts and crops from an earlier run of the same video after uploading (`dm.remove_stale_remote_files`). Readers only ever use the local `outputs/<session_id>/`, which is wiped before each run, so this keeps the Drive copy accurate rather than fixing a double-count.
- [x] Cross-camera identity tracking (continuation of the point above) — see `cross_camera.py` / Boundary Rendezvous
  - [x] Implement tier-3 motion-magnitude correlation — `make_motion_corr_fn` / `motion_magnitude_corr` in `cross_camera.py`. On synthetic tracks (same cow seen at different scales and opposite directions, 1 px jitter): same cow r≈0.998, different cows ≤0.48; still matches with up to ~1.5 s clock offset, fails at 10 s. `motion_corr_threshold` not yet checked against real crossings.
- [x] Reset the tracker between videos in batch mode.
- [x] Collapse match_identity.py and reconcile's inline correlation logic into one implementation — `kinetic_match.py`; `score_up_to()` and `kinetic_match.live_min_active_bins` removed (no callers). Old and new verified identical on cam26's real tracks (assignments, r, all 325 scored pairs).
- [x] Move vision-feature constants into the config file, including CLI plumbing — `config/params.yaml` + `config/cli.yaml`, see Configuration
- [ ] Move from strict hierarchical override to a fused confidence score once ≥3–4 matching signals exist, keeping manual as a hard override.
- [x] Short-gap tracklet stitching — `tracklet_stitch.py`, runs before the noise filter and the Kinetic Matcher. Purely geometric; velocity extrapolation was tested and dropped (no gain).
- [x] Minimum-occurrence-count filter — reconcile's Noise Filter step (`noise_filter:` in params.yaml). Box size was tested as an extra criterion and dropped: short tracks aren't smaller than long ones on the June 16 data, so size belongs in a per-frame appearance weight instead.
- [x] Config file for constants/thresholds — done in one pass, not incrementally (all scripts)
- [ ] Configurable ID-matching hierarchy (e.g. manual > kinetic = cosine > ocr) with tie-break rules.
- [x] Gallery contribution ledger, robust to drift — `gallery_ledger.py` / `gallery_contributions`: per-session rows, day-based decay, evidence weighting, per-cow drift gate with review, exact removal, re-run safe for flat and pose galleries (Key Design Decision 26)
  - [ ] Calibrate `gallery.contribution.drift_gate_cosine` on the stored `drift` of kinetic contributions; consider a per-cow adaptive threshold (from each cow's own drift history) instead of one global value
- [x] Robust session mean + ID-switch flag — `appearance_quality.robust_mean` / `id_switch`, flags in `id_switch_flags` (Key Design Decision 27)
  - [ ] Require the two groups to be split in time (one before, one after a switch point) — one cow whose left and right sides embed differently currently gives interleaved groups and a false flag. Calibrate `id_switch.min_gap` on real switches
  - [ ] Check unresolved tracks too (only tracks that reach the Gallery commit are checked)
- [ ] Persist per-session cosine drift and populate embed_mean — drift now stored per gallery contribution (`gallery_contributions.drift`); timeline `embed_mean` still pending
- [ ] Synthetic-id backpropagation in the Cosine Resolver looks unreachable: it only runs for a temp_id that has a synthetic id AND a kinetic/manual one, but the Cosine Resolver never queries kinetic/manual temp_ids, so that combination can't occur. Verify and fix (pre-existing)
---
- [ ] retrain pose estimator (open question: masked crops vs. raw crops as input — masked-crop training would be an out-of-distribution shift for the current pose model; not yet decided)
- [ ] Cross-check a Boundary Rendezvous pairing against independent identity claims already on record elsewhere (non-adjacent-camera conflict case) — explicitly deferred, not yet designed
- [ ] decide what to do with clumping — **blocked on evaluation**, see below
---
- [x] **convert yolo detector with yolo segmentor**
  - [x] Labeled ~858 images via Roboflow (autolabeling / Smart Polygon, SAM-assisted), pre-split train/valid/test by Roboflow export
  - [x] Trained YOLOv8n-seg locally via Ultralytics (not Roboflow-hosted training — avoids version-pinning/export-format uncertainty; see training notes). Validation mask mAP50 ≈ 0.93, mAP50-95 ≈ 0.75 on the general val split
  - [x] Integrated into `track_and_dump.py` — segmentation masks build background-removed crops for the **embedder only** (Problem A: background contamination); pose input and saved crops unaffected
  - [x] QA/visualization path added: `save_masks` (track_and_dump.py, local-only, never on Drive) + `draw_masks` (display_tracks.py, toggleable overlay)
  - [ ] **No dedicated held-out clumped/touching-cow eval set was built** (explicit decision, not an oversight) — the standard random `test` split (~86 images) is the only current signal on clumping performance, and it may under-represent clumped frames since they're rare in the data. **Clumping-fix quality is therefore still unverified** — this is the open item blocking "decide what to do with clumping" above
  - [ ] Tracker mask-IoU — still box-IoU only (ByteTrack/BoT-SORT default); only worth escalating if ID-switch rate is still bad after the segmentation swap, per original plan
  - [ ] Coat-pattern fingerprinting
  - [ ] fp16/explicit-device optimization?
  - [ ] **Collar data — handle overlapping files on upload** (`drive_manager.py` → `_cmd_upload_collar`):
  - [ ] When two CSV files cover the same animals and overlapping time windows (e.g. an export was re-run with a wider date range), the current upload creates separate files on Drive with different names but partially duplicate data. Need to detect overlap on upload and decide a strategy — options include: merge rows and re-export as a single file with the combined time range, skip files whose entire window is already covered by an existing Drive file, or flag for manual review and skip. Overlapping files are identified by matching animal ID sets and intersecting `[s_timestamp, e_timestamp]` ranges parsed from the canonical filename.