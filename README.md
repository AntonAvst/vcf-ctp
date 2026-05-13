# vcf-ctp
**V**ision–**C**ollar **F**usion for **C**alving-**T**ype **P**rediction in Dairy Cows

# Full Architecture Reference

## Project
**Thesis:** Predictive Modelling of Calving Outcomes in Dairy Cows Using Multi-Modal Sensor and Vision Data  
**Author:** Anton Avstreikh, University of Haifa  
**Advisors:** Prof. Ilan Shimshoni  
**Output:** 4-class calving type prediction — Unassisted · Assisted · Twin · Veterinarian-assisted

---

## Existing Code

### `track_and_dump.py`
- Runs YOLO detection + ByteTrack/BoT-SORT tracking + YOLOv8-Pose (19 kp) + Embedder128 (MobileNetV3 → 128D L2-norm)
- CLI flags: `--save_embed`, `--pose_model`, `--crop_every`, `--min_crop_wh`
- Outputs: `tracks.csv` / `tracks.jsonl` — per-frame rows with temp_id, bbox, det_conf, embed[128], kps[19×3], kps_kconf[19], kps_norm[19×3], frame_datetime
- Parses wall-clock start time from filename token `_S<YYYYMMDDHHmmss>`

### `display_tracks.py`
- Visualization tool: overlays tracks + pose skeleton on video
- Imports `match_identity` module for live kinetic matching (Pearson r + Hungarian)
- Draws score table overlay: temp_id × AnimalId correlation matrix
- Supports ffplay / cv2 / mp4 sinks, TkControls for pause/FF/quit

### `match_identity.py` (not uploaded, but imported)
- Kinetic matching: bbox centroid speed ↔ collar ΔKineticsCountR
- Pearson correlation per 15-min bin, Hungarian assignment
- `score_up_to(tracks_df, kinetics_df, up_to_datetime, bin_minutes, ...)` → assignment dict + scores_df

---

## Sensor Data

### `behavior_data_*.csv`
- Columns: AnimalId, datetime, f_1_2, f_2_3, v
- Interval: ~90 seconds
- Proprietary collar-derived behavioral features

### `kinetic_data_*.csv`
- Columns: AnimalId, datetime, KineticsCountX, KineticsCountY, KineticsCountZ, KineticsCountR
- Interval: ~15 minutes
- Cumulative accelerometer counts; deltas computed for matching/features

---

## Three-Stage Pipeline

### Stage 1 — Inference (track_and_dump.py, per video)
**Inputs:** Raw MP4, YOLO .pt models, cow_registry  
**Process:** Detect → Track (temp_id) → Pose (kps) → Embed (128D)  
**Outputs written:**
- `raw_tracks` — append-only, one row per detection
- `video_sessions` — one row per video file registered

**Key rule:** No identity resolution at inference time. Just save everything. `--save_embed` must be ON.

### Stage 2 — Post-processing (reconcile.py, once per video after Stage 1)
**Steps in order:**
- **A. Kinetic matcher** (existing match_identity.py logic): bbox speed ↔ ΔR · Pearson r · Hungarian → temp_id ↔ AnimalId
- **B. Pose feature extractor** (new): kps[19×3] → scalar biomechanical features per time window
- **C. Gallery builder** (new): group embeds by confirmed AnimalId → EMA mean → gallery.npz
- **D. Cosine resolver** (new): embed vs gallery.npz · sim ≥ threshold → real_id (heals within-video temp_id switches)
- **E. Sensor feature sequencer** (new): Δf_12/f_23/v/kinR → forward-fill to video time grid

**Output written:** `resolved_cow_timeline` — central join table

**Key rule:** Kinetic matching is always the primary confirmation signal. Cosine resolver is secondary (heals switches, handles cross-video). Both run post-hoc, never during inference.

### Stage 3 — Model (dataset builder + CNN-LSTM/GRU)
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
- Written when track_and_dump.py is run

### `collar_signals` — raw sensor time-series
- AnimalId (FK → cow_registry), datetime, signal_type ('behavior'|'kinetic')
- f_1_2, f_2_3, v (behavior) + kin_X, kin_Y, kin_Z, kin_R (kinetics)
- Ingested separately from video pipeline

### `raw_tracks` — append-only inference output
- session_id (FK → video_sessions), frame_datetime, temp_id
- bbox (x1/y1/x2/y2), det_conf
- embed[128]
- **kps[19×3]** — raw pixel coords (NEVER discarded, source of truth for recomputing features)
- **kps_kconf[19]** — per-keypoint confidence
- kps_norm[19×3] — normalised coords

### `reid_registry` — one row per confirmed real identity
- real_id (PK = AnimalId from collar), gallery_embed[128] (EMA mean), gallery_n
- known_temp_ids (list of session+tid pairs), first_seen_dt, match_method ('kinetic'|'cosine')
- Updated by reconcile.py after each video

### `resolved_cow_timeline` — CENTRAL JOIN TABLE
- real_id (FK → reid_registry, nullable), window_start_dt, session_id
- modality_mask (bits: sensor_ok | vision_ok | reid_ok)
- **Sensor cols** (forward-filled): Δf_12, Δf_23, Δv, Δkin_X, Δkin_Y, Δkin_Z, Δkin_R
- **Vision cols (derived from kps by reconcile.py):**
  - spine_angle (kp2→kp3→kp4)
  - pelvic_tilt (kp7 ↔ kp10)
  - tail_elevation (kp5/kp6 vs kp4)
  - limb_symmetry (L/R hock distance ratio)
  - head_drop (kp0/kp1 vs kp2)
  - lying_flag (bbox aspect ratio heuristic)
  - restlessness (variance of spine_angle over window)
  - kps_coverage (mean conf across 19 kp — reliability indicator)
  - embed_mean[128] (mean pool of embed rows in window)
- Raw kps arrays are NOT copied here — they stay in raw_tracks

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

## Key Design Decisions

1. **No ReID during inference** — temp_ids are saved raw; all identity resolution is post-hoc in reconcile.py
2. **Wall-clock as universal join key** — frame_datetime links video to collar data; filename _S<timestamp> is the source
3. **real_id is nullable** — pipeline doesn't block on unresolved identities; modality_mask signals quality
4. **Two-stage identity resolution** — kinetic matching (primary, offline) confirms AnimalId; cosine resolver (secondary) heals temp_id switches and enables cross-video continuity
5. **Pose raw data stays in raw_tracks** — reconcile.py extracts scalar features into resolved_cow_timeline; raw kps recomputable any time
6. **Sensor temporal mismatch handled by forward-fill** — behavior ~90s, kinetics ~15min → upsampled into video time grid in resolved_cow_timeline
7. **kps_coverage column** — tells model how reliable vision features are per window (partial occlusion awareness)
8. **display_tracks.py is the integration testbed** — use it to visually validate identity assignments before committing to feature extraction

---

## Milestone Schedule (from proposal)
- Pose Estimation: March 29, 2026 ✓ (appears done)
- Sensor Pipeline: April 12, 2026
- Re-Identification Module: May 3, 2026
- Vision Feature Extraction: May 24, 2026
- System Integration: June 14, 2026
- Temporal Prediction Model: July 26, 2026
- Empirical Evaluation: August 30, 2026
- Research Paper: September 30, 2026
- Thesis Document: October 30, 2026

---

## Next Open Question (interrupted)
**Training window construction** — how to build labeled multi-hour input sequences per calving event from resolved_cow_timeline. Key challenges: temporal alignment to calving_dt, handling variable cow visibility in video, sensor resolution mismatch, class imbalance for rare outcomes (Twin, Vet).
