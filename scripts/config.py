"""
config.py — single loader for config/params.yaml and config/cli.yaml.

    params.yaml  everything that changes what the pipeline computes
                 (thresholds, model paths, camera topology, ...)
    cli.yaml     how scripts run and what QA output they produce
                 (save_crops, display overlays, storage locations, ...)

Every script imports this module. Nothing in the codebase keeps its own
default for a value that lives here: a missing key, an unknown key (typo) or
a wrong type is an error at load time, never a silent fallback.

Usage
─────
    import config
    config.add_params_arg(ap)         # adds --params to an argparse parser
    args = ap.parse_args()
    config.init(args.params)          # once, in main(), before any work

    P = config.params()               # anywhere afterwards
    P.kinetic_match.corr_threshold
    config.cli().track_and_dump.save_crops

config.params() / config.cli() load the default files lazily if init() was
never called (library use, tests). Values are read at call time, never at
import time, so --params always takes effect even for modules imported before
main() ran.

--params swaps only params.yaml (for threshold sweeps). cli.yaml always comes
from config/cli.yaml next to this file — drive_manager reads its storage
section at import time, before any argument is parsed.

Path rules
──────────
Relative paths in either file resolve against the scripts directory (the
directory holding this file); "~" is expanded. detection.tracker is the one
exception: a bare tracker filename resolves against config/, where
bytetrack.yaml / botsort.yaml live.
"""

from __future__ import annotations

import hashlib
import typing
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Optional

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPTS_DIR / "config"
DEFAULT_PARAMS_PATH = CONFIG_DIR / "params.yaml"
CLI_PATH = CONFIG_DIR / "cli.yaml"


class ConfigError(ValueError):
    pass


# ═════════════════════════════════════════════════════════════════════════════
# params.yaml schema
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Models:
    detector: str
    pose: Optional[str]            # null → run without pose


@dataclass(frozen=True)
class Detection:
    imgsz: int
    conf: float
    iou: float
    tracker: str


@dataclass(frozen=True)
class Pose:
    imgsz: int
    conf: float
    kp_conf_thresh: float


@dataclass(frozen=True)
class Embedder:
    input_size: int
    crop_margin: float


@dataclass(frozen=True)
class Sampling:
    save_every: int
    flush_every: int
    parquet_rows_per_part: int


@dataclass(frozen=True)
class Timeline:
    bin_minutes: int
    embed_mean_min_frames: float   # effective frames a cow needs in a window for embed_mean


@dataclass(frozen=True)
class CameraMode:
    chroma_tol: int
    color_pixel_frac: float
    debounce_seconds: float
    sample_stride: int


@dataclass(frozen=True)
class Session:
    night_start_hour: int
    night_end_hour: int
    night_fraction: float


@dataclass(frozen=True)
class Stitching:
    enabled: bool
    max_gap_s: float
    max_dist: float
    dist_per_s: float
    end_window_s: float
    max_size_ratio: float
    ambiguity_margin: float


@dataclass(frozen=True)
class NoiseFilter:
    min_duration_s: float
    min_detections: int


@dataclass(frozen=True)
class KineticMatch:
    corr_threshold: float
    min_active_bins: int
    min_temp_id_frames: float
    activity_pct: float


@dataclass(frozen=True)
class SourceWeights:
    manual: float
    kinetic: float
    boundary_rendezvous: float
    cosine: float


@dataclass(frozen=True)
class Contribution:
    """Gallery contribution ledger (gallery_ledger.py): galleries are rebuilt
    as a weighted mean over every session's stored contribution, rather than
    an irreversible running EMA. See gallery_ledger.py module docstring."""
    half_life_days: float        # a contribution's weight halves every this many days
    drift_gate_cosine: float     # cosine(current gallery vector, new session vector) below
                                  # this holds the contribution for review instead of
                                  # folding it into the rebuilt gallery
    frames_weight_power: float   # weight *= n_effective_frames ** this (0 = frame count
                                  # doesn't matter beyond the existing min_embeds gate)
    min_weight: float            # a contribution decayed below this is excluded from the
                                  # rebuild sum (numerically faded away, row still kept)
    legacy_seed_frames: float    # n_effective_frames given to a pre-ledger gallery vector
                                  # when it is seeded as a 'legacy' row


@dataclass(frozen=True)
class CosineGate:
    """Which cosine-resolved ids may update a gallery (ToDo item 9). The id
    itself is assigned either way; this only decides gallery updates."""
    enabled: bool                 # false → every cosine id updates (old behaviour)
    threshold_margin: float       # cosine to the cow's vector must reach
                                  # cosine_resolver.threshold + this
    second_best_margin: float     # ... and beat the best OTHER cow by at least this


@dataclass(frozen=True)
class Gallery:
    ema_alpha: float              # still used by TempPoseGallery (unresolved/ephemeral) and
                                   # the synthetic-id merge fold; real_id-keyed flat + pose
                                   # galleries are rebuilt from gallery_contributions instead
    min_embeds: float
    synthetic_merge_alpha: float
    source_weights: SourceWeights
    contribution: Contribution
    cosine_gate: CosineGate


@dataclass(frozen=True)
class RobustMean:
    """Outlier trimming for a track's session-mean embedding (gallery_commit,
    cosine-resolver query vectors, boundary embeddings — anywhere
    appearance_quality.weighted_mean_embed is used)."""
    enabled: bool
    min_frames_for_trim: float    # below this many effective frames, skip trimming entirely
                                   # (not enough points to trust an outlier statistic)
    trim_mad_k: float             # drop frames more than k robust SDs (1.4826 * MAD)
                                   # below the median cosine-to-preliminary-mean
    max_trim_frac: float          # never trim more than this fraction of effective frames


@dataclass(frozen=True)
class IdSwitchCheck:
    """Two-cluster split check on a track's frames — flags a likely tracker
    ID switch (one temp_id secretly covering two animals) for human review.
    Never auto-acts; see gallery_ledger.id_switch_flags."""
    enabled: bool
    min_frames: float             # below this many effective frames, skip the check
    min_cluster_frac: float       # each of the two clusters must hold at least this
                                   # fraction of effective frames (else it's a few outliers,
                                   # not a split — robust_mean trims those separately)
    min_gap: float                # (mean within-cluster cosine) - (between-cluster cosine)
                                   # must exceed this to flag
    max_interleave_frac: float    # in frame order, at most this fraction of the frames may
                                   # sit on the "wrong" side of the best single switch point


@dataclass(frozen=True)
class AppearanceQuality:
    min_px: float
    full_px: float
    edge_margin_px: float
    edge_weight: float
    robust_mean: RobustMean
    id_switch: IdSwitchCheck


@dataclass(frozen=True)
class CosineResolver:
    threshold: float
    min_embeds: float


@dataclass(frozen=True)
class Cooccurrence:
    iou_same_animal: float
    frame_tolerance_min: int
    gap_multiplier: float


@dataclass(frozen=True)
class Identity:
    source_priority: list
    protected_sources: list
    resolve_synthetic_sources: list   # sources trusted to resolve a synthetic id to a real one


@dataclass(frozen=True)
class PostureParams:
    standing_max_ar: float
    lying_min_ar: float
    min_kps_coverage: float
    min_det_conf: float
    ar_conf_weight: float


@dataclass(frozen=True)
class FacingParams:
    min_kp_conf: float
    min_body_axis_px: float
    toward_center: float
    away_center: float
    depth_half_width: float
    lying_conf_penalty: float


@dataclass(frozen=True)
class Vision:
    posture: PostureParams
    facing: FacingParams
    aggregate_min_conf: float
    min_slot_conf: float
    min_slot_frames: float


@dataclass(frozen=True)
class Motion:
    corr_threshold: float
    bin_seconds: float
    window_pad_seconds: float
    min_samples: int
    min_std: float
    max_lag_bins: int


@dataclass(frozen=True)
class CrossCamera:
    boundary_dwell_seconds: float
    boundary_dwell_min_frames: int
    conflict_confidence_gap: float
    single_cow_sanity_floor: float
    motion: Motion


@dataclass(frozen=True)
class Camera:
    id: str
    code: str


@dataclass(frozen=True)
class ZonePair:
    camera_a: str
    camera_b: str
    zone_a_poly: list
    zone_b_poly: list
    direction_relation: str


@dataclass(frozen=True)
class Params:
    models: Models
    detection: Detection
    pose: Pose
    embedder: Embedder
    sampling: Sampling
    camera_mode: CameraMode
    timeline: Timeline
    session: Session
    stitching: Stitching
    noise_filter: NoiseFilter
    appearance_quality: AppearanceQuality
    kinetic_match: KineticMatch
    gallery: Gallery
    cosine_resolver: CosineResolver
    cooccurrence: Cooccurrence
    identity: Identity
    vision: Vision
    cross_camera: CrossCamera
    cameras: list[Camera]
    camera_topology: list[ZonePair]

    # convenience views ------------------------------------------------------
    @property
    def camera_ids(self) -> list[str]:
        return [c.id for c in self.cameras]

    @property
    def camera_codes(self) -> dict[str, str]:
        return {c.id: c.code for c in self.cameras}

    @property
    def source_priority(self) -> dict[str, int]:
        """{source -> rank}, lower wins (manual=0, ...)."""
        return {s: i for i, s in enumerate(self.identity.source_priority)}


# ═════════════════════════════════════════════════════════════════════════════
# cli.yaml schema
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrackAndDumpCli:
    save_crops: bool
    crops_local: bool
    crop_tags: bool
    crop_every: int
    min_crop_wh: list
    save_masks: bool


@dataclass(frozen=True)
class DisplayTracksCli:
    port: int
    start: int
    limit: int
    draw_pose: bool
    draw_masks: bool
    kp_radius: int
    sk_thickness: int
    kp_thresh: float
    hide_occluded: bool
    kp_conf_thresh: float
    show_lowconf: bool
    kp_index: bool
    kp_index_scale: float
    kp_index_thickness: int
    kp_index_offset: int


@dataclass(frozen=True)
class MetricsExportCli:
    out: str
    watch: bool
    pretty: bool


@dataclass(frozen=True)
class StorageCli:
    rclone_remote: str
    drive_folder: str
    local_root: str
    buffer_dir: str
    max_attempts: int
    backoff_base: int


@dataclass(frozen=True)
class Cli:
    track_and_dump: TrackAndDumpCli
    display_tracks: DisplayTracksCli
    metrics_export: MetricsExportCli
    storage: StorageCli


# ═════════════════════════════════════════════════════════════════════════════
# Strict builder
# ═════════════════════════════════════════════════════════════════════════════

def _check_scalar(value, typ, where: str):
    if typ is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if typ is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    if typ is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported type {typ}")


def _build(cls, data, where: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{where or 'top level'}: expected a mapping, "
                          f"got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    missing = [n for n in names if n not in data]
    here = where or "top level"
    if unknown:
        raise ConfigError(f"{here}: unknown key(s) {unknown} — typo, or a key "
                          f"that needs adding to config.py")
    if missing:
        raise ConfigError(f"{here}: missing key(s) {missing}")

    kwargs = {}
    for name in names:
        typ, val, w = hints[name], data[name], (f"{where}.{name}" if where else name)
        origin = typing.get_origin(typ)
        if origin is typing.Union:                     # Optional[X]
            if val is None:
                kwargs[name] = None
                continue
            typ = next(a for a in typing.get_args(typ) if a is not type(None))
            origin = typing.get_origin(typ)
        if is_dataclass(typ):
            kwargs[name] = _build(typ, val, w)
        elif origin is list and is_dataclass(typing.get_args(typ)[0]):
            item = typing.get_args(typ)[0]
            _check_scalar(val, list, w)
            kwargs[name] = [_build(item, v, f"{w}[{i}]") for i, v in enumerate(val)]
        else:
            kwargs[name] = _check_scalar(val, typ, w)
    return cls(**kwargs)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


# ═════════════════════════════════════════════════════════════════════════════
# Cross-field validation
# ═════════════════════════════════════════════════════════════════════════════

def _unit(v: float, where: str) -> None:
    if not 0.0 <= v <= 1.0:
        raise ConfigError(f"{where}: must be in [0, 1], got {v}")


def _validate_params(p: Params) -> None:
    for where, v in [
        ("detection.conf", p.detection.conf), ("detection.iou", p.detection.iou),
        ("pose.conf", p.pose.conf), ("pose.kp_conf_thresh", p.pose.kp_conf_thresh),
        ("kinetic_match.min_temp_id_frames", p.kinetic_match.min_temp_id_frames),
        ("kinetic_match.activity_pct", p.kinetic_match.activity_pct),
        ("gallery.ema_alpha", p.gallery.ema_alpha),
        ("gallery.synthetic_merge_alpha", p.gallery.synthetic_merge_alpha),
        ("session.night_fraction", p.session.night_fraction),
        ("camera_mode.color_pixel_frac", p.camera_mode.color_pixel_frac),
        ("cooccurrence.iou_same_animal", p.cooccurrence.iou_same_animal),
        ("vision.aggregate_min_conf", p.vision.aggregate_min_conf),
        ("vision.min_slot_conf", p.vision.min_slot_conf),
        ("vision.facing.lying_conf_penalty", p.vision.facing.lying_conf_penalty),
        ("vision.posture.ar_conf_weight", p.vision.posture.ar_conf_weight),
    ]:
        _unit(v, where)

    for h, where in [(p.session.night_start_hour, "session.night_start_hour"),
                     (p.session.night_end_hour, "session.night_end_hour")]:
        if not 0 <= h <= 23:
            raise ConfigError(f"{where}: must be 0..23, got {h}")

    sw = p.gallery.source_weights
    for name in ("manual", "kinetic", "boundary_rendezvous", "cosine"):
        _unit(getattr(sw, name), f"gallery.source_weights.{name}")
    if set(p.identity.source_priority) != {"manual", "kinetic", "boundary_rendezvous", "cosine"}:
        raise ConfigError("gallery.source_weights has a weight for exactly manual, kinetic, "
                          "boundary_rendezvous and cosine; identity.source_priority must "
                          f"list the same sources, got {p.identity.source_priority}")

    ct = p.gallery.contribution
    if ct.half_life_days <= 0:
        raise ConfigError(f"gallery.contribution.half_life_days: must be > 0, got {ct.half_life_days}")
    if not -1.0 <= ct.drift_gate_cosine <= 1.0:
        raise ConfigError(f"gallery.contribution.drift_gate_cosine: must be in [-1, 1], "
                          f"got {ct.drift_gate_cosine}")
    if ct.frames_weight_power < 0:
        raise ConfigError(f"gallery.contribution.frames_weight_power: must be >= 0, "
                          f"got {ct.frames_weight_power}")
    if ct.min_weight < 0:
        raise ConfigError(f"gallery.contribution.min_weight: must be >= 0, got {ct.min_weight}")
    if ct.legacy_seed_frames <= 0:
        raise ConfigError(f"gallery.contribution.legacy_seed_frames: must be > 0, "
                          f"got {ct.legacy_seed_frames}")

    cg = p.gallery.cosine_gate
    for name in ("threshold_margin", "second_best_margin"):
        v = getattr(cg, name)
        if not 0.0 <= v <= 1.0:
            raise ConfigError(f"gallery.cosine_gate.{name}: must be in [0, 1], got {v}")
    if p.cosine_resolver.threshold + cg.threshold_margin > 1.0:
        raise ConfigError("gallery.cosine_gate.threshold_margin: cosine_resolver.threshold + "
                          "threshold_margin is above 1 — no cosine id could ever pass")

    aq = p.appearance_quality
    if aq.min_px < 0 or aq.full_px <= aq.min_px:
        raise ConfigError(f"appearance_quality: need 0 <= min_px < full_px, "
                          f"got min_px={aq.min_px}, full_px={aq.full_px}")
    if aq.edge_margin_px < 0:
        raise ConfigError(f"appearance_quality.edge_margin_px: must be >= 0, got {aq.edge_margin_px}")
    _unit(aq.edge_weight, "appearance_quality.edge_weight")

    rm = aq.robust_mean
    if rm.min_frames_for_trim < 2:
        raise ConfigError(f"appearance_quality.robust_mean.min_frames_for_trim: "
                          f"must be >= 2, got {rm.min_frames_for_trim}")
    if rm.trim_mad_k <= 0:
        raise ConfigError(f"appearance_quality.robust_mean.trim_mad_k: must be > 0, "
                          f"got {rm.trim_mad_k}")
    if not 0.0 <= rm.max_trim_frac < 1.0:
        raise ConfigError(f"appearance_quality.robust_mean.max_trim_frac: must be in "
                          f"[0, 1), got {rm.max_trim_frac}")

    isc = aq.id_switch
    if isc.min_frames < 4:
        raise ConfigError(f"appearance_quality.id_switch.min_frames: must be >= 4, "
                          f"got {isc.min_frames}")
    if not 0.0 < isc.min_cluster_frac <= 0.5:
        raise ConfigError(f"appearance_quality.id_switch.min_cluster_frac: must be in "
                          f"(0, 0.5], got {isc.min_cluster_frac}")
    if not 0.0 <= isc.max_interleave_frac <= 0.5:
        raise ConfigError(f"appearance_quality.id_switch.max_interleave_frac: must be in "
                          f"[0, 0.5], got {isc.max_interleave_frac}")
    if p.timeline.embed_mean_min_frames < 0:
        raise ConfigError(f"timeline.embed_mean_min_frames: must be >= 0, "
                          f"got {p.timeline.embed_mean_min_frames}")
    if isc.min_gap < 0:
        raise ConfigError(f"appearance_quality.id_switch.min_gap: must be >= 0, "
                          f"got {isc.min_gap}")
    if p.vision.min_slot_frames < 0:
        raise ConfigError(f"vision.min_slot_frames: must be >= 0, got {p.vision.min_slot_frames}")
    if p.gallery.min_embeds < 0 or p.cosine_resolver.min_embeds < 0:
        raise ConfigError("gallery.min_embeds / cosine_resolver.min_embeds must be >= 0")

    st = p.stitching
    for name in ("max_gap_s", "max_dist", "dist_per_s", "end_window_s",
                 "ambiguity_margin"):
        if getattr(st, name) < 0:
            raise ConfigError(f"stitching.{name}: must be >= 0, got {getattr(st, name)}")
    if st.max_size_ratio < 1:
        raise ConfigError(f"stitching.max_size_ratio: must be >= 1, got {st.max_size_ratio}")

    nf = p.noise_filter
    if nf.min_duration_s < 0:
        raise ConfigError(f"noise_filter.min_duration_s: must be >= 0, got {nf.min_duration_s}")
    if nf.min_detections < 1:
        raise ConfigError(f"noise_filter.min_detections: must be >= 1, got {nf.min_detections}")

    cm = p.camera_mode
    if not 0 <= cm.chroma_tol <= 255:
        raise ConfigError(f"camera_mode.chroma_tol: must be 0..255, got {cm.chroma_tol}")
    if cm.debounce_seconds < 0:
        raise ConfigError(f"camera_mode.debounce_seconds: must be >= 0, got {cm.debounce_seconds}")
    if cm.sample_stride < 1:
        raise ConfigError(f"camera_mode.sample_stride: must be >= 1, got {cm.sample_stride}")

    if p.vision.posture.standing_max_ar >= p.vision.posture.lying_min_ar:
        raise ConfigError("vision.posture: standing_max_ar must be < lying_min_ar "
                          "(the gap between them is the UNCERTAIN band)")

    f = p.vision.facing
    if not 0 < f.depth_half_width < 90:
        raise ConfigError("vision.facing.depth_half_width must be in (0, 90)")

    pri = p.identity.source_priority
    for where, lst in (("identity.source_priority", pri),
                       ("identity.protected_sources", p.identity.protected_sources)):
        if not lst or not all(isinstance(s, str) for s in lst):
            raise ConfigError(f"{where}: must be a non-empty list of source names")
    if len(set(pri)) != len(pri):
        raise ConfigError(f"identity.source_priority has duplicates: {pri}")
    if pri[0] != "manual":
        raise ConfigError("identity.source_priority: 'manual' must stay first "
                          "(manual is a hard override)")
    stray = set(p.identity.protected_sources) - set(pri)
    if stray:
        raise ConfigError(f"identity.protected_sources {sorted(stray)} are not in "
                          f"identity.source_priority")
    stray = set(p.identity.resolve_synthetic_sources) - set(pri)
    if stray:
        raise ConfigError(f"identity.resolve_synthetic_sources {sorted(stray)} are not in "
                          f"identity.source_priority")

    ids = p.camera_ids
    if len(set(ids)) != len(ids):
        raise ConfigError(f"cameras: duplicate id in {ids}")
    codes = [c.code for c in p.cameras]
    if len(set(codes)) != len(codes):
        raise ConfigError(f"cameras: duplicate code in {codes}")
    for c in p.cameras:
        if not c.code.isascii() or not c.code.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(f"cameras: code {c.code!r} must be ASCII letters/digits/_/-")

    for i, z in enumerate(p.camera_topology):
        where = f"camera_topology[{i}]"
        for cam in (z.camera_a, z.camera_b):
            if cam not in ids:
                raise ConfigError(f"{where}: camera {cam!r} is not listed under cameras")
        if z.direction_relation not in ("same", "opposite"):
            raise ConfigError(f"{where}.direction_relation must be 'same' or 'opposite'")
        for name, poly in (("zone_a_poly", z.zone_a_poly), ("zone_b_poly", z.zone_b_poly)):
            if len(poly) < 3 or not all(
                isinstance(pt, list) and len(pt) == 2
                and all(isinstance(c, (int, float)) for c in pt) for pt in poly
            ):
                raise ConfigError(f"{where}.{name}: need >= 3 [x, y] points")


def _validate_cli(c: Cli) -> None:
    t = c.track_and_dump
    if len(t.min_crop_wh) != 2 or not all(isinstance(v, int) for v in t.min_crop_wh):
        raise ConfigError("track_and_dump.min_crop_wh must be [W, H] integers")
    if t.crop_every < 1:
        raise ConfigError("track_and_dump.crop_every must be >= 1")


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

_params: Optional[Params] = None
_params_path: Optional[Path] = None
_cli: Optional[Cli] = None


def resolve_path(p: str) -> Path:
    """Relative → scripts dir; '~' expanded."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else (SCRIPTS_DIR / path)


def load_params(path: Optional[str | Path] = None) -> Params:
    path = Path(path).expanduser().resolve() if path else DEFAULT_PARAMS_PATH
    try:
        p = _build(Params, _read_yaml(path), "")
        _validate_params(p)
    except ConfigError as e:
        raise ConfigError(f"{path.name}: {e}") from None
    return p


def load_cli() -> Cli:
    try:
        c = _build(Cli, _read_yaml(CLI_PATH), "")
        _validate_cli(c)
    except ConfigError as e:
        raise ConfigError(f"{CLI_PATH.name}: {e}") from None
    return c


def init(params_path: Optional[str | Path] = None) -> Params:
    """Load (or reload) params. Call once in main() with args.params."""
    global _params, _params_path
    _params_path = Path(params_path).expanduser().resolve() if params_path else DEFAULT_PARAMS_PATH
    _params = load_params(_params_path)
    if params_path:
        print(f"[config] params: {_params_path}  (override)", flush=True)
    return _params


def params() -> Params:
    if _params is None:
        init()
    return _params


def cli() -> Cli:
    global _cli
    if _cli is None:
        _cli = load_cli()
    return _cli


def params_path() -> Path:
    params()
    return _params_path


def params_snapshot() -> tuple[str, str]:
    """(raw params.yaml text, short sha256) — exactly the file that was loaded,
    comments included, for the per-session reproducibility record."""
    text = params_path().read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def save_params_snapshot(dm, session_id: str, stage: str, dry_run: bool = False) -> str:
    """Write the loaded params.yaml to outputs/<session_id>/params_<stage>.yaml
    (local + Drive) and return its hash. stage is 'track' or 'reconcile':
    they are separate files because reconcile can be re-run with different
    params long after tracking, and both settings matter for a result.
    dm is a DriveManager (duck-typed, so this module never imports it)."""
    text, digest = params_snapshot()
    if dry_run:
        return digest
    local = Path(dm.get_session_dir(session_id)) / f"params_{stage}.yaml"
    local.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# params snapshot — stage={stage} session={session_id}\n"
              f"# source: {params_path()}\n# sha256: {digest}\n")
    local.write_text(header + text, encoding="utf-8")
    dm.write_file(local, f"outputs/{session_id}/{local.name}")
    return digest


def tracker_path() -> str:
    """detection.tracker resolved to a path Ultralytics can open."""
    t = params().detection.tracker
    p = Path(t).expanduser()
    if p.is_absolute():
        return str(p)
    if len(p.parts) == 1 and (CONFIG_DIR / p).exists():
        return str(CONFIG_DIR / p)
    return str(resolve_path(t))


def add_params_arg(ap) -> None:
    ap.add_argument("--params", default=None, metavar="YAML",
                    help=f"Use this params file instead of {DEFAULT_PARAMS_PATH} "
                         f"(e.g. a threshold sweep). cli.yaml is not affected.")


if __name__ == "__main__":
    # `python3 config.py` — validate both files and print a summary.
    import sys
    try:
        p = init(sys.argv[1] if len(sys.argv) > 1 else None)
        c = cli()
    except ConfigError as e:
        print(f"[config] INVALID: {e}")
        sys.exit(1)
    text, digest = params_snapshot()
    print(f"[config] params.yaml OK  ({params_path()}, sha256 {digest})")
    print(f"[config] cli.yaml    OK  ({CLI_PATH})")
    print(f"[config] cameras: {p.camera_ids}")
    print(f"[config] tracker: {tracker_path()}")