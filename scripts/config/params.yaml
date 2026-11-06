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
class KineticMatch:
    corr_threshold: float
    min_active_bins: int
    live_min_active_bins: int
    min_temp_id_frames: float
    activity_pct: float


@dataclass(frozen=True)
class Gallery:
    ema_alpha: float
    min_embeds: int
    synthetic_merge_alpha: float


@dataclass(frozen=True)
class CosineResolver:
    threshold: float
    min_embeds: int


@dataclass(frozen=True)
class Cooccurrence:
    iou_same_animal: float
    frame_tolerance_min: int
    gap_multiplier: float


@dataclass(frozen=True)
class Identity:
    source_priority: list
    protected_sources: list


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