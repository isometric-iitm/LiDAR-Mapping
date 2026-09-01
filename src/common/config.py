"""Config loading with env-first overrides and pc2d/ repo-root path resolution.

Precedence (highest wins):
    1. environment variable (set in the shell or in pc2d/.env via python-dotenv)
    2. value from the committed config YAML
    3. absolute paths are used as-is; relative paths resolve against the pc2d/
       repo root (the folder three levels above this package), NOT the process
       CWD — so commands can be run from any directory.

Committed config YAMLs carry relative defaults. Local .env holds machine-specific
absolute paths (e.g. F:/sih/...). See .env.example for the full contract.
"""
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

_CONFIG_CACHE: dict[str, dict] = {}


@dataclass(frozen=True)
class GridParams:
    """Ring/sector geometry for the log-polar grid (config/grid.yaml -> grid:)."""
    r_min: float = 0.5
    r_max: float = 100.0
    dr_0: float = 0.05
    r_transition: float = 10.0
    alpha: float = 1.004994
    n_theta: int = 720
    z_min: float = -5.0
    z_max: float = 10.0
    n_classes: int = 4
    occupancy_gain: float = 1.0
    occ_threshold: float = 0.2

    @classmethod
    def from_dict(cls, g: dict) -> "GridParams":
        return cls(
            r_min=g["r_min"],
            r_max=g["r_max"],
            dr_0=g["dr_0"],
            r_transition=g.get("r_transition", g["r_max"]),
            alpha=g["alpha"],
            n_theta=int(g["n_theta"]),
            z_min=g["z_min"],
            z_max=g["z_max"],
            n_classes=int(g["n_classes"]),
            occupancy_gain=g.get("occupancy_gain", 1.0),
            occ_threshold=g.get("occ_threshold", 0.2),
        )


@dataclass(frozen=True)
class TraversabilityParams:
    """Drivability scoring weights/thresholds (config/grid.yaml -> traversability:)."""
    enabled: bool = True
    weights: tuple = (0.25, 0.25, 0.35, 0.15)
    z_diff_thresh: float = 0.5
    slope_thresh: float = 0.4
    class_scores: tuple = (1.0, 0.6, 0.2, 0.1)

    @classmethod
    def from_dict(cls, t: dict) -> "TraversabilityParams":
        return cls(
            enabled=t.get("enabled", True),
            weights=tuple(t.get("weights", (0.25, 0.25, 0.35, 0.15))),
            z_diff_thresh=t.get("z_diff_thresh", 0.5),
            slope_thresh=t.get("slope_thresh", 0.4),
            class_scores=tuple(t.get("class_scores", (1.0, 0.6, 0.2, 0.1))),
        )


@dataclass(frozen=True)
class MemoryParams:
    """Uniform-grid equivalence used for the compression report (config/grid.yaml -> memory:)."""
    uniform_cell_guess: float = 200.0
    uniform_cell_size: float = 0.05

    @classmethod
    def from_dict(cls, m: dict) -> "MemoryParams":
        return cls(
            uniform_cell_guess=m.get("uniform_cell_guess", 200.0),
            uniform_cell_size=m.get("uniform_cell_size", 0.05),
        )


@dataclass(frozen=True)
class GridConfig:
    """Typed view of config/grid.yaml (grid/traversability/memory sections)."""
    grid: GridParams = field(default_factory=GridParams)
    traversability: TraversabilityParams = field(default_factory=TraversabilityParams)
    memory: MemoryParams = field(default_factory=MemoryParams)

    @classmethod
    def from_dict(cls, cfg: dict) -> "GridConfig":
        return cls(
            grid=GridParams.from_dict(cfg["grid"]),
            traversability=TraversabilityParams.from_dict(cfg.get("traversability", {})),
            memory=MemoryParams.from_dict(cfg.get("memory", {})),
        )


def as_grid_config(cfg: dict | GridConfig) -> GridConfig:
    """Accept a raw grid.yaml dict or an already-typed GridConfig."""
    return cfg if isinstance(cfg, GridConfig) else GridConfig.from_dict(cfg)


def repo_root() -> Path:
    """Absolute path to the pc2d/ repo root (= parent.parent.parent of this file)."""
    return Path(__file__).resolve().parent.parent.parent


def _load_env() -> None:
    """Parse pc2d/.env into os.environ if present (never overrides an existing var)."""
    load_dotenv(repo_root() / ".env", override=False)


def resolve_path(value: str | Path) -> Path:
    """Return an absolute Path; relative inputs resolve against the repo root."""
    p = Path(value)
    if p.is_absolute():
        return p
    return repo_root() / p


# env var -> list of dotted config paths it can override. Both ckpt dirs map to
# the same var because pipeline.yaml and train_range_image.yaml name it differently.
_ENV_TARGETS: dict[str, list[list[str]]] = {
    "PC2D_SEQ_DIR": [["source", "seq_dir"]],
    "PC2D_CHECKPOINT": [["model", "checkpoint"]],
    "PC2D_CKPT_DIR": [["server", "ckpt_dir"], ["checkpoint", "dir"]],
    "PC2D_RAW_ROOT": [["data", "raw_root"]],
    "PC2D_PROCESSED_ROOT": [["data", "processed_root"]],
    "PC2D_DEVICE": [["model", "device"]],
    "PC2D_PRECISION": [["model", "precision"]],
    "PC2D_PLAYBACK_SPEED": [["source", "playback_speed"]],
}

# Which sections each env var can touch, used to restrict overrides to the YAML
# actually loaded (pipeline vs train). Mapping env -> top-level section names.
_ENV_SECTIONS: dict[str, set[str]] = {
    "PC2D_SEQ_DIR": {"source"},
    "PC2D_CHECKPOINT": {"model"},
    "PC2D_CKPT_DIR": {"server", "checkpoint"},
    "PC2D_RAW_ROOT": {"data"},
    "PC2D_PROCESSED_ROOT": {"data"},
    "PC2D_DEVICE": {"model"},
    "PC2D_PRECISION": {"model"},
    "PC2D_PLAYBACK_SPEED": {"source"},
}

# env var -> cast applied to the string value before it overrides the YAML.
_ENV_CASTS: dict[str, callable] = {
    "PC2D_PLAYBACK_SPEED": float,
}


def _safe_set(cfg: dict, path: list[str], value) -> None:
    """Set cfg[*path] = value, tolerating missing intermediate sections."""
    node = cfg
    for i, k in enumerate(path):
        if i == len(path) - 1:
            node[k] = value
            return
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            return
        node = nxt


def apply_env_overrides(cfg: dict, sections: set[str] | None = None) -> dict:
    """Overlay env vars (top priority) onto a config dict in place.

    ``sections`` (optional) restricts which env vars apply based on the top-level
    config sections they target (e.g. ``{"server","checkpoint"}`` for the training
    YAML). If None, every registered env var that targets an existing section is
    applied — safe because each target is guarded by ``_safe_set``.
    """
    _load_env()
    for env, targets in _ENV_TARGETS.items():
        value = os.getenv(env)
        if value is None or value == "":
            continue
        if sections is not None and not _ENV_SECTIONS[env].intersection(sections):
            continue
        cast = _ENV_CASTS.get(env)
        try:
            val = cast(value) if cast is not None else value
        except (TypeError, ValueError):
            val = value
        for t in targets:
            _safe_set(cfg, t, val)
    return cfg


def load_config(name: str) -> dict:
    """Load a config YAML by name (no extension), env overrides applied & cached."""
    if name in _CONFIG_CACHE:
        return _CONFIG_CACHE[name]
    config_dir = repo_root() / "config"
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    _CONFIG_CACHE[name] = apply_env_overrides(cfg)
    return _CONFIG_CACHE[name]


def reset_cache() -> None:
    _CONFIG_CACHE.clear()


def merge_configs(*names: str) -> dict:
    result = {}
    for name in names:
        result.update(load_config(name))
    return result


def resolve_device(requested: str | None = None) -> str:
    """Normalize a `model.device` value (config/env/CLI) into 'cuda' or 'cpu'.

    ``auto`` -> 'cuda' if a GPU is available else 'cpu'. An explicit 'cuda'
    on a machine without a GPU falls back to CPU with a warning rather than
    crashing. Invalid values warn and fall back to 'auto' behavior.
    """
    import torch

    val = (requested or "auto").strip().lower()
    if val == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        warnings.warn("device=cuda requested but CUDA is unavailable; falling back to cpu")
        return "cpu"
    if val == "cpu":
        return "cpu"
    if val != "auto":
        warnings.warn(f"unknown device '{requested}' (expected auto|cuda|cpu); using auto")
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_precision(requested: str | None = None) -> str:
    """Normalize a `model.precision` value (config/env/CLI) into 'fp32'|'fp16'."""
    val = (requested or "auto").strip().lower()
    if val in ("fp16", "16", "half"):
        return "fp16"
    if val in ("fp32", "32", "float32"):
        return "fp32"
    return "fp16"
