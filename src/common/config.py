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
from pathlib import Path

import yaml
from dotenv import load_dotenv

_CONFIG_CACHE: dict[str, dict] = {}


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
