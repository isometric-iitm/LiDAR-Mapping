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
}

# Which sections each env var can touch, used to restrict overrides to the YAML
# actually loaded (pipeline vs train). Mapping env -> top-level section names.
_ENV_SECTIONS: dict[str, set[str]] = {
    "PC2D_SEQ_DIR": {"source"},
    "PC2D_CHECKPOINT": {"model"},
    "PC2D_CKPT_DIR": {"server", "checkpoint"},
    "PC2D_RAW_ROOT": {"data"},
    "PC2D_PROCESSED_ROOT": {"data"},
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
        for t in targets:
            _safe_set(cfg, t, value)
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
