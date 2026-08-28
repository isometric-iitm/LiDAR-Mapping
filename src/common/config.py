import yaml
from pathlib import Path
from typing import Any

_CONFIG_CACHE: dict[str, Any] = {}

def load_config(name: str) -> dict:
    if name in _CONFIG_CACHE:
        return _CONFIG_CACHE[name]
    config_dir = Path(__file__).resolve().parent.parent.parent / "config"
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _CONFIG_CACHE[name] = cfg
    return cfg

def merge_configs(*names: str) -> dict:
    result = {}
    for name in names:
        result.update(load_config(name))
    return result
