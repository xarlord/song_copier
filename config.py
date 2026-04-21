"""Global configuration loader."""

import os
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).parent.resolve()


def load_config(config_path: str | None = None) -> dict:
    """Load config from YAML, falling back to defaults."""
    path = Path(config_path) if config_path else ROOT_DIR / "config" / "default.yaml"
    if path.exists():
        with open(path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Resolve paths relative to project root
    for key in ("cache", "output", "temp", "rvc_models"):
        if key in cfg.get("paths", {}):
            p = Path(cfg["paths"][key])
            if not p.is_absolute():
                cfg["paths"][key] = ROOT_DIR / p

    # Ensure output directories exist
    for key in ("cache", "output", "temp"):
        dir_key = key if key != "rvc_models" else key
        dir_path = cfg.get("paths", {}).get(dir_key if key != "rvc_models" else "rvc_models", ROOT_DIR / key)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return cfg


def get_device(config: dict) -> str:
    """Resolve device string. 'auto' -> 'cuda' if available, else 'cpu'."""
    device = config.get("device", "auto")
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device