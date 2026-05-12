"""Preset management for stem volumes and effects."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PRESETS_DIR = Path(__file__).resolve().parent.parent / "config" / "presets"


def _ensure_presets_dir() -> Path:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    return PRESETS_DIR


def list_presets() -> list[dict]:
    """List all available presets (built-in + custom).

    Returns:
        List of dicts with keys: name, description, custom, path
    """
    _ensure_presets_dir()
    presets = []
    for f in sorted(PRESETS_DIR.glob("*.yaml")):
        try:
            with open(f) as fh:
                data = yaml.safe_load(fh)
            presets.append({
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "custom": data.get("custom", True),
                "path": str(f),
            })
        except Exception as e:
            logger.warning(f"Skipping broken preset {f}: {e}")
    return presets


def load_preset(name: str) -> dict | None:
    """Load a preset by name.

    Returns:
        Preset dict with keys: name, description, custom, stems
        or None if not found.
    """
    _ensure_presets_dir()
    preset_path = PRESETS_DIR / f"{name}.yaml"
    if not preset_path.exists():
        logger.warning(f"Preset not found: {name}")
        return None
    try:
        with open(preset_path) as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load preset {name}: {e}")
        return None


def save_preset(name: str, description: str, stems: dict) -> str:
    """Save a custom preset.

    Args:
        name: Preset name (used as filename)
        description: Human-readable description
        stems: dict mapping stem name to {volume, effects: {...}}

    Returns:
        Path to saved preset file.
    """
    _ensure_presets_dir()
    # Sanitize name for filename
    safe_name = "".join(c for c in name if c.isalnum() or c in " -_").strip()
    if not safe_name:
        safe_name = "custom_preset"

    data = {
        "name": name,
        "description": description,
        "custom": True,
        "stems": stems,
    }
    path = PRESETS_DIR / f"{safe_name}.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"Saved preset: {safe_name}")
    return str(path)


def delete_preset(name: str) -> bool:
    """Delete a custom preset. Built-in presets cannot be deleted.

    Returns:
        True if deleted, False if not found or is built-in.
    """
    data = load_preset(name)
    if data is None:
        return False
    if not data.get("custom", True):
        logger.warning(f"Cannot delete built-in preset: {name}")
        return False
    safe_name = "".join(c for c in name if c.isalnum() or c in " -_").strip()
    if not safe_name:
        safe_name = "custom_preset"
    path = PRESETS_DIR / f"{safe_name}.yaml"
    path.unlink(missing_ok=True)
    logger.info(f"Deleted preset: {name}")
    return True
