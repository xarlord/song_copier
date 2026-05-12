"""Plugin manager — discover, validate, load, and run third-party audio processing plugins."""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------
_plugin_cache: dict[str, dict] = {}      # name → manifest dict
_module_cache: dict[str, object] = {}    # name → imported module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plugin_base_dir(plugin_dir: str = "plugins") -> Path:
    """Return absolute path to the plugin root directory."""
    # Resolve relative to the project root (where app.py lives)
    base = Path(plugin_dir)
    if not base.is_absolute():
        base = Path(_project_root()) / base
    return base


def _project_root() -> Path:
    """Best-effort project root (directory containing app.py)."""
    # Walk up from this file until we find app.py
    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        if (parent / "app.py").exists():
            return parent
    # Fallback: two levels up from services/
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_plugins(plugin_dir: str = "plugins") -> list[dict]:
    """Scan *plugin_dir* for valid plugin directories.

    Each plugin lives in a sub-directory that contains at minimum:
      - ``plugin.json``  (manifest)
      - ``__init__.py``  (Python module with ``process_audio`` & ``get_info``)

    Returns a list of manifest dicts (one per discovered plugin).
    """
    _plugin_cache.clear()

    base = _plugin_base_dir(plugin_dir)
    if not base.exists():
        logger.info("Plugin directory %s does not exist — creating it.", base)
        base.mkdir(parents=True, exist_ok=True)
        return []

    found: list[dict] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name.startswith("."):
            continue
        manifest_path = child / "plugin.json"
        init_path = child / "__init__.py"
        if not manifest_path.exists() or not init_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s — bad plugin.json: %s", child.name, exc)
            continue
        # Ensure required fields
        for field in ("name", "version"):
            if field not in manifest:
                logger.warning("Skipping %s — missing '%s' in plugin.json", child.name, field)
                manifest = None
                break
        if manifest is None:
            continue
        manifest["_dir"] = str(child)
        _plugin_cache[manifest["name"]] = manifest
        found.append(manifest)

    logger.info("Discovered %d plugin(s) in %s", len(found), base)
    return found


def validate_plugin(plugin_dir: str) -> bool:
    """Return *True* if *plugin_dir* is a valid plugin.

    A valid plugin directory must contain:
      1. ``plugin.json`` with at least ``name`` and ``version`` keys.
      2. ``__init__.py`` defining ``process_audio(audio, sr, **kwargs)`` and
         ``get_info()``.
    """
    p = Path(plugin_dir)
    if not p.is_dir():
        return False

    # Check manifest
    manifest_path = p / "plugin.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if "name" not in manifest or "version" not in manifest:
        return False

    # Check module
    init_path = p / "__init__.py"
    if not init_path.exists():
        return False

    # Attempt import to verify required callables exist
    try:
        module = _import_plugin_module(p)
    except Exception:
        return False

    if not hasattr(module, "process_audio"):
        return False
    if not hasattr(module, "get_info"):
        return False
    return True


def load_plugin(plugin_name: str) -> object:
    """Import and return the plugin module for *plugin_name*.

    Raises ``KeyError`` if the plugin has not been discovered yet.
    """
    if plugin_name in _module_cache:
        return _module_cache[plugin_name]

    if plugin_name not in _plugin_cache:
        raise KeyError(
            f"Plugin '{plugin_name}' not found. Run discover_plugins() first."
        )

    plugin_dir = Path(_plugin_cache[plugin_name]["_dir"])
    module = _import_plugin_module(plugin_dir)
    _module_cache[plugin_name] = module
    return module


def apply_plugin(
    plugin_name: str,
    audio: np.ndarray,
    sr: int,
    **kwargs: Any,
) -> np.ndarray:
    """Load and run *plugin_name*'s ``process_audio`` on the given audio.

    Returns the processed audio as a numpy array.
    """
    module = load_plugin(plugin_name)
    result = module.process_audio(audio, sr, **kwargs)

    if not isinstance(result, np.ndarray):
        raise TypeError(
            f"Plugin '{plugin_name}' process_audio must return np.ndarray, "
            f"got {type(result).__name__}"
        )
    return result


def list_available_plugins() -> list[dict]:
    """Return a list of all discovered plugin info dicts.

    Automatically runs ``discover_plugins()`` on the first call if the cache
    is empty.
    """
    if not _plugin_cache:
        discover_plugins()
    return list(_plugin_cache.values())


def get_plugin_info(plugin_name: str) -> dict | None:
    """Return the manifest dict for a single plugin, or *None*."""
    if not _plugin_cache:
        discover_plugins()
    return _plugin_cache.get(plugin_name)


# ---------------------------------------------------------------------------
# Internal import machinery
# ---------------------------------------------------------------------------

def _import_plugin_module(plugin_dir: Path) -> object:
    """Dynamically import a plugin directory as a Python module."""
    plugin_dir = plugin_dir.resolve()
    dir_name = plugin_dir.name
    module_name = f"_audio_gen_plugin_{dir_name}"

    # Add the parent (plugins/) to sys.path so relative imports work
    parent = str(plugin_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # If already loaded, reload to pick up changes during development
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])

    # Ensure we import the directory package
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(plugin_dir / "__init__.py"),
        submodule_search_locations=[str(plugin_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {plugin_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
