"""GPU detection, VRAM monitoring, and model lifecycle management."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_active_models: dict[str, object] = {}


def get_device() -> str:
    """Get best available device string."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {name}")
            return "cuda"
    except ImportError:
        pass
    logger.warning("No GPU detected, using CPU — inference will be slow")
    return "cpu"


def get_vram_info() -> dict:
    """Get current VRAM usage in MB."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "device": "cpu"}
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024 ** 2)
        used = torch.cuda.memory_allocated(0) / (1024 ** 2)
        return {
            "total_mb": round(total, 1),
            "used_mb": round(used, 1),
            "free_mb": round(total - used, 1),
            "device": torch.cuda.get_device_name(0),
        }
    except ImportError:
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "device": "none"}


def register_model(name: str, model: object) -> None:
    _active_models[name] = model


def unregister_model(name: str) -> None:
    _active_models.pop(name, None)


def unload_all() -> None:
    """Unload all registered models and free GPU memory."""
    import gc
    for name in list(_active_models.keys()):
        model = _active_models.pop(name)
        if hasattr(model, "unload_model"):
            model.unload_model()
        del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    except ImportError:
        pass


def can_fit_vram(required_mb: int) -> bool:
    """Check if there's enough VRAM for a model."""
    info = get_vram_info()
    if info["device"] == "cpu":
        return True  # CPU mode, no VRAM limit
    return info["free_mb"] >= required_mb * 1.1  # 10% safety margin