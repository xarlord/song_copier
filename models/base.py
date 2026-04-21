"""Abstract base class for model wrappers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseModelWrapper(ABC):
    def __init__(self, model_name: str, device: str = "cuda", cache_dir: str | Path = "cache"):
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.model = None
        self._loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Download (if needed) and load model weights."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Release model from GPU memory."""
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def get_vram_requirement_mb(self) -> int:
        """Estimated VRAM needed in MB."""
        ...

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, *args):
        self.unload_model()