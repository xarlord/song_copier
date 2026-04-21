"""RVC voice conversion wrapper."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


class RVCWrapper(BaseModelWrapper):
    """Wraps rvc_python for voice conversion."""

    def __init__(
        self,
        device: str = "cuda",
        cache_dir: str | Path = "cache",
        f0method: str = "rmvpe",
        f0up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.33,
    ):
        super().__init__("rvc", device, cache_dir)
        self.f0method = f0method
        self.f0up_key = f0up_key
        self.index_rate = index_rate
        self.protect = protect
        self._voice_model_path = None

    def load_model(self) -> None:
        if self._loaded:
            return
        from rvc_python.infer import RVCInference

        logger.info("Initializing RVC inference engine")
        self.model = RVCInference(device=self.device if self.device != "cpu" else "cpu:0")
        self._loaded = True
        logger.info("RVC engine ready")

    def unload_model(self) -> None:
        if not self._loaded:
            return
        del self.model
        self.model = None
        self._loaded = False
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("RVC unloaded")

    def get_vram_requirement_mb(self) -> int:
        return 1500

    def load_voice_model(self, model_path: str) -> None:
        """Load a voice model (.pth file) for conversion."""
        if not self._loaded:
            self.load_model()
        logger.info(f"Loading voice model: {model_path}")
        self.model.load_model(model_path)
        self._voice_model_path = model_path

    def convert(
        self,
        input_path: str,
        output_path: str,
        f0method: str | None = None,
        f0up_key: int | None = None,
        index_rate: float | None = None,
        protect: float | None = None,
    ) -> str:
        """Convert vocals using loaded voice model.

        Returns path to converted audio.
        """
        self.model.infer_file(
            input_path,
            output_path,
            f0method=f0method or self.f0method,
            f0up_key=f0up_key if f0up_key is not None else self.f0up_key,
            index_rate=index_rate if index_rate is not None else self.index_rate,
            protect=protect if protect is not None else self.protect,
        )
        logger.info(f"Voice conversion complete: {output_path}")
        return output_path


def list_voice_models(models_dir: str | Path) -> list[dict]:
    """List available .pth voice models in the models directory."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for p in models_dir.glob("**/*.pth"):
        models.append({"name": p.stem, "path": str(p)})
    return models