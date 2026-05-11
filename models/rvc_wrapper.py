"""RVC voice conversion wrapper — uses subprocess bridge to Python 3.10 venv."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from models.base import BaseModelWrapper

logger = logging.getLogger(__name__)

# Resolve paths relative to project root (where this repo lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RVC_ENV_PYTHON = _PROJECT_ROOT / "rvc_env" / "bin" / "python"
_BRIDGE_SCRIPT = _PROJECT_ROOT / "services" / "rvc_bridge.py"


def _run_bridge(config: dict) -> dict:
    """Run rvc_bridge.py in the Python 3.10 venv, passing config via stdin."""
    if not _RVC_ENV_PYTHON.exists():
        raise RuntimeError(
            f"RVC Python 3.10 venv not found at {_RVC_ENV_PYTHON}. "
            f"Create it with: python3.10 -m venv rvc_env && rvc_env/bin/pip install rvc-python torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )

    proc = subprocess.run(
        [str(_RVC_ENV_PYTHON), str(_BRIDGE_SCRIPT)],
        input=json.dumps(config),
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max for long audio
    )

    if proc.returncode != 0:
        raise RuntimeError(f"RVC bridge failed (exit {proc.returncode}): {proc.stderr}")

    result = json.loads(proc.stdout)
    if not result.get("success"):
        raise RuntimeError(f"RVC bridge error: {result.get('error', 'unknown')}")

    return result


class RVCWrapper(BaseModelWrapper):
    """Wraps rvc_python for voice conversion via subprocess bridge to Python 3.10."""

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
        """Verify the RVC bridge is operational."""
        if self._loaded:
            return

        # Ping the bridge to verify rvc-python is importable
        result = _run_bridge({"action": "ping"})
        logger.info(f"RVC bridge health check: {result.get('message', 'ok')}")

        self._loaded = True
        logger.info("RVC engine ready (via subprocess bridge)")

    def unload_model(self) -> None:
        """No persistent model to unload — subprocess is ephemeral."""
        if not self._loaded:
            return
        self._loaded = False
        logger.info("RVC unloaded")

    def get_vram_requirement_mb(self) -> int:
        return 1500

    def load_voice_model(self, model_path: str) -> None:
        """Store voice model path for conversion (actual load happens in bridge)."""
        if not self._loaded:
            self.load_model()
        logger.info(f"Voice model queued: {model_path}")
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
        """Convert vocals using loaded voice model via subprocess bridge.

        Returns path to converted audio.
        """
        if not self._voice_model_path:
            raise RuntimeError("No voice model loaded — call load_voice_model() first")

        config = {
            "action": "convert",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model_path": str(self._voice_model_path),
            "f0method": f0method or self.f0method,
            "f0up_key": f0up_key if f0up_key is not None else self.f0up_key,
            "index_rate": index_rate if index_rate is not None else self.index_rate,
            "protect": protect if protect is not None else self.protect,
            "device": self.device if self.device != "cpu" else "cpu:0",
        }

        result = _run_bridge(config)
        output = result["output_path"]
        logger.info(f"Voice conversion complete: {output}")
        return output


def list_voice_models(models_dir: str | Path) -> list[dict]:
    """List available .pth voice models in the models directory."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for p in models_dir.glob("**/*.pth"):
        models.append({"name": p.stem, "path": str(p)})
    return models
