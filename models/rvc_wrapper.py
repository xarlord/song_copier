"""RVC voice conversion wrapper — uses subprocess bridge to Python 3.10 venv.

Supports two execution environments:
- Native WSL (Linux): calls rvc_env/bin/python directly
- Windows (via WSL interop): calls wsl to invoke the Linux Python 3.10 venv
"""

import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

from models.base import BaseModelWrapper

logger = logging.getLogger(__name__)

# Resolve paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BRIDGE_SCRIPT = _PROJECT_ROOT / "services" / "rvc_bridge.py"

# Detect environment
_IS_WINDOWS = platform.system() == "Windows"


def _windows_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path (D:\\foo) to a WSL path (/mnt/d/foo)."""
    p = Path(win_path)
    # D:\foo -> /mnt/d/foo
    drive = p.drive.rstrip(":").lower()
    return f"/mnt/{drive}{p.as_posix()[2:]}"  # strip drive letter part


def _wsl_to_windows_path(wsl_path: str) -> str:
    """Convert a WSL path (/mnt/d/foo) to a Windows path (D:\\foo)."""
    if wsl_path.startswith("/mnt/"):
        parts = wsl_path[5:].split("/", 1)
        drive = parts[0].upper()
        rest = parts[1] if len(parts) > 1 else ""
        return f"{drive}:\\{rest}".replace("/", "\\")
    return wsl_path


def _get_rvc_python_path() -> str:
    """Get the path to the RVC venv Python, in the correct format for the current OS."""
    if _IS_WINDOWS:
        # Running on Windows — the venv is in WSL filesystem
        # Project root will be D:\Claude_projects\audio_generator
        wsl_root = _windows_to_wsl_path(str(_PROJECT_ROOT))
        return f"{wsl_root}/rvc_env/bin/python"
    else:
        return str(_PROJECT_ROOT / "rvc_env" / "bin" / "python")


def _get_bridge_script_path() -> str:
    """Get the bridge script path in the correct format for the current OS."""
    if _IS_WINDOWS:
        return _windows_to_wsl_path(str(_BRIDGE_SCRIPT))
    return str(_BRIDGE_SCRIPT)


def _translate_path_for_bridge(path: str) -> str:
    """Translate a file path so the bridge process can access it.

    If running on Windows, convert Windows paths to WSL paths for the bridge.
    If running on WSL, paths are already Linux-compatible.
    """
    if _IS_WINDOWS:
        return _windows_to_wsl_path(path)
    return path


def _translate_path_from_bridge(path: str) -> str:
    """Translate a file path from the bridge back to the host OS format."""
    if _IS_WINDOWS:
        return _wsl_to_windows_path(path)
    return path


def _run_bridge(config: dict) -> dict:
    """Run rvc_bridge.py in the Python 3.10 venv, passing config via stdin.

    Handles Windows→WSL bridge calls automatically.
    """
    rvc_python = _get_rvc_python_path()
    bridge_script = _get_bridge_script_path()

    # Translate file paths in config for the bridge environment
    for key in ("input_path", "output_path", "model_path"):
        if key in config and config[key]:
            config[key] = _translate_path_for_bridge(config[key])

    if _IS_WINDOWS:
        # Run via wsl command
        cmd = ["wsl", "-e", rvc_python, bridge_script]
    else:
        cmd = [rvc_python, bridge_script]

    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(config),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max for long audio
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"RVC bridge not found. WSL Python 3.10 venv expected at: {rvc_python}. "
            f"Create it with: python3.10 -m venv rvc_env && rvc_env/bin/pip install "
            f"rvc-python torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )

    if proc.returncode != 0:
        raise RuntimeError(f"RVC bridge failed (exit {proc.returncode}): {proc.stderr}")

    result = json.loads(proc.stdout)
    if not result.get("success"):
        raise RuntimeError(f"RVC bridge error: {result.get('error', 'unknown')}")

    # Translate output paths back to host format
    if "output_path" in result:
        result["output_path"] = _translate_path_from_bridge(result["output_path"])

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
