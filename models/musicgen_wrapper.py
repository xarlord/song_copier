"""MusicGen wrapper for text-to-music with melody conditioning."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from models.base import BaseModelWrapper
from services.audio_io import tensor_to_audio

logger = logging.getLogger(__name__)

VRAM_ESTIMATES = {
    "facebook/musicgen-melody": 4000,
    "facebook/musicgen-small": 1000,
    "facebook/musicgen-medium": 3000,
    "facebook/musicgen-large": 6000,
}


class MusicGenWrapper(BaseModelWrapper):
    """Wraps audiocraft.models.MusicGen for music generation."""

    def __init__(
        self,
        model_name: str = "facebook/musicgen-melody",
        device: str = "cuda",
        cache_dir: str | Path = "cache",
        duration: float = 15.0,
        temperature: float = 1.0,
        cfg_coef: float = 3.0,
        top_k: int = 250,
    ):
        super().__init__(model_name, device, cache_dir)
        self.duration = duration
        self.temperature = temperature
        self.cfg_coef = cfg_coef
        self.top_k = top_k

    def load_model(self) -> None:
        if self._loaded:
            return

        # Install a minimal xformers stub so audiocraft can import without
        # the real xformers package (which is incompatible with torch 2.7+).
        import sys
        import types

        if 'xformers' not in sys.modules:
            xf = types.ModuleType('xformers')
            xf_ops = types.ModuleType('xformers.ops')
            xf.ops = xf_ops
            import torch
            xf_ops.memory_efficient_attention = lambda *a, **kw: None
            xf_ops.unbind = torch.unbind
            xf_ops.LowerTriangularMask = type('LowerTriangularMask', (), {})
            sys.modules['xformers'] = xf
            sys.modules['xformers.ops'] = xf_ops

        # Now import and patch the verification to skip the xformers check
        from audiocraft.modules import transformer as _tf
        _tf._verify_xformers_memory_efficient_compat = lambda: None
        _tf._verify_xformers_internal_compat = lambda: None

        from audiocraft.models import MusicGen

        logger.info(f"Loading MusicGen model: {self.model_name}")
        self.model = MusicGen.get_pretrained(self.model_name)
        self._set_params()
        self._loaded = True
        logger.info(f"MusicGen {self.model_name} loaded")

    def _set_params(self):
        self.model.set_generation_params(
            duration=self.duration,
            temperature=self.temperature,
            cfg_coef=self.cfg_coef,
            top_k=self.top_k,
        )

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
        logger.info("MusicGen unloaded")

    def get_vram_requirement_mb(self) -> int:
        return VRAM_ESTIMATES.get(self.model_name, 4000)

    def generate(
        self,
        prompt: str,
        melody_path: str | None = None,
        melody_sample_rate: int = 32000,
        duration: float | None = None,
        temperature: float | None = None,
        cfg_coef: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Generate music from text prompt, optionally conditioned on melody.

        Returns:
            Tuple of (waveform [channels, samples], sample_rate)
        """
        if not self._loaded:
            self.load_model()

        # Update params if overridden
        if duration is not None:
            self.duration = duration
        if temperature is not None:
            self.temperature = temperature
        if cfg_coef is not None:
            self.cfg_coef = cfg_coef
        self._set_params()

        import torch
        import torchaudio

        if melody_path:
            logger.info(f"Generating with melody conditioning from: {melody_path}")
            melody, sr = torchaudio.load(melody_path)
            # Resample if needed
            if sr != melody_sample_rate:
                melody = torchaudio.functional.resample(melody, sr, melody_sample_rate)
            # Truncate to 30s max for conditioning
            max_samples = melody_sample_rate * 30
            if melody.shape[-1] > max_samples:
                melody = melody[..., :max_samples]
            # Convert to batch format
            if melody.dim() == 1:
                melody = melody.unsqueeze(0)

            wav = self.model.generate_with_chroma(
                descriptions=[prompt],
                melody_wavs=melody[None].to(self.device),
                melody_sample_rate=melody_sample_rate,
                progress=True,
            )
        else:
            logger.info(f"Generating from prompt: {prompt}")
            wav = self.model.generate(
                descriptions=[prompt],
                progress=True,
            )

        audio = tensor_to_audio(wav[0])
        return audio, self.model.sample_rate

    def generate_with_style(
        self,
        prompt: str,
        style_audio_path: str,
        eval_q: int = 1,
        duration: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Generate with style conditioning (requires musicgen-style model)."""
        if not self._loaded:
            self.load_model()

        if duration is not None:
            self.duration = duration
            self._set_params()

        import torch
        import torchaudio

        style_wav, sr = torchaudio.load(style_audio_path)
        style_wav = style_wav[..., : sr * 30]  # max 30s

        wav = self.model.generate_with_chroma(
            descriptions=[prompt],
            melody_wavs=style_wav[None].to(self.device),
            melody_sample_rate=sr,
            progress=True,
        )

        audio = tensor_to_audio(wav[0])
        return audio, self.model.sample_rate