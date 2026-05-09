"""Demucs source separation wrapper."""

import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio as ta

from models.base import BaseModelWrapper
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, convert_audio

logger = logging.getLogger(__name__)

VRAM_ESTIMATES = {
    "htdemucs": 3000,
    "htdemucs_ft": 5000,
    "htdemucs_6s": 6000,
    "mdx": 2000,
}


class DemucsWrapper(BaseModelWrapper):
    """Wraps Demucs for stem separation using pretrained.get_model + apply.apply_model."""

    def __init__(
        self,
        model_name: str = "htdemucs_ft",
        device: str = "cuda",
        cache_dir: str | Path = "cache",
        segment: float = 10,
        shifts: int = 5,  # Average 5 shifts for better quality (removes artifacts)
        overlap: float = 0.25,
    ):
        super().__init__(model_name, device, cache_dir)
        self.segment = segment
        self.shifts = shifts
        self.overlap = overlap

    def load_model(self) -> None:
        if self._loaded:
            return
        logger.info(f"Loading Demucs model: {self.model_name}")
        self.model = get_model(self.model_name)
        self.model.eval()
        # Override segment on the inner model if specified
        if self.segment and self.segment > 0:
            from demucs.apply import BagOfModels
            if isinstance(self.model, BagOfModels):
                for m in self.model.models:
                    m.segment = self.segment
            elif hasattr(self.model, 'segment'):
                self.model.segment = self.segment
        self._loaded = True
        logger.info(f"Demucs {self.model_name} loaded (sources: {self.model.sources})")

    def unload_model(self) -> None:
        if not self._loaded:
            return
        self.model.cpu()
        del self.model
        self.model = None
        self._loaded = False
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Demucs unloaded")

    def get_vram_requirement_mb(self) -> int:
        return VRAM_ESTIMATES.get(self.model_name, 4000)

    def _load_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        """Load audio file into tensor compatible with the model."""
        ref = self.model.samplerate
        audio_channels = self.model.audio_channels

        # Try ffmpeg-based loading first, fall back to torchaudio
        wav = None
        try:
            wav = AudioFile(audio_path).read(
                streams=0, samplerate=ref, channels=audio_channels
            )
        except Exception:
            pass

        if wav is None:
            wav, sr = ta.load(str(audio_path))
            wav = convert_audio(wav, sr, ref, audio_channels)

        return wav, ref

    def separate(self, audio_path: str) -> tuple[dict[str, np.ndarray], int]:
        """Separate audio file into stems.

        Returns:
            Tuple of (stems dict mapping name to waveform [channels, samples], sample_rate)
        """
        if not self._loaded:
            self.load_model()

        logger.info(f"Separating: {audio_path}")

        wav, sr = self._load_audio(audio_path)
        # Ensure batch dim: [1, channels, length]
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        device = torch.device(self.device)
        wav = wav.to(device)

        with torch.no_grad():
            estimates = apply_model(
                self.model,
                wav,
                shifts=self.shifts,
                overlap=self.overlap,
                progress=True,
                device=device,
            )

        # estimates shape: [batch, sources, channels, length]
        stems = {}
        source_names = self.model.sources
        for i, name in enumerate(source_names):
            stems[name] = estimates[0, i].cpu().numpy()

        logger.info(f"Separation complete: {list(stems.keys())}")
        return stems, sr