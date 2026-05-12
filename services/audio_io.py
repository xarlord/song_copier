"""Audio I/O utilities — load, save, resample, format conversion."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_audio(path: str | Path, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.
    Returns (waveform as numpy array [channels, samples], sample_rate).
    """
    import torchaudio

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform.numpy(), sr


def save_audio(
    waveform: np.ndarray,
    path: str | Path,
    sample_rate: int,
    format: str = "wav",
) -> Path:
    """Save numpy waveform to file. waveform shape: [channels, samples] or [samples]."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure 2D: [channels, samples]
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]

    # soundfile expects [samples, channels]
    data = waveform.T

    if format == "mp3":
        # soundfile doesn't write mp3, use pydub
        from pydub import AudioSegment
        import tempfile

        wav_path = path.with_suffix(".wav")
        sf.write(str(wav_path), data, sample_rate)
        audio = AudioSegment.from_wav(str(wav_path))
        mp3_path = path.with_suffix(".mp3")
        audio.export(str(mp3_path), format="mp3")
        wav_path.unlink(missing_ok=True)
        return mp3_path
    else:
        subtype = "FLOAT" if format == "wav" else "PCM_16"
        sf.write(str(path.with_suffix(f".{format}")), data, sample_rate, subtype=subtype)
        return path.with_suffix(f".{format}")


def audio_to_tensor(audio: np.ndarray):
    """Convert numpy array to PyTorch tensor."""
    import torch
    return torch.from_numpy(audio)


def tensor_to_audio(tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return tensor.numpy()