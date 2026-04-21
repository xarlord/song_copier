"""Audio analysis — BPM detection, key detection, segment info."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def analyze_audio(audio_path: str) -> dict:
    """Analyze audio file for musical properties.

    Returns dict with: bpm, key, duration, rms_energy
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr

    # BPM detection
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # Key detection (chroma-based)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = chroma.mean(axis=1)
    key_idx = int(np.argmax(chroma_avg))
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key = note_names[key_idx]

    # Major/minor detection
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    # Rotate chroma to detected key and correlate
    rotated = np.roll(chroma_avg, -key_idx)
    major_corr = np.corrcoef(rotated, major_profile)[0, 1]
    minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]
    mode = "major" if major_corr > minor_corr else "minor"

    # RMS energy
    rms = librosa.feature.rms(y=y).mean()

    logger.info(f"Analysis: BPM={tempo:.1f}, Key={key} {mode}, Duration={duration:.1f}s")

    return {
        "bpm": round(tempo, 1),
        "key": f"{key} {mode}",
        "duration": round(duration, 1),
        "rms_energy": float(rms),
    }


def build_prompt_from_analysis(analysis: dict, user_prompt: str = "") -> str:
    """Build a MusicGen text prompt from analysis + user description."""
    parts = []
    if user_prompt:
        parts.append(user_prompt)
    else:
        parts.append(f"{analysis['key']} key")
        parts.append(f"{analysis['bpm']} BPM")

    return ", ".join(parts)