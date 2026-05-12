"""Example plugin — soft-clipping distortion with dry/wet mix control."""

from __future__ import annotations

import numpy as np


def get_info() -> dict:
    """Return plugin metadata."""
    return {
        "name": "example-distortion",
        "version": "1.0.0",
        "description": "Soft-clipping distortion effect — warm saturation to hard clip",
        "author": "Audio Generator Studio",
    }


def process_audio(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Apply soft-clipping distortion.

    Parameters
    ----------
    audio : np.ndarray
        Input audio (mono or stereo), float values.
    sr : int
        Sample rate in Hz (unused by this effect but part of the interface).
    intensity : float
        0 = clean, 1 = maximum distortion. Drives the input gain into the
        soft-clipper. Default 0.5.
    mix : float
        Dry/wet blend. 0 = fully dry, 1 = fully wet. Default 1.0.

    Returns
    -------
    np.ndarray
        Distorted audio, same shape as input, clamped to [-1, 1].
    """
    intensity: float = float(kwargs.get("intensity", 0.5))
    mix: float = float(kwargs.get("mix", 1.0))

    # Copy to avoid mutating the original
    x = audio.astype(np.float64, copy=True)

    # Drive: map intensity 0..1 → gain 1..20
    drive = 1.0 + intensity * 19.0
    x *= drive

    # Soft clip using tanh — smooth saturation curve
    wet = np.tanh(x)

    # Blend dry/wet
    dry = audio.astype(np.float64)
    out = dry * (1.0 - mix) + wet * mix

    # Clamp to valid range
    return np.clip(out, -1.0, 1.0).astype(np.float32)
