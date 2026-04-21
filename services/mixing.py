"""Stem recombination and mixing utilities."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def mix_stems(
    stems: dict[str, np.ndarray],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Mix selected stems together.

    Args:
        stems: dict mapping stem name to waveform array [channels, samples]
        include: only include these stems (default: all)
        exclude: exclude these stems (default: none)
        normalize: normalize output to prevent clipping
    """
    exclude = exclude or []
    include = include or list(stems.keys())

    result = None
    for name in include:
        if name in exclude:
            continue
        stem = stems[name]
        if result is None:
            result = np.zeros_like(stem, dtype=np.float32)
        # Pad shorter stem if lengths differ
        if stem.shape[-1] < result.shape[-1]:
            padded = np.zeros_like(result)
            padded[..., : stem.shape[-1]] = stem
            stem = padded
        elif stem.shape[-1] > result.shape[-1]:
            result = np.pad(result, ((0, 0), (0, stem.shape[-1] - result.shape[-1])))

        result = result + stem

    if result is None:
        raise ValueError("No stems to mix")

    if normalize:
        max_val = np.abs(result).max()
        if max_val > 0.95:
            result = result * (0.95 / max_val)

    return result


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to target peak level."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio * (target_peak / max_val)
    return audio