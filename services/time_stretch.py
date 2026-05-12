"""Time-stretching and pitch-shifting for audio stems.

Uses librosa.effects for high-quality time-stretch (phase vocoder) and
pitch-shift (fast quadratic interpolation). All functions handle mono
(shape (samples,)) and stereo (shape (channels, samples)) input, returning
the same shape.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Change tempo without changing pitch.

    Args:
        audio: numpy array, shape (samples,) for mono or (channels, samples) for stereo.
        sr: sample rate.
        rate: stretch factor. 0.5 = half speed (longer), 2.0 = double speed (shorter).
              1.0 = no change.

    Returns:
        Time-stretched audio with the same dimensionality as input.
    """
    if rate == 1.0:
        return audio

    import librosa

    if audio.ndim == 1:
        # Mono
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched.astype(np.float32)

    # Stereo (or multi-channel): process each channel independently
    channels = []
    for ch in range(audio.shape[0]):
        ch_stretched = librosa.effects.time_stretch(audio[ch], rate=rate)
        channels.append(ch_stretched)

    # Channels may have slightly different lengths after stretch; pad to match
    max_len = max(ch.shape[0] for ch in channels)
    result = np.zeros((len(channels), max_len), dtype=np.float32)
    for i, ch in enumerate(channels):
        result[i, : ch.shape[0]] = ch

    return result


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Change pitch without changing tempo.

    Args:
        audio: numpy array, shape (samples,) for mono or (channels, samples) for stereo.
        sr: sample rate.
        semitones: pitch shift in semitones. Positive = higher, negative = lower.

    Returns:
        Pitch-shifted audio with the same shape as input.
    """
    if semitones == 0.0:
        return audio

    import librosa

    if audio.ndim == 1:
        # Mono
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        return shifted.astype(np.float32)

    # Stereo: process each channel independently
    channels = []
    for ch in range(audio.shape[0]):
        ch_shifted = librosa.effects.pitch_shift(audio[ch], sr=sr, n_steps=semitones)
        channels.append(ch_shifted)

    # Channels should have the same length, but pad just in case
    max_len = max(ch.shape[0] for ch in channels)
    result = np.zeros((len(channels), max_len), dtype=np.float32)
    for i, ch in enumerate(channels):
        result[i, : ch.shape[0]] = ch

    return result


def process_stem(
    audio: np.ndarray,
    sr: int,
    stretch_rate: float = 1.0,
    pitch_semitones: float = 0.0,
) -> np.ndarray:
    """Apply pitch shift then time stretch to a single stem.

    Order: pitch shift first, then time stretch. This avoids pitch artifacts
    that can arise from stretching before shifting.

    If both parameters are at their defaults, returns the original array unchanged.

    Args:
        audio: numpy array (mono or stereo).
        sr: sample rate.
        stretch_rate: time stretch factor (1.0 = no change).
        pitch_semitones: semitones to shift (0.0 = no change).

    Returns:
        Processed audio with the same dimensionality as input.
    """
    if stretch_rate == 1.0 and pitch_semitones == 0.0:
        return audio

    logger.debug(
        "Processing stem: pitch=%.1f semitones, stretch=%.2fx",
        pitch_semitones,
        stretch_rate,
    )

    result = audio.copy()

    # Step 1: Pitch shift
    if pitch_semitones != 0.0:
        result = pitch_shift(result, sr, pitch_semitones)

    # Step 2: Time stretch
    if stretch_rate != 1.0:
        result = time_stretch(result, sr, stretch_rate)

    return result


def batch_process_stems(
    stems: dict[str, np.ndarray],
    sr: int,
    settings: dict[str, dict],
) -> dict[str, np.ndarray]:
    """Process multiple stems with per-stem time-stretch and pitch-shift settings.

    Args:
        stems: dict mapping stem name to numpy audio array.
               Example: {"vocals": np.array(...), "drums": np.array(...)}
        sr: sample rate (same for all stems).
        settings: dict mapping stem name to processing parameters.
                  Example: {
                      "vocals": {"stretch": 1.0, "pitch": 0},
                      "drums": {"stretch": 0.9, "pitch": -2},
                  }
                  Missing stems use defaults (no processing).
                  Each settings dict supports:
                    - "stretch" (float, default 1.0): time stretch rate
                    - "pitch" (float, default 0.0): semitones to shift

    Returns:
        dict mapping stem name to processed numpy array.
        Stems not listed in settings are returned unchanged.
    """
    processed = {}

    for name, audio in stems.items():
        stem_settings = settings.get(name, {})
        stretch_rate = stem_settings.get("stretch", 1.0)
        pitch_semitones = stem_settings.get("pitch", 0.0)

        logger.info(
            "Processing stem '%s': stretch=%.2fx, pitch=%.1f semitones",
            name,
            stretch_rate,
            pitch_semitones,
        )

        processed[name] = process_stem(audio, sr, stretch_rate, pitch_semitones)

    return processed
