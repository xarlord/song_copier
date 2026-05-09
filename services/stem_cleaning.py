"""Stem post-processing for artifact reduction and quality enhancement."""

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase bandpass filter using scipy.

    Args:
        audio: shape (channels, samples) or (samples,)
        sr: sample rate
        low_hz: low cutoff frequency (Hz)
        high_hz: high cutoff frequency (Hz)
        order: filter order (4 is good balance)

    Returns:
        Filtered audio, same shape as input
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(order, [low_hz, high_hz], btype='band', fs=sr, output='sos')

    # Handle both mono and stereo
    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)

    filtered = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        filtered[ch] = sosfiltfilt(sos, audio[ch])
    return filtered


def highpass_filter(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 3,
) -> np.ndarray:
    """Apply zero-phase highpass filter (removes low frequencies)."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(order, cutoff_hz, btype='highpass', fs=sr, output='sos')

    if audio.ndim == 1:
        return sosfiltfilt(sos, audio)

    filtered = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        filtered[ch] = sosfiltfilt(sos, audio[ch])
    return filtered


def reduce_artifacts_noisereduce(
    audio: np.ndarray,
    sr: int,
    prop_decrease: float = 0.6,
    freq_mask_smooth_hz: int = 500,
    stationary: bool = False,
) -> np.ndarray:
    """Apply noisereduce spectral gating for artifact reduction.

    Args:
        audio: shape (channels, samples) or (samples,)
        sr: sample rate
        prop_decrease: 0.0-1.0, how much to reduce noise (lower = gentler)
        freq_mask_smooth_hz: frequency smoothing to prevent musical noise
        stationary: False for non-stationary (better for sporadic artifacts)

    Returns:
        Cleaned audio, same shape as input
    """
    try:
        import noisereduce as nr
    except ImportError:
        logger.warning("noisereduce not installed, skipping artifact reduction")
        return audio

    # Handle stereo: process each channel
    if audio.ndim == 2:
        cleaned = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            cleaned[ch] = nr.reduce_noise(
                y=audio[ch],
                sr=sr,
                stationary=stationary,
                prop_decrease=prop_decrease,
                freq_mask_smooth_hz=freq_mask_smooth_hz,
                time_mask_smooth_ms=50,
                thresh_n_mult_nonstationary=2,
                n_fft=2048,
            )
        return cleaned
    else:
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=50,
            thresh_n_mult_nonstationary=2,
            n_fft=2048,
        )


def clean_bass(audio: np.ndarray, sr: int) -> np.ndarray:
    """Clean bass stem: bandpass 30-250Hz to remove rumble and high-freq bleed."""
    logger.info("Applying bass stem cleaning (bandpass 30-250Hz)")
    return bandpass_filter(audio, sr, low_hz=30, high_hz=250, order=4)


def clean_drums(audio: np.ndarray, sr: int) -> np.ndarray:
    """Clean drums stem: highpass at 60Hz + gentle noisereduce."""
    logger.info("Applying drums stem cleaning (highpass 60Hz + noise reduction)")
    # First remove bass bleed
    audio = highpass_filter(audio, sr, cutoff_hz=60, order=3)
    # Then gentle noise reduction (drums have transients, use low prop_decrease)
    audio = reduce_artifacts_noisereduce(
        audio, sr,
        prop_decrease=0.4,
        freq_mask_smooth_hz=300,
        stationary=False,
    )
    return audio


def clean_vocals(audio: np.ndarray, sr: int) -> np.ndarray:
    """Clean vocals: moderate noisereduce to suppress artifacts."""
    logger.info("Applying vocals stem cleaning (noise reduction)")
    return reduce_artifacts_noisereduce(
        audio, sr,
        prop_decrease=0.7,
        freq_mask_smooth_hz=500,
        stationary=False,
    )


def clean_guitar(audio: np.ndarray, sr: int) -> np.ndarray:
    """Clean guitar: moderate noisereduce."""
    logger.info("Applying guitar stem cleaning (noise reduction)")
    return reduce_artifacts_noisereduce(
        audio, sr,
        prop_decrease=0.6,
        freq_mask_smooth_hz=400,
        stationary=False,
    )


def clean_piano(audio: np.ndarray, sr: int) -> np.ndarray:
    """Clean piano: moderate noisereduce."""
    logger.info("Applying piano stem cleaning (noise reduction)")
    return reduce_artifacts_noisereduce(
        audio, sr,
        prop_decrease=0.6,
        freq_mask_smooth_hz=400,
        stationary=False,
    )


def clean_other(audio: np.ndarray, sr: int) -> np.ndarray:
    """Clean other: light noisereduce."""
    logger.info("Applying other stem cleaning (light noise reduction)")
    return reduce_artifacts_noisereduce(
        audio, sr,
        prop_decrease=0.5,
        freq_mask_smooth_hz=300,
        stationary=False,
    )


def clean_stem(
    stem_name: str,
    audio: np.ndarray,
    sr: int,
    enabled: bool = True,
) -> np.ndarray:
    """Apply per-stem cleaning based on stem type.

    Args:
        stem_name: one of 'vocals', 'drums', 'bass', 'guitar', 'piano', 'other'
        audio: shape (channels, samples)
        sr: sample rate
        enabled: if False, skip cleaning (return audio unchanged)

    Returns:
        Cleaned audio
    """
    if not enabled:
        return audio

    cleaners = {
        "vocals": clean_vocals,
        "drums": clean_drums,
        "bass": clean_bass,
        "guitar": clean_guitar,
        "piano": clean_piano,
        "other": clean_other,
    }

    cleaner = cleaners.get(stem_name)
    if cleaner:
        return cleaner(audio, sr)

    logger.warning(f"No cleaner for stem '{stem_name}', returning original")
    return audio
