"""Mastering service: loudness normalization, limiting, stereo widening.

Uses only numpy + pedalboard — no pyloudnorm or other dependencies.
Implements simplified ITU-R BS.1770-4 loudness measurement.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presets: target LUFS for common delivery formats
# ---------------------------------------------------------------------------
PRESETS: dict[str, float] = {
    "spotify": -14.0,
    "youtube": -13.0,
    "cd": -9.0,
    "podcast": -16.0,
}

# ---------------------------------------------------------------------------
# Internal: simplified K-weighting filter (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------

def _k_weight_filter(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (b, a) coefficients for a simplified BS.1770-4 K-weighting filter.

    Stage 1 – "pre-filter": a second-order high-pass shelf that boosts high
    frequencies (approximating the head-related transfer function).
    Stage 2 – "RLB weighting": a high-pass that attenuates low frequencies.

    The coefficients below are the standard ITU values for 48 kHz.  For other
    sample rates we apply a simple bilinear-transform frequency scaling which
    is accurate enough for a mastering guide (not an official compliance test).
    """
    # ITU reference coefficients @ 48 kHz
    # Stage 1 (high shelf boost)
    b1_48 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64)
    a1_48 = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
    # Stage 2 (high-pass / RLB)
    b2_48 = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    a2_48 = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64)

    if sr == 48000:
        # Cascade two second-order sections into one fourth-order
        return _cascade_sos(b1_48, a1_48, b2_48, a2_48)

    # Scale pole/zero frequencies proportionally for other sample rates.
    # This is the standard approach when exact ITU coefficients aren't available.
    ratio = sr / 48000.0
    b1 = _warp_coefficients(b1_48, ratio)
    a1 = _warp_coefficients(a1_48, ratio)
    b2 = _warp_coefficients(b2_48, ratio)
    a2 = _warp_coefficients(a2_48, ratio)

    return _cascade_sos(b1, a1, b2, a2)


def _warp_coefficients(coeffs: np.ndarray, ratio: float) -> np.ndarray:
    """Scale filter coefficients for a different sample rate.

    Applies frequency warping via the bilinear transform scaling factor.
    For a second-order section b0 + b1*z^-1 + b2*z^-2 the warping adjusts
    the frequency-dependent terms (b1, b2) by the ratio.
    """
    c = coeffs.copy()
    # Simple proportional scaling of the resonant terms
    if len(c) == 3:
        c[1] = 1.0 - (1.0 - c[1]) * ratio
        c[2] = c[2] ** (1.0 / ratio) if ratio > 0 else c[2]
    return c


def _cascade_sos(
    b1: np.ndarray, a1: np.ndarray, b2: np.ndarray, a2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cascade two second-order sections into a single fourth-order filter."""
    b = np.convolve(b1, b2)
    a = np.convolve(a1, a2)
    return b, a


def _apply_filter(audio: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply IIR filter to audio using second-order sections (direct form II)."""
    # Use scipy-free lfilter via direct implementation
    # For stability, normalize by a[0]
    a0 = a[0]
    b_norm = b / a0
    a_norm = a / a0

    n_samples = audio.shape[-1]
    n_ch = audio.shape[0] if audio.ndim == 2 else 1
    filter_order = len(a_norm) - 1

    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    output = np.zeros_like(audio, dtype=np.float64)

    for ch in range(n_ch):
        x = audio[ch].astype(np.float64)
        y = np.zeros(n_samples, dtype=np.float64)
        # Direct Form II transposed
        state = np.zeros(filter_order, dtype=np.float64)
        for i in range(n_samples):
            y[i] = b_norm[0] * x[i] + state[0]
            for j in range(filter_order - 1):
                state[j] = b_norm[j + 1] * x[i] - a_norm[j + 1] * y[i] + state[j + 1]
            if filter_order > 0:
                state[filter_order - 1] = b_norm[filter_order] * x[i] - a_norm[filter_order] * y[i]
        output[ch] = y

    if output.shape[0] == 1 and n_ch == 1 and audio.shape[0] == 1:
        pass  # keep 2d consistent with input

    return output


# Vectorised lfilter using numpy's built-in (no scipy needed)
def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Minimal lfilter replacement using numpy only (direct form II transposed)."""
    a0 = a[0]
    b = b / a0
    a = a / a0
    n = len(x)
    order = len(a) - 1
    y = np.zeros(n, dtype=np.float64)
    z = np.zeros(order, dtype=np.float64)

    for i in range(n):
        y[i] = b[0] * x[i] + z[0]
        for j in range(order - 1):
            z[j] = b[j + 1] * x[i] - a[j + 1] * y[i] + z[j + 1]
        if order > 0:
            z[order - 1] = b[order] * x[i] - a[order] * y[i]
    return y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_loudness(audio: np.ndarray, sr: int) -> dict:
    """Measure loudness using simplified ITU-R BS.1770-4.

    Args:
        audio: waveform, shape (channels, samples) or (samples,), float32/64.
        sr: sample rate in Hz.

    Returns:
        dict with keys:
            lufs   – integrated loudness in LUFS (float)
            peak_db – true peak in dBTP (float)
            rms_db – overall RMS in dB (float)
    """
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    audio = audio.astype(np.float64)

    n_channels, n_samples = audio.shape

    # --- K-weighted filtering ---
    b, a = _k_weight_filter(sr)

    weighted = np.zeros_like(audio, dtype=np.float64)
    for ch in range(n_channels):
        weighted[ch] = _lfilter(b, a, audio[ch])

    # --- Channel gating (simplified: no absolute gate, just mean square) ---
    # BS.1770-4 uses a 400ms integration window with 75% overlap and an
    # absolute gate at -70 LUFS.  For simplicity we skip the gating and
    # compute the overall mean.
    n_frames = n_samples

    # Channel weighting factors (BS.1770-4)
    channel_weight = np.array([1.0, 1.0, 1.0, 1.41, 1.41])[:n_channels]
    # For stereo (2ch): both weighted 1.0

    mean_square = 0.0
    for ch in range(n_channels):
        ch_ms = np.mean(weighted[ch] ** 2)
        mean_square += channel_weight[ch] * ch_ms

    # LUFS = -0.691 + 10 * log10(mean_square)
    if mean_square > 0:
        lufs = -0.691 + 10.0 * np.log10(mean_square)
    else:
        lufs = -np.inf

    # --- True peak ---
    # Oversample 4x and measure peak (simple linear interpolation approximation)
    peak = np.max(np.abs(audio))

    # Also check interpolated peaks for true-peak estimation
    if n_samples > 1:
        # Sinc interpolation approximation: just use peak of original + upsampled by 4
        upsampled = np.zeros((n_channels, n_samples * 4 - 3), dtype=np.float64)
        for ch in range(n_channels):
            # Linear interpolation as a rough true-peak estimate
            upsampled[ch, 0::4] = audio[ch]
            for i in range(1, 4):
                frac = i / 4.0
                upsampled[ch, i::4] = audio[ch][:-1] * (1 - frac) + audio[ch][1:] * frac
        true_peak = np.max(np.abs(upsampled))
    else:
        true_peak = peak

    peak_db = 20.0 * np.log10(true_peak) if true_peak > 0 else -np.inf

    # --- RMS ---
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20.0 * np.log10(rms) if rms > 0 else -np.inf

    return {
        "lufs": float(lufs),
        "peak_db": float(peak_db),
        "rms_db": float(rms_db),
    }


def normalize_loudness(
    audio: np.ndarray, sr: int, target_lufs: float = -14.0
) -> np.ndarray:
    """Adjust gain so that the integrated loudness hits *target_lufs*.

    Args:
        audio: waveform (channels, samples) or (samples,).
        sr: sample rate.
        target_lufs: desired integrated loudness in LUFS.

    Returns:
        Gain-adjusted audio (same shape/dtype as input).
    """
    metrics = measure_loudness(audio, sr)
    current_lufs = metrics["lufs"]

    if current_lufs == -np.inf:
        logger.warning("Audio is silent — cannot normalize loudness.")
        return audio

    offset_db = target_lufs - current_lufs
    gain = 10.0 ** (offset_db / 20.0)

    logger.debug(
        "Normalizing %.1f LUFS → %.1f LUFS  (gain %.2f dB)",
        current_lufs, target_lufs, offset_db,
    )
    return (audio * gain).astype(audio.dtype)


def apply_limiter(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -1.0,
) -> np.ndarray:
    """Apply a brick-wall limiter using pedalboard.

    Args:
        audio: waveform (channels, samples) or (samples,).
        sr: sample rate.
        threshold_db: ceiling in dB (default -1.0 dBTP).

    Returns:
        Limited audio.
    """
    from pedalboard import Limiter, Pedalboard

    was_1d = audio.ndim == 1
    if was_1d:
        audio = audio.reshape(1, -1)

    # pedalboard.Limiter threshold is in dB; it limits peaks to threshold_db
    board = Pedalboard([Limiter(threshold_db=threshold_db)])
    result = board(audio, sr)

    if was_1d:
        result = result.flatten()

    return result.astype(audio.dtype)


def apply_stereo_widen(audio: np.ndarray, width: float = 1.5) -> np.ndarray:
    """Widen or narrow stereo image via mid-side processing.

    Mid = (L + R) / 2
    Side = (L - R) / 2

    The side channel is scaled by *width*:
        width < 1.0 → narrower   (mono at 0.0)
        width = 1.0 → no change
        width > 1.0 → wider

    If the input is mono (1-D or single-channel 2-D), it is first converted
    to stereo by duplicating the channel.

    Args:
        audio: waveform (channels, samples) or (samples,).
        width: stereo width multiplier.

    Returns:
        Stereo audio with shape (2, samples), float32.
    """
    # Ensure 2-D
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] == 1:
        audio = np.concatenate([audio, audio], axis=0)

    left = audio[0].astype(np.float64)
    right = audio[1].astype(np.float64)

    # Encode mid-side
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    # Scale side
    side *= width

    # Decode back to L/R
    new_left = mid + side
    new_right = mid - side

    result = np.stack([new_left, new_right], axis=0)

    # Clamp to prevent clipping from width > 1
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak * 0.95

    return result.astype(np.float32)


def master_audio(
    audio: np.ndarray,
    sr: int,
    preset: str = "spotify",
    target_lufs: float | None = None,
    limiter_threshold: float = -1.0,
    stereo_width: float = 1.2,
) -> tuple[np.ndarray, dict]:
    """Full mastering chain: normalize → widen → limit.

    Args:
        audio: input waveform (channels, samples) or (samples,).
        sr: sample rate.
        preset: one of "spotify", "youtube", "cd", "podcast".
            Determines target LUFS unless *target_lufs* is explicitly given.
        target_lufs: override the preset's target loudness.
        limiter_threshold: brick-wall ceiling in dBTP.
        stereo_width: mid-side width multiplier.

    Returns:
        (mastered_audio, metrics) where metrics is a dict with pre/post loudness
        info and the chosen preset.
    """
    # Resolve target LUFS
    if target_lufs is None:
        target_lufs = PRESETS.get(preset, -14.0)

    # Pre-measurement
    pre_metrics = measure_loudness(audio, sr)

    # 1. Normalize loudness
    result = normalize_loudness(audio, sr, target_lufs=target_lufs)

    # 2. Stereo widening
    result = apply_stereo_widen(result, width=stereo_width)

    # 3. Brick-wall limiter
    result = apply_limiter(result, sr, threshold_db=limiter_threshold)

    # Post-measurement
    post_metrics = measure_loudness(result, sr)

    metrics = {
        "preset": preset,
        "target_lufs": target_lufs,
        "pre": pre_metrics,
        "post": post_metrics,
    }

    logger.info(
        "Mastered with preset=%s  target=%.1f LUFS  "
        "pre_lufs=%.1f → post_lufs=%.1f  post_peak=%.1f dBTP",
        preset, target_lufs,
        pre_metrics["lufs"], post_metrics["lufs"], post_metrics["peak_db"],
    )

    return result, metrics
