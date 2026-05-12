"""Harmonizer plugin — add vocal harmonies at configurable intervals."""

from __future__ import annotations

import numpy as np
import librosa


def get_info() -> dict:
    """Return plugin metadata."""
    return {
        "name": "harmonizer",
        "version": "1.0.0",
        "description": "Add vocal harmonies at configurable intervals",
        "author": "Audio Generator Studio",
    }


def process_audio(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Apply harmonizer effect — create pitch-shifted harmony voices.

    Parameters
    ----------
    audio : np.ndarray
        Input audio array. Mono (1D) or stereo (2D with shape (2, N)).
    sr : int
        Sample rate in Hz.
    interval_1 : str
        Harmony 1 interval in semitones (e.g. "3", "7", "none" to disable).
        Default "3".
    interval_2 : str
        Harmony 2 interval in semitones (e.g. "3", "7", "none" to disable).
        Default "7".
    gain_1 : float
        Volume of harmony voice 1 (0-1). Default 0.7.
    gain_2 : float
        Volume of harmony voice 2 (0-1). Default 0.5.
    mix : float
        Dry/wet mix. 0 = original only, 1 = harmonies only (no original).
        Default 0.5.

    Returns
    -------
    np.ndarray
        Audio with added harmonies, same shape as input, float32.
    """
    interval_1_str: str = str(kwargs.get("interval_1", "3"))
    interval_2_str: str = str(kwargs.get("interval_2", "7"))
    gain_1: float = float(kwargs.get("gain_1", 0.7))
    gain_2: float = float(kwargs.get("gain_2", 0.5))
    mix: float = float(kwargs.get("mix", 0.5))

    # Parse intervals
    interval_1 = _parse_interval(interval_1_str)
    interval_2 = _parse_interval(interval_2_str)

    # Determine if stereo
    was_stereo = audio.ndim == 2
    if was_stereo:
        # Mix stereo to mono for harmony generation, then re-apply stereo
        mono = np.mean(audio, axis=0).astype(np.float64)
    else:
        mono = audio.astype(np.float64)

    n_samples = len(mono)

    # Build the wet signal (harmonies)
    wet = np.zeros(n_samples, dtype=np.float64)

    if interval_1 is not None and gain_1 > 0.001:
        harmony_1 = _pitch_shift_audio(mono, sr, interval_1)
        harmony_1 = _match_length(harmony_1, n_samples)
        wet += harmony_1 * gain_1

    if interval_2 is not None and gain_2 > 0.001:
        harmony_2 = _pitch_shift_audio(mono, sr, interval_2)
        harmony_2 = _match_length(harmony_2, n_samples)
        wet += harmony_2 * gain_2

    # Blend dry and wet
    dry = mono
    result = dry * (1.0 - mix) + wet * mix

    # If original was stereo, create a stereo output:
    # - Left channel: original + slight pan of harmonies left
    # - Right channel: original + slight pan of harmonies right
    if was_stereo:
        # Create stereo image with harmonies slightly panned
        orig_left = audio[0].astype(np.float64)
        orig_right = audio[1].astype(np.float64)

        out_left = orig_left * (1.0 - mix) + wet * mix * 0.6
        out_right = orig_right * (1.0 - mix) + wet * mix * 0.6

        # Add center harmony from mono mix
        center = result * mix * 0.4
        out_left += center * 0.3
        out_right += center * 0.3

        out = np.stack([out_left, out_right], axis=0)
    else:
        out = result

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _parse_interval(interval_str: str) -> int | None:
    """Parse an interval string like '3', '-7', 'none' to an int or None."""
    if not interval_str or interval_str.lower() == "none":
        return None
    try:
        return int(interval_str)
    except (ValueError, TypeError):
        return None


def _pitch_shift_audio(audio: np.ndarray, sr: int, n_semitones: int) -> np.ndarray:
    """Pitch shift the entire audio signal by n_semitones using librosa.

    Uses a segmented approach for long audio to manage memory, with
    crossfading between segments to avoid artifacts.
    """
    n_samples = len(audio)

    if n_samples < 256:
        return audio.copy()

    # For shorter audio (< 30 seconds), process in one go
    if n_samples <= sr * 30:
        return _pitch_shift_single(audio, sr, n_semitones)

    # For longer audio, process in segments with overlap
    segment_len = sr * 10  # 10-second segments
    overlap = sr // 2      # 0.5-second overlap
    hop = segment_len - overlap
    fade_len = overlap

    # Create fade-in / fade-out windows for crossfading
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float64)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float64)

    output = np.zeros(n_samples, dtype=np.float64)
    positions = []

    # Collect segment start positions
    start = 0
    while start < n_samples:
        positions.append(start)
        start += hop

    for seg_start in positions:
        seg_end = min(seg_start + segment_len, n_samples)
        segment = audio[seg_start:seg_end]

        shifted = _pitch_shift_single(segment, sr, n_semitones)
        shifted = _match_length(shifted, seg_end - seg_start)

        if seg_start > 0 and fade_len > 0:
            # Crossfade with existing output
            fade_region = min(fade_len, seg_end - seg_start, seg_start)
            if fade_region > 0:
                # Fade out existing
                output[seg_start:seg_start + fade_region] *= fade_out[:fade_region]
                # Fade in new
                shifted[:fade_region] *= fade_in[:fade_region]
                output[seg_start:seg_start + fade_region] += shifted[:fade_region]
                # Copy rest
                remaining = shifted[fade_region:]
                out_end = seg_start + fade_region + len(remaining)
                if out_end <= n_samples:
                    output[seg_start + fade_region:out_end] = remaining
            else:
                end_copy = min(seg_start + len(shifted), n_samples)
                output[seg_start:end_copy] = shifted[:end_copy - seg_start]
        else:
            end_copy = min(seg_start + len(shifted), n_samples)
            output[seg_start:end_copy] = shifted[:end_copy - seg_start]

    return output


def _pitch_shift_single(audio: np.ndarray, sr: int, n_semitones: int) -> np.ndarray:
    """Pitch shift a single segment using librosa.effects.pitch_shift."""
    try:
        shifted = librosa.effects.pitch_shift(
            audio,
            sr=sr,
            n_steps=n_semitones,
        )
        return shifted
    except Exception:
        # Fallback: resample-based approach
        return _pitch_shift_resample(audio, sr, n_semitones)


def _pitch_shift_resample(audio: np.ndarray, sr: int, n_semitones: int) -> np.ndarray:
    """Simple resample-based pitch shift as a fallback."""
    rate = 2.0 ** (n_semitones / 12.0)
    target_sr = int(sr * rate)
    if target_sr <= 0:
        return audio.copy()
    try:
        resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    except Exception:
        return audio.copy()

    # Resample back to original length
    n_orig = len(audio)
    if len(resampled) > 1:
        indices = np.linspace(0, len(resampled) - 1, n_orig)
        return np.interp(indices, np.arange(len(resampled)), resampled)
    return audio.copy()


def _match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Ensure audio is exactly target_len samples by padding or trimming."""
    current_len = len(audio)
    if current_len == target_len:
        return audio
    if current_len > target_len:
        return audio[:target_len]
    # Pad with zeros
    padded = np.zeros(target_len, dtype=audio.dtype)
    padded[:current_len] = audio
    return padded
