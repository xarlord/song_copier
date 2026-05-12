"""Autotune plugin — pitch correction with configurable key, scale, speed, and formant shift."""

from __future__ import annotations

import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Note / scale helpers
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Semitone offsets from root for each scale type (0-indexed relative to root)
_SCALE_INTERVALS = {
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major":     [0, 2, 4, 5, 7, 9, 11],
    "minor":     [0, 2, 3, 5, 7, 8, 10],
    "pentatonic":[0, 2, 4, 7, 9],
    "blues":     [0, 3, 5, 6, 7, 10],
}


def _note_to_midi(note_name: str) -> int:
    """Convert a note name like 'C#' to its pitch-class index (0-11)."""
    return _NOTE_NAMES.index(note_name)


def _build_target_notes(key: str, scale: str) -> list[int]:
    """Return the set of pitch-class indices (0-11) for the given key/scale."""
    root = _note_to_midi(key)
    intervals = _SCALE_INTERVALS.get(scale, _SCALE_INTERVALS["chromatic"])
    return [(root + iv) % 12 for iv in intervals]


def _freq_to_pitch_class(freq: float) -> int | None:
    """Map a frequency in Hz to a pitch-class index (0-11), or None if unpitched."""
    if freq <= 0 or not np.isfinite(freq):
        return None
    midi_float = 12.0 * np.log2(freq / 440.0) + 69.0
    return int(round(midi_float)) % 12


def _freq_to_midi(freq: float) -> float:
    """Convert frequency to fractional MIDI note number."""
    if freq <= 0:
        return 0.0
    return 12.0 * np.log2(freq / 440.0) + 69.0


def _midi_to_freq(midi: float) -> float:
    """Convert MIDI note number to frequency."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _nearest_target_midi(midi_float: float, target_pcs: list[int]) -> float:
    """Find the nearest MIDI note whose pitch class is in target_pcs."""
    midi_round = round(midi_float)
    best_dist = float("inf")
    best_midi = midi_round
    # Check a few candidates in each direction
    for offset in range(-2, 3):
        candidate = midi_round + offset
        if candidate % 12 in target_pcs:
            dist = abs(candidate - midi_float)
            if dist < best_dist:
                best_dist = dist
                best_midi = float(candidate)
    return best_midi


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------

def get_info() -> dict:
    """Return plugin metadata."""
    return {
        "name": "autotune",
        "version": "1.0.0",
        "description": "Pitch correction / autotune effect",
        "author": "Audio Generator Studio",
    }


def process_audio(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Apply autotune pitch correction.

    Parameters
    ----------
    audio : np.ndarray
        Input audio array. Mono (1D) or stereo (2D with shape (2, N)).
    sr : int
        Sample rate in Hz.
    key : str
        Target musical key (e.g. "C", "F#"). Default "C".
    scale : str
        Scale type: chromatic, major, minor, pentatonic, blues. Default "chromatic".
    speed : float
        Correction speed. 0 = subtle/natural, 1 = hard T-Pain snap. Default 0.8.
    formant_shift : float
        Formant shift in semitones. Default 0.0.

    Returns
    -------
    np.ndarray
        Pitch-corrected audio, same shape as input, float32.
    """
    key: str = str(kwargs.get("key", "C"))
    scale: str = str(kwargs.get("scale", "chromatic"))
    speed: float = float(kwargs.get("speed", 0.8))
    formant_shift: float = float(kwargs.get("formant_shift", 0.0))

    # Build target pitch-class set
    target_pcs = _build_target_notes(key, scale)

    # Handle stereo: process each channel independently
    was_stereo = audio.ndim == 2
    if was_stereo:
        channels = [audio[0].copy(), audio[1].copy()]
    else:
        channels = [audio.copy()]

    output_channels = []
    for ch_idx, ch_data in enumerate(channels):
        corrected = _autotune_mono(ch_data, sr, target_pcs, speed)
        output_channels.append(corrected)

    if was_stereo:
        out = np.stack(output_channels, axis=0)
    else:
        out = output_channels[0]

    # Apply formant shift if nonzero
    if abs(formant_shift) > 0.05:
        out = _apply_formant_shift(out, sr, formant_shift)

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _autotune_mono(
    y: np.ndarray,
    sr: int,
    target_pcs: list[int],
    speed: float,
) -> np.ndarray:
    """Autotune a single mono channel using frame-by-frame pitch detection and shifting."""
    # Ensure mono float64
    y = y.astype(np.float64)
    n_samples = len(y)

    if n_samples < sr // 10:  # Very short signal — return as-is
        return y.astype(np.float32)

    # Pitch detection parameters
    frame_length = 2048
    hop_length = 512

    # Run pyin for pitch detection
    try:
        f0, voiced_flags, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        # If pyin fails, return original
        return y.astype(np.float32)

    n_frames = len(f0)
    if n_frames == 0:
        return y.astype(np.float32)

    # Build output buffer
    output = np.zeros(n_samples, dtype=np.float64)

    # Process in overlapping windows
    # Use a Hann window for overlap-add
    win_len = hop_length * 4  # Analysis window length
    window = np.hanning(win_len)

    for i in range(n_frames):
        freq = f0[i]
        if np.isnan(freq) or freq <= 0:
            # Unvoiced frame: copy original
            start = i * hop_length
            end = min(start + win_len, n_samples)
            seg = y[start:end]
            w = window[: len(seg)]
            output[start:end] += seg * w
            continue

        # Current pitch in MIDI
        current_midi = _freq_to_midi(freq)

        # Find nearest target note
        target_midi = _nearest_target_midi(current_midi, target_pcs)

        # Calculate needed shift in semitones
        shift_semitones = target_midi - current_midi

        # Apply speed: blend between no shift (speed=0) and full shift (speed=1)
        effective_shift = shift_semitones * speed

        if abs(effective_shift) < 0.01:
            # No significant shift needed
            start = i * hop_length
            end = min(start + win_len, n_samples)
            seg = y[start:end]
            w = window[: len(seg)]
            output[start:end] += seg * w
            continue

        # Extract the window for this frame
        start = i * hop_length
        end = min(start + win_len, n_samples)
        actual_len = end - start
        if actual_len < hop_length:
            continue

        segment = y[start:end]

        # Pitch shift this segment by effective_shift semitones
        try:
            shifted = _pitch_shift_segment(segment, sr, effective_shift)
        except Exception:
            shifted = segment

        # Ensure shifted matches expected length
        if len(shifted) != actual_len:
            # Resample to match
            if len(shifted) > 0:
                indices = np.linspace(0, len(shifted) - 1, actual_len)
                shifted = np.interp(indices, np.arange(len(shifted)), shifted)
            else:
                shifted = np.zeros(actual_len, dtype=np.float64)

        w = window[:actual_len]
        output[start:end] += shifted * w

    # Normalize by the overlap-add window sum to avoid amplitude changes
    window_sum = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + win_len, n_samples)
        actual_len = end - start
        window_sum[start:end] += window[:actual_len] ** 2

    # Avoid division by zero
    mask = window_sum > 1e-8
    output[mask] /= window_sum[mask]
    # For silence gaps, just copy original
    output[~mask] = y[~mask]

    return output.astype(np.float32)


def _pitch_shift_segment(segment: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    """Pitch shift a short audio segment using librosa.time_stretch + resample technique.

    Uses the standard phase vocoder approach via librosa.effects.pitch_shift.
    Falls back to a simple resampling approach if the segment is too short.
    """
    if len(segment) < 256:
        return segment

    # For very small shifts, use simple resampling
    if abs(n_semitones) < 0.1:
        return segment

    # Use librosa pitch_shift (phase vocoder based)
    try:
        shifted = librosa.effects.pitch_shift(
            segment,
            sr=sr,
            n_steps=n_semitones,
        )
        return shifted
    except Exception:
        # Fallback: resample-based pitch shift
        rate = 2.0 ** (-n_semitones / 12.0)
        stretched = librosa.resample(
            segment,
            orig_sr=sr,
            target_sr=int(sr * rate),
        )
        # Resample back to original length
        if len(stretched) > 1:
            indices = np.linspace(0, len(stretched) - 1, len(segment))
            return np.interp(indices, np.arange(len(stretched)), stretched)
        return segment


def _apply_formant_shift(audio: np.ndarray, sr: int, shift_semitones: float) -> np.ndarray:
    """Apply a simple formant shift by pitch shifting and mixing back.

    This uses a spectral envelope approach: shift the spectral envelope
    independently from the fundamental frequency. A practical approximation
    is to use librosa.effects.pitch_shift with a small window.
    """
    if abs(shift_semitones) < 0.05:
        return audio

    was_stereo = audio.ndim == 2
    if was_stereo:
        channels = [audio[0], audio[1]]
    else:
        channels = [audio]

    result = []
    for ch in channels:
        ch = ch.astype(np.float64)
        # Formant shift approximation: pitch shift the entire signal
        # then resample to maintain original duration
        try:
            shifted = librosa.effects.pitch_shift(
                ch,
                sr=sr,
                n_steps=shift_semitones,
            )
        except Exception:
            shifted = ch

        # Match length to original
        if len(shifted) != len(ch):
            if len(shifted) > 1:
                indices = np.linspace(0, len(shifted) - 1, len(ch))
                shifted = np.interp(indices, np.arange(len(shifted)), shifted)
            else:
                shifted = ch

        result.append(shifted)

    if was_stereo:
        return np.stack(result, axis=0).astype(np.float32)
    return result[0].astype(np.float32)
