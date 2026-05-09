"""Per-stem audio effects using pedalboard (CPU-based, no VRAM needed)."""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Default effect configs (all effects off)
DEFAULT_EFFECTS = {
    "eq": {"low_gain": 0.0, "mid_gain": 0.0, "high_gain": 0.0},
    "reverb": {"room_size": 0.0, "damping": 0.5, "wet_level": 0.0},
    "delay": {"delay_seconds": 0.0, "feedback": 0.0, "mix": 0.0},
    "compressor": {"threshold_db": 0.0, "ratio": 1.0, "attack_ms": 10.0},
    "pan": 0.0,
}


def apply_eq(audio: np.ndarray, sr: int, low_gain: float = 0.0, mid_gain: float = 0.0, high_gain: float = 0.0) -> np.ndarray:
    """Apply 3-band parametric EQ: low shelf (200Hz), mid bell (1kHz), high shelf (4kHz)."""
    if low_gain == 0.0 and mid_gain == 0.0 and high_gain == 0.0:
        return audio

    from pedalboard import LowShelfFilter, PeakFilter, HighShelfFilter, Pedalboard

    filters = []
    if low_gain != 0.0:
        filters.append(LowShelfFilter(cutoff_frequency_hz=200, gain_db=low_gain))
    if mid_gain != 0.0:
        filters.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=mid_gain, q=1.0))
    if high_gain != 0.0:
        filters.append(HighShelfFilter(cutoff_frequency_hz=4000, gain_db=high_gain))

    if not filters:
        return audio

    board = Pedalboard(filters)
    return board(audio, sr)


def apply_reverb(audio: np.ndarray, sr: int, room_size: float = 0.0, damping: float = 0.5, wet_level: float = 0.0) -> np.ndarray:
    """Apply plate reverb."""
    if wet_level <= 0.0:
        return audio

    from pedalboard import Reverb, Pedalboard

    board = Pedalboard([Reverb(room_size=room_size, damping=damping, wet_level=wet_level)])
    return board(audio, sr)


def apply_delay(audio: np.ndarray, sr: int, delay_seconds: float = 0.0, feedback: float = 0.0, mix: float = 0.0) -> np.ndarray:
    """Apply stereo delay effect."""
    if mix <= 0.0 or delay_seconds <= 0.0:
        return audio

    from pedalboard import Delay, Pedalboard

    delay_ms = delay_seconds * 1000.0
    board = Pedalboard([Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix)])
    return board(audio, sr)


def apply_compressor(audio: np.ndarray, sr: int, threshold_db: float = 0.0, ratio: float = 1.0, attack_ms: float = 10.0) -> np.ndarray:
    """Apply dynamic range compression."""
    if ratio <= 1.0 or threshold_db >= 0.0:
        return audio

    from pedalboard import Compressor, Pedalboard

    board = Pedalboard([Compressor(threshold_db=threshold_db, ratio=ratio, attack_ms=attack_ms)])
    return board(audio, sr)


def apply_pan(audio: np.ndarray, sr: int, pan: float = 0.0) -> np.ndarray:
    """Apply stereo panning. -1.0=full left, 0.0=center, 1.0=full right.

    For mono input, converts to stereo first.
    For stereo input, adjusts L/R gain balance.
    """
    if pan == 0.0:
        return audio

    pan = np.clip(pan, -1.0, 1.0)

    if audio.ndim == 1:
        # Mono to stereo with panning
        left_gain = np.cos((pan + 1.0) * np.pi / 4.0)
        right_gain = np.sin((pan + 1.0) * np.pi / 4.0)
        stereo = np.stack([audio * left_gain, audio * right_gain], axis=0)
        return stereo.astype(np.float32)

    # Stereo: adjust L/R balance
    left_gain = np.cos((pan + 1.0) * np.pi / 4.0)
    right_gain = np.sin((pan + 1.0) * np.pi / 4.0)
    result = audio.copy()
    result[0] = audio[0] * left_gain
    result[1] = audio[1] * right_gain
    return result.astype(np.float32)


def apply_effects(audio: np.ndarray, sr: int, effects_config: dict) -> np.ndarray:
    """Apply a chain of effects to an audio stem.

    Args:
        audio: shape (channels, samples) or (samples,), float32
        sr: sample rate
        effects_config: dict with keys matching DEFAULT_EFFECTS.
            Missing keys use defaults (effect bypassed).

    Returns:
        Processed audio. If panning is applied, may change from mono to stereo.
    """
    result = audio.copy()

    eq_cfg = effects_config.get("eq", DEFAULT_EFFECTS["eq"])
    result = apply_eq(result, sr, **eq_cfg)

    rev_cfg = effects_config.get("reverb", DEFAULT_EFFECTS["reverb"])
    result = apply_reverb(result, sr, **rev_cfg)

    del_cfg = effects_config.get("delay", DEFAULT_EFFECTS["delay"])
    result = apply_delay(result, sr, **del_cfg)

    comp_cfg = effects_config.get("compressor", DEFAULT_EFFECTS["compressor"])
    result = apply_compressor(result, sr, **comp_cfg)

    pan_val = effects_config.get("pan", DEFAULT_EFFECTS["pan"])
    result = apply_pan(result, sr, pan=pan_val)

    return result


def effects_config_from_ui(eq_low=0, eq_mid=0, eq_high=0,
                           rev_room=0, rev_damp=0.5, rev_wet=0,
                           del_sec=0, del_fb=0, del_mix=0,
                           comp_thresh=0, comp_ratio=1, comp_attack=10,
                           pan=0) -> dict:
    """Build an effects_config dict from UI slider values."""
    return {
        "eq": {"low_gain": eq_low, "mid_gain": eq_mid, "high_gain": eq_high},
        "reverb": {"room_size": rev_room, "damping": rev_damp, "wet_level": rev_wet},
        "delay": {"delay_seconds": del_sec, "feedback": del_fb, "mix": del_mix},
        "compressor": {"threshold_db": comp_thresh, "ratio": comp_ratio, "attack_ms": comp_attack},
        "pan": pan,
    }
