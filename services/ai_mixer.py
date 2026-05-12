"""AI Mixing Assistant — heuristic-based stem analysis and mix suggestions.

Analyzes individual stems using audio features (RMS, peak, spectral centroid)
and suggests balanced volume, EQ, reverb, compressor, and pan settings.
Does NOT use an LLM — all logic is rule-based audio analysis heuristics.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stem-role defaults (the "mixing recipe")
# ---------------------------------------------------------------------------
ROLE_DEFAULTS: dict[str, dict] = {
    "vocals": {
        "volume": 1.0,
        "pan": 0.0,
        "reverb": {"room_size": 0.3, "damping": 0.5, "wet": 0.2},
        "compressor": {"threshold": 0.0, "ratio": 1.0},  # no compression
        "eq": {"low": 0.0, "mid": 0.0, "high": 0.0},
    },
    "drums": {
        "volume": 0.85,
        "pan": 0.0,
        "reverb": {"room_size": 0.0, "damping": 0.5, "wet": 0.0},
        "compressor": {"threshold": -10.0, "ratio": 4.0},
        "eq": {"low": 0.0, "mid": 0.0, "high": 0.0},
    },
    "bass": {
        "volume": 0.8,
        "pan": 0.0,
        "reverb": {"room_size": 0.0, "damping": 0.5, "wet": 0.0},
        "compressor": {"threshold": -8.0, "ratio": 3.0},
        "eq": {"low": 2.0, "mid": 0.0, "high": 0.0},
    },
    "guitar": {
        "volume": 0.75,
        "pan": 0.3,  # will be negated for one side in a pair
        "reverb": {"room_size": 0.2, "damping": 0.5, "wet": 0.1},
        "compressor": {"threshold": 0.0, "ratio": 1.0},
        "eq": {"low": 0.0, "mid": 2.0, "high": 0.0},
    },
    "piano": {
        "volume": 0.7,
        "pan": 0.2,
        "reverb": {"room_size": 0.25, "damping": 0.5, "wet": 0.15},
        "compressor": {"threshold": 0.0, "ratio": 1.0},
        "eq": {"low": 0.0, "mid": 0.0, "high": 0.0},
    },
    "other": {
        "volume": 0.65,
        "pan": 0.1,
        "reverb": {"room_size": 0.15, "damping": 0.5, "wet": 0.1},
        "compressor": {"threshold": 0.0, "ratio": 1.0},
        "eq": {"low": 0.0, "mid": 0.0, "high": 0.0},
    },
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is 1-D mono. Accepts (samples,) or (channels, samples)."""
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def _rms_db(audio: np.ndarray) -> float:
    """Compute RMS level in dB. Returns -120 for silence."""
    mono = _to_mono(audio)
    rms = np.sqrt(np.mean(mono ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20.0 * np.log10(rms))


def _peak_db(audio: np.ndarray) -> float:
    """Compute peak level in dB. Returns -120 for silence."""
    peak = np.abs(_to_mono(audio)).max()
    if peak < 1e-10:
        return -120.0
    return float(20.0 * np.log10(peak))


def _spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Compute spectral centroid in Hz (brightness indicator)."""
    mono = _to_mono(audio)
    n_fft = min(2048, len(mono))
    if n_fft < 4:
        return 0.0

    window = np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(mono[:n_fft] * window))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    mag_sum = spectrum.sum()
    if mag_sum < 1e-10:
        return 0.0

    return float((freqs * spectrum).sum() / mag_sum)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_stem(audio: np.ndarray, sr: int, stem_name: str) -> dict:
    """Analyze a single stem and return measured features + suggested settings.

    Args:
        audio: waveform array, shape (channels, samples) or (samples,), float32
        sr: sample rate
        stem_name: one of vocals, drums, bass, guitar, piano, other

    Returns:
        dict with keys:
            rms_db, peak_db, spectral_centroid,
            suggested_volume, suggested_eq, suggested_reverb,
            suggested_compressor, suggested_pan
    """
    stem_name = stem_name.lower().strip()
    defaults = ROLE_DEFAULTS.get(stem_name, ROLE_DEFAULTS["other"])

    rms = _rms_db(audio)
    peak = _peak_db(audio)
    centroid = _spectral_centroid(audio, sr)

    # --- Volume auto-balance: louder stems get reduced volume ---
    # Reference level: -18 dBFS is a typical healthy stem level.
    # For every dB above -18, reduce suggested volume by 3%.
    volume_offset = 0.0
    if rms > -120.0:
        delta_db = rms - (-18.0)
        volume_offset = -delta_db * 0.03  # negative = lower volume for loud stems

    suggested_volume = float(np.clip(defaults["volume"] + volume_offset, 0.05, 1.5))

    # --- EQ fine-tuning based on spectral centroid ---
    eq_suggestion = dict(defaults["eq"])
    if centroid > 0:
        # Very bright stem → tame highs slightly
        if centroid > 5000.0:
            eq_suggestion["high"] = eq_suggestion.get("high", 0.0) - 1.0
        # Very dark stem → boost highs slightly
        elif centroid < 800.0 and stem_name not in ("bass",):
            eq_suggestion["high"] = eq_suggestion.get("high", 0.0) + 1.0

    # --- Reverb (copy defaults) ---
    suggested_reverb = dict(defaults["reverb"])

    # --- Compressor (copy defaults) ---
    suggested_compressor = dict(defaults["compressor"])

    # Adjust compressor threshold based on dynamics (peak - rms = crest factor)
    crest = peak - rms if (peak > -120 and rms > -120) else 0.0
    if suggested_compressor["ratio"] > 1.0 and crest > 15.0:
        # Very dynamic — lower threshold a bit to catch more peaks
        suggested_compressor["threshold"] -= 2.0

    # --- Pan (copy defaults) ---
    suggested_pan = defaults["pan"]

    result = {
        "rms_db": round(rms, 1),
        "peak_db": round(peak, 1),
        "spectral_centroid": round(centroid, 1),
        "suggested_volume": round(suggested_volume, 3),
        "suggested_eq": {
            "low": round(eq_suggestion["low"], 1),
            "mid": round(eq_suggestion["mid"], 1),
            "high": round(eq_suggestion["high"], 1),
        },
        "suggested_reverb": {
            "room_size": round(suggested_reverb["room_size"], 2),
            "damping": round(suggested_reverb["damping"], 2),
            "wet": round(suggested_reverb["wet"], 2),
        },
        "suggested_compressor": {
            "threshold": round(suggested_compressor["threshold"], 1),
            "ratio": round(suggested_compressor["ratio"], 1),
        },
        "suggested_pan": round(suggested_pan, 2),
    }

    logger.info(
        f"Analyze '{stem_name}': RMS={rms:.1f}dB  Peak={peak:.1f}dB  "
        f"Centroid={centroid:.0f}Hz  Vol={suggested_volume:.2f}  Pan={suggested_pan}"
    )
    return result


# ---------------------------------------------------------------------------
# Full mix suggestion
# ---------------------------------------------------------------------------

def suggest_mix(stems: dict[str, np.ndarray], sr: int) -> dict[str, dict]:
    """Analyze all stems and return per-stem mix suggestions.

    Rules:
      - Vocals loudest, drums 2nd, bass centered
      - Guitars panned, reverb on vocals only (role defaults)
      - Compressor on bass/drums only
      - Auto-balance: louder stems get reduced volume to normalize

    Args:
        stems: dict mapping stem name → waveform array
        sr: sample rate

    Returns:
        dict mapping stem name → analysis dict (same shape as analyze_stem output)
    """
    suggestions: dict[str, dict] = {}

    # First pass: analyze each stem independently
    analyses: dict[str, dict] = {}
    for name, audio in stems.items():
        analyses[name] = analyze_stem(audio, sr, name)

    # Second pass: global RMS-based normalization
    # Compute average RMS across all stems
    rms_values = [a["rms_db"] for a in analyses.values() if a["rms_db"] > -120.0]
    if rms_values:
        avg_rms = float(np.mean(rms_values))
    else:
        avg_rms = -18.0

    for name, analysis in analyses.items():
        if analysis["rms_db"] <= -120.0:
            # Silent stem — keep defaults
            suggestions[name] = analysis
            continue

        # Additional volume adjustment: if this stem is louder than average,
        # reduce volume proportionally.
        delta_from_avg = analysis["rms_db"] - avg_rms
        global_offset = -delta_from_avg * 0.05  # gentler than per-stem
        adjusted_vol = float(np.clip(
            analysis["suggested_volume"] + global_offset, 0.05, 1.5
        ))
        analysis["suggested_volume"] = round(adjusted_vol, 3)

        # Panning: alternate guitars and "other" for stereo width
        if name == "guitar":
            analysis["suggested_pan"] = 0.3
        elif name == "other":
            analysis["suggested_pan"] = -0.1

        suggestions[name] = analysis

    logger.info(
        f"Mix suggestions for {len(suggestions)} stems: "
        + ", ".join(f"{n} vol={s['suggested_volume']:.2f}" for n, s in suggestions.items())
    )
    return suggestions


# ---------------------------------------------------------------------------
# Apply suggestions
# ---------------------------------------------------------------------------

def apply_suggested_mix(
    stems: dict[str, np.ndarray],
    sr: int,
    suggestions: dict[str, dict],
) -> np.ndarray:
    """Apply all suggested settings to stems and return the final stereo mix.

    Processing order per stem:
      1. Volume scaling
      2. EQ (via services.effects.apply_eq)
      3. Reverb (via services.effects.apply_reverb)
      4. Compressor (via services.effects.apply_compressor)
      5. Pan (via services.effects.apply_pan)

    All stems are then summed and normalized.

    Args:
        stems: dict mapping stem name → waveform array (mono or stereo)
        sr: sample rate
        suggestions: output of suggest_mix()

    Returns:
        Mixed stereo audio as np.ndarray, shape (2, samples), float32.
    """
    from services.effects import apply_eq, apply_reverb, apply_compressor, apply_pan

    max_length = max(s.shape[-1] for s in stems.values())
    mix = np.zeros((2, max_length), dtype=np.float64)

    for name, audio in stems.items():
        sugg = suggestions.get(name)
        if sugg is None:
            # No suggestion — add at full volume
            stem = audio.astype(np.float64)
        else:
            stem = audio.copy().astype(np.float64)

            # 1. Volume
            vol = sugg["suggested_volume"]
            stem = stem * vol

            # 2. EQ
            eq = sugg["suggested_eq"]
            if any(v != 0.0 for v in eq.values()):
                stem = apply_eq(stem.astype(np.float32), sr, **eq).astype(np.float64)

            # 3. Reverb
            rev = sugg["suggested_reverb"]
            if rev["wet"] > 0.0:
                stem = apply_reverb(
                    stem.astype(np.float32), sr,
                    room_size=rev["room_size"],
                    damping=rev["damping"],
                    wet_level=rev["wet"],
                ).astype(np.float64)

            # 4. Compressor
            comp = sugg["suggested_compressor"]
            if comp["ratio"] > 1.0 and comp["threshold"] < 0.0:
                stem = apply_compressor(
                    stem.astype(np.float32), sr,
                    threshold_db=comp["threshold"],
                    ratio=comp["ratio"],
                ).astype(np.float64)

            # 5. Pan
            pan_val = sugg["suggested_pan"]
            if pan_val != 0.0:
                stem = apply_pan(stem.astype(np.float32), sr, pan=pan_val)

        # Ensure stereo for mixing
        if stem.ndim == 1:
            stem = np.stack([stem, stem], axis=0)
        elif stem.shape[0] == 1:
            stem = np.concatenate([stem, stem], axis=0)

        stem = stem.astype(np.float64)
        slen = stem.shape[-1]
        mix[:, :slen] += stem

    # Normalize to prevent clipping
    peak = np.abs(mix).max()
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    return mix.astype(np.float32)
