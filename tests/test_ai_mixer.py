"""Tests for services/ai_mixer.py — stem analysis, mix suggestions."""

import numpy as np
import pytest

from services.ai_mixer import (
    analyze_stem,
    suggest_mix,
    _to_mono,
    _rms_db,
    _peak_db,
    _spectral_centroid,
    ROLE_DEFAULTS,
)


class TestHelpers:
    """Tests for internal helper functions."""

    def test_to_mono_stereo(self, sample_audio_1s):
        """Stereo → mono conversion."""
        mono = _to_mono(sample_audio_1s)
        assert mono.ndim == 1
        assert mono.shape[0] == sample_audio_1s.shape[1]

    def test_to_mono_passthrough(self, sample_audio_mono):
        """Mono input should pass through unchanged."""
        mono = _to_mono(sample_audio_mono)
        np.testing.assert_array_equal(mono, sample_audio_mono)

    def test_rms_db_loud_audio(self):
        """Loud audio should have higher RMS dB."""
        loud = np.ones(44100, dtype=np.float32) * 0.5
        rms = _rms_db(loud)
        assert rms > -10.0

    def test_rms_db_silence(self):
        """Silent audio should return -120 dB."""
        silence = np.zeros(44100, dtype=np.float32)
        rms = _rms_db(silence)
        assert rms == -120.0

    def test_peak_db(self):
        """Peak dB should be correct for known amplitude."""
        audio = np.ones(44100, dtype=np.float32) * 0.5
        peak = _peak_db(audio)
        assert abs(peak - (-6.02)) < 0.1

    def test_peak_db_silence(self):
        """Silent audio peak should be -120 dB."""
        silence = np.zeros(44100, dtype=np.float32)
        peak = _peak_db(silence)
        assert peak == -120.0

    def test_spectral_centroid_returns_float(self, sample_audio_1s, sample_rate):
        """Spectral centroid should return a float."""
        centroid = _spectral_centroid(sample_audio_1s, sample_rate)
        assert isinstance(centroid, float)
        assert centroid >= 0.0


class TestAnalyzeStem:
    """Tests for analyze_stem."""

    def test_returns_expected_keys(self, sample_audio_1s, sample_rate):
        """analyze_stem should return dict with all expected keys."""
        result = analyze_stem(sample_audio_1s, sample_rate, "vocals")
        expected_keys = {
            "rms_db", "peak_db", "spectral_centroid",
            "suggested_volume", "suggested_eq",
            "suggested_reverb", "suggested_compressor",
            "suggested_pan",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_rms_db_is_float(self, sample_audio_1s, sample_rate):
        """rms_db should be a float."""
        result = analyze_stem(sample_audio_1s, sample_rate, "drums")
        assert isinstance(result["rms_db"], float)

    def test_suggested_volume_in_range(self, sample_audio_1s, sample_rate):
        """Suggested volume should be between 0.05 and 1.5."""
        for stem_name in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
            result = analyze_stem(sample_audio_1s, sample_rate, stem_name)
            assert 0.05 <= result["suggested_volume"] <= 1.5

    def test_eq_has_low_mid_high(self, sample_audio_1s, sample_rate):
        """suggested_eq should have low, mid, high keys."""
        result = analyze_stem(sample_audio_1s, sample_rate, "vocals")
        assert "low" in result["suggested_eq"]
        assert "mid" in result["suggested_eq"]
        assert "high" in result["suggested_eq"]

    def test_reverb_has_expected_keys(self, sample_audio_1s, sample_rate):
        """suggested_reverb should have room_size, damping, wet."""
        result = analyze_stem(sample_audio_1s, sample_rate, "vocals")
        assert "room_size" in result["suggested_reverb"]
        assert "damping" in result["suggested_reverb"]
        assert "wet" in result["suggested_reverb"]

    def test_unknown_stem_uses_other_defaults(self, sample_audio_1s, sample_rate):
        """Unknown stem name should fall back to 'other' defaults."""
        result = analyze_stem(sample_audio_1s, sample_rate, "unknown_instrument")
        assert isinstance(result, dict)
        assert "suggested_volume" in result

    def test_silence_handling(self, sample_rate):
        """Silent stem should return very low RMS."""
        silence = np.zeros((2, sample_rate), dtype=np.float32)
        result = analyze_stem(silence, sample_rate, "vocals")
        assert result["rms_db"] == -120.0

    def test_mono_input(self, sample_audio_mono, sample_rate):
        """Mono (1D) input should work."""
        result = analyze_stem(sample_audio_mono, sample_rate, "vocals")
        assert isinstance(result["rms_db"], float)


class TestSuggestMix:
    """Tests for suggest_mix."""

    def test_returns_dict_per_stem(self, sample_stems, sample_rate):
        """suggest_mix should return a dict with one entry per stem."""
        result = suggest_mix(sample_stems, sample_rate)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(sample_stems.keys())

    def test_each_entry_has_expected_keys(self, sample_stems, sample_rate):
        """Each entry should have the same keys as analyze_stem."""
        result = suggest_mix(sample_stems, sample_rate)
        for name, analysis in result.items():
            assert "suggested_volume" in analysis
            assert "suggested_eq" in analysis
            assert "suggested_pan" in analysis

    def test_vocals_generally_loudest(self, sample_rate):
        """Vocals should generally have highest suggested volume."""
        rng = np.random.default_rng(42)
        stems = {
            "vocals": (rng.standard_normal((2, sample_rate)) * 0.3).astype(np.float32),
            "drums": (rng.standard_normal((2, sample_rate)) * 0.3).astype(np.float32),
            "bass": (rng.standard_normal((2, sample_rate)) * 0.3).astype(np.float32),
        }
        result = suggest_mix(stems, sample_rate)
        assert result["vocals"]["suggested_volume"] >= result["bass"]["suggested_volume"]

    def test_empty_stems(self, sample_rate):
        """Empty stems dict should return empty suggestions."""
        result = suggest_mix({}, sample_rate)
        assert result == {}

    def test_role_defaults_exist(self):
        """ROLE_DEFAULTS should have entries for standard stem types."""
        for name in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
            assert name in ROLE_DEFAULTS
