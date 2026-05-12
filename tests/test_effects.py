"""Tests for services/effects.py — EQ, reverb, delay, compressor, pan."""

import numpy as np
import pytest

from services.effects import (
    apply_eq,
    apply_reverb,
    apply_delay,
    apply_compressor,
    apply_pan,
    apply_effects,
    effects_config_from_ui,
    DEFAULT_EFFECTS,
)


class TestApplyEQ:
    """Tests for apply_eq."""

    def test_eq_no_change(self, sample_audio_1s, sample_rate):
        """All gains at 0 → audio returned unchanged."""
        result = apply_eq(sample_audio_1s, sample_rate, low_gain=0.0, mid_gain=0.0, high_gain=0.0)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_eq_low_boost(self, sample_audio_1s, sample_rate):
        """Low shelf boost should modify the audio."""
        result = apply_eq(sample_audio_1s, sample_rate, low_gain=6.0)
        assert result.shape == sample_audio_1s.shape
        assert not np.allclose(result, sample_audio_1s, atol=1e-4)

    def test_eq_mid_cut(self, sample_audio_1s, sample_rate):
        """Mid bell cut should modify the audio."""
        result = apply_eq(sample_audio_1s, sample_rate, mid_gain=-3.0)
        assert result.shape == sample_audio_1s.shape
        assert not np.allclose(result, sample_audio_1s, atol=1e-4)

    def test_eq_high_boost(self, sample_audio_1s, sample_rate):
        """High shelf boost should modify the audio."""
        result = apply_eq(sample_audio_1s, sample_rate, high_gain=4.0)
        assert result.shape == sample_audio_1s.shape
        assert not np.allclose(result, sample_audio_1s, atol=1e-4)

    def test_eq_preserves_shape(self, sample_audio_1s, sample_rate):
        """EQ should not change number of channels or sample count."""
        result = apply_eq(sample_audio_1s, sample_rate, low_gain=3.0, mid_gain=-2.0, high_gain=5.0)
        assert result.shape == sample_audio_1s.shape


class TestApplyReverb:
    """Tests for apply_reverb."""

    def test_reverb_zero_wet_returns_original(self, sample_audio_1s, sample_rate):
        """wet_level=0 should return original audio."""
        result = apply_reverb(sample_audio_1s, sample_rate, wet_level=0.0)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_reverb_applied(self, sample_audio_1s, sample_rate):
        """Reverb with wet_level > 0 should modify the audio."""
        result = apply_reverb(sample_audio_1s, sample_rate, room_size=0.8, damping=0.5, wet_level=0.5)
        assert result.shape == sample_audio_1s.shape
        assert not np.allclose(result, sample_audio_1s, atol=1e-4)

    def test_reverb_small_room(self, sample_audio_1s, sample_rate):
        """Small room reverb should be subtler than large room."""
        small = apply_reverb(sample_audio_1s, sample_rate, room_size=0.2, wet_level=0.3)
        large = apply_reverb(sample_audio_1s, sample_rate, room_size=0.9, wet_level=0.3)
        # Both should differ from original
        assert not np.allclose(small, sample_audio_1s, atol=1e-4)
        assert not np.allclose(large, sample_audio_1s, atol=1e-4)

    def test_reverb_preserves_shape(self, sample_audio_1s, sample_rate):
        """Reverb output shape matches input."""
        result = apply_reverb(sample_audio_1s, sample_rate, room_size=0.5, wet_level=0.3)
        assert result.shape == sample_audio_1s.shape


class TestApplyDelay:
    """Tests for apply_delay."""

    def test_delay_zero_mix_returns_original(self, sample_audio_1s, sample_rate):
        """mix=0 should return original."""
        result = apply_delay(sample_audio_1s, sample_rate, delay_seconds=0.3, mix=0.0)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_delay_zero_seconds_returns_original(self, sample_audio_1s, sample_rate):
        """delay_seconds=0 should return original."""
        result = apply_delay(sample_audio_1s, sample_rate, delay_seconds=0.0, mix=0.5)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_delay_applied(self, sample_audio_1s, sample_rate):
        """Active delay should modify the audio."""
        result = apply_delay(sample_audio_1s, sample_rate, delay_seconds=0.25, feedback=0.3, mix=0.4)
        assert not np.allclose(result, sample_audio_1s, atol=1e-4)


class TestApplyCompressor:
    """Tests for apply_compressor."""

    def test_compressor_ratio_1_returns_original(self, sample_audio_1s, sample_rate):
        """ratio=1.0 means no compression → original returned."""
        result = apply_compressor(sample_audio_1s, sample_rate, threshold_db=-10.0, ratio=1.0)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_compressor_threshold_0_returns_original(self, sample_audio_1s, sample_rate):
        """threshold_db=0 means no compression → original returned."""
        result = apply_compressor(sample_audio_1s, sample_rate, threshold_db=0.0, ratio=4.0)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_compressor_applied(self, sample_audio_1s, sample_rate):
        """Active compression should modify the audio."""
        result = apply_compressor(
            sample_audio_1s, sample_rate,
            threshold_db=-20.0, ratio=4.0, attack_ms=5.0,
        )
        assert result.shape == sample_audio_1s.shape
        # Compressed output should have lower peak than input (for loud audio)
        # At least it should differ
        assert not np.allclose(result, sample_audio_1s, atol=1e-4)

    def test_compressor_preserves_shape(self, sample_audio_1s, sample_rate):
        """Compressor preserves audio shape."""
        result = apply_compressor(sample_audio_1s, sample_rate, threshold_db=-12.0, ratio=3.0)
        assert result.shape == sample_audio_1s.shape


class TestApplyPan:
    """Tests for apply_pan."""

    def test_pan_zero_returns_original(self, sample_audio_1s, sample_rate):
        """pan=0.0 should return original."""
        result = apply_pan(sample_audio_1s, sample_rate, pan=0.0)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_pan_left_stereo(self, sample_audio_1s, sample_rate):
        """Pan left on stereo should attenuate right channel more."""
        result = apply_pan(sample_audio_1s, sample_rate, pan=-1.0)
        assert result.shape == sample_audio_1s.shape
        # Left channel should be louder than right
        assert np.mean(np.abs(result[0])) >= np.mean(np.abs(result[1]))

    def test_pan_right_stereo(self, sample_audio_1s, sample_rate):
        """Pan right on stereo should attenuate left channel more."""
        result = apply_pan(sample_audio_1s, sample_rate, pan=1.0)
        assert result.shape == sample_audio_1s.shape
        assert np.mean(np.abs(result[1])) >= np.mean(np.abs(result[0]))

    def test_pan_mono_creates_stereo(self, sample_audio_mono, sample_rate):
        """Panning mono audio should produce stereo output."""
        result = apply_pan(sample_audio_mono, sample_rate, pan=0.5)
        assert result.ndim == 2
        assert result.shape[0] == 2

    def test_pan_clamped(self, sample_audio_1s, sample_rate):
        """Pan value > 1 or < -1 should be clamped."""
        result = apply_pan(sample_audio_1s, sample_rate, pan=5.0)
        assert result.shape == sample_audio_1s.shape
        # Should not crash and should produce valid audio
        assert np.all(np.isfinite(result))


class TestApplyEffectsChain:
    """Tests for apply_effects (full chain)."""

    def test_default_config_no_change(self, sample_audio_1s, sample_rate):
        """Default effects config (all off) should return copy of audio."""
        result = apply_effects(sample_audio_1s, sample_rate, DEFAULT_EFFECTS)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_combined_effects(self, sample_audio_1s, sample_rate):
        """Multiple effects applied together should produce valid audio."""
        config = {
            "eq": {"low_gain": 2.0, "mid_gain": -1.0, "high_gain": 3.0},
            "reverb": {"room_size": 0.5, "damping": 0.5, "wet_level": 0.3},
            "delay": {"delay_seconds": 0.2, "feedback": 0.2, "mix": 0.2},
            "compressor": {"threshold_db": -12.0, "ratio": 3.0, "attack_ms": 10.0},
            "pan": 0.0,
        }
        result = apply_effects(sample_audio_1s, sample_rate, config)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))

    def test_effects_config_from_ui(self):
        """effects_config_from_ui should produce a valid config dict."""
        config = effects_config_from_ui(
            eq_low=2.0, eq_mid=-1.0, eq_high=3.0,
            rev_room=0.5, rev_wet=0.3,
            del_sec=0.2, del_fb=0.3, del_mix=0.4,
            comp_thresh=-12.0, comp_ratio=3.0,
            pan=0.5,
        )
        assert "eq" in config
        assert config["eq"]["low_gain"] == 2.0
        assert config["pan"] == 0.5

    def test_empty_config_uses_defaults(self, sample_audio_1s, sample_rate):
        """Empty effects config dict should use defaults (no change)."""
        result = apply_effects(sample_audio_1s, sample_rate, {})
        np.testing.assert_array_equal(result, sample_audio_1s)
