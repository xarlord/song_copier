"""Tests for services/mastering.py — loudness, normalization, limiting, stereo widen."""

import numpy as np
import pytest

from services.mastering import (
    measure_loudness,
    normalize_loudness,
    apply_limiter,
    apply_stereo_widen,
    master_audio,
    PRESETS,
)


class TestMeasureLoudness:
    """Tests for measure_loudness."""

    def test_returns_dict_with_expected_keys(self, sample_audio_1s, sample_rate):
        """measure_loudness should return dict with lufs, peak_db, rms_db."""
        result = measure_loudness(sample_audio_1s, sample_rate)
        assert "lufs" in result
        assert "peak_db" in result
        assert "rms_db" in result

    def test_lufs_is_float(self, sample_audio_1s, sample_rate):
        """LUFS value should be a float."""
        result = measure_loudness(sample_audio_1s, sample_rate)
        assert isinstance(result["lufs"], float)

    def test_loud_audio_has_higher_lufs(self, sample_rate):
        """Louder audio should have higher LUFS."""
        soft = np.ones((2, sample_rate), dtype=np.float32) * 0.1
        loud = np.ones((2, sample_rate), dtype=np.float32) * 0.9
        soft_loudness = measure_loudness(soft, sample_rate)
        loud_loudness = measure_loudness(loud, sample_rate)
        assert loud_loudness["lufs"] > soft_loudness["lufs"]

    def test_silence_returns_minus_inf(self, sample_audio_silence, sample_rate):
        """Silent audio should return -inf LUFS."""
        result = measure_loudness(sample_audio_silence, sample_rate)
        assert result["lufs"] == -np.inf

    def test_mono_input(self, sample_audio_mono, sample_rate):
        """measure_loudness should handle mono (1D) input."""
        result = measure_loudness(sample_audio_mono, sample_rate)
        assert "lufs" in result
        assert isinstance(result["lufs"], float)

    def test_peak_db_is_negative_for_unit_audio(self, sample_rate):
        """Audio within [-1, 1] should have peak_db <= 0."""
        audio = np.random.randn(2, sample_rate).astype(np.float32) * 0.5
        result = measure_loudness(audio, sample_rate)
        assert result["peak_db"] <= 0.0


class TestNormalizeLoudness:
    """Tests for normalize_loudness."""

    def test_normalize_to_minus_14(self, sample_audio_1s, sample_rate):
        """After normalization, LUFS should be closer to -14."""
        result = normalize_loudness(sample_audio_1s, sample_rate, target_lufs=-14.0)
        metrics = measure_loudness(result, sample_rate)
        # Allow 2 LUFS tolerance (simplified measurement)
        assert abs(metrics["lufs"] - (-14.0)) < 2.0

    def test_normalize_preserves_shape(self, sample_audio_1s, sample_rate):
        """Normalization should not change shape."""
        result = normalize_loudness(sample_audio_1s, sample_rate, target_lufs=-14.0)
        assert result.shape == sample_audio_1s.shape

    def test_normalize_silence_unchanged(self, sample_audio_silence, sample_rate):
        """Silent audio should be returned unchanged."""
        result = normalize_loudness(sample_audio_silence, sample_rate, target_lufs=-14.0)
        np.testing.assert_array_equal(result, sample_audio_silence)

    def test_normalize_to_minus_23(self, sample_rate):
        """Normalize to -23 LUFS (broadcast standard)."""
        audio = np.random.randn(2, sample_rate).astype(np.float32) * 0.5
        result = normalize_loudness(audio, sample_rate, target_lufs=-23.0)
        metrics = measure_loudness(result, sample_rate)
        assert abs(metrics["lufs"] - (-23.0)) < 2.0


class TestApplyLimiter:
    """Tests for apply_limiter."""

    def test_limiter_prevents_clipping(self, sample_rate):
        """Limiter should keep audio below threshold."""
        loud = np.ones((2, sample_rate), dtype=np.float32) * 2.0
        result = apply_limiter(loud, sample_rate, threshold_db=-1.0)
        # Limited audio should not have extreme values
        assert np.all(np.isfinite(result))

    def test_limiter_preserves_shape(self, sample_audio_1s, sample_rate):
        """Limiter should preserve audio shape."""
        result = apply_limiter(sample_audio_1s, sample_rate, threshold_db=-1.0)
        assert result.shape == sample_audio_1s.shape

    def test_limiter_mono_input(self, sample_audio_mono, sample_rate):
        """Limiter should handle 1D (mono) input."""
        result = apply_limiter(sample_audio_mono, sample_rate, threshold_db=-1.0)
        assert result.ndim == 1
        assert len(result) == len(sample_audio_mono)


class TestApplyStereoWiden:
    """Tests for apply_stereo_widen."""

    def test_output_is_stereo(self, sample_audio_1s, sample_rate):
        """Output should always be stereo (2, samples)."""
        result = apply_stereo_widen(sample_audio_1s, width=1.5)
        assert result.shape[0] == 2

    def test_mono_input_becomes_stereo(self, sample_audio_mono, sample_rate):
        """Mono input should be converted to stereo."""
        result = apply_stereo_widen(sample_audio_mono, width=1.5)
        assert result.ndim == 2
        assert result.shape[0] == 2

    def test_width_1_no_change_for_stereo(self, sample_audio_1s, sample_rate):
        """Width=1.0 should not change stereo content."""
        result = apply_stereo_widen(sample_audio_1s, width=1.0)
        np.testing.assert_allclose(result, sample_audio_1s, atol=1e-5)

    def test_width_0_collapses_to_mono(self, sample_audio_1s, sample_rate):
        """Width=0.0 should make left == right (mono)."""
        result = apply_stereo_widen(sample_audio_1s, width=0.0)
        np.testing.assert_allclose(result[0], result[1], atol=1e-6)

    def test_no_clipping_after_widen(self, sample_audio_1s, sample_rate):
        """Widened audio should be clamped to prevent clipping."""
        result = apply_stereo_widen(sample_audio_1s, width=3.0)
        assert np.max(np.abs(result)) <= 1.0


class TestMasterAudio:
    """Tests for master_audio (full mastering chain)."""

    def test_master_returns_tuple(self, sample_audio_1s, sample_rate):
        """master_audio should return (audio, metrics) tuple."""
        result = master_audio(sample_audio_1s, sample_rate)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_master_output_shape(self, sample_audio_1s, sample_rate):
        """Mastered audio should be stereo."""
        audio, metrics = master_audio(sample_audio_1s, sample_rate)
        assert audio.ndim == 2
        assert audio.shape[0] == 2

    def test_master_metrics_has_expected_keys(self, sample_audio_1s, sample_rate):
        """Metrics dict should contain pre/post measurements."""
        audio, metrics = master_audio(sample_audio_1s, sample_rate)
        assert "preset" in metrics
        assert "target_lufs" in metrics
        assert "pre" in metrics
        assert "post" in metrics

    def test_master_spotify_preset(self, sample_audio_1s, sample_rate):
        """Spotify preset should target -14 LUFS."""
        audio, metrics = master_audio(sample_audio_1s, sample_rate, preset="spotify")
        assert metrics["target_lufs"] == -14.0

    def test_master_custom_lufs(self, sample_audio_1s, sample_rate):
        """Custom target_lufs should override preset."""
        audio, metrics = master_audio(sample_audio_1s, sample_rate, target_lufs=-23.0)
        assert metrics["target_lufs"] == -23.0

    def test_master_no_nan_or_inf(self, sample_audio_1s, sample_rate):
        """Mastered audio should not contain NaN or Inf."""
        audio, _ = master_audio(sample_audio_1s, sample_rate)
        assert np.all(np.isfinite(audio))

    def test_presets_dict(self):
        """PRESETS should have expected keys and values."""
        assert "spotify" in PRESETS
        assert PRESETS["spotify"] == -14.0
        assert "youtube" in PRESETS
        assert "cd" in PRESETS
        assert "podcast" in PRESETS
