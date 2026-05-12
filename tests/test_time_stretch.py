"""Tests for services/time_stretch.py — time stretch, pitch shift, process_stem."""

import numpy as np
import pytest

from services.time_stretch import time_stretch, pitch_shift, process_stem


class TestTimeStretch:
    """Tests for time_stretch."""

    def test_rate_1_returns_original(self, sample_audio_1s, sample_rate):
        """Rate 1.0 should return original audio."""
        result = time_stretch(sample_audio_1s, sample_rate, rate=1.0)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_stretch_2x_doubles_length(self, sample_audio_1s, sample_rate):
        """Rate 0.5 (half speed) should approximately double the length."""
        result = time_stretch(sample_audio_1s, sample_rate, rate=0.5)
        expected_len = int(sample_audio_1s.shape[-1] * 2)
        # Allow 5% tolerance for phase vocoder artifacts
        assert abs(result.shape[-1] - expected_len) < expected_len * 0.05

    def test_stretch_half_halves_length(self, sample_audio_1s, sample_rate):
        """Rate 2.0 (double speed) should approximately halve the length."""
        result = time_stretch(sample_audio_1s, sample_rate, rate=2.0)
        expected_len = int(sample_audio_1s.shape[-1] * 0.5)
        assert abs(result.shape[-1] - expected_len) < expected_len * 0.1

    def test_mono_input(self, sample_audio_mono, sample_rate):
        """Mono input should return mono output."""
        result = time_stretch(sample_audio_mono, sample_rate, rate=1.5)
        assert result.ndim == 1

    def test_stereo_preserves_channels(self, sample_audio_1s, sample_rate):
        """Stereo input should preserve channel count."""
        result = time_stretch(sample_audio_1s, sample_rate, rate=1.5)
        assert result.shape[0] == 2

    def test_output_is_valid_audio(self, sample_audio_1s, sample_rate):
        """Stretched audio should not contain NaN or Inf."""
        result = time_stretch(sample_audio_1s, sample_rate, rate=0.7)
        assert np.all(np.isfinite(result))


class TestPitchShift:
    """Tests for pitch_shift."""

    def test_zero_semitones_returns_original(self, sample_audio_1s, sample_rate):
        """0 semitones should return original."""
        result = pitch_shift(sample_audio_1s, sample_rate, semitones=0.0)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_shift_up_12_semitones(self, sample_audio_1s, sample_rate):
        """+12 semitones (octave up) should produce valid audio."""
        result = pitch_shift(sample_audio_1s, sample_rate, semitones=12.0)
        assert result.shape == sample_audio_1s.shape
        assert np.all(np.isfinite(result))

    def test_shift_down_12_semitones(self, sample_audio_1s, sample_rate):
        """-12 semitones (octave down) should produce valid audio."""
        result = pitch_shift(sample_audio_1s, sample_rate, semitones=-12.0)
        assert result.shape == sample_audio_1s.shape
        assert np.all(np.isfinite(result))

    def test_mono_input(self, sample_audio_mono, sample_rate):
        """Mono pitch shift should return mono."""
        result = pitch_shift(sample_audio_mono, sample_rate, semitones=5.0)
        assert result.ndim == 1

    def test_stereo_preserves_channels(self, sample_audio_1s, sample_rate):
        """Stereo pitch shift should preserve channel count."""
        result = pitch_shift(sample_audio_1s, sample_rate, semitones=5.0)
        assert result.shape[0] == 2

    def test_small_shift(self, sample_audio_1s, sample_rate):
        """Small pitch shift (1 semitone) should work."""
        result = pitch_shift(sample_audio_1s, sample_rate, semitones=1.0)
        assert np.all(np.isfinite(result))


class TestProcessStem:
    """Tests for process_stem."""

    def test_defaults_return_original(self, sample_audio_1s, sample_rate):
        """Default params should return original audio."""
        result = process_stem(sample_audio_1s, sample_rate)
        np.testing.assert_array_equal(result, sample_audio_1s)

    def test_pitch_only(self, sample_audio_1s, sample_rate):
        """Only pitch shift should be applied."""
        result = process_stem(sample_audio_1s, sample_rate, pitch_semitones=3.0)
        assert result.shape == sample_audio_1s.shape
        assert np.all(np.isfinite(result))

    def test_stretch_only(self, sample_audio_1s, sample_rate):
        """Only time stretch should be applied."""
        result = process_stem(sample_audio_1s, sample_rate, stretch_rate=1.5)
        expected_len = int(sample_audio_1s.shape[-1] / 1.5)
        assert abs(result.shape[-1] - expected_len) < expected_len * 0.1

    def test_both_pitch_and_stretch(self, sample_audio_1s, sample_rate):
        """Both pitch shift and stretch should be applied."""
        result = process_stem(
            sample_audio_1s, sample_rate,
            stretch_rate=0.8, pitch_semitones=-3.0,
        )
        assert np.all(np.isfinite(result))

    def test_mono_input(self, sample_audio_mono, sample_rate):
        """Mono input should be handled."""
        result = process_stem(sample_audio_mono, sample_rate, pitch_semitones=5.0)
        assert result.ndim == 1
        assert np.all(np.isfinite(result))
