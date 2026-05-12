"""Tests for services/visualization.py — waveform, spectrogram, spectrum, comparison."""

import numpy as np
import pytest

from services.visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_spectrum,
    plot_comparison,
)


class TestPlotWaveform:
    """Tests for plot_waveform."""

    def test_returns_ndarray(self, sample_audio_1s, sample_rate):
        """plot_waveform should return a numpy ndarray."""
        result = plot_waveform(sample_audio_1s, sample_rate)
        assert isinstance(result, np.ndarray)

    def test_output_shape_hw3(self, sample_audio_1s, sample_rate):
        """Output should be (H, W, 3) RGB image."""
        result = plot_waveform(sample_audio_1s, sample_rate)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_mono_input(self, sample_audio_mono, sample_rate):
        """Mono input should produce valid image."""
        result = plot_waveform(sample_audio_mono, sample_rate)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_short_audio(self, sample_audio_short, sample_rate):
        """Very short audio should still produce valid image."""
        result = plot_waveform(sample_audio_short, sample_rate)
        assert result.ndim == 3

    def test_custom_title(self, sample_audio_1s, sample_rate):
        """Custom title should not cause errors."""
        result = plot_waveform(sample_audio_1s, sample_rate, title="My Waveform")
        assert isinstance(result, np.ndarray)


class TestPlotSpectrogram:
    """Tests for plot_spectrogram."""

    def test_returns_ndarray(self, sample_audio_1s, sample_rate):
        """plot_spectrogram should return numpy ndarray."""
        result = plot_spectrogram(sample_audio_1s, sample_rate)
        assert isinstance(result, np.ndarray)

    def test_output_shape_hw3(self, sample_audio_1s, sample_rate):
        """Output should be (H, W, 3)."""
        result = plot_spectrogram(sample_audio_1s, sample_rate)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_mono_input(self, sample_audio_mono, sample_rate):
        """Mono input should work."""
        result = plot_spectrogram(sample_audio_mono, sample_rate)
        assert result.ndim == 3

    def test_custom_title(self, sample_audio_1s, sample_rate):
        """Custom title should work."""
        result = plot_spectrogram(sample_audio_1s, sample_rate, title="My Spectrogram")
        assert isinstance(result, np.ndarray)


class TestPlotSpectrum:
    """Tests for plot_spectrum."""

    def test_returns_ndarray(self, sample_audio_1s, sample_rate):
        """plot_spectrum should return numpy ndarray."""
        result = plot_spectrum(sample_audio_1s, sample_rate)
        assert isinstance(result, np.ndarray)

    def test_output_shape_hw3(self, sample_audio_1s, sample_rate):
        """Output should be (H, W, 3)."""
        result = plot_spectrum(sample_audio_1s, sample_rate)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_mono_input(self, sample_audio_mono, sample_rate):
        """Mono input should work."""
        result = plot_spectrum(sample_audio_mono, sample_rate)
        assert result.ndim == 3

    def test_silence_input(self, sample_audio_silence, sample_rate):
        """Silence should still produce a valid spectrum plot."""
        result = plot_spectrum(sample_audio_silence, sample_rate)
        assert result.ndim == 3


class TestPlotComparison:
    """Tests for plot_comparison."""

    def test_returns_ndarray(self, sample_audio_1s, sample_rate):
        """plot_comparison should return numpy ndarray."""
        result = plot_comparison(
            sample_audio_1s, sample_rate,
            sample_audio_1s, sample_rate,
        )
        assert isinstance(result, np.ndarray)

    def test_output_shape_hw3(self, sample_audio_1s, sample_rate):
        """Output should be (H, W, 3) — comparison is larger."""
        result = plot_comparison(
            sample_audio_1s, sample_rate,
            sample_audio_1s, sample_rate,
        )
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_different_audio(self, sample_audio_1s, sample_audio_short, sample_rate):
        """Comparison of different audio lengths should work."""
        result = plot_comparison(
            sample_audio_1s, sample_rate,
            sample_audio_short, sample_rate,
        )
        assert result.ndim == 3

    def test_custom_labels(self, sample_audio_1s, sample_rate):
        """Custom labels should not cause errors."""
        result = plot_comparison(
            sample_audio_1s, sample_rate,
            sample_audio_1s, sample_rate,
            label_a="Original", label_b="Processed",
        )
        assert isinstance(result, np.ndarray)

    def test_mono_vs_stereo(self, sample_audio_mono, sample_audio_1s, sample_rate):
        """Comparing mono vs stereo should work."""
        result = plot_comparison(
            sample_audio_mono, sample_rate,
            sample_audio_1s, sample_rate,
        )
        assert result.ndim == 3
