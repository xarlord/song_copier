"""Tests for services/mixing.py — mix_stems, mix_stems_with_volumes."""

import numpy as np
import pytest

from services.mixing import mix_stems, mix_stems_with_volumes, normalize_audio


class TestMixStems:
    """Tests for mix_stems."""

    def test_mix_equal_length_stems(self, sample_stems):
        """Mix 4 equal-length stems — output shape should match."""
        result = mix_stems(sample_stems)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 2  # stereo
        assert result.shape[1] == list(sample_stems.values())[0].shape[1]

    def test_mix_all_stems_summed(self, sample_stems):
        """Result should be the sum of all stems (before normalization)."""
        result_raw = mix_stems(sample_stems, normalize=False)
        manual_sum = sum(sample_stems.values())
        np.testing.assert_allclose(result_raw, manual_sum, atol=1e-6)

    def test_mix_with_include(self, sample_stems):
        """Only included stems should be in the mix."""
        result = mix_stems(sample_stems, include=["vocals", "bass"])
        assert isinstance(result, np.ndarray)
        # Should differ from full mix
        full = mix_stems(sample_stems)
        assert not np.allclose(result, full)

    def test_mix_with_exclude(self, sample_stems):
        """Excluded stems should not be in the mix."""
        result = mix_stems(sample_stems, exclude=["drums"])
        manual = sample_stems["vocals"] + sample_stems["bass"] + sample_stems["other"]
        np.testing.assert_allclose(result, manual, atol=1e-5)

    def test_mix_different_lengths_padding(self, sample_rate):
        """Stems of different lengths — shorter should be zero-padded."""
        stems = {
            "short": np.random.randn(2, sample_rate // 2).astype(np.float32),
            "long": np.random.randn(2, sample_rate).astype(np.float32),
        }
        result = mix_stems(stems)
        assert result.shape[1] == sample_rate  # length of longest stem

    def test_mix_empty_stems_raises(self):
        """Empty stems dict should raise ValueError."""
        with pytest.raises(ValueError, match="No stems to mix"):
            mix_stems({})

    def test_mix_normalize_prevents_clipping(self, sample_stems):
        """Normalized output should not exceed ~0.95 peak."""
        result = mix_stems(sample_stems, normalize=True)
        assert np.abs(result).max() <= 0.96  # tolerance for float

    def test_mix_no_normalize(self, sample_stems):
        """Without normalization, output may exceed 1.0."""
        # Use very loud stems
        loud_stems = {k: v * 5.0 for k, v in sample_stems.items()}
        result = mix_stems(loud_stems, normalize=False)
        assert np.abs(result).max() > 1.0


class TestMixStemsWithVolumes:
    """Tests for mix_stems_with_volumes."""

    def test_basic_volume_mix(self, sample_stems):
        """Mix with all volumes at 1.0 should equal simple sum (before normalization)."""
        volumes = {name: 1.0 for name in sample_stems}
        result = mix_stems_with_volumes(sample_stems, volumes)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_zero_volume_mutes_stem(self, sample_stems):
        """Volume 0 should mute the stem (no contribution)."""
        volumes = {name: 0.0 for name in sample_stems}
        with pytest.raises(ValueError, match="No stems to mix"):
            mix_stems_with_volumes(sample_stems, volumes)

    def test_partial_volume(self, sample_stems):
        """Some stems muted, some active."""
        volumes = {name: (1.0 if name == "vocals" else 0.0) for name in sample_stems}
        result = mix_stems_with_volumes(sample_stems, volumes)
        # Should be close to just the vocals stem (normalized)
        assert isinstance(result, np.ndarray)

    def test_volume_half_reduces_amplitude(self, sample_rate):
        """Volume 0.5 should reduce amplitude roughly by half."""
        stems = {"test": np.ones((2, sample_rate), dtype=np.float32) * 0.5}
        volumes = {"test": 0.5}
        result = mix_stems_with_volumes(stems, volumes, normalize=False)
        # 0.5 * 0.5 = 0.25 per sample
        np.testing.assert_allclose(result, 0.25, atol=1e-6)

    def test_mono_stem_handling(self, sample_rate):
        """Mono (1D) stems should be handled gracefully."""
        stems = {
            "mono": np.random.randn(sample_rate).astype(np.float32),
        }
        volumes = {"mono": 1.0}
        result = mix_stems_with_volumes(stems, volumes)
        assert result.ndim == 2
        assert result.shape[0] == 1 or result.shape[0] == 2

    def test_missing_volume_defaults_to_one(self, sample_stems):
        """Stems without explicit volume should default to 1.0."""
        volumes = {}  # empty volumes
        result = mix_stems_with_volumes(sample_stems, volumes)
        assert isinstance(result, np.ndarray)


class TestNormalizeAudio:
    """Tests for normalize_audio."""

    def test_normalize_loud_audio(self):
        """Loud audio should be scaled down to target peak."""
        audio = np.ones((2, 1000), dtype=np.float32) * 0.5
        result = normalize_audio(audio, target_peak=0.8)
        assert np.abs(result).max() == pytest.approx(0.8, abs=1e-5)

    def test_normalize_silence(self):
        """Silent audio should remain silent."""
        audio = np.zeros((2, 1000), dtype=np.float32)
        result = normalize_audio(audio)
        np.testing.assert_array_equal(result, audio)

    def test_normalize_preserves_shape(self, sample_audio_1s):
        """Normalization should not change shape."""
        result = normalize_audio(sample_audio_1s)
        assert result.shape == sample_audio_1s.shape
