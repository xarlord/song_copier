"""Tests for services/audio_io.py — load, save, resample."""

import numpy as np
import pytest

from services.audio_io import load_audio, save_audio


class TestSaveAudio:
    """Tests for save_audio."""

    def test_save_stereo_wav(self, sample_audio_1s, sample_rate, tmp_output_dir):
        """Save stereo audio as WAV and verify file exists."""
        path = save_audio(sample_audio_1s, tmp_output_dir / "out.wav", sample_rate)
        assert path.exists()
        assert path.suffix == ".wav"

    def test_save_mono_wav(self, sample_audio_mono, sample_rate, tmp_output_dir):
        """Save 1D mono audio — should auto-expand to (1, N)."""
        path = save_audio(sample_audio_mono, tmp_output_dir / "mono.wav", sample_rate)
        assert path.exists()

    def test_save_creates_parent_dirs(self, sample_audio_1s, sample_rate, tmp_output_dir):
        """save_audio should create missing parent directories."""
        deep_path = tmp_output_dir / "nested" / "dir" / "audio.wav"
        path = save_audio(sample_audio_1s, deep_path, sample_rate)
        assert path.exists()

    def test_save_flac_format(self, sample_audio_1s, sample_rate, tmp_output_dir):
        """Save audio in FLAC format."""
        path = save_audio(sample_audio_1s, tmp_output_dir / "out.flac", sample_rate, format="flac")
        assert path.exists()
        assert path.suffix == ".flac"


class TestLoadAudio:
    """Tests for load_audio."""

    def test_load_nonexistent_raises(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")

    def test_save_load_roundtrip_stereo(self, sample_audio_1s, sample_rate, tmp_output_dir):
        """Save then load stereo audio — shape and values should match."""
        path = save_audio(sample_audio_1s, tmp_output_dir / "rt.wav", sample_rate)
        loaded, sr = load_audio(path, target_sr=sample_rate)
        assert sr == sample_rate
        assert loaded.shape == sample_audio_1s.shape
        np.testing.assert_allclose(loaded, sample_audio_1s, atol=1e-5)

    def test_save_load_roundtrip_mono(self, sample_audio_mono, sample_rate, tmp_output_dir):
        """Save then load mono audio."""
        path = save_audio(sample_audio_mono, tmp_output_dir / "rt_mono.wav", sample_rate)
        loaded, sr = load_audio(path, target_sr=sample_rate)
        assert sr == sample_rate
        # mono gets expanded to (1, N) on save, torchaudio returns (1, N)
        assert loaded.shape[0] == 1
        np.testing.assert_allclose(loaded.squeeze(), sample_audio_mono, atol=1e-5)

    def test_load_resample(self, sample_audio_1s, sample_rate, tmp_output_dir):
        """Load with a different target_sr triggers resampling."""
        path = save_audio(sample_audio_1s, tmp_output_dir / "rs.wav", sample_rate)
        loaded, new_sr = load_audio(path, target_sr=22050)
        assert new_sr == 22050
        # Resampled should have roughly half the samples
        assert loaded.shape[-1] < sample_audio_1s.shape[-1]

    def test_load_default_target_sr(self, sample_audio_1s, tmp_output_dir):
        """Load without specifying target_sr defaults to 44100."""
        sr_original = 44100
        path = save_audio(sample_audio_1s, tmp_output_dir / "def.wav", sr_original)
        loaded, sr = load_audio(path)
        assert sr == 44100


class TestEdgeCases:
    """Edge case tests for audio I/O."""

    def test_save_zero_length_audio(self, tmp_output_dir, sample_rate):
        """Saving zero-length audio should still work (empty file)."""
        audio = np.zeros((2, 0), dtype=np.float32)
        path = save_audio(audio, tmp_output_dir / "empty.wav", sample_rate)
        assert path.exists()

    def test_save_very_short_audio(self, tmp_output_dir, sample_rate):
        """Save very short audio (10 samples)."""
        audio = np.random.randn(2, 10).astype(np.float32)
        path = save_audio(audio, tmp_output_dir / "short.wav", sample_rate)
        assert path.exists()
        loaded, sr = load_audio(path, target_sr=sample_rate)
        assert loaded.shape[-1] == 10
