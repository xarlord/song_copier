"""Shared test fixtures for audio_generator test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 44100


@pytest.fixture
def sample_audio_1s(sample_rate):
    """1-second stereo audio (2 channels, 44100 samples each)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((2, sample_rate)).astype(np.float32)


@pytest.fixture
def sample_audio_mono(sample_rate):
    """1-second mono audio (1D array)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(sample_rate).astype(np.float32)


@pytest.fixture
def sample_audio_short(sample_rate):
    """0.1-second stereo audio."""
    n_samples = int(sample_rate * 0.1)
    rng = np.random.default_rng(42)
    return rng.standard_normal((2, n_samples)).astype(np.float32)


@pytest.fixture
def sample_audio_sine(sample_rate):
    """1-second 440Hz sine wave (mono) for pitch detection tests."""
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def sample_audio_silence(sample_rate):
    """1-second silence (stereo)."""
    return np.zeros((2, sample_rate), dtype=np.float32)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Dedicated output directory inside tmp_path."""
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture
def sample_wav_file(tmp_path, sample_audio_1s, sample_rate):
    """Write sample_audio_1s to a temporary .wav file, return path."""
    import soundfile as sf

    wav_path = tmp_path / "test_audio.wav"
    # soundfile expects (samples, channels)
    sf.write(str(wav_path), sample_audio_1s.T, sample_rate)
    return wav_path


@pytest.fixture
def sample_stems(sample_audio_1s, sample_rate):
    """4 equal-length stereo stems for mixing tests."""
    rng = np.random.default_rng(123)
    return {
        "vocals": (rng.standard_normal((2, sample_rate)) * 0.5).astype(np.float32),
        "drums": (rng.standard_normal((2, sample_rate)) * 0.4).astype(np.float32),
        "bass": (rng.standard_normal((2, sample_rate)) * 0.3).astype(np.float32),
        "other": (rng.standard_normal((2, sample_rate)) * 0.2).astype(np.float32),
    }


@pytest.fixture
def sample_project_dir(tmp_path):
    """Create a fake project directory structure with stem WAV files."""
    import soundfile as sf

    proj = tmp_path / "project_stems"
    proj.mkdir()
    sr = 44100
    rng = np.random.default_rng(99)
    for name in ("vocals", "drums", "bass", "other"):
        audio = (rng.standard_normal((2, sr)) * 0.3).astype(np.float32)
        sf.write(str(proj / f"{name}.wav"), audio.T, sr)
    return proj
