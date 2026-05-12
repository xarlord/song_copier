"""Audio visualization — waveform, spectrogram, spectrum, and comparison plots."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Dark theme colours
_BG_COLOR = "#1a1a2e"
_GRID_COLOR = "#2a2a4a"
_WAVEFORM_COLOR_L = "#00d2ff"  # left / mono channel
_WAVEFORM_COLOR_R = "#ff6bcb"  # right channel
_SPECTRUM_COLOR = "#7b68ee"
_LABEL_COLOR = "white"


def _fig_to_numpy(fig: "matplotlib.figure.Figure") -> np.ndarray:
    """Convert a matplotlib Figure to an RGB numpy array (H, W, 3)."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()  # (H, W, 4)
    data = np.asarray(buf)[..., :3].copy()  # drop alpha → (H, W, 3)
    return data


def _apply_dark_style(ax):
    """Apply dark-theme styling to an Axes."""
    ax.set_facecolor(_BG_COLOR)
    ax.tick_params(colors=_LABEL_COLOR, which="both")
    ax.xaxis.label.set_color(_LABEL_COLOR)
    ax.yaxis.label.set_color(_LABEL_COLOR)
    ax.title.set_color(_LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, alpha=0.3, linewidth=0.5)


def _setup_figure(figsize=(10, 3)):
    """Create a Figure with Agg backend and dark background."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG_COLOR)
    _apply_dark_style(ax)
    return fig, ax, plt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_waveform(audio: np.ndarray, sr: int, title: str = "Waveform") -> np.ndarray:
    """Plot waveform (time vs amplitude). For stereo, overlay both channels.

    Returns numpy RGB image array (H, W, 3) suitable for Gradio Image component.
    """
    fig, ax, plt = _setup_figure(figsize=(10, 3))

    n_samples = audio.shape[-1]
    time = np.linspace(0, n_samples / sr, num=n_samples, endpoint=False)

    if audio.ndim == 1:
        ax.plot(time, audio, color=_WAVEFORM_COLOR_L, linewidth=0.6)
    else:
        # Stereo — overlay both channels
        ax.plot(time, audio[0], color=_WAVEFORM_COLOR_L, linewidth=0.6, label="Left")
        ax.plot(time, audio[1], color=_WAVEFORM_COLOR_R, linewidth=0.6, label="Right")
        ax.legend(facecolor=_BG_COLOR, edgecolor=_GRID_COLOR, labelcolor=_LABEL_COLOR, fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()

    img = _fig_to_numpy(fig)
    plt.close(fig)
    return img


def plot_spectrogram(audio: np.ndarray, sr: int, title: str = "Spectrogram") -> np.ndarray:
    """Mel spectrogram using librosa. Shows frequency (Hz) vs time.

    Returns numpy RGB image array (H, W, 3).
    """
    import librosa

    fig, ax, plt = _setup_figure(figsize=(10, 3))

    # Ensure mono for spectrogram
    y = audio.mean(axis=0) if audio.ndim > 1 else audio

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img_spec = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr,
                                        fmax=sr // 2, ax=ax, cmap="inferno")

    cbar = fig.colorbar(img_spec, ax=ax, format="%+2.0f dB", pad=0.01)
    cbar.ax.yaxis.set_tick_params(color=_LABEL_COLOR)
    cbar.outline.set_edgecolor(_GRID_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=_LABEL_COLOR, fontsize=7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()

    img = _fig_to_numpy(fig)
    plt.close(fig)
    return img


def plot_spectrum(audio: np.ndarray, sr: int, title: str = "Frequency Spectrum") -> np.ndarray:
    """FFT magnitude spectrum (frequency vs magnitude in dB).

    Returns numpy RGB image array (H, W, 3).
    """
    fig, ax, plt = _setup_figure(figsize=(10, 3))

    # Ensure mono
    y = audio.mean(axis=0) if audio.ndim > 1 else audio

    n = len(y)
    # Apply Hann window for smoother spectrum
    window = np.hanning(n)
    fft_data = np.fft.rfft(y * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    magnitude = np.abs(fft_data)

    # Convert to dB (avoid log of zero)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    ax.plot(freqs, magnitude_db, color=_SPECTRUM_COLOR, linewidth=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.set_xlim(0, sr // 2)
    fig.tight_layout()

    img = _fig_to_numpy(fig)
    plt.close(fig)
    return img


def plot_comparison(
    audio_a: np.ndarray,
    sr_a: int,
    audio_b: np.ndarray,
    sr_b: int,
    label_a: str = "A",
    label_b: str = "B",
) -> np.ndarray:
    """Side-by-side comparison: 2×2 grid (waveforms top, spectrograms bottom).

    Returns numpy RGB image array (H, W, 3).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor=_BG_COLOR)

    pairs = [
        (audio_a, sr_a, label_a),
        (audio_b, sr_b, label_b),
    ]

    for col, (audio, sr, label) in enumerate(pairs):
        y = audio.mean(axis=0) if audio.ndim > 1 else audio
        time = np.linspace(0, len(y) / sr, num=len(y), endpoint=False)

        # ---- Top row: waveform ----
        ax_wav = axes[0, col]
        _apply_dark_style(ax_wav)
        if audio.ndim == 1:
            ax_wav.plot(time, audio, color=_WAVEFORM_COLOR_L, linewidth=0.5)
        else:
            ax_wav.plot(time, audio[0], color=_WAVEFORM_COLOR_L, linewidth=0.5, label="L")
            ax_wav.plot(time, audio[1], color=_WAVEFORM_COLOR_R, linewidth=0.5, label="R")
            ax_wav.legend(facecolor=_BG_COLOR, edgecolor=_GRID_COLOR,
                          labelcolor=_LABEL_COLOR, fontsize=7)
        ax_wav.set_xlabel("Time (s)")
        ax_wav.set_ylabel("Amplitude")
        ax_wav.set_title(f"Waveform — {label}")

        # ---- Bottom row: spectrogram ----
        ax_spec = axes[1, col]
        _apply_dark_style(ax_spec)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
        S_dB = librosa.power_to_db(S, ref=np.max)

        img_spec = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel",
                                            sr=sr, fmax=sr // 2, ax=ax_spec,
                                            cmap="inferno")

        cbar = fig.colorbar(img_spec, ax=ax_spec, format="%+2.0f dB", pad=0.01)
        cbar.ax.yaxis.set_tick_params(color=_LABEL_COLOR)
        cbar.outline.set_edgecolor(_GRID_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=_LABEL_COLOR, fontsize=7)

        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_title(f"Spectrogram — {label}")

    fig.tight_layout()
    img = _fig_to_numpy(fig)
    plt.close(fig)
    return img
