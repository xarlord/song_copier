# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local audio generation studio with 12 tabs: recording, stem separation, voice swapping, song spin-offs, time-stretching, mastering, AI mixing, visualization, MIDI export, project library, plugins, and batch processing. Python-based with Gradio web UI. All ML inference runs locally on GPU.

## Tech Stack

- **Python 3.12** (main app) + **Python 3.10** (RVC via WSL subprocess bridge)
- **PyTorch 2.7.1 + CUDA 11.8** — GPU inference backend
- **Demucs** — source separation (4 or 6 stems via htdemucs / htdemucs_6s)
- **MusicGen** (audiocraft 1.3.0) — text-to-music with melody conditioning (requires xformers monkey-patch)
- **RVC v2** — voice conversion via `rvc_env/` Python 3.10 venv + subprocess bridge
- **Gradio 6.x** — web UI with 12 tabbed sections
- **pedalboard** — effects rack (EQ, reverb, delay, compressor, pan) + mastering chain
- **librosa** — analysis (BPM, key), time-stretch, pitch-shift, pitch detection
- **mido** — MIDI file creation from pitch tracking
- **matplotlib** — audio visualization (waveforms, spectrograms, piano rolls)
- **noisereduce + scipy** — post-processing for stem artifact reduction
- **pydub / soundfile** — audio I/O and format conversion

## Commands

```bash
# Setup
py -3.12 -m venv venv
venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# RVC setup (WSL only, one-time)
python3.10 -m venv rvc_env
rvc_env/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
rvc_env/bin/pip install rvc-python

# Run
python app.py

# Run with external sharing
python app.py --share

# Docker (GPU)
docker compose up

# Docker (GPU with specific device)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## Architecture

```
app.py                → Gradio Blocks entry point with 12 tabs
config.py             → Config loader (YAML → dict)
config/default.yaml   → Model parameters, paths, device, cleanup settings
models/
  base.py             → Abstract base class (load_model, unload_model, get_vram_requirement_mb)
  demucs_wrapper.py   → Demucs source separation (htdemucs, htdemucs_ft, htdemucs_6s)
  musicgen_wrapper.py → MusicGen text-to-music (with xformers monkey-patch)
  rvc_wrapper.py      → RVC voice conversion (subprocess bridge to Python 3.10 venv)
pipelines/
  instrumental.py     → Vocal removal pipeline (Demucs 4-stem)
  stem_separator.py   → 6-stem separation + per-stem cleaning pipeline
  voice_swap.py       → Voice swap pipeline (Demucs + RVC)
  spinoff.py          → Song variation pipeline (Demucs + MusicGen + optional stem mix)
services/
  audio_io.py         → Load, save, resample, format conversion
  mixing.py           → Stem recombination, volume-controlled mixing
  analysis.py         → BPM/key detection via librosa
  stem_cleaning.py    → Per-stem artifact reduction (bandpass, noisereduce)
  effects.py          → Per-stem effects chain (EQ, reverb, delay, compressor, pan)
  presets.py          → YAML preset CRUD (load/save/delete)
  mastering.py        → LUFS normalization, limiter, stereo widening
  visualization.py    → Waveform, spectrogram, FFT, A/B comparison plots
  midi_export.py      → Pitch tracking → MIDI file export (librosa.pyin + mido)
  time_stretch.py     → Independent per-stem time-stretch + pitch-shift
  ai_mixer.py         → Heuristic AI mixing assistant (volume/EQ/reverb suggestions)
  project_manager.py  → Save/load/search separation projects (JSON storage)
  plugin_manager.py   → Plugin discovery, validation, loading
  cleanup.py          → Auto-cleanup of old temp/output files
  rvc_bridge.py       → Subprocess bridge for rvc-python in Python 3.10 venv
ui/
  tabs/
    recording_tab.py     → Mic recording → separate/master
    instrumental_tab.py  → Stem Separator & Mixer (6-stem + volume sliders + effects + presets)
    voice_swap_tab.py    → Voice Swap tab
    spinoff_tab.py       → Song Spin-off tab
    time_stretch_tab.py  → Per-stem time-stretch & pitch-shift
    mastering_tab.py     → Mastering chain (LUFS normalization + limiter + stereo widen)
    ai_mixer_tab.py      → AI mixing assistant with tweakable sliders
    visualization_tab.py → Waveform/spectrogram/spectrum + A/B comparison
    midi_export_tab.py   → Audio → MIDI export with piano roll visualization
    project_library_tab.py → Save/load/browse separation projects
    plugins_tab.py       → Browse and run community plugins
    batch_tab.py         → Batch process multiple files
  components.py       → Shared UI components
  styles.py           → Custom CSS (dark theme)
utils/
  gpu.py              → GPU detection, VRAM monitoring, model lifecycle
  logging.py          → Structured logging setup
plugins/
  example_distortion/ → Example plugin (soft-clipping distortion)
  README.md           → Plugin author guide
```

### Key Design Decisions

- **Models loaded on demand, not at startup** — prevents 12-16GB VRAM spike
- **Pipeline pattern** — each use case is a pipeline that orchestrates model wrappers and services
- **Model wrapper base class** (`models/base.py`) — enforces `load_model()`, `unload_model()`, `get_vram_requirement_mb()` interface
- **VRAM manager** (`utils/gpu.py`) — tracks memory, auto-unloads models, prevents OOM
- **RVC via subprocess bridge** — `rvc_wrapper.py` detects Windows vs WSL, translates paths, calls `rvc_env/bin/python` via WSL interop
- **Post-processing pipeline** (`services/stem_cleaning.py`) — per-stem bandpass filtering + noisereduce for artifact reduction
- **All cache/output on D: drive** — model cache in project `cache/` dir, not user home
- **xformers monkey-patch** (`models/musicgen_wrapper.py`) — audiocraft requires xformers which is incompatible with torch 2.7.1, so a fake module is injected before import
- **Plugin system** — plugins live in `plugins/<name>/` with `plugin.json` manifest + `__init__.py` implementing `process_audio()`
- **Docker deployment** — multi-stage Dockerfile (nvidia/cuda:11.8), docker-compose with GPU/CPU profiles

### Known Issues

- `audiocraft 1.3.0` expects `torch==2.1.0` and `xformers` — we use `--no-deps` install + monkey-patch
- `transformers` must be pinned to `~4.42.0` — newer 5.x breaks audiocraft's T5EncoderModel import
- MIDI export works best with monophonic audio (single melody lines) — polyphonic sources will produce dominant-pitch-only output
- RVC base models (~700MB) download automatically on first use from HuggingFace

### Data Flow

1. **Record**: Mic input → save → route to stem separator or mastering
2. **Stem Separator**: Input → Demucs 6s (separate 6 stems) → Clean each stem → Save individually → Volume mixer UI → Preview mix
3. **Voice Swap**: Input → Demucs (extract vocals) → RVC bridge (Python 3.10) → Mix back → Output
4. **Spin-off**: Input → Demucs (separate) → Analyze (BPM/key) → MusicGen (melody conditioning) → Mix with original stems → Output
5. **Mastering**: Input → Measure LUFS → Normalize → Stereo widen → Limit → Output (with before/after metrics)
6. **AI Mix**: Input → Separate → Analyze each stem → Suggest volumes/EQ/reverb/pan → Apply → Output
7. **MIDI Export**: Input → Pitch detection (librosa.pyin) → Note events → MIDI file (mido) → Piano roll visualization
8. **Time-Stretch**: Input → Per-stem pitch shift → Per-stem time stretch → Remix → Output
9. **Plugin**: Input → Load plugin → Apply process_audio() → Output
