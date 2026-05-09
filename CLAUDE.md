# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local audio generation studio that creates song spin-offs: stem separation with mixer, voice swapping, and AI-generated variations. Python-based with Gradio web UI. All ML inference runs locally on GPU.

## Tech Stack

- **Python 3.12** (installed, despite rvc-python preferring 3.10)
- **PyTorch 2.7.1 + CUDA 11.8** — GPU inference backend
- **Demucs** — source separation (4 or 6 stems via htdemucs / htdemucs_6s)
- **MusicGen** (audiocraft 1.3.0) — text-to-music with melody conditioning (requires xformers monkey-patch)
- **Gradio 6.x** — web UI with tabbed interface
- **noisereduce + scipy** — post-processing for stem artifact reduction
- **librosa / pydub / soundfile** — audio processing
- **RVC v2** — voice conversion (NOT installed — rvc-python fails to build on Windows + Python 3.12)

## Commands

```bash
# Setup
py -3.12 -m venv venv
venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Run
python app.py

# Run with external sharing
python app.py --share
```

## Architecture

```
app.py                → Gradio Blocks entry point with 3 tabs
config.py             → Config loader (YAML → dict)
config/default.yaml   → Model parameters, paths, device settings
models/
  base.py             → Abstract base class (load_model, unload_model, get_vram_requirement_mb)
  demucs_wrapper.py   → Demucs source separation (htdemucs, htdemucs_ft, htdemucs_6s)
  musicgen_wrapper.py → MusicGen text-to-music (with xformers monkey-patch)
  rvc_wrapper.py      → RVC voice conversion (not functional — rvc-python build fails)
pipelines/
  instrumental.py     → Vocal removal pipeline (Demucs 4-stem)
  stem_separator.py   → 6-stem separation + per-stem cleaning pipeline
  voice_swap.py       → Voice swap pipeline (Demucs + RVC)
  spinoff.py          → Song variation pipeline (Demucs + MusicGen)
services/
  audio_io.py         → Load, save, resample, format conversion
  mixing.py           → Stem recombination, volume-controlled mixing
  analysis.py         → BPM/key detection via librosa
  stem_cleaning.py    → Per-stem artifact reduction (bandpass, noisereduce)
ui/
  tabs/
    instrumental_tab.py  → Stem Separator & Mixer (6-stem + volume sliders)
    voice_swap_tab.py     → Voice Swap tab
    spinoff_tab.py        → Song Spin-off tab
  components.py       → Shared UI components
  styles.py           → Custom CSS
utils/
  gpu.py              → GPU detection, VRAM monitoring, model lifecycle
  logging.py          → Structured logging setup
```

### Key Design Decisions

- **Models loaded on demand, not at startup** — prevents 12-16GB VRAM spike
- **Pipeline pattern** — each use case is a pipeline that orchestrates model wrappers and services
- **Model wrapper base class** (`models/base.py`) — enforces `load_model()`, `unload_model()`, `get_vram_requirement_mb()` interface
- **VRAM manager** (`utils/gpu.py`) — tracks memory, auto-unloads models, prevents OOM
- **Demucs shifts=5 by default** — averages 5 random temporal shifts to reduce artifacts
- **Post-processing pipeline** (`services/stem_cleaning.py`) — per-stem bandpass filtering + noisereduce for artifact reduction
- **All cache/output on D: drive** — model cache in project `cache/` dir, not user home
- **xformers monkey-patch** (`models/musicgen_wrapper.py`) — audiocraft requires xformers which is incompatible with torch 2.7.1, so a fake module is injected before import
- **RVC requires pre-trained .pth model files** — users place them in `cache/rvc/models/` (Voice Swap tab non-functional until rvc-python builds)

### Known Issues

- `rvc-python` fails to build on Windows + Python 3.12 — Voice Swap tab is non-functional
- `audiocraft 1.3.0` expects `torch==2.1.0` and `xformers` — we use `--no-deps` install + monkey-patch
- `transformers` must be pinned to `~4.42.0` — newer 5.x breaks audiocraft's T5EncoderModel import

### Data Flow

1. **Stem Separator**: Input → Demucs 6s (separate 6 stems) → Clean each stem → Save individually → Volume mixer UI → Preview mix
2. **Voice Swap**: Input → Demucs (extract vocals) → RVC (convert vocals) → Mix back → Output
3. **Spin-off**: Input → Demucs (separate) → Analyze (BPM/key) → MusicGen (melody conditioning) → Mix → Output
