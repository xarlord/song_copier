# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local audio generation studio that creates song spin-offs: instrumental extraction, voice swapping, and AI-generated variations. Python-based with Gradio web UI. All ML inference runs locally on GPU.

## Tech Stack

- **Python 3.10** (required for RVC/audiocraft compatibility)
- **PyTorch 2.1.x + CUDA 11.8** — GPU inference backend
- **Demucs** — source separation (vocals/drums/bass/other stems)
- **RVC v2** (rvc-python) — voice conversion/cloning
- **MusicGen** (audiocraft) — text-to-music with melody conditioning
- **Gradio 4.x** — web UI with tabbed interface
- **librosa / pydub / soundfile** — audio processing

## Commands

```bash
# Setup
py -3.10 -m venv venv
venv\Scripts\activate
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Run
python app.py

# Run with external sharing
python app.py --share
```

## Architecture

```
app.py              → Gradio Blocks entry point with 3 tabs
models/             → Model wrappers (base class + demucs, rvc, musicgen)
pipelines/          → End-to-end pipelines (instrumental, voice_swap, spinoff)
services/           → Audio I/O, mixing, analysis utilities
ui/tabs/            → One Gradio tab per use case
utils/              → GPU detection, model download, logging
config/default.yaml → Model parameters, paths, device settings
```

### Key Design Decisions

- **Models loaded on demand, not at startup** — prevents 12-16GB VRAM spike
- **Pipeline pattern** — each use case is a pipeline that orchestrates model wrappers and services
- **Model wrapper base class** (`models/base.py`) — enforces `load_model()`, `unload_model()`, `get_vram_requirement_mb()` interface
- **VRAM manager** (`utils/gpu.py`) — tracks memory, auto-unloads models, prevents OOM
- **RVC requires pre-trained .pth model files** — users place them in `cache/rvc/models/`
- **All cache/output on D: drive** — model cache in project `cache/` dir, not user home

### Data Flow

1. **Instrumental**: Input → Demucs (separate stems) → Mix drums+bass+other → Output
2. **Voice Swap**: Input → Demucs (extract vocals) → RVC (convert vocals) → Mix back → Output
3. **Spin-off**: Input → Demucs (separate) → Analyze (BPM/key) → MusicGen (melody conditioning) → Mix → Output
