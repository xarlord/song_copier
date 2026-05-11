# Phase 2: Production Quality — Implementation Plan

> **For Hermes:** Execute directly — do not delegate further.

**Goal:** Add mastering chain, audio visualization, and real-time recording to make the app production-quality.

**Architecture:** Three independent features built as services + UI tabs. Mastering uses pedalboard (already installed). Visualization uses matplotlib + numpy. Recording uses Gradio's built-in microphone component.

**Tech Stack:** pedalboard (already in requirements), matplotlib, numpy, Gradio 6.x mic component

---

## Task 1: Mastering Service (`services/mastering.py`)

**Create:** `services/mastering.py`

Loudness normalization to LUFS targets + limiter + stereo widening. Uses pedalboard.

Targets:
- Spotify: -14 LUFS
- YouTube: -13 LUFS
- CD/Apple Music: -9 LUFS
- Custom: user-specified

Functions:
- `measure_loudness(audio, sr)` → returns {lufs, peak, rms}
- `normalize_loudness(audio, sr, target_lufs)` → normalized audio
- `apply_limiter(audio, sr, threshold_db=-1.0)` → limited audio
- `apply_stereo_widen(audio, width=1.5)` → widened audio (stereo only)
- `master_audio(audio, sr, preset="spotify")` → full mastering chain

## Task 2: Mastering UI Tab (`ui/tabs/mastering_tab.py`)

**Create:** `ui/tabs/mastering_tab.py`

Upload audio → select target preset → adjust limiter/widen → preview → download.

Components:
- Audio file upload
- Preset dropdown (Spotify, YouTube, CD, Custom)
- Custom LUFS slider (-24 to -3)
- Limiter threshold slider (-6 to 0 dB)
- Stereo width slider (0.5 to 2.0)
- Loudness meter (before/after LUFS display)
- Preview output audio player
- Download button

## Task 3: Visualization Service (`services/visualization.py`)

**Create:** `services/visualization.py`

Generate waveform and spectrogram images from numpy audio arrays.

Functions:
- `plot_waveform(audio, sr, title="")` → returns matplotlib figure
- `plot_spectrogram(audio, sr, title="")` → returns matplotlib figure
- `plot_waveform_comparison(audio_a, audio_b, sr, titles=[])` → side-by-side
- `fig_to_image(fig)` → convert matplotlib figure to numpy array for Gradio

## Task 4: Visualization UI Tab (`ui/tabs/visualization_tab.py`)

**Create:** `ui/tabs/visualization_tab.py`

Upload audio → display waveform + spectrogram + frequency spectrum. Optional: upload two files for A/B comparison.

Components:
- Audio file upload
- Waveform display (Gradio Image)
- Spectrogram display (Gradio Image)
- Frequency spectrum display
- Stem selector (if separated file)
- A/B comparison mode (second upload)

## Task 5: Recording Tab (`ui/tabs/recording_tab.py`)

**Create:** `ui/tabs/recording_tab.py`

Record from microphone → route to stem separator or voice swap.

Components:
- Gradio Microphone component (browser recording)
- Or file upload as alternative input
- "Record & Separate" button → saves temp → runs stem separator
- "Record & Preview" button → plays back raw recording
- Output stems display

## Task 6: Wire Everything Into `app.py`

**Modify:** `app.py` — add 3 new tabs, import new tab creators.

## Task 7: Update requirements.txt

Add matplotlib if not already present.

## Task 8: Commit & Push
