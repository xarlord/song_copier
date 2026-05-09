# Premium Effects Rack + Presets + Batch Processing

**Date:** 2026-05-09
**Status:** Approved
**Target:** Musicians / hobbyists, local desktop app, RTX 3060 Laptop (6GB VRAM)

## Overview

Add a per-stem effects rack, preset system, and batch processing to the Stem Separator tab. After separating a song into 6 stems, users get an effects panel for each stem with adjustable parameters (EQ, reverb, delay, compression, panning). Effects are applied on CPU via pedalboard during mix preview and final export. Presets save/load effect chains. Batch processing separates multiple files with the same settings.

## Architecture

Three layers between separation output and final mix:

```
Separated stems (numpy)
  -> apply per-stem effects (pedalboard on CPU)
  -> apply per-stem volume
  -> mix all stems
  -> normalize
  -> output
```

### Layer 1: `services/effects.py`

Pure audio processing. `apply_effects(stem_audio, sr, effects_config) -> processed_audio`.

Effects via `pedalboard`:

| Effect | Parameters | Range |
|--------|-----------|-------|
| Parametric EQ (3-band) | low_gain (-12..+12 dB @200Hz), mid_gain (-12..+12 dB @1kHz), high_gain (-12..+12 dB @4kHz) | Shelf/Bell filters |
| Reverb (plate) | room_size (0..1), damping (0..1), wet_level (0..1) | Plate reverb |
| Delay (stereo) | delay_seconds (0..1), feedback (0..0.9), mix (0..1) | Simple delay |
| Compressor | threshold_db (-40..0), ratio (1..10), attack_ms (0.1..50) | Dynamic range |
| Panning | pan (-1..+1) | Stereo position |

Default state: all effects off (EQ 0dB, reverb/delay wet=0, compressor bypassed, pan=center).

### Layer 2: `services/presets.py`

Preset management. Save/load/delete as YAML in `config/presets/`.

Structure:
```yaml
name: Karaoke
description: Remove vocals, slight reverb on drums
custom: false
stems:
  vocals:
    volume: 0.0
    effects:
      eq: {low: 0, mid: 0, high: 0}
      reverb: {room_size: 0, damping: 0, wet_level: 0}
      delay: {delay_seconds: 0, feedback: 0, mix: 0}
      compressor: {threshold_db: 0, ratio: 1, attack_ms: 10}
      pan: 0
  drums:
    volume: 1.0
    effects:
      reverb: {room_size: 0.3, damping: 0.5, wet_level: 0.15}
  # ... other stems
```

Built-in presets:
- **Karaoke** — vocals=0, others=1.0, slight reverb on drums
- **Bass Boost** — bass EQ low=+6, bass volume=1.3, rest=0.8
- **Vocal Isolation** — only vocals at 1.0 with reverb, rest=0.1
- **Live Feel** — reverb on everything, delay on vocals, guitar left/piano right

### Layer 3: UI Changes

**`ui/tabs/instrumental_tab.py`** — Effects panel per stem (collapsible accordion), preset dropdown, save/delete preset buttons.

**`ui/tabs/batch_tab.py`** — New tab for batch processing. Multi-file upload, preset selector, "Process All" button, progress tracking, ZIP download.

## Non-Destructive Editing

Effects are applied lazily during preview/export. Original stem files are never modified. Users can tweak freely.

## Error Handling

- **pedalboard not installed** — graceful fallback to volume-only mode with warning
- **Invalid effect params** — clamped to valid ranges at service layer
- **Corrupt preset YAML** — skip broken presets, show warning, never crash
- **Batch failure** — continue on error, report per-file status, generate partial ZIP
- **VRAM during batch** — full load/unload cycle per file to prevent leaks
- **Large files (>10 min)** — show estimated time warning

## Testing Strategy

- `tests/test_effects.py` — Unit tests per effect with sine wave input, verify output shape/no NaN
- `tests/test_presets.py` — Save/load round-trip, invalid YAML handling, built-in presets load
- `tests/test_pipeline.py` — End-to-end with 5s sine wave test input
- Manual: effects sound right on real music, presets apply correctly, batch handles failures

## Files

| File | Change |
|------|--------|
| `services/effects.py` | New — pedalboard effects processing |
| `services/presets.py` | New — preset save/load/list |
| `config/presets/*.yaml` | New — built-in preset files |
| `ui/tabs/instrumental_tab.py` | Modify — add effects UI + preset dropdown |
| `ui/tabs/batch_tab.py` | New — batch processing tab |
| `requirements.txt` | Add `pedalboard>=0.9.0` |

## Implementation Phases

### Phase 1: Effects Service (Medium)
- Create `services/effects.py` with all 5 effects
- Add `pedalboard` to requirements.txt
- Unit tests for each effect

### Phase 2: Effects UI (Medium)
- Add effects panels to instrumental_tab.py
- Wire effects into Preview Mix flow
- Each stem gets collapsible accordion with effect sliders

### Phase 3: Presets (Small)
- Create `services/presets.py`
- Create built-in preset YAML files
- Add preset dropdown + save/delete buttons to UI

### Phase 4: Batch Processing (Small)
- Create `ui/tabs/batch_tab.py`
- Multi-file upload + preset selection
- ZIP output with progress tracking
