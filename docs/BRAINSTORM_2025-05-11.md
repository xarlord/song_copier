# 🎵 Audio Generator Studio — Brainstorm & Roadmap

## Current State Summary

### ✅ What Works
- **6-stem separation** (Demucs htdemucs_6s: vocals, drums, bass, guitar, piano, other)
- **Stem cleaning** (bandpass + noisereduce artifact reduction)
- **Volume mixer** (per-stem sliders with mute/reset)
- **Effects rack** (EQ, reverb, delay, compressor, stereo pan — via pedalboard)
- **Preset system** (4 built-in presets, YAML-based)
- **Song Spin-off** (MusicGen text-to-music with melody conditioning)
- **Batch processing** (multi-file upload → separate → apply preset → ZIP)
- **Audio analysis** (BPM + key detection)
- **VRAM management** (on-demand loading, auto-unload)

### ❌ Blocked
- **Voice Swap / RVC** — `rvc-python` won't build on Windows + Python 3.12

### ⚠️ Incomplete
- **Spin-off mix_with_original_stems** — loads drums but never uses them (stub)
- **Preset save UI** — backend exists but no UI save button

---

## 🧠 BRAINSTORM: New Features & Directions

### TIER 1 — Unblock & Fix (High Priority)

#### 1. 🔓 Fix Voice Conversion (Unblock Voice Swap)
The RVC blocker has multiple solution paths:

| Approach | Pros | Cons |
|---|---|---|
| **A) Docker RVC container** | Full isolation, no build issues, can use Python 3.10 inside | Docker overhead, inter-process audio transfer |
| **B) RVC-WebUI as API server** | Battle-tested, community-maintained, HTTP-based | Extra process to manage, dependency bloat |
| **C) WSL RVC environment** | Already on WSL, can use Python 3.10 natively | Two Python envs, path complexity |
| **D) Fallback to so-vits-svc** | Alternative voice conversion | Older, less quality, same build issues possible |
| **E) Cloud API (ElevenLabs/Replicate)** | Zero local build issues | Costs money, latency, privacy concerns |
| **F) XTTS v2 (Coqui)** | TTS-based voice cloning, builds on 3.12 | Different approach (TTS not conversion) |

**Recommended:** Option A (Docker) for cleanest isolation, or C (WSL) for simplest.

#### 2. 🐛 Fix Spin-off mix_with_original_stems
- drums loaded but never mixed — quick 5-line fix
- unlocks "keep original drums + AI-generated melody" mode

#### 3. 💾 Add Preset Save UI Button
- Backend CRUD exists, just needs a Gradio button + name input

---

### TIER 2 — New Features (Medium Priority)

#### 4. 🎤 Real-Time Recording Tab
- Record vocals/instruments directly in browser via Gradio microphone component
- Auto-route to stem separator or voice swap pipeline
- Zero-friction workflow: record → separate → mix

#### 5. 🎛️ Mastering Chain
- Final output processing: loudness normalization (LUFS), limiter, stereo widener
- Target loudness presets (Spotify -14 LUFS, YouTube -13, CD -9)
- Uses pedalboard (already installed) for processing
- Makes the app a complete production pipeline: separate → mix → master

#### 6. 🎼 Stem Library / Project Manager
- Save separated stems as named "projects"
- Browse and reload previous work
- Compare different separation runs or effect chains
- Persistent metadata (BPM, key, date, original file)

#### 7. 🔄 A/B Comparison Mode
- Upload reference track → compare your mix against it
- Loudness-matched A/B switching
- Spectrum overlay visualization
- Great for mixing decisions

#### 8. 📊 Audio Visualization Dashboard
- Waveform display per stem
- Spectrogram (mel/log-power)
- Real-time audio preview with playback position
- Frequency spectrum analyzer
- Makes the app feel professional

#### 9. 🎸 Guitar/Bass Tab Transcription
- Use basic pitch or spotify's polyphonic transcription
- Convert separated guitar/bass stems → MIDI/tab
- Huge value for musicians learning songs

---

### TIER 3 — Advanced / Stretch Goals

#### 10. 🤖 AI Mixing Assistant
- Analyze stems → suggest volume levels, EQ settings, effect chains
- Use LLM to explain mixing decisions
- "One-click mix" that auto-applies suggested settings

#### 11. 🌐 Web Deployment Mode
- Docker compose for deployment on cloud GPU (RunPod, Vast.ai)
- Queue system for multiple users
- Authentication + usage limits

#### 12. 🎵 MIDI Export
- Convert separated stems to MIDI via pitch tracking
- Export drums → MIDI drum map
- Enables use in DAWs (FL Studio, Ableton, etc.)

#### 13. 🎧 Stem Time-Stretching & Pitch-Shifting
- Independent tempo/key change per stem
- rubberband or pyrubberband for high-quality stretching
- Create mashups from different songs

#### 14. 🔗 Plugin Architecture
- Allow third-party audio processing modules
- Python entry_points for community extensions
- Could enable autotune, harmonizer, etc.

---

## 📋 RECOMMENDED PLAN

### Phase 1: Unblock & Stabilize
1. Fix Voice Swap (Docker RVC or WSL Python 3.10 env)
2. Fix spin-off mix_with_original_stems bug
3. Add preset save button to UI
4. Clean up temp/output directories (add auto-cleanup)

### Phase 2: Production Quality
5. Add mastering chain (LUFS normalization + limiter)
6. Add audio visualization (waveforms + spectrograms)
7. Add A/B comparison mode
8. Add real-time recording input

### Phase 3: Musician Tools
9. Add stem library / project manager
10. Add pitch transcription → MIDI export
11. Add per-stem time-stretch & pitch-shift
12. AI mixing assistant

### Phase 4: Scale & Share
13. Docker deployment config
14. Plugin architecture
15. Web deployment mode

---

## Priority Matrix (Impact vs Effort)

```
HIGH IMPACT, LOW EFFORT (DO FIRST):
  ✅ Fix spin-off bug (5 lines)
  ✅ Preset save UI button
  ✅ Fix voice swap via Docker

HIGH IMPACT, MEDIUM EFFORT:
  🔧 Mastering chain
  🔧 Audio visualization
  🔧 Real-time recording

HIGH IMPACT, HIGH EFFORT:
  🚀 AI mixing assistant
  🚀 Plugin architecture
  🚀 MIDI export

LOW IMPACT, LOW EFFORT (NICE-TO-HAVE):
  💡 Auto-cleanup temp files
  💡 Project naming

LOW IMPACT, HIGH EFFORT (DEFER):
  ⏳ Web deployment
  ⏳ Stem library browser
```
