"""MIDI export service — convert audio stems to MIDI files using pitch tracking."""

import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# MIDI note name lookup
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# General MIDI program numbers for instruments
_INSTRUMENT_PROGRAMS = {
    "auto": 0,
    "piano": 0,
    "guitar": 24,
    "bass": 32,
    "drums": 0,  # drums use channel 10
    "vocals": 52,  # choir aahs
}


def freq_to_midi_note(freq: float) -> int:
    """Convert frequency in Hz to MIDI note number.

    Uses the standard formula: 69 + 12 * log2(freq / 440).
    Returns 0 for zero or negative frequencies (silence).

    Args:
        freq: Frequency in Hz.

    Returns:
        MIDI note number (0–127), or 0 if freq <= 0.
    """
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_note_to_name(note: int) -> str:
    """Convert MIDI note number to human-readable name.

    Examples:
        >>> midi_note_to_name(60)
        'C4'
        >>> midi_note_to_name(69)
        'A4'
        >>> midi_note_to_name(0)
        'C-1'

    Args:
        note: MIDI note number (0–127).

    Returns:
        Note name string like 'C4', 'A#3', etc.
    """
    if note < 0 or note > 127:
        return f"?{note}"
    name = _NOTE_NAMES[note % 12]
    octave = (note // 12) - 1
    return f"{name}{octave}"


def detect_pitch(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Detect fundamental frequency over time using librosa.pyin.

    Works best on monophonic audio (single melody line).
    For polyphonic audio, the dominant pitch is estimated.

    Args:
        audio: Audio waveform as 1D numpy array. If stereo (2D), mixed to mono.
        sr: Sample rate.

    Returns:
        Tuple of (times, frequencies) — both 1D numpy arrays.
        `frequencies` contains NaN where no pitch is detected.
    """
    import librosa

    # Ensure mono
    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    # Ensure float32 for librosa
    audio = audio.astype(np.float32)

    logger.info(f"Detecting pitch: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.1f}s)")

    # Use pyin for monophonic pitch tracking
    fmin = librosa.note_to_hz("C2")   # ~65 Hz — covers bass guitar
    fmax = librosa.note_to_hz("C7")   # ~2093 Hz — covers most melody

    frequencies, voiced_flags, voiced_probs = librosa.pyin(
        y=audio,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=2048,
        hop_length=512,
    )

    # Build time array matching the frame positions
    times = librosa.times_like(frequencies, sr=sr, hop_length=512)

    logger.info(
        f"Pitch detection complete: {len(frequencies)} frames, "
        f"{np.sum(~np.isnan(frequencies))} voiced"
    )

    return times, frequencies


def audio_to_midi(
    audio: np.ndarray,
    sr: int,
    instrument: str = "auto",
    min_note_duration: float = 0.05,
    pitch_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> 'MidiFile':
    """Convert audio to MIDI using fundamental frequency detection.

    Detects pitch over time and quantizes into discrete MIDI note events.
    Best results with monophonic audio; for polyphonic, the dominant pitch
    is tracked.

    Args:
        audio: Audio waveform as numpy array (1D mono or 2D multi-channel).
        sr: Sample rate in Hz.
        instrument: Instrument name for General MIDI program assignment.
            One of: 'auto', 'piano', 'guitar', 'bass', 'drums', 'vocals'.
        min_note_duration: Minimum note duration in seconds. Notes shorter
            than this are merged with the previous note or discarded.
        pitch_data: Optional pre-computed (times, frequencies) tuple from
            detect_pitch(). If provided, skips redundant pitch detection.

    Returns:
        A mido.MidiFile object containing the transcription.
    """
    import mido

    # Detect pitch (or use pre-computed results)
    if pitch_data is not None:
        times, frequencies = pitch_data
    else:
        times, frequencies = detect_pitch(audio, sr)

    # Quantize frequencies to MIDI notes
    midi_notes = np.array([freq_to_midi_note(f) if not np.isnan(f) else 0 for f in frequencies])

    # Build note events: group consecutive frames with same note
    notes = []
    if len(midi_notes) == 0:
        # No frames — return empty MIDI file
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        return mid

    current_note = midi_notes[0]
    start_time = times[0]

    for i in range(1, len(midi_notes)):
        if midi_notes[i] != current_note:
            end_time = times[i]
            duration = end_time - start_time

            # Only add notes that are non-zero (voiced) and long enough
            if current_note > 0 and duration >= min_note_duration:
                velocity = _estimate_velocity(frequencies, start_time, end_time, times)
                notes.append({
                    "note": int(current_note),
                    "start": start_time,
                    "duration": duration,
                    "velocity": velocity,
                })

            current_note = midi_notes[i]
            start_time = times[i]

    # Handle last note segment
    if current_note > 0 and len(times) > 0:
        end_time = times[-1]
        duration = end_time - start_time
        if duration >= min_note_duration:
            velocity = _estimate_velocity(frequencies, start_time, end_time, times)
            notes.append({
                "note": int(current_note),
                "start": start_time,
                "duration": duration,
                "velocity": velocity,
            })

    # Build MIDI file
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo (default 120 BPM, will be overridden in export_midi)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

    # Set instrument program
    program = _INSTRUMENT_PROGRAMS.get(instrument, 0)
    channel = 9 if instrument == "drums" else 0  # Channel 10 for drums (0-indexed: 9)
    track.append(mido.Message("program_change", program=program, channel=channel, time=0))

    # Convert note events to MIDI messages with delta times
    ticks_per_beat = 480
    microseconds_per_beat = 500000  # 120 BPM default

    midi_events = []
    for n in notes:
        start_tick = int(mido.second2tick(n["start"], ticks_per_beat, microseconds_per_beat))
        end_tick = int(mido.second2tick(
            n["start"] + n["duration"], ticks_per_beat, microseconds_per_beat
        ))
        midi_events.append(("note_on", start_tick, n["note"], n["velocity"]))
        midi_events.append(("note_off", end_tick, n["note"], 0))

    # Sort by tick, with note_off before note_on at same tick
    midi_events.sort(key=lambda e: (e[1], 0 if e[0] == "note_off" else 1))

    # Write sorted events with delta times
    last_tick = 0
    for event_type, tick, note, velocity in midi_events:
        delta = tick - last_tick
        track.append(mido.Message(
            event_type, note=note, velocity=velocity,
            channel=channel, time=delta,
        ))
        last_tick = tick

    # End of track
    track.append(mido.MetaMessage("end_of_track", time=0))

    logger.info(
        f"MIDI transcription: {len(notes)} notes, "
        f"range {midi_note_to_name(min((n['note'] for n in notes), default=60))}"
        f"-{midi_note_to_name(max((n['note'] for n in notes), default=60))}"
    )

    return mid


def _estimate_velocity(
    frequencies: np.ndarray,
    start_time: float,
    end_time: float,
    times: np.ndarray,
) -> int:
    """Estimate MIDI velocity based on pitch confidence / frequency presence.

    Since pyin doesn't return amplitude directly, we use a default moderate
    velocity with slight variation based on frame count.
    """
    # Count voiced frames in this note segment
    mask = (times >= start_time) & (times < end_time)
    voiced_count = np.sum(~np.isnan(frequencies[mask]))
    total_count = np.sum(mask)

    if total_count == 0:
        return 80

    # Higher ratio of voiced frames → higher velocity
    confidence = voiced_count / total_count
    velocity = int(60 + 60 * confidence)  # Range 60–120
    return max(1, min(127, velocity))


def export_midi(
    audio: np.ndarray,
    sr: int,
    output_path: str,
    instrument: str = "auto",
    bpm: float = 120.0,
) -> str:
    """Full pipeline: detect pitch → create MIDI → save to file.

    Args:
        audio: Audio waveform as numpy array.
        sr: Sample rate in Hz.
        output_path: Path to write the .mid file.
        instrument: Instrument name for MIDI program (see audio_to_midi).
        bpm: Tempo in beats per minute. Used for MIDI timing.

    Returns:
        Absolute path to the saved MIDI file.
    """
    import mido

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .mid extension
    if output_path.suffix.lower() not in (".mid", ".midi"):
        output_path = output_path.with_suffix(".mid")

    logger.info(f"Exporting MIDI: {output_path} (BPM={bpm}, instrument={instrument})")

    # Detect pitch once and reuse results
    pitch_data = detect_pitch(audio, sr)

    # Generate MIDI with pre-computed pitch data
    mid = audio_to_midi(audio, sr, instrument=instrument, pitch_data=pitch_data)

    # Update tempo to requested BPM
    tempo_msg = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo_msg = msg
                break
        if tempo_msg:
            break

    if tempo_msg:
        tempo_msg.tempo = mido.bpm2tempo(bpm)

    # Recalculate timing with the correct BPM
    # Rebuild the MIDI file with proper tempo using the already-detected pitch data
    ticks_per_beat = mid.ticks_per_beat
    microseconds_per_beat = mido.bpm2tempo(bpm)

    # Reuse the pitch data already detected above
    times, frequencies = pitch_data
    midi_notes = np.array([freq_to_midi_note(f) if not np.isnan(f) else 0 for f in frequencies])

    # Build note events
    notes = []
    if len(midi_notes) > 0:
        current_note = midi_notes[0]
        start_time = times[0]

        for i in range(1, len(midi_notes)):
            if midi_notes[i] != current_note:
                end_time = times[i]
                duration = end_time - start_time
                if current_note > 0 and duration >= 0.05:
                    velocity = _estimate_velocity(frequencies, start_time, end_time, times)
                    notes.append({
                        "note": int(current_note),
                        "start": start_time,
                        "duration": duration,
                        "velocity": velocity,
                    })
                current_note = midi_notes[i]
                start_time = times[i]

        if current_note > 0:
            end_time = times[-1]
            duration = end_time - start_time
            if duration >= 0.05:
                velocity = _estimate_velocity(frequencies, start_time, end_time, times)
                notes.append({
                    "note": int(current_note),
                    "start": start_time,
                    "duration": duration,
                    "velocity": velocity,
                })

    # Build final MIDI
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))

    program = _INSTRUMENT_PROGRAMS.get(instrument, 0)
    channel = 9 if instrument == "drums" else 0
    track.append(mido.Message("program_change", program=program, channel=channel, time=0))

    # Track name
    track.append(mido.MetaMessage(
        "track_name", name=f"Exported ({instrument}, {bpm} BPM)", time=0
    ))

    # Note events with correct tempo
    midi_events = []
    for n in notes:
        start_tick = int(mido.second2tick(n["start"], ticks_per_beat, microseconds_per_beat))
        end_tick = int(mido.second2tick(
            n["start"] + n["duration"], ticks_per_beat, microseconds_per_beat
        ))
        midi_events.append(("note_on", start_tick, n["note"], n["velocity"]))
        midi_events.append(("note_off", end_tick, n["note"], 0))

    midi_events.sort(key=lambda e: (e[1], 0 if e[0] == "note_off" else 1))

    last_tick = 0
    for event_type, tick, note, velocity in midi_events:
        delta = tick - last_tick
        track.append(mido.Message(
            event_type, note=note, velocity=velocity,
            channel=channel, time=max(0, delta),
        ))
        last_tick = tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    # Save
    mid.save(str(output_path))
    logger.info(f"MIDI saved: {output_path} ({len(notes)} notes)")

    return str(output_path.resolve())
