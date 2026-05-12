"""Tab transcription service — convert audio stems to guitar/bass tablature."""

import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# MIDI note name lookup
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# String tuning presets: list of MIDI note numbers from lowest string to highest
TUNINGS = {
    "guitar_standard": [40, 45, 50, 55, 59, 64],   # E2 A2 D3 G3 B3 E4
    "guitar_drop_d":   [38, 45, 50, 55, 59, 64],   # D2 A2 D3 G3 B3 E4
    "guitar_open_g":   [38, 43, 50, 55, 59, 62],   # D2 G2 D3 G3 B3 D4
    "bass_standard":   [28, 33, 38, 43],            # E1 A1 D2 G2
    "bass_drop_d":     [26, 33, 38, 43],            # D1 A1 D2 G2
}

# Human-readable tuning labels
TUNING_LABELS = {
    "guitar_standard": ["E2", "A2", "D3", "G3", "B3", "E4"],
    "guitar_drop_d":   ["D2", "A2", "D3", "G3", "B3", "E4"],
    "guitar_open_g":   ["D2", "G2", "D3", "G3", "B3", "D4"],
    "bass_standard":   ["E1", "A1", "D2", "G2"],
    "bass_drop_d":     ["D1", "A1", "D2", "G2"],
}


def freq_to_midi_note(freq: float) -> int:
    """Convert frequency in Hz to MIDI note number."""
    if freq <= 0 or np.isnan(freq):
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_note_to_name(note: int) -> str:
    """Convert MIDI note number to human-readable name (e.g. C4, A#3)."""
    if note < 0 or note > 127:
        return f"?{note}"
    name = _NOTE_NAMES[note % 12]
    octave = (note // 12) - 1
    return f"{name}{octave}"


def detect_notes(
    audio: np.ndarray,
    sr: int,
    min_duration: float = 0.05,
) -> list[dict]:
    """Detect individual notes from audio using librosa.pyin pitch tracking.

    Args:
        audio: Audio waveform as 1D numpy array. If stereo (2D), mixed to mono.
        sr: Sample rate in Hz.
        min_duration: Minimum note duration in seconds. Shorter notes are discarded.

    Returns:
        List of note dicts with keys:
            start (float), end (float), pitch (str), midi_note (int),
            frequency (float), velocity (int)
    """
    import librosa

    # Ensure mono
    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    audio = audio.astype(np.float32)

    logger.info(f"Detecting notes: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.1f}s)")

    # Choose fmin/fmax based on likely instrument range
    fmin = librosa.note_to_hz("C1")   # ~33 Hz — covers bass guitar low notes
    fmax = librosa.note_to_hz("C7")   # ~2093 Hz — covers guitar high frets

    frequencies, voiced_flags, voiced_probs = librosa.pyin(
        y=audio,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=2048,
        hop_length=512,
    )

    times = librosa.times_like(frequencies, sr=sr, hop_length=512)

    logger.info(
        f"Pitch detection complete: {len(frequencies)} frames, "
        f"{np.sum(~np.isnan(frequencies))} voiced"
    )

    # Quantize to MIDI notes and group consecutive same-pitch frames
    midi_notes_arr = np.array([
        freq_to_midi_note(f) if not np.isnan(f) else 0 for f in frequencies
    ])

    notes: list[dict] = []
    if len(midi_notes_arr) == 0:
        return notes

    current_note = midi_notes_arr[0]
    current_freq = frequencies[0] if not np.isnan(frequencies[0]) else 0.0
    start_time = times[0]
    freq_sum = current_freq
    freq_count = 1 if current_freq > 0 else 0

    for i in range(1, len(midi_notes_arr)):
        if midi_notes_arr[i] != current_note:
            end_time = times[i]
            duration = end_time - start_time

            if current_note > 0 and duration >= min_duration:
                avg_freq = freq_sum / max(1, freq_count)
                velocity = _estimate_velocity_from_probs(
                    voiced_probs, start_time, end_time, times
                )
                notes.append({
                    "start": float(start_time),
                    "end": float(end_time),
                    "pitch": midi_note_to_name(int(current_note)),
                    "midi_note": int(current_note),
                    "frequency": float(avg_freq),
                    "velocity": velocity,
                })

            current_note = midi_notes_arr[i]
            current_freq = frequencies[i] if not np.isnan(frequencies[i]) else 0.0
            start_time = times[i]
            freq_sum = current_freq
            freq_count = 1 if current_freq > 0 else 0
        else:
            f = frequencies[i]
            if not np.isnan(f) and f > 0:
                freq_sum += f
                freq_count += 1

    # Handle last note segment
    if current_note > 0 and len(times) > 0:
        end_time = times[-1]
        duration = end_time - start_time
        if duration >= min_duration:
            avg_freq = freq_sum / max(1, freq_count)
            velocity = _estimate_velocity_from_probs(
                voiced_probs, start_time, end_time, times
            )
            notes.append({
                "start": float(start_time),
                "end": float(end_time),
                "pitch": midi_note_to_name(int(current_note)),
                "midi_note": int(current_note),
                "frequency": float(avg_freq),
                "velocity": velocity,
            })

    logger.info(f"Detected {len(notes)} notes")
    return notes


def _estimate_velocity_from_probs(
    voiced_probs: np.ndarray,
    start_time: float,
    end_time: float,
    times: np.ndarray,
) -> int:
    """Estimate MIDI velocity from pyin voiced probability."""
    mask = (times >= start_time) & (times < end_time)
    if np.sum(mask) == 0:
        return 80
    avg_prob = float(np.mean(voiced_probs[mask]))
    velocity = int(50 + 77 * avg_prob)  # Range 50–127
    return max(1, min(127, velocity))


def _find_string_fret(
    midi_note: int,
    tuning_name: str = "guitar_standard",
    num_frets: int = 22,
) -> tuple[int, int] | None:
    """Find the best (string_index, fret) for a MIDI note, preferring lower frets.

    Args:
        midi_note: MIDI note number.
        tuning_name: Key from TUNINGS dict.
        num_frets: Maximum fret number.

    Returns:
        Tuple of (string_index, fret) or None if note is out of range.
        string_index is 0-based from lowest string.
    """
    tuning = TUNINGS.get(tuning_name, TUNINGS["guitar_standard"])

    best = None
    best_fret = num_frets + 1

    for string_idx, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret <= num_frets:
            # Prefer lowest fret; among ties prefer higher string (thinner)
            if fret < best_fret or (fret == best_fret and best is not None and string_idx > best[0]):
                best_fret = fret
                best = (string_idx, fret)

    return best


def midi_to_guitar_tab(
    notes: list[dict],
    tuning: str = "standard",
    num_frets: int = 22,
    instrument: str = "guitar",
) -> dict:
    """Convert a list of detected notes to guitar/bass tablature data.

    Args:
        notes: List of note dicts from detect_notes().
        tuning: Tuning name: 'standard', 'drop_d', 'open_g'.
        num_frets: Maximum fret number on the instrument.
        instrument: 'guitar' or 'bass'.

    Returns:
        Dict with:
            strings: list of per-string note lists [{fret, start, end}, ...]
            tuning: list of note names for each string
            tuning_name: the resolved tuning key used
            techniques: list of detected techniques [{type, string, fret, time}, ...]
    """
    # Resolve tuning key
    tuning_name = _resolve_tuning_key(instrument, tuning)
    tuning_midi = TUNINGS[tuning_name]
    tuning_labels = TUNING_LABELS[tuning_name]
    num_strings = len(tuning_midi)

    # Initialize strings
    strings_data: list[list[dict]] = [[] for _ in range(num_strings)]

    # Track string assignments for technique detection
    assignments: list[dict] = []

    for note in notes:
        midi = note["midi_note"]
        result = _find_string_fret(midi, tuning_name, num_frets)

        if result is None:
            # Note out of range — skip or assign to nearest fret on lowest string
            logger.debug(f"Note {midi_note_to_name(midi)} ({midi}) out of range for {tuning_name}")
            continue

        string_idx, fret = result
        entry = {
            "fret": fret,
            "start": note["start"],
            "end": note["end"],
            "midi_note": midi,
            "velocity": note.get("velocity", 80),
        }
        strings_data[string_idx].append(entry)
        assignments.append({
            "string": string_idx,
            "fret": fret,
            "start": note["start"],
            "end": note["end"],
        })

    # Sort each string by start time
    for s in strings_data:
        s.sort(key=lambda x: x["start"])

    # Detect techniques (hammer-ons, pull-offs, slides)
    techniques = _detect_techniques(assignments)

    return {
        "strings": strings_data,
        "tuning": tuning_labels,
        "tuning_name": tuning_name,
        "techniques": techniques,
        "num_frets": num_frets,
        "instrument": instrument,
    }


def _resolve_tuning_key(instrument: str, tuning: str) -> str:
    """Resolve instrument + tuning name to a TUNINGS key."""
    instrument = instrument.lower().strip()
    tuning = tuning.lower().strip()

    if instrument == "bass":
        if tuning in ("drop_d", "drop d"):
            return "bass_drop_d"
        return "bass_standard"

    # Guitar
    if tuning in ("drop_d", "drop d"):
        return "guitar_drop_d"
    if tuning in ("open_g", "open g"):
        return "guitar_open_g"
    return "guitar_standard"


def _detect_techniques(assignments: list[dict]) -> list[dict]:
    """Detect hammer-ons, pull-offs, and slides from sequential note assignments.

    A hammer-on: same string, next note fret is 1-3 higher, gap < 0.05s
    A pull-off: same string, next note fret is 1-3 lower, gap < 0.05s
    A slide: same string, fret difference 2-5, gap < 0.08s
    """
    techniques = []
    # Sort by start time
    sorted_assignments = sorted(assignments, key=lambda x: x["start"])

    for i in range(len(sorted_assignments) - 1):
        curr = sorted_assignments[i]
        nxt = sorted_assignments[i + 1]

        if curr["string"] != nxt["string"]:
            continue

        gap = nxt["start"] - curr["end"]
        fret_diff = nxt["fret"] - curr["fret"]

        if abs(fret_diff) < 1:
            continue

        if 0 < fret_diff <= 3 and gap < 0.05:
            techniques.append({
                "type": "hammer_on",
                "string": curr["string"],
                "from_fret": curr["fret"],
                "to_fret": nxt["fret"],
                "time": nxt["start"],
            })
        elif -3 <= fret_diff < 0 and gap < 0.05:
            techniques.append({
                "type": "pull_off",
                "string": curr["string"],
                "from_fret": curr["fret"],
                "to_fret": nxt["fret"],
                "time": nxt["start"],
            })
        elif 2 <= abs(fret_diff) <= 5 and gap < 0.08:
            techniques.append({
                "type": "slide",
                "string": curr["string"],
                "from_fret": curr["fret"],
                "to_fret": nxt["fret"],
                "time": nxt["start"],
            })

    return techniques


def render_ascii_tab(
    tab_data: dict,
    measures: int = 4,
    beats_per_measure: int = 4,
    bpm: float = 120.0,
) -> str:
    """Render tab data as ASCII tablature string.

    Args:
        tab_data: Output from midi_to_guitar_tab().
        measures: Number of measures to display per line.
        beats_per_measure: Time signature numerator.
        bpm: Beats per minute for measure timing.

    Returns:
        Multi-line ASCII tab string.
    """
    strings_data = tab_data["strings"]
    tuning_labels = tab_data["tuning"]
    num_strings = len(strings_data)
    techniques = tab_data.get("techniques", [])

    if not any(strings_data):
        return "No notes to display in tablature."

    beat_duration = 60.0 / bpm if bpm > 0 else 0.5
    measure_duration = beat_duration * beats_per_measure
    section_duration = measure_duration * measures

    # Find overall time range
    all_notes = []
    for s_idx, s_notes in enumerate(strings_data):
        for n in s_notes:
            all_notes.append((n["start"], n["end"], s_idx, n["fret"]))

    if not all_notes:
        return "No notes to display in tablature."

    max_time = max(n[1] for n in all_notes)

    # Build technique lookup: (string, time) -> technique type
    tech_lookup: dict[tuple[int, float], str] = {}
    for tech in techniques:
        tech_lookup[(tech["string"], round(tech["time"], 4))] = tech["type"]

    # Quantize to a grid for display
    # Use 16th note resolution (4 subdivisions per beat)
    subdivision = beat_duration / 4.0
    total_steps = int(np.ceil(max_time / subdivision)) + 1

    # Create grid: grid[string_idx][step] = fret number or None
    grid: list[list[int | None]] = [[None] * total_steps for _ in range(num_strings)]

    for start, end, s_idx, fret in all_notes:
        start_step = int(round(start / subdivision))
        end_step = int(round(end / subdivision))
        start_step = max(0, min(start_step, total_steps - 1))
        end_step = max(0, min(end_step, total_steps - 1))
        # Place the fret number at the start step
        grid[s_idx][start_step] = fret
        # Fill intermediate steps with -1 (sustain marker)
        for step in range(start_step + 1, end_step + 1):
            if step < total_steps and grid[s_idx][step] is None:
                grid[s_idx][step] = -1  # sustain

    # Render sections
    lines: list[str] = []
    num_sections = max(1, int(np.ceil(max_time / section_duration)))

    for sec in range(num_sections):
        step_start = int(sec * section_duration / subdivision)
        step_end = min(total_steps, int((sec + 1) * section_duration / subdivision))

        if step_start >= total_steps:
            break

        # Measure headers
        for m in range(measures):
            measure_num = sec * measures + m + 1
            measure_start_step = int((sec * section_duration + m * measure_duration) / subdivision)
            if measure_start_step < total_steps:
                lines.append(f"  Measure {measure_num}")
        lines.append("")

        # Build string lines
        string_lines: list[str] = []
        for s_idx in range(num_strings):
            label = tuning_labels[s_idx].ljust(3)
            parts = [f"{label}|"]
            for step in range(step_start, step_end):
                val = grid[s_idx][step] if step < total_steps else None
                if val is None:
                    parts.append("-")
                elif val == -1:
                    parts.append("-")  # sustain shown as dash
                else:
                    fret_str = str(val)
                    parts.append(fret_str)
            parts.append("|")
            string_lines.append("".join(parts))

        lines.extend(string_lines)
        lines.append("")

    # Add technique legend if any
    if techniques:
        lines.append("Techniques:")
        tech_counts: dict[str, int] = {}
        for t in techniques:
            tech_counts[t["type"]] = tech_counts.get(t["type"], 0) + 1
        for t_type, count in tech_counts.items():
            label = t_type.replace("_", " ").title()
            lines.append(f"  {label}: {count}x")
        lines.append("")

    return "\n".join(lines)


def render_chord_diagram(
    notes: list[dict],
    start_time: float,
    end_time: float,
    tuning: str = "standard",
    instrument: str = "guitar",
    num_frets: int = 22,
) -> str | None:
    """Generate a chord fingering diagram as a matplotlib figure.

    Shows notes active between start_time and end_time as a chord diagram
    with string positions and fret indicators.

    Args:
        notes: List of note dicts from detect_notes().
        start_time: Start of time window.
        end_time: End of time window.
        tuning: Tuning name.
        instrument: 'guitar' or 'bass'.
        num_frets: Max frets.

    Returns:
        Path to saved PNG image, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        logger.warning("matplotlib not available for chord diagram")
        return None

    tuning_name = _resolve_tuning_key(instrument, tuning)
    tuning_midi = TUNINGS[tuning_name]
    tuning_labels = TUNING_LABELS[tuning_name]
    num_strings = len(tuning_midi)

    # Find notes active in the time window
    active_notes = [
        n for n in notes
        if n["start"] < end_time and n["end"] > start_time
    ]

    if not active_notes:
        return None

    # Map to string/fret
    positions: dict[int, int] = {}  # string_idx -> fret
    for note in active_notes:
        result = _find_string_fret(note["midi_note"], tuning_name, num_frets)
        if result is not None:
            s_idx, fret = result
            if s_idx not in positions or fret < positions[s_idx]:
                positions[s_idx] = fret

    if not positions:
        return None

    # Determine fret range to display (5-fret window around the chord)
    active_frets = [f for f in positions.values() if f > 0]
    if not active_frets:
        # All open strings
        min_fret_display = 0
        max_fret_display = 5
    else:
        min_fret_display = max(0, min(active_frets) - 1)
        max_fret_display = min_fret_display + 5
        if max_fret_display < max(active_frets) + 1:
            max_fret_display = max(active_frets) + 1

    num_display_frets = max_fret_display - min_fret_display

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 5.5), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    margin_left = 0.8
    margin_right = 0.4
    margin_top = 0.6
    margin_bottom = 0.8

    fret_spacing = 1.0
    string_spacing = 0.8

    # Draw nut (thick line at top if starting from fret 0)
    if min_fret_display == 0:
        nut_y = margin_top
        ax.plot(
            [margin_left - string_spacing / 2, margin_left + (num_strings - 1) * string_spacing + string_spacing / 2],
            [nut_y, nut_y],
            color="#e0e0ff",
            linewidth=4,
            solid_capstyle="round",
        )
    else:
        # Show fret number indicator
        ax.text(
            margin_left - string_spacing,
            margin_top + fret_spacing / 2,
            f"{min_fret_display + 1}fr",
            color="#b8b8d0",
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
        )

    # Draw frets (horizontal lines)
    for f in range(num_display_frets + 1):
        y = margin_top - f * fret_spacing
        ax.plot(
            [margin_left - string_spacing / 2, margin_left + (num_strings - 1) * string_spacing + string_spacing / 2],
            [y, y],
            color="#555570",
            linewidth=1,
        )

    # Draw strings (vertical lines)
    for s in range(num_strings):
        x = margin_left + s * string_spacing
        ax.plot(
            [x, x],
            [margin_top, margin_top - num_display_frets * fret_spacing],
            color="#aaaacc",
            linewidth=1.5 - s * 0.1,  # Thicker for lower strings
        )
        # String label at bottom
        ax.text(
            x,
            margin_top - num_display_frets * fret_spacing - 0.35,
            tuning_labels[s],
            color="#8888bb",
            fontsize=8,
            ha="center",
            va="top",
        )

    # Draw finger positions
    for s_idx, fret in positions.items():
        x = margin_left + s_idx * string_spacing
        if fret == 0:
            # Open string — draw circle above nut
            ax.plot(x, margin_top + 0.25, "o", color="#66bbff", markersize=8)
            ax.text(x, margin_top + 0.25, "O", color="#1a1a2e", fontsize=6, ha="center", va="center", fontweight="bold")
        else:
            display_fret = fret - min_fret_display
            y = margin_top - (display_fret - 0.5) * fret_spacing
            circle = plt.Circle((x, y), 0.28, color="#ff6688", ec="#ffffff40", linewidth=1)
            ax.add_patch(circle)
            ax.text(x, y, str(fret), color="white", fontsize=10, ha="center", va="center", fontweight="bold")

    # Mark muted strings (not in chord)
    for s in range(num_strings):
        if s not in positions:
            x = margin_left + s * string_spacing
            ax.text(x, margin_top + 0.25, "X", color="#ff6666", fontsize=9, ha="center", va="center", fontweight="bold")

    # Title
    chord_name = _identify_chord(positions, tuning_midi)
    ax.set_title(
        f"Chord: {chord_name}\n({start_time:.2f}s – {end_time:.2f}s)",
        color="#e0e0ff",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    ax.set_xlim(margin_left - string_spacing * 1.2, margin_left + num_strings * string_spacing)
    ax.set_ylim(margin_top - num_display_frets * fret_spacing - 0.8, margin_top + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()

    # Save
    output_dir = Path(tempfile.gettempdir()) / "audio_generator" / "tabs"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(output_dir / "chord_diagram.png")
    fig.savefig(img_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return img_path


def _identify_chord(positions: dict[int, int], tuning_midi: list[int]) -> str:
    """Attempt to identify a chord name from string/fret positions."""
    if not positions:
        return "N/A"

    # Build list of MIDI notes in the chord
    midi_notes = []
    for s_idx, fret in positions.items():
        midi_notes.append(tuning_midi[s_idx] + fret)

    if not midi_notes:
        return "N/A"

    # Normalize to pitch classes
    pitch_classes = sorted(set(n % 12 for n in midi_notes))

    # Simple chord detection by interval pattern
    _PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    root = min(midi_notes) % 12
    root_name = _PC_NAMES[root]

    intervals = sorted([(pc - root) % 12 for pc in pitch_classes])

    # Major triad: [0, 4, 7]
    if set(intervals) == {0, 4, 7}:
        return f"{root_name}"
    # Minor triad: [0, 3, 7]
    if set(intervals) == {0, 3, 7}:
        return f"{root_name}m"
    # Power chord: [0, 7]
    if set(intervals) == {0, 7}:
        return f"{root_name}5"
    # Diminished: [0, 3, 6]
    if set(intervals) == {0, 3, 6}:
        return f"{root_name}dim"
    # Augmented: [0, 4, 8]
    if set(intervals) == {0, 4, 8}:
        return f"{root_name}aug"
    # Major 7th: [0, 4, 7, 11]
    if set(intervals) == {0, 4, 7, 11}:
        return f"{root_name}maj7"
    # Minor 7th: [0, 3, 7, 10]
    if set(intervals) == {0, 3, 7, 10}:
        return f"{root_name}m7"
    # Dominant 7th: [0, 4, 7, 10]
    if set(intervals) == {0, 4, 7, 10}:
        return f"{root_name}7"
    # Suspended 2: [0, 2, 7]
    if set(intervals) == {0, 2, 7}:
        return f"{root_name}sus2"
    # Suspended 4: [0, 5, 7]
    if set(intervals) == {0, 5, 7}:
        return f"{root_name}sus4"

    # Fallback: list the notes
    note_names = sorted(set(midi_note_to_name(n) for n in midi_notes))
    return "/".join(note_names)


def tab_to_text(tab_data: dict, bpm: float = 120.0) -> str:
    """Export tab data as a plain text file content.

    Args:
        tab_data: Output from midi_to_guitar_tab().
        bpm: Beats per minute.

    Returns:
        Plain text string suitable for saving as .txt.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("GUITAR TAB TRANSCRIPTION")
    lines.append("=" * 60)
    lines.append(f"Instrument: {tab_data.get('instrument', 'guitar').title()}")
    lines.append(f"Tuning: {' - '.join(tab_data['tuning'])}")
    lines.append(f"BPM: {bpm}")
    lines.append(f"Frets: 0-{tab_data.get('num_frets', 22)}")
    lines.append("")

    # Technique summary
    techniques = tab_data.get("techniques", [])
    if techniques:
        lines.append("Techniques detected:")
        tech_counts: dict[str, int] = {}
        for t in techniques:
            tech_counts[t["type"]] = tech_counts.get(t["type"], 0) + 1
        for t_type, count in tech_counts.items():
            label = t_type.replace("_", " ").title()
            lines.append(f"  {label}: {count}")
        lines.append("")

    # Per-string note listing
    lines.append("-" * 60)
    lines.append("NOTE LISTING BY STRING")
    lines.append("-" * 60)

    for s_idx, s_notes in enumerate(tab_data["strings"]):
        string_label = tab_data["tuning"][s_idx]
        lines.append(f"\nString {s_idx + 1} ({string_label}):")
        if not s_notes:
            lines.append("  (no notes)")
            continue
        for n in s_notes:
            lines.append(
                f"  Fret {n['fret']:>2d}  |  "
                f"{n['start']:.3f}s – {n['end']:.3f}s  "
                f"(dur: {n['end'] - n['start']:.3f}s)"
            )

    # ASCII tab
    lines.append("")
    lines.append("-" * 60)
    lines.append("TABLATURE")
    lines.append("-" * 60)
    lines.append("")
    ascii_tab = render_ascii_tab(tab_data, bpm=bpm)
    lines.append(ascii_tab)

    return "\n".join(lines)


def export_musicxml(
    notes: list[dict],
    tab_data: dict,
    output_path: str,
    bpm: float = 120.0,
) -> str:
    """Export transcription as MusicXML format (readable by Guitar Pro, MuseScore, etc.).

    Args:
        notes: List of note dicts from detect_notes().
        tab_data: Output from midi_to_guitar_tab().
        output_path: Path to save the .musicxml file.
        bpm: Tempo in BPM.

    Returns:
        Absolute path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".musicxml":
        output_path = output_path.with_suffix(".musicxml")

    tuning_name = tab_data.get("tuning_name", "guitar_standard")
    tuning_midi = TUNINGS[tuning_name]
    tuning_labels = tab_data["tuning"]
    num_strings = len(tuning_midi)
    instrument = tab_data.get("instrument", "guitar")
    techniques = tab_data.get("techniques", [])

    divisions = 4  # Quarter note divisions (16th note resolution)

    # Build XML
    root = ET.Element("score-partwise", version="4.0")

    # Work
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = "Tab Transcription"

    # Part list
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = instrument.title()

    # Staff details (string tuning)
    score_instrument = ET.SubElement(score_part, "score-instrument", id="P1-I1")
    ET.SubElement(score_instrument, "instrument-name").text = instrument.title()
    midi_instrument = ET.SubElement(score_part, "midi-instrument", id="P1-I1")
    ET.SubElement(midi_instrument, "midi-channel").text = "1"
    program = 24 if instrument == "guitar" else 32
    ET.SubElement(midi_instrument, "midi-program").text = str(program + 1)

    # Part
    part = ET.SubElement(root, "part", id="P1")

    # Attributes for first measure
    measure_num = 1
    measure = ET.SubElement(part, "measure", number=str(measure_num))
    attributes = ET.SubElement(measure, "attributes")

    ET.SubElement(attributes, "divisions").text = str(divisions)

    # Key (C major)
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = "0"

    # Time signature
    time_sig = ET.SubElement(attributes, "time")
    ET.SubElement(time_sig, "beats").text = "4"
    ET.SubElement(time_sig, "beat-type").text = "4"

    # Clef (tab)
    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "TAB"
    ET.SubElement(clef, "line").text = "5"

    # Standard clef for pitch display
    clef2 = ET.SubElement(attributes, "clef", number="2")
    ET.SubElement(clef2, "sign").text = "F"
    ET.SubElement(clef2, "line").text = "4"

    # Staff details
    staff_details = ET.SubElement(attributes, "staff-details")
    ET.SubElement(staff_details, "staff-lines").text = str(num_strings)
    for s_idx in range(num_strings):
        staff_tuning = ET.SubElement(staff_details, "staff-tuning", line=str(s_idx + 1))
        midi = tuning_midi[s_idx]
        note_name = _NOTE_NAMES[midi % 12]
        octave = (midi // 12) - 1
        ET.SubElement(staff_tuning, "tuning-step").text = note_name.replace("#", "")
        if "#" in note_name:
            ET.SubElement(staff_tuning, "tuning-alter").text = "1"
        ET.SubElement(staff_tuning, "tuning-octave").text = str(octave)

    # Tempo
    direction = ET.SubElement(measure, "direction", placement="above")
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(int(bpm))

    # Technique lookup for annotations
    tech_times: dict[float, str] = {}
    for tech in techniques:
        tech_times[round(tech["time"], 4)] = tech["type"]

    # Quantize notes to measures
    beat_duration = 60.0 / bpm if bpm > 0 else 0.5
    measure_duration = beat_duration * 4  # 4/4 time
    quarter_duration = beat_duration
    division_duration = quarter_duration / divisions

    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda n: n["start"])

    current_measure_start = 0.0
    current_position_in_measure = 0  # in divisions

    for note_data in sorted_notes:
        start = note_data["start"]
        duration = note_data["end"] - note_data["start"]
        midi = note_data["midi_note"]

        # Which measure?
        note_measure = int(start / measure_duration)
        note_measure_start = note_measure * measure_duration

        # Create new measure if needed
        if note_measure + 1 > measure_num:
            measure_num = note_measure + 1
            measure = ET.SubElement(part, "measure", number=str(measure_num))

        # Position within measure in divisions
        offset_from_measure_start = start - note_measure_start
        start_div = int(round(offset_from_measure_start / division_duration))
        duration_div = max(1, int(round(duration / division_duration)))

        # Create note element
        note_el = ET.SubElement(measure, "note")

        # Check if this is a chord (overlapping with previous)
        # Simple approach: add chord tag if same start time as previous

        # Pitch
        pitch = ET.SubElement(note_el, "pitch")
        step_name = _NOTE_NAMES[midi % 12]
        octave = (midi // 12) - 1
        ET.SubElement(pitch, "step").text = step_name.replace("#", "")
        if "#" in step_name:
            ET.SubElement(pitch, "alter").text = "1"
        ET.SubElement(pitch, "octave").text = str(octave)

        ET.SubElement(note_el, "duration").text = str(duration_div)

        # Velocity
        velocity = note_data.get("velocity", 80)
        dynamics = max(1, min(127, velocity))
        ET.SubElement(note_el, "velocity").text = str(dynamics)

        # Tab notation
        notations = ET.SubElement(note_el, "notations")

        result = _find_string_fret(midi, tuning_name, tab_data.get("num_frets", 22))
        if result is not None:
            s_idx, fret = result
            technical = ET.SubElement(notations, "technical")
            string_el = ET.SubElement(technical, "string")
            string_el.text = str(s_idx + 1)
            fret_el = ET.SubElement(technical, "fret")
            fret_el.text = str(fret)

        # Technique annotations
        rounded_start = round(start, 4)
        if rounded_start in tech_times:
            tech_type = tech_times[rounded_start]
            if tech_type == "hammer_on":
                ET.SubElement(notations, "technical")
                # Add hammer-on articulation
                ET.SubElement(notations, "articulations")
            elif tech_type == "slide":
                ET.SubElement(notations, "technical")

    # Write XML
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml_str += '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
    xml_str += ET.tostring(root, encoding="unicode", xml_declaration=False)

    output_path.write_text(xml_str, encoding="utf-8")

    logger.info(f"MusicXML exported: {output_path}")
    return str(output_path.resolve())


def export_guitar_pro(
    notes: list[dict],
    tab_data: dict,
    output_path: str,
    bpm: float = 120.0,
) -> str:
    """Export transcription — delegates to MusicXML format.

    Guitar Pro 7+ can import MusicXML directly.

    Args:
        notes: List of note dicts from detect_notes().
        tab_data: Output from midi_to_guitar_tab().
        output_path: Path to save the file.
        bpm: Tempo in BPM.

    Returns:
        Absolute path to the saved file.
    """
    return export_musicxml(notes, tab_data, output_path, bpm)


def auto_detect_bpm(audio: np.ndarray, sr: int) -> float:
    """Auto-detect BPM from audio using librosa.

    Args:
        audio: Audio waveform (mono or stereo).
        sr: Sample rate.

    Returns:
        Detected BPM as float.
    """
    import librosa

    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    audio = audio.astype(np.float32)

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    return float(tempo)
