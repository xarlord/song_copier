"""MIDI Export tab — convert audio to MIDI files with pitch tracking."""

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)


def create_tab() -> gr.Blocks:
    """Build the MIDI Export tab UI and wire all callbacks."""

    with gr.Blocks() as tab:
        gr.Markdown("## 🎼 MIDI Export")
        gr.Markdown(
            "Upload audio and convert it to a MIDI file using pitch detection. "
            "Works best with **monophonic** audio (single melody, bass line, vocal)."
        )

        # === Input Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Upload Audio")
            input_audio = gr.Audio(
                label="Audio Input",
                type="filepath",
                elem_classes=["upload-zone"],
            )

        # === Settings Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Settings")
            with gr.Row():
                bpm_input = gr.Number(
                    value=120,
                    label="BPM (Tempo)",
                    minimum=30,
                    maximum=300,
                    step=1,
                    info="Beats per minute for MIDI timing",
                )
                instrument_dropdown = gr.Dropdown(
                    choices=["auto", "piano", "guitar", "bass", "drums", "vocals"],
                    value="auto",
                    label="Instrument",
                    info="General MIDI instrument assignment",
                )
            with gr.Row():
                export_btn = gr.Button(
                    "🎼 Export MIDI",
                    variant="primary",
                    size="lg",
                )

        # === Status ===
        status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2,
            elem_classes=["status-box"],
        )

        # === Output Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Output")
            midi_file = gr.File(
                label="Download MIDI",
                file_types=[".mid", ".midi"],
            )

            with gr.Row():
                with gr.Column(scale=2):
                    notes_display = gr.Textbox(
                        label="Detected Notes",
                        interactive=False,
                        lines=10,
                        elem_classes=["status-box"],
                    )
                with gr.Column(scale=3):
                    piano_roll_img = gr.Image(
                        label="Piano Roll",
                        type="filepath",
                    )

        # === Callbacks ===

        def do_export(audio_path, bpm, instrument, progress=gr.Progress()):
            """Handle the export button click."""
            if not audio_path:
                return (
                    "⚠️ Please upload an audio file first.",
                    None,
                    "",
                    None,
                )

            try:
                import librosa
                from services.midi_export import (
                    export_midi,
                    midi_note_to_name,
                    detect_pitch,
                    freq_to_midi_note,
                )
            except ImportError as e:
                return (
                    f"❌ Missing dependency: {e}. Please install librosa and mido.",
                    None,
                    "",
                    None,
                )

            progress(0.1, desc="Loading audio…")

            try:
                # Load audio
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
                logger.info(f"Loaded audio: {duration:.1f}s at {sr} Hz")

                if duration < 0.1:
                    return ("⚠️ Audio is too short for pitch detection.", None, "", None)

                progress(0.3, desc="Detecting pitch…")

                # Detect pitch for display
                times, frequencies = detect_pitch(y, sr)

                progress(0.6, desc="Generating MIDI…")

                # Export MIDI
                output_dir = Path(tempfile.gettempdir()) / "audio_generator" / "midi"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                midi_path = str(output_dir / f"midi_export_{timestamp}.mid")

                bpm = float(bpm) if bpm else 120.0
                saved_path = export_midi(
                    audio=y,
                    sr=sr,
                    output_path=midi_path,
                    instrument=instrument,
                    bpm=bpm,
                )

                progress(0.8, desc="Building note summary…")

                # Build detected notes summary
                notes_summary = _build_notes_summary(times, frequencies)

                progress(0.9, desc="Generating piano roll…")

                # Generate piano roll image
                piano_roll_path = _generate_piano_roll(times, frequencies, bpm)

                progress(1.0, desc="Done!")

                total_notes = notes_summary.count("\n") - 1
                return (
                    f"✅ MIDI exported successfully!\n"
                    f"   Duration: {duration:.1f}s | BPM: {bpm:.0f} | "
                    f"Notes detected: {total_notes}",
                    saved_path,
                    notes_summary,
                    piano_roll_path,
                )

            except Exception as e:
                logger.error(f"MIDI export error: {e}", exc_info=True)
                return (
                    f"❌ Error during export: {e}",
                    None,
                    "",
                    None,
                )

        export_btn.click(
            fn=do_export,
            inputs=[input_audio, bpm_input, instrument_dropdown],
            outputs=[status, midi_file, notes_display, piano_roll_img],
        )

    return tab


def _build_notes_summary(times: np.ndarray, frequencies: np.ndarray) -> str:
    """Build a human-readable summary of detected notes."""
    from services.midi_export import freq_to_midi_note, midi_note_to_name

    voiced_mask = ~np.isnan(frequencies)
    voiced_freqs = frequencies[voiced_mask]
    voiced_times = times[voiced_mask]

    if len(voiced_freqs) == 0:
        return "No pitched notes detected."

    # Group into discrete notes
    midi_notes = [freq_to_midi_note(f) for f in voiced_freqs]
    note_events = []
    current_note = midi_notes[0]
    start_time = voiced_times[0]

    for i in range(1, len(midi_notes)):
        if midi_notes[i] != current_note:
            duration = voiced_times[i] - start_time
            if current_note > 0 and duration >= 0.05:
                note_events.append({
                    "note": current_note,
                    "name": midi_note_to_name(current_note),
                    "start": start_time,
                    "duration": duration,
                })
            current_note = midi_notes[i]
            start_time = voiced_times[i]

    # Last note
    if current_note > 0 and len(voiced_times) > 0:
        duration = voiced_times[-1] - start_time
        if duration >= 0.05:
            note_events.append({
                "note": current_note,
                "name": midi_note_to_name(current_note),
                "start": start_time,
                "duration": duration,
            })

    if not note_events:
        return "No notes detected (audio may be unpitched or too noisy)."

    # Summary header
    unique_notes = sorted(set(e["note"] for e in note_events))
    note_range = f"{midi_note_to_name(min(unique_notes))} – {midi_note_to_name(max(unique_notes))}"

    lines = [
        f"Total notes: {len(note_events)} | Range: {note_range}",
        f"Unique pitches: {len(unique_notes)}",
        "",
        "  Time (s)    Duration   Note",
        "  ─────────────────────────────",
    ]

    # Show up to 50 notes
    for event in note_events[:50]:
        lines.append(
            f"  {event['start']:6.2f}      {event['duration']:5.3f}     {event['name']}"
        )

    if len(note_events) > 50:
        lines.append(f"  … and {len(note_events) - 50} more notes")

    return "\n".join(lines)


def _generate_piano_roll(
    times: np.ndarray,
    frequencies: np.ndarray,
    bpm: float,
) -> str | None:
    """Generate a piano roll visualization as a PNG image.

    Args:
        times: Time positions from pitch detection.
        frequencies: Detected frequencies (may contain NaN).
        bpm: Beats per minute (for beat grid lines).

    Returns:
        Path to the generated image, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from services.midi_export import freq_to_midi_note, midi_note_to_name
    except ImportError:
        logger.warning("matplotlib not available, skipping piano roll")
        return None

    voiced_mask = ~np.isnan(frequencies)
    if not np.any(voiced_mask):
        return None

    voiced_times = times[voiced_mask]
    voiced_notes = np.array([freq_to_midi_note(f) for f in frequencies[voiced_mask]])

    if len(voiced_notes) == 0:
        return None

    # Group into note segments for horizontal bars
    note_events = []
    current_note = voiced_notes[0]
    start_time = voiced_times[0]

    for i in range(1, len(voiced_notes)):
        if voiced_notes[i] != current_note:
            duration = voiced_times[i] - start_time
            if current_note > 0 and duration >= 0.03:
                note_events.append((start_time, duration, int(current_note)))
            current_note = voiced_notes[i]
            start_time = voiced_times[i]

    if current_note > 0:
        duration = voiced_times[-1] - start_time
        if duration >= 0.03:
            note_events.append((start_time, duration, int(current_note)))

    if not note_events:
        return None

    # Determine plot range
    all_notes = [e[2] for e in note_events]
    min_note = max(0, min(all_notes) - 2)
    max_note = min(127, max(all_notes) + 2)
    total_duration = times[-1]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    # Draw beat grid lines
    beat_duration = 60.0 / bpm
    beat_time = 0
    while beat_time <= total_duration:
        color = "#ffffff" if beat_time == 0 else "#333355"
        ax.axvline(x=beat_time, color=color, linewidth=0.5, alpha=0.6)
        beat_time += beat_duration

    # Draw note bars
    for start, duration, note in note_events:
        # Color by pitch height (low=blue, high=red)
        pitch_norm = (note - min_note) / max(1, max_note - min_note)
        color = plt.cm.plasma(pitch_norm)
        ax.barh(
            y=note,
            width=duration,
            left=start,
            height=0.8,
            color=color,
            edgecolor="#ffffff20",
            linewidth=0.5,
        )

    # Piano key labels on Y axis
    y_ticks = range(min_note, max_note + 1)
    y_labels = [midi_note_to_name(n) for n in y_ticks]
    ax.set_yticks(list(y_ticks))
    ax.set_yticklabels(y_labels, fontsize=7, color="#b8b8d0")

    ax.set_xlabel("Time (seconds)", color="#b8b8d0", fontsize=10)
    ax.set_ylabel("Note", color="#b8b8d0", fontsize=10)
    ax.set_title(
        "Piano Roll — Detected Notes",
        color="#e0e0ff",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlim(0, total_duration)
    ax.set_ylim(min_note - 0.5, max_note + 0.5)

    # Style
    ax.tick_params(colors="#8888aa", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333355")
    ax.spines["left"].set_color("#333355")

    fig.tight_layout()

    # Save to temp file
    import tempfile
    output_dir = Path(tempfile.gettempdir()) / "audio_generator" / "midi"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(output_dir / "piano_roll.png")
    fig.savefig(img_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return img_path
