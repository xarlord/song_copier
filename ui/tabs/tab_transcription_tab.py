"""Tab Transcription tab — convert audio to guitar/bass tablature with visualization."""

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)


def create_tab() -> gr.Blocks:
    """Build the Tab Transcription tab UI and wire all callbacks."""

    with gr.Blocks() as tab:
        gr.Markdown("## 🎸 Tab Transcription")
        gr.Markdown(
            "Upload audio and convert it to guitar or bass tablature using pitch detection. "
            "Works best with **monophonic** audio (single melody line, bass line, or solo). "
            "Supports standard guitar, drop D, open G, and bass tunings."
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
            gr.Markdown("### Instrument & Settings")
            with gr.Row():
                instrument_dropdown = gr.Dropdown(
                    choices=[
                        "Guitar (Standard)",
                        "Guitar (Drop D)",
                        "Guitar (Open G)",
                        "Bass (Standard)",
                    ],
                    value="Guitar (Standard)",
                    label="Instrument & Tuning",
                    info="Select instrument and tuning preset",
                )
                bpm_input = gr.Number(
                    value=120,
                    label="BPM (Tempo)",
                    minimum=30,
                    maximum=300,
                    step=1,
                    info="Beats per minute. Set to 0 to auto-detect.",
                )
            with gr.Row():
                min_duration_slider = gr.Slider(
                    minimum=0.02,
                    maximum=0.2,
                    value=0.05,
                    step=0.01,
                    label="Min Note Duration (s)",
                    info="Notes shorter than this are discarded",
                )
                num_frets_slider = gr.Slider(
                    minimum=12,
                    maximum=24,
                    value=22,
                    step=1,
                    label="Number of Frets",
                    info="Maximum fret number on the instrument",
                )
            with gr.Row():
                transcribe_btn = gr.Button(
                    "🎸 Transcribe Tab",
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
            gr.Markdown("### Tablature Output")

            ascii_tab_output = gr.Textbox(
                label="ASCII Tablature",
                interactive=False,
                lines=18,
                max_lines=40,
                elem_classes=["status-box"],
                show_copy_button=True,
            )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Chord Diagram")
                chord_img = gr.Image(
                    label="Chord Fingering",
                    type="filepath",
                )
                with gr.Row():
                    chord_time_slider = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=0.0,
                        step=0.1,
                        label="Chord Window Start (s)",
                    )
                    chord_window_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        label="Chord Window Duration (s)",
                    )
                    render_chord_btn = gr.Button("🎹 Render Chord")

            with gr.Column(scale=3):
                gr.Markdown("### Detected Notes")
                notes_table = gr.Dataframe(
                    headers=["Start (s)", "End (s)", "Note", "String", "Fret", "Technique"],
                    label="Note List",
                    interactive=False,
                    wrap=True,
                )

        # === Export Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Export")
            with gr.Row():
                export_txt_btn = gr.Button("📄 Export .txt")
                export_musicxml_btn = gr.Button("🎵 Export .musicxml")
                export_midi_preview_btn = gr.Button("🔊 MIDI Preview")

            with gr.Row():
                txt_file = gr.File(label="Download .txt")
                musicxml_file = gr.File(label="Download .musicxml")
                midi_preview_file = gr.File(label="Download MIDI Preview")

        # === Hidden state for sharing between callbacks ===
        detected_notes_state = gr.State([])
        tab_data_state = gr.State({})
        instrument_state = gr.State("Guitar (Standard)")
        bpm_state = gr.State(120.0)
        audio_duration_state = gr.State(0.0)

        # === Callbacks ===

        def _parse_instrument(instrument_str: str) -> tuple[str, str]:
            """Parse dropdown value to (instrument, tuning) pair."""
            mapping = {
                "Guitar (Standard)": ("guitar", "standard"),
                "Guitar (Drop D)": ("guitar", "drop_d"),
                "Guitar (Open G)": ("guitar", "open_g"),
                "Bass (Standard)": ("bass", "standard"),
            }
            return mapping.get(instrument_str, ("guitar", "standard"))

        def do_transcribe(audio_path, instrument_str, bpm, min_duration, num_frets, progress=gr.Progress()):
            """Handle the transcribe button click."""
            if not audio_path:
                return (
                    "⚠️ Please upload an audio file first.",
                    "",
                    None,
                    [],
                    {},
                    "Guitar (Standard)",
                    120.0,
                    0.0,
                    None,
                    gr.update(maximum=0.0),
                )

            try:
                import librosa
                from services.tab_transcription import (
                    detect_notes,
                    midi_to_guitar_tab,
                    render_ascii_tab,
                    auto_detect_bpm,
                )
            except ImportError as e:
                return (
                    f"❌ Missing dependency: {e}. Please install librosa.",
                    "",
                    None,
                    [],
                    {},
                    instrument_str,
                    float(bpm) if bpm else 120.0,
                    0.0,
                    None,
                    gr.update(maximum=0.0),
                )

            progress(0.1, desc="Loading audio…")

            try:
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
                logger.info(f"Loaded audio: {duration:.1f}s at {sr} Hz")

                if duration < 0.1:
                    return (
                        "⚠️ Audio is too short for pitch detection.",
                        "",
                        None,
                        [],
                        {},
                        instrument_str,
                        float(bpm) if bpm else 120.0,
                        0.0,
                        None,
                        gr.update(maximum=0.0),
                    )

                # Auto-detect BPM if set to 0
                actual_bpm = float(bpm) if bpm else 120.0
                if actual_bpm <= 0:
                    progress(0.15, desc="Auto-detecting BPM…")
                    actual_bpm = auto_detect_bpm(y, sr)
                    logger.info(f"Auto-detected BPM: {actual_bpm:.1f}")

                progress(0.3, desc="Detecting notes…")

                instrument, tuning = _parse_instrument(instrument_str)
                notes = detect_notes(y, sr, min_duration=float(min_duration))

                if not notes:
                    return (
                        "⚠️ No notes detected. Audio may be unpitched or too noisy.",
                        "",
                        None,
                        [],
                        {},
                        instrument_str,
                        actual_bpm,
                        duration,
                        None,
                        gr.update(maximum=duration),
                    )

                progress(0.6, desc="Generating tablature…")

                tab_data = midi_to_guitar_tab(
                    notes,
                    tuning=tuning,
                    num_frets=int(num_frets),
                    instrument=instrument,
                )

                progress(0.8, desc="Rendering ASCII tab…")

                ascii_tab = render_ascii_tab(tab_data, bpm=actual_bpm)

                progress(0.9, desc="Building note table…")

                # Build note table with string/fret info
                table_rows = _build_note_table(notes, tab_data)

                # Generate initial chord diagram
                chord_path = _render_initial_chord(notes, instrument, tuning, num_frets)

                progress(1.0, desc="Done!")

                technique_count = len(tab_data.get("techniques", []))
                return (
                    f"✅ Tab transcription complete!\n"
                    f"   Duration: {duration:.1f}s | BPM: {actual_bpm:.0f} | "
                    f"Notes: {len(notes)} | Techniques: {technique_count}",
                    ascii_tab,
                    chord_path,
                    notes,
                    tab_data,
                    instrument_str,
                    actual_bpm,
                    duration,
                    table_rows,
                    gr.update(maximum=duration),
                )

            except Exception as e:
                logger.error(f"Tab transcription error: {e}", exc_info=True)
                return (
                    f"❌ Error during transcription: {e}",
                    "",
                    None,
                    [],
                    {},
                    instrument_str,
                    float(bpm) if bpm else 120.0,
                    0.0,
                    None,
                    gr.update(maximum=0.0),
                )

        transcribe_btn.click(
            fn=do_transcribe,
            inputs=[
                input_audio,
                instrument_dropdown,
                bpm_input,
                min_duration_slider,
                num_frets_slider,
            ],
            outputs=[
                status,
                ascii_tab_output,
                chord_img,
                detected_notes_state,
                tab_data_state,
                instrument_state,
                bpm_state,
                audio_duration_state,
                notes_table,
                chord_time_slider,
            ],
        )

        def do_render_chord(notes, instrument_str, start_time, window_dur):
            """Render a chord diagram for the selected time window."""
            if not notes:
                return None

            try:
                from services.tab_transcription import render_chord_diagram

                instrument, tuning = _parse_instrument(instrument_str)
                img_path = render_chord_diagram(
                    notes,
                    start_time=float(start_time),
                    end_time=float(start_time) + float(window_dur),
                    tuning=tuning,
                    instrument=instrument,
                )
                return img_path
            except Exception as e:
                logger.error(f"Chord render error: {e}", exc_info=True)
                return None

        render_chord_btn.click(
            fn=do_render_chord,
            inputs=[
                detected_notes_state,
                instrument_state,
                chord_time_slider,
                chord_window_slider,
            ],
            outputs=[chord_img],
        )

        def do_export_txt(tab_data, instrument_str, bpm):
            """Export tab as .txt file."""
            if not tab_data or not tab_data.get("strings"):
                return None

            try:
                from services.tab_transcription import tab_to_text

                content = tab_to_text(tab_data, bpm=float(bpm))

                output_dir = Path(tempfile.gettempdir()) / "audio_generator" / "tabs"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                txt_path = str(output_dir / f"tab_export_{timestamp}.txt")
                Path(txt_path).write_text(content, encoding="utf-8")

                return txt_path
            except Exception as e:
                logger.error(f"TXT export error: {e}", exc_info=True)
                return None

        export_txt_btn.click(
            fn=do_export_txt,
            inputs=[tab_data_state, instrument_state, bpm_state],
            outputs=[txt_file],
        )

        def do_export_musicxml(notes, tab_data, instrument_str, bpm):
            """Export as MusicXML file."""
            if not notes or not tab_data:
                return None

            try:
                from services.tab_transcription import export_musicxml

                output_dir = Path(tempfile.gettempdir()) / "audio_generator" / "tabs"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                xml_path = str(output_dir / f"tab_export_{timestamp}.musicxml")

                saved = export_musicxml(
                    notes, tab_data, xml_path, bpm=float(bpm)
                )
                return saved
            except Exception as e:
                logger.error(f"MusicXML export error: {e}", exc_info=True)
                return None

        export_musicxml_btn.click(
            fn=do_export_musicxml,
            inputs=[
                detected_notes_state,
                tab_data_state,
                instrument_state,
                bpm_state,
            ],
            outputs=[musicxml_file],
        )

        def do_midi_preview(notes, bpm):
            """Generate a MIDI preview of detected notes."""
            if not notes:
                return None

            try:
                import mido
                from services.tab_transcription import midi_note_to_name

                output_dir = Path(tempfile.gettempdir()) / "audio_generator" / "tabs"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                midi_path = str(output_dir / f"tab_preview_{timestamp}.mid")

                bpm_val = float(bpm) if bpm else 120.0
                ticks_per_beat = 480
                microseconds_per_beat = mido.bpm2tempo(bpm_val)

                mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
                track = mido.MidiTrack()
                mid.tracks.append(track)

                track.append(mido.MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))
                track.append(mido.Message("program_change", program=25, channel=0, time=0))  # Steel guitar

                midi_events = []
                for n in notes:
                    midi_note = n["midi_note"]
                    if midi_note <= 0:
                        continue
                    start_tick = int(mido.second2tick(n["start"], ticks_per_beat, microseconds_per_beat))
                    end_tick = int(mido.second2tick(n["end"], ticks_per_beat, microseconds_per_beat))
                    velocity = n.get("velocity", 80)
                    midi_events.append(("note_on", start_tick, midi_note, velocity))
                    midi_events.append(("note_off", end_tick, midi_note, 0))

                midi_events.sort(key=lambda e: (e[1], 0 if e[0] == "note_off" else 1))

                last_tick = 0
                for event_type, tick, note, velocity in midi_events:
                    delta = max(0, tick - last_tick)
                    track.append(mido.Message(
                        event_type, note=note, velocity=velocity,
                        channel=0, time=delta,
                    ))
                    last_tick = tick

                track.append(mido.MetaMessage("end_of_track", time=0))
                mid.save(midi_path)

                return midi_path
            except Exception as e:
                logger.error(f"MIDI preview error: {e}", exc_info=True)
                return None

        export_midi_preview_btn.click(
            fn=do_midi_preview,
            inputs=[detected_notes_state, bpm_state],
            outputs=[midi_preview_file],
        )

    return tab


def _build_note_table(notes: list[dict], tab_data: dict) -> list[list]:
    """Build a list of rows for the notes dataframe.

    Returns list of [start, end, note_name, string, fret, technique] rows.
    """
    from services.tab_transcription import _find_string_fret, midi_note_to_name

    tuning_name = tab_data.get("tuning_name", "guitar_standard")
    num_frets = tab_data.get("num_frets", 22)
    techniques = tab_data.get("techniques", [])

    # Build technique lookup: (string, approximate time) -> technique
    tech_lookup: dict[tuple[int, float], str] = {}
    for tech in techniques:
        key = (tech["string"], round(tech["time"], 3))
        tech_lookup[key] = tech["type"].replace("_", " ").title()

    rows = []
    for note in notes:
        midi = note["midi_note"]
        result = _find_string_fret(midi, tuning_name, num_frets)

        if result is not None:
            s_idx, fret = result
            string_label = f"String {s_idx + 1}"
            fret_label = str(fret) if fret > 0 else "Open"

            # Check for technique
            technique = ""
            for offset in [round(note["start"] + i * 0.001, 3) for i in range(-5, 6)]:
                tech = tech_lookup.get((s_idx, offset), "")
                if tech:
                    technique = tech
                    break
        else:
            string_label = "—"
            fret_label = "—"
            technique = ""

        rows.append([
            round(note["start"], 3),
            round(note["end"], 3),
            note.get("pitch", midi_note_to_name(midi)),
            string_label,
            fret_label,
            technique,
        ])

    return rows


def _render_initial_chord(
    notes: list[dict],
    instrument: str,
    tuning: str,
    num_frets: int,
) -> str | None:
    """Render a chord diagram for the first group of simultaneous notes."""
    try:
        from services.tab_transcription import render_chord_diagram
    except ImportError:
        return None

    if not notes:
        return None

    # Find the first time region with the most overlapping notes
    # Use a simple sliding window approach
    best_start = notes[0]["start"]
    best_count = 0
    window = 0.2

    for n in notes:
        start = n["start"]
        count = sum(1 for m in notes if m["start"] < start + window and m["end"] > start)
        if count > best_count:
            best_count = count
            best_start = start

    if best_count < 2:
        # Try single note diagram
        return render_chord_diagram(
            notes,
            start_time=best_start,
            end_time=best_start + 0.3,
            tuning=tuning,
            instrument=instrument,
            num_frets=int(num_frets),
        )

    return render_chord_diagram(
        notes,
        start_time=best_start,
        end_time=best_start + window,
        tuning=tuning,
        instrument=instrument,
        num_frets=int(num_frets),
    )
