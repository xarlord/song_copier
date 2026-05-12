"""Time Stretch & Pitch Shift tab — global and per-stem processing."""

import logging

import gradio as gr
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

from config import load_config
from services.audio_io import load_audio, save_audio
from services.time_stretch import process_stem, batch_process_stems
from services.mixing import mix_stems_with_volumes

STEM_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"]
STEM_LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano",
    "other": "Other",
}
STEM_ICONS = {
    "vocals": "🎤",
    "drums": "🥁",
    "bass": "🎸",
    "guitar": "🎶",
    "piano": "🎹",
    "other": "🎻",
}

# Default per-stem settings
_DEFAULT_SETTINGS = {name: {"stretch": 1.0, "pitch": 0.0} for name in STEM_NAMES}


def create_tab() -> gr.Column:
    """Build the Time Stretch / Pitch Shift tab UI."""

    with gr.Column() as tab:
        gr.Markdown("## ⏱️ Time Stretch & Pitch Shift")
        gr.Markdown(
            "Change tempo without affecting pitch, or shift pitch without changing tempo. "
            "Works on full mixes or individual stems."
        )

        # ── Global Processing Section ──────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Global Processing")
            gr.Markdown("Apply time-stretch and pitch-shift to the entire audio file.")

            input_audio = gr.Audio(
                label="📁 Upload Audio",
                type="filepath",
                elem_classes=["upload-zone"],
            )

            with gr.Row():
                global_stretch = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="⏱️ Time Stretch Rate",
                    info="0.5 = half speed, 1.0 = original, 2.0 = double speed",
                )
                global_pitch = gr.Slider(
                    minimum=-12.0,
                    maximum=12.0,
                    value=0.0,
                    step=0.5,
                    label="🎵 Pitch Shift (semitones)",
                    info="-12 to +12 semitones. 0 = original pitch",
                )

            with gr.Row():
                process_btn = gr.Button(
                    "⏱️ Process",
                    variant="primary",
                    size="lg",
                )
                reset_global_btn = gr.Button("↩️ Reset", size="sm")

            status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
                elem_classes=["status-box"],
            )

        # ── Output Section ─────────────────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Output")
            output_audio = gr.Audio(
                label="🎧 Processed Audio",
                type="filepath",
            )

        # ── Advanced: Per-Stem Processing ──────────────────────────────
        with gr.Accordion("🔬 Advanced: Per-Stem Processing", open=False):
            gr.Markdown(
                "Separate audio into stems first, then apply independent "
                "time-stretch and pitch-shift to each stem. Finally, remix all stems."
            )

            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("#### Upload & Separate")

                with gr.Row():
                    stem_input_audio = gr.Audio(
                        label="📁 Upload Audio for Stem Separation",
                        type="filepath",
                        elem_classes=["upload-zone"],
                    )

                with gr.Row():
                    separate_btn = gr.Button(
                        "🔀 Separate & Load Stems",
                        variant="primary",
                        size="lg",
                    )
                    stem_status = gr.Textbox(
                        label="Separation Status",
                        interactive=False,
                        lines=1,
                        elem_classes=["status-box"],
                        scale=2,
                    )

                # Hidden state to store separated stems
                stems_state = gr.State(value=None)

            # ── Per-Stem Controls ──────────────────────────────────────
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("#### Per-Stem Controls")
                gr.Markdown(
                    "Adjust time-stretch and pitch-shift independently for each stem."
                )

                stem_stretch_sliders = {}
                stem_pitch_sliders = {}

                for stem_name in STEM_NAMES:
                    with gr.Row():
                        gr.Markdown(
                            f"**{STEM_ICONS[stem_name]} {STEM_LABELS[stem_name]}**",
                            elem_classes=[f"stem-{stem_name}"],
                        )
                        stem_stretch_sliders[stem_name] = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.05,
                            label="Stretch Rate",
                            scale=2,
                        )
                        stem_pitch_sliders[stem_name] = gr.Slider(
                            minimum=-12.0,
                            maximum=12.0,
                            value=0.0,
                            step=0.5,
                            label="Pitch (semi)",
                            scale=2,
                        )

                with gr.Row():
                    reset_stems_btn = gr.Button("↩️ Reset All Stems", size="sm")
                    process_stems_btn = gr.Button(
                        "⏱️ Process All Stems & Remix",
                        variant="primary",
                        size="lg",
                    )

                stems_status = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    lines=2,
                    elem_classes=["status-box"],
                )

            # ── Per-Stem Output ────────────────────────────────────────
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("#### Stem Output")
                stem_output_audio = gr.Audio(
                    label="🎧 Remixed Output",
                    type="filepath",
                )

                with gr.Accordion("Individual Processed Stems", open=False):
                    stem_outputs = {}
                    with gr.Row():
                        stem_outputs["vocals"] = gr.Audio(label="🎤 Vocals", type="filepath")
                        stem_outputs["drums"] = gr.Audio(label="🥁 Drums", type="filepath")
                    with gr.Row():
                        stem_outputs["bass"] = gr.Audio(label="🎸 Bass", type="filepath")
                        stem_outputs["guitar"] = gr.Audio(label="🎶 Guitar", type="filepath")
                    with gr.Row():
                        stem_outputs["piano"] = gr.Audio(label="🎹 Piano", type="filepath")
                        stem_outputs["other"] = gr.Audio(label="🎻 Other", type="filepath")

        # ═══════════════════════════════════════════════════════════════
        # Callbacks
        # ═══════════════════════════════════════════════════════════════

        def _do_global_process(audio_path, stretch_rate, pitch_semitones, progress=gr.Progress()):
            """Process the entire audio with global stretch/pitch settings."""
            if not audio_path:
                return "⚠️ Please upload an audio file.", None

            if stretch_rate == 1.0 and pitch_semitones == 0.0:
                return "No processing needed (default values).", None

            try:
                progress(0.1, desc="Loading audio...")
                audio, sr = load_audio(audio_path)

                progress(0.4, desc="Processing audio...")
                processed = process_stem(audio, sr, stretch_rate, pitch_semitones)

                progress(0.8, desc="Saving output...")
                config = load_config()
                output_dir = config["paths"]["output"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"timestretch_{timestamp}.wav"
                save_audio(processed, output_path, sr)

                duration_orig = audio.shape[-1] / sr
                duration_new = processed.shape[-1] / sr
                msg = (
                    f"✅ Done! Stretch: {stretch_rate:.2f}x, Pitch: {pitch_semitones:+.1f} semitones.\n"
                    f"Duration: {duration_orig:.1f}s → {duration_new:.1f}s"
                )
                return msg, str(output_path)

            except Exception as e:
                logger.error(f"Global process error: {e}", exc_info=True)
                return f"❌ Error: {e}", None

        def _do_reset_global():
            return 1.0, 0.0

        def _do_separate_stems(audio_path, progress=gr.Progress()):
            """Separate audio into stems and store in state."""
            if not audio_path:
                return "⚠️ Please upload an audio file.", None

            try:
                from pipelines.stem_separator import separate_stems

                progress(0.1, desc="Loading and separating stems...")

                def on_progress(pct, msg):
                    progress(pct, desc=msg)

                result = separate_stems(
                    input_path=audio_path,
                    model_name="htdemucs_6s",
                    output_format="wav",
                    apply_cleaning=True,
                    progress_callback=on_progress,
                )

                stem_files = result["stem_files"]
                stems_list = result["stems"]

                # Load each stem into a dict of numpy arrays
                stems_data = {}
                for name in stems_list:
                    fpath = stem_files.get(name)
                    if fpath:
                        audio_arr, _ = load_audio(fpath)
                        stems_data[name] = audio_arr

                msg = f"✅ Separated {len(stems_list)} stems: {', '.join(stems_list)}"
                return msg, stems_data

            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Separation error: {e}", exc_info=True)
                return f"❌ Error: {e}", None

        def _do_process_stems(
            stems_data,
            v_stretch, v_pitch,
            d_stretch, d_pitch,
            b_stretch, b_pitch,
            g_stretch, g_pitch,
            p_stretch, p_pitch,
            o_stretch, o_pitch,
            progress=gr.Progress(),
        ):
            """Process all stems with per-stem settings and remix."""
            if not stems_data:
                return "⚠️ No stems loaded. Upload and separate first.", None, *[None] * 6

            try:
                slider_values = {
                    "vocals": (v_stretch, v_pitch),
                    "drums": (d_stretch, d_pitch),
                    "bass": (b_stretch, b_pitch),
                    "guitar": (g_stretch, g_pitch),
                    "piano": (p_stretch, p_pitch),
                    "other": (o_stretch, o_pitch),
                }

                # Build settings dict
                settings = {}
                for name, (stretch, pitch) in slider_values.items():
                    settings[name] = {"stretch": stretch, "pitch": pitch}

                # Determine SR from any loaded stem
                sample_rate = 44100  # default

                progress(0.1, desc="Processing stems...")

                # Process each stem
                processed = batch_process_stems(stems_data, sample_rate, settings)

                progress(0.6, desc="Mixing stems...")

                # Mix with unity volume
                volumes = {name: 1.0 for name in processed}
                mixed = mix_stems_with_volumes(processed, volumes, normalize=True)

                # Save outputs
                config = load_config()
                output_dir = config["paths"]["output"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save remix
                remix_path = output_dir / f"timestretch_remix_{timestamp}.wav"
                save_audio(mixed, remix_path, sample_rate)

                # Save individual stems
                individual_paths = []
                for name in STEM_NAMES:
                    if name in processed:
                        stem_path = output_dir / f"timestretch_{name}_{timestamp}.wav"
                        save_audio(processed[name], stem_path, sample_rate)
                        individual_paths.append(str(stem_path))
                    else:
                        individual_paths.append(None)

                processed_names = [
                    f"{STEM_ICONS[n]} {STEM_LABELS[n]}" for n in STEM_NAMES
                    if n in processed and (settings[n]["stretch"] != 1.0 or settings[n]["pitch"] != 0.0)
                ]

                detail = ", ".join(processed_names) if processed_names else "no changes"
                msg = f"✅ Processed stems ({detail}) and remixed."
                return msg, str(remix_path), *individual_paths

            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Stem processing error: {e}", exc_info=True)
                return f"❌ Error: {e}", None, *[None] * 6

        def _do_reset_stems():
            """Reset all per-stem sliders to defaults."""
            values = []
            for _ in STEM_NAMES:
                values.extend([1.0, 0.0])
            return values

        # ── Wire Global Callbacks ──────────────────────────────────────
        process_btn.click(
            fn=_do_global_process,
            inputs=[input_audio, global_stretch, global_pitch],
            outputs=[status, output_audio],
        )

        reset_global_btn.click(
            fn=_do_reset_global,
            outputs=[global_stretch, global_pitch],
        )

        # ── Wire Stem Callbacks ────────────────────────────────────────
        separate_btn.click(
            fn=_do_separate_stems,
            inputs=[stem_input_audio],
            outputs=[stem_status, stems_state],
        )

        all_stem_slider_outputs = []
        for name in STEM_NAMES:
            all_stem_slider_outputs.append(stem_stretch_sliders[name])
            all_stem_slider_outputs.append(stem_pitch_sliders[name])

        reset_stems_btn.click(
            fn=_do_reset_stems,
            outputs=all_stem_slider_outputs,
        )

        all_stem_slider_inputs = []
        for name in STEM_NAMES:
            all_stem_slider_inputs.append(stem_stretch_sliders[name])
            all_stem_slider_inputs.append(stem_pitch_sliders[name])

        process_stems_btn.click(
            fn=_do_process_stems,
            inputs=[stems_state] + all_stem_slider_inputs,
            outputs=[stems_status, stem_output_audio] + [stem_outputs[s] for s in STEM_NAMES],
        )

    return tab
