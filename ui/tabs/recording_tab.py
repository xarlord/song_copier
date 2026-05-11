"""Recording tab — record/upload audio, preview, separate stems, and master."""

import logging
from datetime import datetime

import gradio as gr

from config import load_config, get_device
from pipelines.stem_separator import separate_stems, mix_stems_from_files
from services.audio_io import load_audio, save_audio
from services.mastering import master_audio, measure_loudness

logger = logging.getLogger(__name__)

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

MASTERING_PRESETS = ["spotify", "youtube", "cd", "podcast"]


def create_tab() -> gr.Column:
    """Build and return the Recording tab as a Gradio Column."""

    with gr.Column() as tab:
        # ── State ────────────────────────────────────────────────────────
        stem_files_state = gr.State(None)

        # ── Recording / Upload Section ───────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 🎙️ Record or Upload Audio")
            input_audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or Upload Audio",
                elem_classes=["upload-zone"],
            )

        # ── Action Buttons ───────────────────────────────────────────────
        with gr.Row():
            preview_btn = gr.Button("🎧 Preview", size="lg", scale=1)
            master_btn = gr.Button("🎛️ Master", variant="secondary", size="lg", scale=1)

        with gr.Row():
            separate_btn = gr.Button(
                "🎚️ Separate Stems",
                variant="primary",
                size="lg",
                scale=2,
            )
            model_dropdown = gr.Dropdown(
                choices=["htdemucs_6s", "htdemucs", "htdemucs_ft"],
                value="htdemucs_6s",
                label="Separation Model",
                info="6s = vocals/drums/bass/guitar/piano/other",
                scale=1,
            )

        # ── Mastering Options ────────────────────────────────────────────
        with gr.Row():
            mastering_preset = gr.Dropdown(
                choices=MASTERING_PRESETS,
                value="spotify",
                label="Mastering Preset",
                info="Target loudness standard",
                scale=2,
            )
            limiter_thresh = gr.Slider(
                -3.0, 0.0, value=-1.0, step=0.1,
                label="Limiter Ceiling (dBTP)",
                scale=1,
            )
            stereo_width = gr.Slider(
                0.0, 2.0, value=1.2, step=0.1,
                label="Stereo Width",
                scale=1,
            )

        # ── Status ───────────────────────────────────────────────────────
        status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=3,
            elem_classes=["status-box"],
        )

        # ── Output: Preview / Mastered ──────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 🔊 Output")
            preview_output = gr.Audio(label="🎧 Preview / Mastered", type="filepath")

        # ── Output: Stem Separation ──────────────────────────────────────
        stem_section = gr.Column(visible=False)

        with stem_section:
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("### 🎚️ Separated Stems")

                # Mix preview
                mix_output = gr.Audio(label="Mix Preview", type="filepath")

                # Volume sliders for each stem
                gr.Markdown("#### Volume Mixer")
                volume_sliders = {}
                with gr.Row():
                    for name in ["vocals", "drums", "bass"]:
                        with gr.Column(elem_classes=[f"stem-{name}"]):
                            volume_sliders[name] = gr.Slider(
                                0, 1.5, value=1.0, step=0.05,
                                label=f"{STEM_ICONS[name]} {STEM_LABELS[name]}",
                            )
                with gr.Row():
                    for name in ["guitar", "piano", "other"]:
                        with gr.Column(elem_classes=[f"stem-{name}"]):
                            volume_sliders[name] = gr.Slider(
                                0, 1.5, value=1.0, step=0.05,
                                label=f"{STEM_ICONS[name]} {STEM_LABELS[name]}",
                            )

                with gr.Row():
                    reset_btn = gr.Button("Reset 100%", size="sm")
                    mute_all_btn = gr.Button("Mute All", size="sm")
                    mix_btn = gr.Button("🔀 Remix & Preview", variant="primary", size="lg")

                # Individual stem players
                with gr.Accordion("Individual Stems", open=False):
                    stem_audio_outputs = {}
                    with gr.Row():
                        stem_audio_outputs["vocals"] = gr.Audio(label="🎤 Vocals", type="filepath")
                        stem_audio_outputs["drums"] = gr.Audio(label="🥁 Drums", type="filepath")
                    with gr.Row():
                        stem_audio_outputs["bass"] = gr.Audio(label="🎸 Bass", type="filepath")
                        stem_audio_outputs["guitar"] = gr.Audio(label="🎶 Guitar", type="filepath")
                    with gr.Row():
                        stem_audio_outputs["piano"] = gr.Audio(label="🎹 Piano", type="filepath")
                        stem_audio_outputs["other"] = gr.Audio(label="🎻 Other", type="filepath")

        # ================================================================
        # Callbacks
        # ================================================================

        def do_preview(audio_path):
            """Simply return the recorded/uploaded audio filepath for playback."""
            if not audio_path:
                return None, "⚠️ No audio recorded or uploaded."
            return audio_path, "✅ Playing back your audio."

        def do_separate(audio_path, model_name, progress=gr.Progress()):
            """Separate the recorded audio into stems."""
            if not audio_path:
                return (
                    "⚠️ Please record or upload audio first.",
                    gr.Column(visible=False),
                    None,
                    *[None] * 6,
                )

            def on_progress(pct, msg):
                progress(pct, desc=msg)

            try:
                config = load_config()

                # If the file came from a microphone recording, it may be a
                # temporary file — copy it to our temp dir for safe-keeping.
                result = separate_stems(
                    input_path=audio_path,
                    model_name=model_name,
                    output_format="wav",
                    apply_cleaning=True,
                    progress_callback=on_progress,
                )

                stem_files = result["stem_files"]
                individual = [stem_files.get(s) for s in STEM_NAMES]
                duration = result.get("duration", 0)

                msg = (
                    f"✅ Separated into {len(result['stems'])} stems "
                    f"(model={model_name}, duration={duration:.1f}s).\n"
                    f"Stems: {', '.join(result['stems'])}\n"
                    "Adjust volumes and click 🔀 Remix & Preview!"
                )

                return (
                    msg,
                    gr.Column(visible=True),
                    stem_files,
                    *individual,
                )
            except Exception as e:
                logger.error("Stem separation error: %s", e, exc_info=True)
                return (
                    f"❌ Error during stem separation: {e}",
                    gr.Column(visible=False),
                    None,
                    *[None] * 6,
                )

        def do_master(audio_path, preset, limiter_db, stereo_w, progress=gr.Progress()):
            """Master the recorded/uploaded audio."""
            if not audio_path:
                return None, "⚠️ No audio recorded or uploaded."

            progress(0.1, desc="Loading audio...")
            try:
                audio, sr = load_audio(audio_path)

                progress(0.3, desc="Measuring loudness...")
                pre = measure_loudness(audio, sr)

                progress(0.5, desc="Mastering...")
                mastered, metrics = master_audio(
                    audio, sr,
                    preset=preset,
                    limiter_threshold=limiter_db,
                    stereo_width=stereo_w,
                )

                progress(0.8, desc="Saving...")
                config = load_config()
                output_dir = config["paths"]["output"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = output_dir / f"mastered_{preset}_{timestamp}.wav"
                save_audio(mastered, out_path, sr, format="wav")

                post = metrics["post"]
                msg = (
                    f"✅ Mastered with preset: {preset}\n"
                    f"Pre:  {pre['lufs']:.1f} LUFS  |  Peak: {pre['peak_db']:.1f} dBTP\n"
                    f"Post: {post['lufs']:.1f} LUFS  |  Peak: {post['peak_db']:.1f} dBTP\n"
                    f"Saved to: {out_path}"
                )
                return str(out_path), msg
            except Exception as e:
                logger.error("Mastering error: %s", e, exc_info=True)
                return None, f"❌ Mastering error: {e}"

        def do_mix(stem_files, v_vol, d_vol, b_vol, g_vol, p_vol, o_vol,
                    progress=gr.Progress()):
            """Mix stems with volume adjustments and return preview."""
            if not stem_files:
                return None

            volumes = {
                "vocals": v_vol, "drums": d_vol, "bass": b_vol,
                "guitar": g_vol, "piano": p_vol, "other": o_vol,
            }

            config = load_config()
            output_dir = config["paths"]["output"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"remix_preview_{timestamp}.wav"

            try:
                result = mix_stems_from_files(
                    stem_files=stem_files,
                    volumes=volumes,
                    output_path=str(output_path),
                    progress_callback=lambda pct, msg: progress(pct, desc=msg),
                )
                return result
            except Exception as e:
                logger.error("Remix preview error: %s", e, exc_info=True)
                return None

        def reset_volumes():
            return [1.0] * 6

        def mute_all():
            return [0.0] * 6

        # ── Wire callbacks ───────────────────────────────────────────────
        preview_btn.click(
            fn=do_preview,
            inputs=[input_audio],
            outputs=[preview_output, status],
        )

        separate_btn.click(
            fn=do_separate,
            inputs=[input_audio, model_dropdown],
            outputs=[
                status,
                stem_section,
                stem_files_state,
                *[stem_audio_outputs[s] for s in STEM_NAMES],
            ],
        )

        master_btn.click(
            fn=do_master,
            inputs=[input_audio, mastering_preset, limiter_thresh, stereo_width],
            outputs=[preview_output, status],
        )

        mix_btn.click(
            fn=do_mix,
            inputs=[stem_files_state] + [volume_sliders[s] for s in STEM_NAMES],
            outputs=[mix_output],
        )

        reset_btn.click(
            fn=reset_volumes,
            outputs=[volume_sliders[s] for s in STEM_NAMES],
        )
        mute_all_btn.click(
            fn=mute_all,
            outputs=[volume_sliders[s] for s in STEM_NAMES],
        )

    return tab
