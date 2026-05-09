"""Stem Separator tab — separate audio into stems with volume mixer."""

import gradio as gr
from pathlib import Path
from datetime import datetime

from config import load_config
from pipelines.stem_separator import separate_stems, mix_stems_from_files


def create_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown(
            "## 🎚️ Stem Separator & Mixer\n"
            "Upload a song to separate it into individual stems, then mix them with volume controls."
        )

        # Store state for separated stems
        stem_files_state = gr.State(None)
        stems_list_state = gr.State(None)

        # === Top Section: Upload & Separate ===
        with gr.Row():
            input_audio = gr.Audio(label="Input Song", type="filepath")

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=[
                        "htdemucs_6s",
                        "htdemucs_ft",
                        "htdemucs",
                    ],
                    value="htdemucs_6s",
                    label="Model",
                    info="htdemucs_6s = 6 stems (vocals,drums,bass,guitar,piano,other)",
                )
                format_dropdown = gr.Dropdown(
                    choices=["wav", "mp3", "flac"],
                    value="wav",
                    label="Stem Format",
                )
                clean_checkbox = gr.Checkbox(
                    value=True,
                    label="Apply stem cleaning",
                    info="Reduces artifacts (slower but better quality)",
                )
                separate_btn = gr.Button("🔪 Separate Stems", variant="primary", size="lg")

        status = gr.Textbox(label="Status", interactive=False, lines=2)

        # === Middle Section: Volume Controls (visible after separation) ===
        mixer_section = gr.Column(visible=False)

        with mixer_section:
            gr.Markdown("### 🎛️ Stem Mixer\nAdjust volumes and preview the mix.")

            # Stem volume sliders
            with gr.Row():
                vocals_vol = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="🎤 Vocals")
                drums_vol = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="🥁 Drums")
                bass_vol = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="🎸 Bass")

            with gr.Row():
                guitar_vol = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="🎸 Guitar")
                piano_vol = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="🎹 Piano")
                other_vol = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="🎻 Other")

            with gr.Row():
                reset_btn = gr.Button("Reset All to 100%", size="sm")
                mute_all_btn = gr.Button("Mute All", size="sm")
                preview_btn = gr.Button("▶️ Preview Mix", variant="primary", size="lg")

        # === Bottom Section: Outputs ===
        with gr.Row():
            mixed_output = gr.Audio(label="Mixed Preview", type="filepath")

        with gr.Accordion("Individual Stems", open=False):
            with gr.Row():
                vocals_audio = gr.Audio(label="Vocals", type="filepath")
                drums_audio = gr.Audio(label="Drums", type="filepath")
            with gr.Row():
                bass_audio = gr.Audio(label="Bass", type="filepath")
                guitar_audio = gr.Audio(label="Guitar", type="filepath")
            with gr.Row():
                piano_audio = gr.Audio(label="Piano", type="filepath")
                other_audio = gr.Audio(label="Other", type="filepath")

        # === Callbacks ===

        def do_separate(audio_path, model_name, fmt, apply_cleaning, progress=gr.Progress()):
            if not audio_path:
                return "Please upload an audio file.", gr.Column(visible=False), None, None, *[None] * 6

            def on_progress(pct, msg):
                progress(pct, desc=msg)

            try:
                result = separate_stems(
                    input_path=audio_path,
                    model_name=model_name,
                    output_format=fmt,
                    apply_cleaning=apply_cleaning,
                    progress_callback=on_progress,
                )

                stem_files = result["stem_files"]
                stems_list = result["stems"]

                # Get individual file paths
                vocals = stem_files.get("vocals")
                drums = stem_files.get("drums")
                bass = stem_files.get("bass")
                guitar = stem_files.get("guitar")
                piano = stem_files.get("piano")
                other = stem_files.get("other")

                msg = f"Separated into {len(stems_list)} stems: {', '.join(stems_list)}. Adjust volumes and click Preview Mix!"
                return (
                    msg,
                    gr.Column(visible=True),
                    stem_files,
                    stems_list,
                    vocals, drums, bass, guitar, piano, other,
                )
            except Exception as e:
                return f"Error: {e}", gr.Column(visible=False), None, None, *[None] * 6

        def do_preview(stem_files, v_vol, d_vol, b_vol, g_vol, p_vol, o_vol, progress=gr.Progress()):
            if not stem_files:
                return None

            volumes = {
                "vocals": v_vol,
                "drums": d_vol,
                "bass": b_vol,
                "guitar": g_vol,
                "piano": p_vol,
                "other": o_vol,
            }

            config = load_config()
            output_dir = config["paths"]["output"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"mix_preview_{timestamp}.wav"

            try:
                result = mix_stems_from_files(
                    stem_files=stem_files,
                    volumes=volumes,
                    output_path=str(output_path),
                    progress_callback=lambda pct, msg: progress(pct, desc=msg),
                )
                return result
            except Exception as e:
                return None

        def reset_volumes():
            return [1.0] * 6

        def mute_all():
            return [0.0] * 6

        separate_btn.click(
            fn=do_separate,
            inputs=[input_audio, model_dropdown, format_dropdown, clean_checkbox],
            outputs=[
                status,
                mixer_section,
                stem_files_state,
                stems_list_state,
                vocals_audio, drums_audio, bass_audio,
                guitar_audio, piano_audio, other_audio,
            ],
        )

        preview_btn.click(
            fn=do_preview,
            inputs=[
                stem_files_state,
                vocals_vol, drums_vol, bass_vol,
                guitar_vol, piano_vol, other_vol,
            ],
            outputs=[mixed_output],
        )

        reset_btn.click(
            fn=reset_volumes,
            outputs=[vocals_vol, drums_vol, bass_vol, guitar_vol, piano_vol, other_vol],
        )

        mute_all_btn.click(
            fn=mute_all,
            outputs=[vocals_vol, drums_vol, bass_vol, guitar_vol, piano_vol, other_vol],
        )

    return tab
