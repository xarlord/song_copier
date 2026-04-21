"""Voice swap tab — replace vocals with a different voice."""

import gradio as gr

from config import load_config
from models.rvc_wrapper import list_voice_models
from pipelines.voice_swap import swap_vocals


def _refresh_voice_models():
    config = load_config()
    models = list_voice_models(config["paths"]["rvc_models"])
    if not models:
        return gr.update(choices=["No models found"], value="No models found")
    choices = [f"{m['name']} ({m['path']})" for m in models]
    return gr.update(choices=choices, value=choices[0])


def create_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown(
            "## 🎤 Voice Swapper\n"
            "Replace vocals in a song with a different voice using RVC.\n\n"
            "**Setup:** Place `.pth` voice model files in `cache/rvc/models/` directory."
        )

        with gr.Row():
            input_audio = gr.Audio(label="Input Song", type="filepath")

        with gr.Row():
            with gr.Column(scale=1):
                config = load_config()
                models = list_voice_models(config["paths"]["rvc_models"])
                model_choices = [f"{m['name']} ({m['path']})" for m in models] if models else ["No models found"]

                voice_model = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else None,
                    label="Voice Model",
                    info=".pth model file from cache/rvc/models/",
                )
                refresh_btn = gr.Button("🔄 Refresh Models", size="sm")

                with gr.Row():
                    upload_model = gr.File(
                        label="Upload Voice Model (.pth)",
                        file_types=[".pth"],
                    )

                with gr.Row():
                    f0method = gr.Dropdown(
                        choices=["rmvpe", "crepe", "harvest"],
                        value="rmvpe",
                        label="Pitch Method",
                    )
                    f0up_key = gr.Slider(
                        minimum=-12, maximum=12, value=0, step=1,
                        label="Pitch Shift (semitones)",
                    )

                with gr.Row():
                    seg_model = gr.Dropdown(
                        choices=["htdemucs_ft", "htdemucs"],
                        value="htdemucs_ft",
                        label="Separation Model",
                    )
                    format_out = gr.Dropdown(
                        choices=["wav", "mp3", "flac"],
                        value="wav",
                        label="Output Format",
                    )

                run_btn = gr.Button("Swap Vocals", variant="primary")

        status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            output_audio = gr.Audio(label="Result (Voice Swapped)", type="filepath")
            instrumental_out = gr.Audio(label="Instrumental", type="filepath")

        def handle_upload(file, current_choices):
            if file is None:
                return current_choices
            import shutil
            from pathlib import Path
            cfg = load_config()
            dest = Path(cfg["paths"]["rvc_models"]) / Path(file.name).name
            shutil.copy2(file.name, str(dest))
            return _refresh_voice_models()

        def process(audio_path, voice_model_str, f0_method, pitch, sep_model, fmt, progress=gr.Progress()):
            if not audio_path:
                return "Please upload an audio file.", None, None
            if "No models" in voice_model_str:
                return "No voice model available. Place .pth files in cache/rvc/models/.", None, None

            # Extract path from dropdown value: "name (path)"
            model_path = voice_model_str.split("(")[-1].rstrip(")")

            def on_progress(pct, msg):
                progress(pct, desc=msg)

            try:
                result = swap_vocals(
                    input_path=audio_path,
                    voice_model_path=model_path,
                    model_name=sep_model,
                    f0method=f0_method,
                    f0up_key=int(pitch),
                    output_format=fmt,
                    progress_callback=on_progress,
                )
                return "Done!", result["output_path"], result["instrumental_path"]
            except Exception as e:
                return f"Error: {e}", None, None

        refresh_btn.click(fn=_refresh_voice_models, outputs=[voice_model])
        upload_model.change(fn=handle_upload, inputs=[upload_model, voice_model], outputs=[voice_model])
        run_btn.click(
            fn=process,
            inputs=[input_audio, voice_model, f0method, f0up_key, seg_model, format_out],
            outputs=[status, output_audio, instrumental_out],
        )

    return tab