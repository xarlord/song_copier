"""Instrumental tab — remove vocals from songs."""

import gradio as gr

from pipelines.instrumental import create_instrumental


def create_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## 🎵 Vocal Remover\nUpload a song and extract the instrumental (vocals removed).")

        with gr.Row():
            input_audio = gr.Audio(label="Input Song", type="filepath")

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=["htdemucs_ft", "htdemucs"],
                    value="htdemucs_ft",
                    label="Model",
                    info="htdemucs_ft = best quality, htdemucs = faster",
                )
                segment_slider = gr.Slider(
                    minimum=5, maximum=20, value=10, step=1,
                    label="Segment Length (seconds)",
                    info="Lower = less VRAM usage",
                )
                format_dropdown = gr.Dropdown(
                    choices=["wav", "mp3", "flac"],
                    value="wav",
                    label="Output Format",
                )
                run_btn = gr.Button("Remove Vocals", variant="primary")

        status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            instrumental_out = gr.Audio(label="Instrumental", type="filepath")
            vocals_out = gr.Audio(label="Extracted Vocals", type="filepath")

        def process(audio_path, model, segment, fmt, progress=gr.Progress()):
            if not audio_path:
                return "Please upload an audio file.", None, None

            def on_progress(pct, msg):
                progress(pct, desc=msg)

            try:
                result = create_instrumental(
                    input_path=audio_path,
                    model_name=model,
                    segment=int(segment),
                    output_format=fmt,
                    progress_callback=on_progress,
                )
                return (
                    f"Done! Stems: {result['stems']}",
                    result["instrumental_path"],
                    result["vocals_path"],
                )
            except Exception as e:
                return f"Error: {e}", None, None

        run_btn.click(
            fn=process,
            inputs=[input_audio, model_dropdown, segment_slider, format_dropdown],
            outputs=[status, instrumental_out, vocals_out],
        )

    return tab