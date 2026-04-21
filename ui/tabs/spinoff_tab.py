"""Spin-off tab — generate song variations."""

import gradio as gr

from pipelines.spinoff import generate_spinoff


def create_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown(
            "## 🎸 Song Spin-off Generator\n"
            "Generate AI variations of your songs using MusicGen with melody conditioning."
        )

        with gr.Row():
            input_audio = gr.Audio(label="Input Song", type="filepath")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Description / Prompt",
                    placeholder="e.g., electronic remix with faster tempo, jazz version, ambient chillout...",
                    lines=2,
                )

                with gr.Row():
                    model = gr.Dropdown(
                        choices=[
                            "facebook/musicgen-melody",
                            "facebook/musicgen-small",
                        ],
                        value="facebook/musicgen-melody",
                        label="Model",
                    )
                    use_melody = gr.Checkbox(
                        value=True,
                        label="Use Melody Conditioning",
                        info="Condition generation on the input song's melody",
                    )

                with gr.Row():
                    duration = gr.Slider(
                        minimum=5, maximum=30, value=15, step=5,
                        label="Duration (seconds)",
                    )
                    temperature = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label="Temperature (creativity)",
                        info="Lower = conservative, Higher = creative",
                    )

                with gr.Row():
                    cfg_coef = gr.Slider(
                        minimum=1.0, maximum=9.0, value=3.0, step=0.5,
                        label="CFG Coefficient",
                        info="How closely to follow the prompt",
                    )
                    format_out = gr.Dropdown(
                        choices=["wav", "mp3", "flac"],
                        value="wav",
                        label="Output Format",
                    )

                run_btn = gr.Button("Generate Spin-off", variant="primary")

        status = gr.Textbox(label="Status", interactive=False, lines=2)
        analysis_display = gr.Textbox(label="Analysis", interactive=False, lines=2)

        with gr.Row():
            output_audio = gr.Audio(label="Generated Variation", type="filepath")
            raw_audio = gr.Audio(label="Raw Generated (before processing)", type="filepath")

        def process(audio_path, prompt_text, model_name, melody, dur, temp, cfg, fmt, progress=gr.Progress()):
            if not audio_path:
                return "Please upload an audio file.", "", None, None

            def on_progress(pct, msg):
                progress(pct, desc=msg)

            try:
                result = generate_spinoff(
                    input_path=audio_path,
                    prompt=prompt_text,
                    model_name=model_name,
                    duration=float(dur),
                    temperature=float(temp),
                    cfg_coef=float(cfg),
                    use_melody_conditioning=melody,
                    output_format=fmt,
                    progress_callback=on_progress,
                )
                analysis = result["analysis"]
                analysis_text = (
                    f"BPM: {analysis['bpm']} | Key: {analysis['key']} | "
                    f"Duration: {analysis['duration']}s | Prompt: '{result['prompt_used']}'"
                )
                return f"Done! Generated {dur}s variation.", analysis_text, result["output_path"], result["raw_generated_path"]
            except Exception as e:
                return f"Error: {e}", "", None, None

        run_btn.click(
            fn=process,
            inputs=[input_audio, prompt, model, use_melody, duration, temperature, cfg_coef, format_out],
            outputs=[status, analysis_display, output_audio, raw_audio],
        )

    return tab