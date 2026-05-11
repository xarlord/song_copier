"""Mastering tab — loudness normalization, limiting, stereo widening."""

import logging
from datetime import datetime
from pathlib import Path

import gradio as gr

from config import load_config
from services.audio_io import load_audio, save_audio
from services.mastering import master_audio, measure_loudness, PRESETS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_metrics(metrics: dict, label: str) -> str:
    """Pretty-print loudness metrics."""
    lufs = metrics.get("lufs", float("-inf"))
    peak = metrics.get("peak_db", float("-inf"))
    rms = metrics.get("rms_db", float("-inf"))
    return (
        f"{'━' * 36}\n"
        f"  {label}\n"
        f"{'━' * 36}\n"
        f"  LUFS:      {lufs:>8.1f} dB\n"
        f"  Peak:      {peak:>8.1f} dBTP\n"
        f"  RMS:       {rms:>8.1f} dB\n"
        f"{'━' * 36}"
    )


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_tab() -> gr.Blocks:
    with gr.Blocks() as tab:

        # === Upload Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 🎛️ Audio Mastering")
            gr.Markdown(
                "Upload a mix and apply loudness normalization, stereo widening, "
                "and brick-wall limiting for streaming or CD delivery."
            )

            input_audio = gr.Audio(
                label="Upload Audio",
                sources=["upload"],
                type="filepath",
                elem_classes=["upload-zone"],
            )

        # === Settings Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Settings")

            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    choices=["spotify", "youtube", "cd", "podcast", "custom"],
                    value="spotify",
                    label="Preset",
                    info="Target loudness for common delivery formats",
                    scale=2,
                )
                custom_lufs = gr.Slider(
                    minimum=-24,
                    maximum=-3,
                    value=-14,
                    step=0.5,
                    label="Custom Target LUFS",
                    info="Only used when preset = custom",
                    visible=False,
                    scale=2,
                )

            with gr.Row():
                limiter_threshold = gr.Slider(
                    minimum=-6,
                    maximum=0,
                    value=-1.0,
                    step=0.1,
                    label="Limiter Threshold (dBTP)",
                    info="Brick-wall ceiling to prevent clipping",
                    scale=1,
                )
                stereo_width = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="Stereo Width",
                    info="< 1 = narrower, > 1 = wider",
                    scale=1,
                )

            with gr.Row():
                master_btn = gr.Button(
                    "🎛️ Master Audio",
                    variant="primary",
                    size="lg",
                )

        # === Results Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Results")

            with gr.Row():
                before_metrics = gr.Textbox(
                    label="📊 Before",
                    lines=6,
                    interactive=False,
                    elem_classes=["status-box"],
                    scale=1,
                )
                after_metrics = gr.Textbox(
                    label="📊 After",
                    lines=6,
                    interactive=False,
                    elem_classes=["status-box"],
                    scale=1,
                )

            mastered_audio = gr.Audio(
                label="Mastered Audio",
                type="filepath",
            )

        status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2,
            elem_classes=["status-box"],
        )

        # === Callbacks ===

        def toggle_custom_lufs(preset: str):
            """Show the custom LUFS slider only when preset is 'custom'."""
            return gr.Slider(visible=(preset == "custom"))

        preset_dropdown.change(
            fn=toggle_custom_lufs,
            inputs=[preset_dropdown],
            outputs=[custom_lufs],
        )

        def do_master(audio_path, preset, custom_lufs_val, limiter_thresh, width,
                      progress=gr.Progress()):
            if not audio_path:
                return (
                    "⚠️ Please upload an audio file first.",
                    "", "", None,
                )

            progress(0.0, desc="Loading audio…")

            try:
                # Load audio
                audio, sr = load_audio(audio_path)

                # Measure original loudness
                progress(0.15, desc="Measuring original loudness…")
                pre = measure_loudness(audio, sr)

                # Resolve target LUFS
                if preset == "custom":
                    target_lufs = custom_lufs_val
                else:
                    target_lufs = None  # let master_audio use the preset

                # Run mastering chain
                progress(0.4, desc="Mastering…")
                mastered, metrics = master_audio(
                    audio=audio,
                    sr=sr,
                    preset=preset,
                    target_lufs=target_lufs,
                    limiter_threshold=limiter_thresh,
                    stereo_width=width,
                )
                progress(0.8, desc="Saving mastered audio…")

                # Save output
                config = load_config()
                output_dir = Path(config["paths"]["output"])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"mastered_{timestamp}.wav"
                saved_path = save_audio(mastered, output_path, sr, format="wav")

                progress(1.0, desc="Done!")

                pre_text = _format_metrics(pre, "Original")
                post_text = _format_metrics(metrics["post"], f"Mastered ({preset})")

                status_msg = (
                    f"✅ Mastered with preset '{preset}' "
                    f"(target {metrics['target_lufs']:.1f} LUFS).  "
                    f"Loudness: {pre['lufs']:.1f} → {metrics['post']['lufs']:.1f} LUFS"
                )

                return status_msg, pre_text, post_text, str(saved_path)

            except Exception as e:
                logger.error("Mastering failed: %s", e, exc_info=True)
                return (
                    f"❌ Error: {e}",
                    "", "", None,
                )

        master_btn.click(
            fn=do_master,
            inputs=[
                input_audio,
                preset_dropdown,
                custom_lufs,
                limiter_threshold,
                stereo_width,
            ],
            outputs=[status, before_metrics, after_metrics, mastered_audio],
        )

    return tab
