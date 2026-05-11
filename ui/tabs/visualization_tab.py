"""Visualization tab — waveform, spectrogram, spectrum, and A/B comparison plots."""

import logging

import gradio as gr

from services.audio_io import load_audio
from services.visualization import plot_waveform, plot_spectrogram, plot_spectrum, plot_comparison

logger = logging.getLogger(__name__)


def create_tab() -> gr.Column:
    """Build and return the Visualization tab as a Gradio Column."""

    with gr.Column() as tab:
        gr.Markdown("## 📊 Audio Visualization")

        # === Upload & Analyze Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Upload & Analyze")
            input_audio = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Upload Audio",
                elem_classes=["upload-zone"],
            )
            analyze_btn = gr.Button(
                "📊 Analyze Audio",
                variant="primary",
                size="lg",
            )

        # Status message
        status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=1,
            elem_classes=["status-box"],
        )

        # === Results Section — 3 plots in a row ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Analysis Results")
            with gr.Row():
                waveform_img = gr.Image(label="Waveform", show_download_button=True)
                spectrogram_img = gr.Image(label="Spectrogram", show_download_button=True)
                spectrum_img = gr.Image(label="Frequency Spectrum", show_download_button=True)

        # === A/B Comparison Section ===
        with gr.Accordion("🔄 A/B Comparison", open=False):
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("### Compare two audio files side by side")
                ref_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Reference Audio (B)",
                    elem_classes=["upload-zone"],
                )
                compare_btn = gr.Button(
                    "🔄 Compare A vs B",
                    variant="secondary",
                    size="lg",
                )
                comparison_status = gr.Textbox(
                    label="Comparison Status",
                    interactive=False,
                    lines=1,
                    elem_classes=["status-box"],
                )
                comparison_img = gr.Image(
                    label="A/B Comparison",
                    show_download_button=True,
                )

        # === Callbacks ===

        def do_analyze(audio_path, progress=gr.Progress()):
            """Analyze a single audio file: waveform, spectrogram, spectrum."""
            if not audio_path:
                return "Please upload an audio file.", None, None, None

            try:
                progress(0.1, desc="Loading audio…")
                audio, sr = load_audio(audio_path)

                progress(0.3, desc="Generating waveform…")
                waveform = plot_waveform(audio, sr)

                progress(0.6, desc="Generating spectrogram…")
                spectrogram = plot_spectrogram(audio, sr)

                progress(0.9, desc="Generating frequency spectrum…")
                spectrum = plot_spectrum(audio, sr)

                duration = audio.shape[-1] / sr
                channels = audio.shape[0] if audio.ndim > 1 else 1
                msg = (
                    f"✅ Analysis complete — {duration:.1f}s, "
                    f"{channels}ch, {sr}Hz"
                )
                return msg, waveform, spectrogram, spectrum

            except Exception as e:
                logger.error("Analysis failed: %s", e, exc_info=True)
                return f"❌ Error: {e}", None, None, None

        def do_compare(audio_a_path, audio_b_path, progress=gr.Progress()):
            """Compare two audio files with side-by-side plots."""
            if not audio_a_path:
                return "Please upload the primary audio (A) first.", None
            if not audio_b_path:
                return "Please upload the reference audio (B).", None

            try:
                progress(0.15, desc="Loading audio A…")
                audio_a, sr_a = load_audio(audio_a_path)

                progress(0.3, desc="Loading audio B…")
                audio_b, sr_b = load_audio(audio_b_path)

                progress(0.5, desc="Generating comparison…")
                comparison = plot_comparison(audio_a, sr_a, audio_b, sr_b)

                dur_a = audio_a.shape[-1] / sr_a
                dur_b = audio_b.shape[-1] / sr_b
                msg = (
                    f"✅ Comparison complete — A: {dur_a:.1f}s ({sr_a}Hz), "
                    f"B: {dur_b:.1f}s ({sr_b}Hz)"
                )
                return msg, comparison

            except Exception as e:
                logger.error("Comparison failed: %s", e, exc_info=True)
                return f"❌ Error: {e}", None

        # Wire analyze callback
        analyze_btn.click(
            fn=do_analyze,
            inputs=[input_audio],
            outputs=[status, waveform_img, spectrogram_img, spectrum_img],
        )

        # Wire comparison callback
        compare_btn.click(
            fn=do_compare,
            inputs=[input_audio, ref_audio],
            outputs=[comparison_status, comparison_img],
        )

    return tab
