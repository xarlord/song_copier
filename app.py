"""Audio Generator Studio — Local song spin-off generation.

Launch: python app.py [--share]
"""

import argparse
import sys

import gradio as gr

sys.path.insert(0, ".")

from utils.logging import setup_logging
from utils.gpu import get_device, get_vram_info
from services.cleanup import run_startup_cleanup
from config import load_config
from ui.styles import CUSTOM_CSS
from ui.tabs.instrumental_tab import create_tab as create_instrumental_tab
from ui.tabs.voice_swap_tab import create_tab as create_voice_swap_tab
from ui.tabs.spinoff_tab import create_tab as create_spinoff_tab
from ui.tabs.batch_tab import create_tab as create_batch_tab
from ui.tabs.mastering_tab import create_tab as create_mastering_tab
from ui.tabs.visualization_tab import create_tab as create_visualization_tab
from ui.tabs.recording_tab import create_tab as create_recording_tab


CUSTOM_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0f0f1a",
    body_background_fill_dark="#0f0f1a",
    button_primary_background_fill="linear-gradient(135deg, #533483, #0f3460)",
    button_primary_background_fill_hover="linear-gradient(135deg, #6b44a8, #1a4a80)",
    button_primary_text_color="white",
    block_background_fill="#16162a",
    block_background_fill_dark="#16162a",
    block_border_color="rgba(255,255,255,0.06)",
    block_title_text_color="#b8b8d0",
    input_background_fill="#1a1a30",
    input_border_color="rgba(255,255,255,0.08)",
    accordion_text_color="#c0c0d8",
    accordion_text_color_dark="#c0c0d8",
)


def create_app() -> gr.Blocks:
    device = get_device()
    vram = get_vram_info()

    with gr.Blocks(title="Audio Generator Studio") as app:
        with gr.Column(elem_classes=["app-header"]):
            gr.Markdown(
                "# Audio Generator Studio\n"
                "Local AI-powered audio manipulation. All processing runs on your GPU."
            )
            gr.Markdown(
                f"**Device:** {vram.get('device', device)} | "
                f"**VRAM:** {vram.get('total_mb', 'N/A')} MB",
                elem_classes=["device-info"],
            )

        with gr.Tabs():
            with gr.Tab("🎙️ Record"):
                create_recording_tab()
            with gr.Tab("Stem Separator"):
                create_instrumental_tab()
            with gr.Tab("Voice Swap"):
                create_voice_swap_tab()
            with gr.Tab("Song Spin-off"):
                create_spinoff_tab()
            with gr.Tab("🎛️ Mastering"):
                create_mastering_tab()
            with gr.Tab("📊 Visualize"):
                create_visualization_tab()
            with gr.Tab("Batch Process"):
                create_batch_tab()

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Generator Studio")
    parser.add_argument("--share", action="store_true", help="Create public sharing link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")

    # Clean up old temp/output files at startup
    config = load_config()
    run_startup_cleanup(config)

    app = create_app()
    app.launch(
        share=args.share,
        server_port=args.port,
        inbrowser=True,
        css=CUSTOM_CSS,
        theme=CUSTOM_THEME,
    )
