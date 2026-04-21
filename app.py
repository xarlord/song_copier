"""Audio Generator Studio — Local song spin-off generation.

Launch: python app.py [--share]
"""

import argparse
import sys

import gradio as gr

# Add project root to path
sys.path.insert(0, ".")

from utils.logging import setup_logging
from utils.gpu import get_device, get_vram_info
from ui.styles import CUSTOM_CSS
from ui.tabs.instrumental_tab import create_tab as create_instrumental_tab
from ui.tabs.voice_swap_tab import create_tab as create_voice_swap_tab
from ui.tabs.spinoff_tab import create_tab as create_spinoff_tab


def create_app() -> gr.Blocks:
    device = get_device()
    vram = get_vram_info()

    with gr.Blocks(
        title="Audio Generator Studio",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 🎵 Audio Generator Studio\n"
            "Local AI-powered audio manipulation. All processing runs on your machine.\n\n"
            f"**Device:** {vram.get('device', device)} | "
            f"**VRAM:** {vram.get('total_mb', 'N/A')} MB"
        )

        with gr.Tabs():
            with gr.Tab("Vocal Remover"):
                create_instrumental_tab()
            with gr.Tab("Voice Swap"):
                create_voice_swap_tab()
            with gr.Tab("Song Spin-off"):
                create_spinoff_tab()

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Generator Studio")
    parser.add_argument("--share", action="store_true", help="Create public sharing link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")

    app = create_app()
    app.launch(
        share=args.share,
        server_port=args.port,
        inbrowser=True,
    )