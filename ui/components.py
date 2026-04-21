"""Shared UI components for Gradio interface."""

import gradio as gr


def audio_output_row(label: str = "Output"):
    """Create a standard audio output component."""
    return gr.Audio(label=label, type="filepath")


def model_status_display():
    """Create a model status textbox."""
    return gr.Textbox(label="Status", interactive=False, lines=1)


def progress_display():
    """Create a progress info box."""
    return gr.Textbox(label="Progress", interactive=False, lines=1)