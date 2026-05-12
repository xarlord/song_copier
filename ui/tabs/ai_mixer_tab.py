"""AI Mixer tab — heuristic-based mixing assistant with per-stem suggestions.

Workflow:
  1. Upload audio → run Demucs separation
  2. Click "🤖 Analyze & Suggest" → heuristic analysis produces per-stem settings
  3. Review / tweak suggested values in the sliders
  4. Click "Apply Mix" → render final mix with all effects applied
"""

import logging
from datetime import datetime

import gradio as gr
import numpy as np

from config import load_config
from pipelines.stem_separator import separate_stems
from services.ai_mixer import analyze_stem, suggest_mix, apply_suggested_mix
from services.audio_io import save_audio

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_suggestions(suggestions: dict[str, dict]) -> str:
    """Pretty-print suggestions as a readable report."""
    lines = []
    for name, s in suggestions.items():
        icon = STEM_ICONS.get(name, "🎵")
        label = STEM_LABELS.get(name, name.title())
        lines.append(f"{icon} {label}:")
        lines.append(f"    RMS: {s['rms_db']:.1f} dB  |  Peak: {s['peak_db']:.1f} dB  |  "
                     f"Brightness: {s['spectral_centroid']:.0f} Hz")
        lines.append(f"    Volume: {s['suggested_volume']:.2f}  |  Pan: {s['suggested_pan']:+.2f}")
        eq = s["suggested_eq"]
        lines.append(f"    EQ  Low:{eq['low']:+.1f}  Mid:{eq['mid']:+.1f}  High:{eq['high']:+.1f} dB")
        rev = s["suggested_reverb"]
        lines.append(f"    Reverb  Room:{rev['room_size']:.2f}  Damp:{rev['damping']:.2f}  Wet:{rev['wet']:.2f}")
        comp = s["suggested_compressor"]
        lines.append(f"    Compressor  Thresh:{comp['threshold']:.1f} dB  Ratio:{comp['ratio']:.1f}")
        lines.append("")
    return "\n".join(lines)


def _suggestion_to_slider_values(suggestions: dict[str, dict]) -> list[float]:
    """Flatten suggestions into an ordered list of slider values for all stems.

    Order per stem: volume, eq_low, eq_mid, eq_high, rev_room, rev_damp,
                     rev_wet, comp_thresh, comp_ratio, pan
    """
    values: list[float] = []
    for name in STEM_NAMES:
        s = suggestions.get(name)
        if s is None:
            values.extend([1.0, 0, 0, 0, 0, 0.5, 0, 0, 1, 0])
            continue
        values.append(s["suggested_volume"])
        values.append(s["suggested_eq"]["low"])
        values.append(s["suggested_eq"]["mid"])
        values.append(s["suggested_eq"]["high"])
        values.append(s["suggested_reverb"]["room_size"])
        values.append(s["suggested_reverb"]["damping"])
        values.append(s["suggested_reverb"]["wet"])
        values.append(s["suggested_compressor"]["threshold"])
        values.append(s["suggested_compressor"]["ratio"])
        values.append(s["suggested_pan"])
    return values


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

def create_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        # ---- State ----
        stem_files_state = gr.State(None)
        stems_data_state = gr.State(None)  # dict[str, np.ndarray]
        sr_state = gr.State(44100)
        suggestions_state = gr.State(None)

        # ================================================================
        # Section 1 — Upload & Separate
        # ================================================================
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 🎵 Upload & Separate")
            input_audio = gr.Audio(
                label="Drop a song here",
                type="filepath",
                elem_classes=["upload-zone"],
            )
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["htdemucs_6s", "htdemucs_ft", "htdemucs"],
                    value="htdemucs_6s",
                    label="Separation Model",
                    scale=2,
                )
                format_dropdown = gr.Dropdown(
                    choices=["wav", "mp3", "flac"],
                    value="wav",
                    label="Format",
                    scale=1,
                )
                clean_checkbox = gr.Checkbox(
                    value=True,
                    label="Clean stems",
                    scale=1,
                )
                separate_btn = gr.Button(
                    "Separate Stems",
                    variant="primary",
                    size="lg",
                    scale=1,
                )

        status_box = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2,
            elem_classes=["status-box"],
        )

        # ================================================================
        # Section 2 — AI Analysis
        # ================================================================
        ai_section = gr.Column(visible=False)

        with ai_section:
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("### 🤖 AI Mix Analysis")
                analyze_btn = gr.Button(
                    "🤖 Analyze & Suggest Mix",
                    variant="primary",
                    size="lg",
                )

                suggestions_report = gr.Textbox(
                    label="AI Suggestions",
                    interactive=False,
                    lines=14,
                    elem_classes=["status-box"],
                    visible=False,
                )

            # ============================================================
            # Section 3 — Per-stem tweakable sliders
            # ============================================================
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("### 🎛️ Tweak Suggestions")
                gr.Markdown(
                    "Values are pre-filled by the AI. Adjust any slider to taste, "
                    "then click **Apply Mix**."
                )

                stem_sliders: dict[str, dict] = {}
                for stem_name in STEM_NAMES:
                    icon = STEM_ICONS[stem_name]
                    label = STEM_LABELS[stem_name]
                    with gr.Accordion(f"{icon} {label}", open=(stem_name in ("vocals", "drums", "bass"))):
                        sl: dict = {}
                        with gr.Row():
                            sl["volume"] = gr.Slider(
                                0, 1.5, value=1.0, step=0.01,
                                label="Volume",
                            )
                            sl["pan"] = gr.Slider(
                                -1, 1, value=0.0, step=0.05,
                                label="Pan (L← →R)",
                            )
                        with gr.Row():
                            sl["eq_low"] = gr.Slider(
                                -12, 12, value=0, step=0.5,
                                label="EQ Low (dB)",
                            )
                            sl["eq_mid"] = gr.Slider(
                                -12, 12, value=0, step=0.5,
                                label="EQ Mid (dB)",
                            )
                            sl["eq_high"] = gr.Slider(
                                -12, 12, value=0, step=0.5,
                                label="EQ High (dB)",
                            )
                        with gr.Row():
                            sl["rev_room"] = gr.Slider(
                                0, 1, value=0, step=0.05,
                                label="Reverb Room",
                            )
                            sl["rev_damp"] = gr.Slider(
                                0, 1, value=0.5, step=0.05,
                                label="Reverb Damp",
                            )
                            sl["rev_wet"] = gr.Slider(
                                0, 1, value=0, step=0.05,
                                label="Reverb Wet",
                            )
                        with gr.Row():
                            sl["comp_thresh"] = gr.Slider(
                                -40, 0, value=0, step=1,
                                label="Compress Thresh (dB)",
                            )
                            sl["comp_ratio"] = gr.Slider(
                                1, 10, value=1, step=0.5,
                                label="Compress Ratio",
                            )
                        stem_sliders[stem_name] = sl

            # Apply button
            apply_btn = gr.Button(
                "✨ Apply Mix",
                variant="primary",
                size="lg",
            )

        # ================================================================
        # Section 4 — Output
        # ================================================================
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 🔊 Output")
            output_audio = gr.Audio(label="AI Mixed Output", type="filepath")

            with gr.Accordion("Individual Stems", open=False):
                stem_audio_outputs: dict[str, gr.Audio] = {}
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

        # ---- Separate ----
        def do_separate(audio_path, model_name, fmt, apply_cleaning, progress=gr.Progress()):
            if not audio_path:
                return (
                    "Please upload an audio file.", gr.Column(visible=False),
                    None, None, 44100, None,
                    *[None] * 6,
                )

            def _progress(pct, msg):
                progress(pct, desc=msg)

            try:
                result = separate_stems(
                    input_path=audio_path,
                    model_name=model_name,
                    output_format=fmt,
                    apply_cleaning=apply_cleaning,
                    progress_callback=_progress,
                )

                stem_files = result["stem_files"]
                stems_list = result["stems"]

                # Load stems into memory for analysis
                from services.audio_io import load_audio
                stems_data: dict[str, np.ndarray] = {}
                sr_val = 44100
                for s_name in STEM_NAMES:
                    if s_name in stem_files:
                        audio_arr, loaded_sr = load_audio(stem_files[s_name])
                        stems_data[s_name] = audio_arr
                        sr_val = loaded_sr

                individual = [stem_files.get(s) for s in STEM_NAMES]
                msg = (
                    f"✅ Separated {len(stems_list)} stems: {', '.join(stems_list)}. "
                    "Click **🤖 Analyze & Suggest Mix** to get AI mixing suggestions!"
                )
                return (
                    msg, gr.Column(visible=True),
                    stem_files, stems_data, sr_val, None,
                    *individual,
                )
            except Exception as e:
                logger.error(f"Separation error: {e}", exc_info=True)
                return (
                    f"❌ Error: {e}", gr.Column(visible=False),
                    None, None, 44100, None,
                    *[None] * 6,
                )

        separate_btn.click(
            fn=do_separate,
            inputs=[input_audio, model_dropdown, format_dropdown, clean_checkbox],
            outputs=[
                status_box, ai_section,
                stem_files_state, stems_data_state, sr_state, suggestions_state,
                *[stem_audio_outputs[s] for s in STEM_NAMES],
            ],
        )

        # ---- Analyze ----
        def do_analyze(stems_data, sr, progress=gr.Progress()):
            if not stems_data or not sr:
                return "⚠️ Separate stems first.", gr.Textbox(visible=False), None, []

            progress(0.3, desc="Analyzing stems…")
            try:
                suggestions = suggest_mix(stems_data, sr)
                report = _format_suggestions(suggestions)
                slider_values = _suggestion_to_slider_values(suggestions)

                progress(1.0, desc="Done!")
                return (
                    f"✅ Analysis complete for {len(suggestions)} stems.",
                    gr.Textbox(value=report, visible=True),
                    suggestions,
                    slider_values,
                )
            except Exception as e:
                logger.error(f"Analysis error: {e}", exc_info=True)
                return f"❌ Error: {e}", gr.Textbox(visible=False), None, []

        # Collect all slider outputs for the analyze callback
        all_slider_components: list = []
        for name in STEM_NAMES:
            sl = stem_sliders[name]
            all_slider_components.extend([
                sl["volume"], sl["pan"],
                sl["eq_low"], sl["eq_mid"], sl["eq_high"],
                sl["rev_room"], sl["rev_damp"], sl["rev_wet"],
                sl["comp_thresh"], sl["comp_ratio"],
            ])

        analyze_btn.click(
            fn=do_analyze,
            inputs=[stems_data_state, sr_state],
            outputs=[status_box, suggestions_report, suggestions_state] + all_slider_components,
        )

        # ---- Apply Mix ----
        def do_apply(stems_data, sr, *slider_vals, progress=gr.Progress()):
            if not stems_data or not sr:
                return None, "⚠️ Separate and analyze stems first."

            progress(0.2, desc="Applying mix settings…")
            try:
                # Reconstruct suggestions from slider values
                slider_per_stem = 10
                user_suggestions: dict[str, dict] = {}
                for i, name in enumerate(STEM_NAMES):
                    vals = slider_vals[i * slider_per_stem:(i + 1) * slider_per_stem]
                    (vol, pan, eq_low, eq_mid, eq_high,
                     rev_room, rev_damp, rev_wet,
                     comp_thresh, comp_ratio) = vals

                    user_suggestions[name] = {
                        "rms_db": 0.0,
                        "peak_db": 0.0,
                        "spectral_centroid": 0.0,
                        "suggested_volume": vol,
                        "suggested_pan": pan,
                        "suggested_eq": {
                            "low": eq_low,
                            "mid": eq_mid,
                            "high": eq_high,
                        },
                        "suggested_reverb": {
                            "room_size": rev_room,
                            "damping": rev_damp,
                            "wet": rev_wet,
                        },
                        "suggested_compressor": {
                            "threshold": comp_thresh,
                            "ratio": comp_ratio,
                        },
                    }

                progress(0.5, desc="Rendering mix…")
                mixed = apply_suggested_mix(stems_data, sr, user_suggestions)

                progress(0.8, desc="Saving…")
                config = load_config()
                output_dir = config["paths"]["output"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = output_dir / f"ai_mix_{timestamp}.wav"
                save_audio(mixed, out_path, sr)

                progress(1.0, desc="Done!")
                return str(out_path), f"✅ Mix saved: {out_path.name}"
            except Exception as e:
                logger.error(f"Apply mix error: {e}", exc_info=True)
                return None, f"❌ Error: {e}"

        apply_btn.click(
            fn=do_apply,
            inputs=[stems_data_state, sr_state] + all_slider_components,
            outputs=[output_audio, status_box],
        )

    return tab
