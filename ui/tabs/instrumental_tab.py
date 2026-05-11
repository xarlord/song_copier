"""Stem Separator tab — separate audio into stems with volume mixer and effects."""

import gradio as gr
from datetime import datetime

from config import load_config
from pipelines.stem_separator import separate_stems, mix_stems_from_files
from services.effects import effects_config_from_ui
from services.presets import list_presets, load_preset, save_preset


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


def _make_effect_sliders():
    """Create effect slider controls for one stem."""
    sliders = {}
    with gr.Row():
        sliders["eq_low"] = gr.Slider(-12, 12, value=0, step=0.5, label="EQ Low (200Hz)")
        sliders["eq_mid"] = gr.Slider(-12, 12, value=0, step=0.5, label="EQ Mid (1kHz)")
        sliders["eq_high"] = gr.Slider(-12, 12, value=0, step=0.5, label="EQ High (4kHz)")
    with gr.Row():
        sliders["rev_room"] = gr.Slider(0, 1, value=0, step=0.05, label="Reverb Size")
        sliders["rev_damp"] = gr.Slider(0, 1, value=0.5, step=0.05, label="Reverb Damp")
        sliders["rev_wet"] = gr.Slider(0, 1, value=0, step=0.05, label="Reverb Wet")
    with gr.Row():
        sliders["del_sec"] = gr.Slider(0, 1, value=0, step=0.05, label="Delay (sec)")
        sliders["del_fb"] = gr.Slider(0, 0.9, value=0, step=0.05, label="Delay Feedback")
        sliders["del_mix"] = gr.Slider(0, 1, value=0, step=0.05, label="Delay Mix")
    with gr.Row():
        sliders["comp_thresh"] = gr.Slider(-40, 0, value=0, step=1, label="Compress Thresh")
        sliders["comp_ratio"] = gr.Slider(1, 10, value=1, step=0.5, label="Compress Ratio")
        sliders["comp_attack"] = gr.Slider(0.1, 50, value=10, step=0.5, label="Compress Attack")
    with gr.Row():
        sliders["pan"] = gr.Slider(-1, 1, value=0, step=0.05, label="Pan (L - R)")
    return sliders


def _collect_effects_from_sliders(sliders_dict: dict) -> list:
    """Extract slider values in the order expected by effects_config_from_ui."""
    return [
        sliders_dict["eq_low"], sliders_dict["eq_mid"], sliders_dict["eq_high"],
        sliders_dict["rev_room"], sliders_dict["rev_damp"], sliders_dict["rev_wet"],
        sliders_dict["del_sec"], sliders_dict["del_fb"], sliders_dict["del_mix"],
        sliders_dict["comp_thresh"], sliders_dict["comp_ratio"], sliders_dict["comp_attack"],
        sliders_dict["pan"],
    ]


def create_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        # State
        stem_files_state = gr.State(None)
        stems_list_state = gr.State(None)

        # === Upload Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Upload & Separate")
            with gr.Row():
                input_audio = gr.Audio(
                    label="Drop a song here",
                    type="filepath",
                    elem_classes=["upload-zone"],
                )

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["htdemucs_6s", "htdemucs_ft", "htdemucs"],
                    value="htdemucs_6s",
                    label="Model",
                    info="6s = vocals/drums/bass/guitar/piano/other",
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
                    info="Reduce artifacts",
                    scale=1,
                )
                separate_btn = gr.Button(
                    "Separate Stems",
                    variant="primary",
                    size="lg",
                    scale=1,
                )

        status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2,
            elem_classes=["status-box"],
        )

        # === Mixer Section (visible after separation) ===
        mixer_section = gr.Column(visible=False)

        with mixer_section:
            # --- Volume Mixer ---
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("### Volume Mixer")
                volume_sliders = {}
                with gr.Row():
                    for name in ["vocals", "drums", "bass"]:
                        with gr.Column(elem_classes=[f"stem-{name}"]):
                            volume_sliders[name] = gr.Slider(
                                0, 1.5, value=1, step=0.05,
                                label=f"{STEM_ICONS[name]} {STEM_LABELS[name]}",
                            )
                with gr.Row():
                    for name in ["guitar", "piano", "other"]:
                        with gr.Column(elem_classes=[f"stem-{name}"]):
                            volume_sliders[name] = gr.Slider(
                                0, 1.5, value=1, step=0.05,
                                label=f"{STEM_ICONS[name]} {STEM_LABELS[name]}",
                            )

                with gr.Row():
                    reset_btn = gr.Button("Reset 100%", size="sm")
                    mute_all_btn = gr.Button("Mute All", size="sm")
                    reset_fx_btn = gr.Button("Reset Effects", size="sm")
                    preview_btn = gr.Button(
                        "Preview Mix",
                        variant="primary",
                        size="lg",
                    )

            # --- Presets ---
            with gr.Column(elem_classes=["preset-bar"]):
                gr.Markdown("### Presets")
                preset_names = [p["name"] for p in list_presets()]
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=preset_names,
                        label="Load Preset",
                        info="Applies to all volume and effect sliders",
                        scale=3,
                    )
                    load_preset_btn = gr.Button("Apply", size="sm", scale=1)

                # Save preset
                with gr.Row():
                    save_preset_name = gr.Textbox(
                        label="Preset Name",
                        placeholder="My custom preset…",
                        scale=2,
                    )
                    save_preset_desc = gr.Textbox(
                        label="Description",
                        placeholder="Optional description",
                        scale=2,
                    )
                    save_preset_btn = gr.Button(
                        "💾 Save Preset", size="sm", scale=1,
                    )
                save_preset_status = gr.Textbox(
                    visible=False, show_label=False,
                )

            # --- Effects Rack ---
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("### Effects Rack")
                stem_effect_sliders = {}
                with gr.Accordion("Per-stem effects", open=False):
                    for stem_name in STEM_NAMES:
                        with gr.Accordion(
                            f"{STEM_ICONS[stem_name]} {STEM_LABELS[stem_name]}",
                            open=False,
                        ):
                            stem_effect_sliders[stem_name] = _make_effect_sliders()

        # === Output Section ===
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Output")
            mixed_output = gr.Audio(label="Mixed Preview", type="filepath")

            with gr.Accordion("Individual Stems", open=False):
                stem_audio_outputs = {}
                with gr.Row():
                    stem_audio_outputs["vocals"] = gr.Audio(label="Vocals", type="filepath")
                    stem_audio_outputs["drums"] = gr.Audio(label="Drums", type="filepath")
                with gr.Row():
                    stem_audio_outputs["bass"] = gr.Audio(label="Bass", type="filepath")
                    stem_audio_outputs["guitar"] = gr.Audio(label="Guitar", type="filepath")
                with gr.Row():
                    stem_audio_outputs["piano"] = gr.Audio(label="Piano", type="filepath")
                    stem_audio_outputs["other"] = gr.Audio(label="Other", type="filepath")

        # === Callbacks ===

        def do_separate(audio_path, model_name, fmt, apply_cleaning, progress=gr.Progress()):
            if not audio_path:
                return ("Please upload an audio file.", gr.Column(visible=False),
                        None, None, *[None] * 6)

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
                individual = [stem_files.get(s) for s in STEM_NAMES]

                msg = (f"Separated into {len(stems_list)} stems: {', '.join(stems_list)}. "
                       "Adjust volumes and effects, then click Preview Mix!")
                return (msg, gr.Column(visible=True),
                        stem_files, stems_list, *individual)
            except Exception as e:
                return (f"Error: {e}", gr.Column(visible=False),
                        None, None, *[None] * 6)

        all_effect_inputs = []
        for stem_name in STEM_NAMES:
            all_effect_inputs.extend(_collect_effects_from_sliders(stem_effect_sliders[stem_name]))

        preview_inputs = (
            [stem_files_state]
            + [volume_sliders[s] for s in STEM_NAMES]
            + all_effect_inputs
        )

        def do_preview(stem_files, v_vol, d_vol, b_vol, g_vol, p_vol, o_vol,
                       *all_fx_values, progress=gr.Progress()):
            if not stem_files:
                return None

            volumes = {
                "vocals": v_vol, "drums": d_vol, "bass": b_vol,
                "guitar": g_vol, "piano": p_vol, "other": o_vol,
            }

            effects = {}
            fx_per_stem = 13
            for i, stem_name in enumerate(STEM_NAMES):
                vals = all_fx_values[i * fx_per_stem:(i + 1) * fx_per_stem]
                effects[stem_name] = effects_config_from_ui(*vals)

            config = load_config()
            output_dir = config["paths"]["output"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"mix_preview_{timestamp}.wav"

            try:
                result = mix_stems_from_files(
                    stem_files=stem_files,
                    volumes=volumes,
                    output_path=str(output_path),
                    effects=effects,
                    progress_callback=lambda pct, msg: progress(pct, desc=msg),
                )
                return result
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Preview mix error: {e}", exc_info=True)
                return None

        def reset_volumes():
            return [1.0] * 6

        def mute_all():
            return [0.0] * 6

        def reset_effects():
            defaults = []
            for _ in STEM_NAMES:
                defaults.extend([0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 1, 10, 0])
            return defaults

        def do_load_preset(preset_name):
            if not preset_name:
                return reset_volumes() + reset_effects()

            data = load_preset(preset_name)
            if not data:
                return reset_volumes() + reset_effects()

            volumes = []
            for stem in STEM_NAMES:
                stem_data = data.get("stems", {}).get(stem, {})
                volumes.append(stem_data.get("volume", 1.0))

            fx_values = []
            for stem in STEM_NAMES:
                stem_data = data.get("stems", {}).get(stem, {})
                fx = stem_data.get("effects", {})
                eq = fx.get("eq", {})
                rev = fx.get("reverb", {})
                dly = fx.get("delay", {})
                comp = fx.get("compressor", {})
                fx_values.extend([
                    eq.get("low_gain", 0), eq.get("mid_gain", 0), eq.get("high_gain", 0),
                    rev.get("room_size", 0), rev.get("damping", 0.5), rev.get("wet_level", 0),
                    dly.get("delay_seconds", 0), dly.get("feedback", 0), dly.get("mix", 0),
                    comp.get("threshold_db", 0), comp.get("ratio", 1), comp.get("attack_ms", 10),
                    fx.get("pan", 0),
                ])

            return volumes + fx_values

        def do_save_preset(preset_name, preset_desc,
                           v_vol, d_vol, b_vol, g_vol, p_vol, o_vol,
                           *all_fx_values):
            """Save current slider values as a custom preset."""
            if not preset_name or not preset_name.strip():
                return gr.update(value="⚠️ Please enter a preset name.", visible=True), gr.Dropdown(choices=[p["name"] for p in list_presets()])

            volumes = {
                "vocals": v_vol, "drums": d_vol, "bass": b_vol,
                "guitar": g_vol, "piano": p_vol, "other": o_vol,
            }

            stems = {}
            fx_per_stem = 13
            for i, stem_name in enumerate(STEM_NAMES):
                vals = all_fx_values[i * fx_per_stem:(i + 1) * fx_per_stem]
                stems[stem_name] = {
                    "volume": volumes[stem_name],
                    "effects": effects_config_from_ui(*vals),
                }

            try:
                path = save_preset(preset_name.strip(), preset_desc.strip(), stems)
                updated = [p["name"] for p in list_presets()]
                return (
                    gr.update(value=f"✅ Saved preset: {preset_name.strip()}", visible=True),
                    gr.Dropdown(choices=updated, value=preset_name.strip()),
                )
            except Exception as e:
                return gr.update(value=f"❌ Error saving preset: {e}", visible=True), gr.Dropdown()

        # Wire callbacks
        separate_btn.click(
            fn=do_separate,
            inputs=[input_audio, model_dropdown, format_dropdown, clean_checkbox],
            outputs=[
                status, mixer_section, stem_files_state, stems_list_state,
                *[stem_audio_outputs[s] for s in STEM_NAMES],
            ],
        )

        preview_btn.click(fn=do_preview, inputs=preview_inputs, outputs=[mixed_output])

        reset_btn.click(fn=reset_volumes, outputs=[volume_sliders[s] for s in STEM_NAMES])
        mute_all_btn.click(fn=mute_all, outputs=[volume_sliders[s] for s in STEM_NAMES])

        all_fx_outputs = []
        for stem_name in STEM_NAMES:
            all_fx_outputs.extend(_collect_effects_from_sliders(stem_effect_sliders[stem_name]))
        reset_fx_btn.click(fn=reset_effects, outputs=all_fx_outputs)

        all_preset_outputs = [volume_sliders[s] for s in STEM_NAMES] + all_fx_outputs
        load_preset_btn.click(fn=do_load_preset, inputs=[preset_dropdown], outputs=all_preset_outputs)

        save_inputs = (
            [save_preset_name, save_preset_desc]
            + [volume_sliders[s] for s in STEM_NAMES]
            + all_effect_inputs
        )
        save_preset_btn.click(
            fn=do_save_preset,
            inputs=save_inputs,
            outputs=[save_preset_status, preset_dropdown],
        )

    return tab
