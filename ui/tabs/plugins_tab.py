"""Plugins tab — browse, configure, and run third-party audio processing plugins."""

from __future__ import annotations

import json as _json
import logging
from typing import Any

import gradio as gr
import numpy as np

from services import plugin_manager
from services.audio_io import load_audio, save_audio

logger = logging.getLogger(__name__)

MAX_PARAM_SLOTS = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plugins_to_table_rows() -> list[list[str]]:
    """Return plugin info as rows for the Dataframe widget."""
    plugins = plugin_manager.list_available_plugins()
    return [
        [
            p.get("name", "—"),
            p.get("version", "—"),
            p.get("description", ""),
            p.get("author", "—"),
        ]
        for p in plugins
    ]


def _param_default(param: dict) -> Any:
    """Return the default value for a parameter definition."""
    if "default" in param:
        return param["default"]
    ptype = param.get("type", "float")
    if ptype == "int":
        return param.get("min", 0)
    if ptype == "bool":
        return False
    if ptype == "choice":
        choices = param.get("choices", [])
        return choices[0] if choices else ""
    return param.get("min", 0.0)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _refresh_plugins() -> list[list[str]]:
    """Re-scan plugin directory and return updated table rows."""
    plugin_manager.discover_plugins()
    return _plugins_to_table_rows()


def _on_plugin_select(evt: gr.SelectData):
    """Handle table row selection — return info + slider updates."""
    rows = _plugins_to_table_rows()
    empty = [gr.update(visible=False)] * MAX_PARAM_SLOTS

    if evt.index is None or evt.index[0] is None:
        return ("_No plugin selected._", "", [], *empty, gr.update(visible=False))

    row_idx = evt.index[0]
    if row_idx < 0 or row_idx >= len(rows):
        return ("_Invalid selection._", "", [], *empty, gr.update(visible=False))

    plugin_name = rows[row_idx][0]
    info = plugin_manager.get_plugin_info(plugin_name)
    if info is None:
        return (
            f"_Plugin '{plugin_name}' not found._", "", [],
            *empty, gr.update(visible=False),
        )

    desc = (
        f"**{info.get('name', '?')}** v{info.get('version', '?')}  \n"
        f"{info.get('description', '')}  \n"
        f"Author: {info.get('author', '—')}"
    )

    parameters = info.get("parameters", [])
    pnames: list[str] = []
    slider_updates: list[dict] = []
    json_extra: dict[str, Any] = {}

    for i in range(MAX_PARAM_SLOTS):
        if i < len(parameters):
            p = parameters[i]
            pname = p["name"]
            ptype = p.get("type", "float")
            label = p.get("label", pname)
            pnames.append(pname)

            if ptype in ("float", "int"):
                step = p.get("step", 0.01 if ptype == "float" else 1)
                slider_updates.append(gr.update(
                    visible=True,
                    label=label,
                    minimum=p.get("min", 0),
                    maximum=p.get("max", 1),
                    value=_param_default(p),
                    step=step,
                ))
            elif ptype == "bool":
                default_val = _param_default(p)
                slider_updates.append(gr.update(
                    visible=True,
                    label=f"{label} (0 = off · 1 = on)",
                    minimum=0, maximum=1,
                    value=1 if default_val else 0,
                    step=1,
                ))
            elif ptype == "choice":
                json_extra[pname] = _param_default(p)
                slider_updates.append(gr.update(visible=False))
            else:
                slider_updates.append(gr.update(visible=False))
        else:
            slider_updates.append(gr.update(visible=False))

    json_visible = len(json_extra) > 0
    json_str = _json.dumps(json_extra, indent=2) if json_extra else ""

    return (desc, plugin_name, pnames, *slider_updates, gr.update(visible=json_visible, value=json_str))


def _run_plugin(
    plugin_name: str,
    pnames: list[str],
    audio_path: str | None,
    json_str: str,
    *slider_values: float,
) -> tuple[str | None, str]:
    """Load audio, apply plugin, save result, return (output_path, status)."""
    if not plugin_name:
        return None, "⚠️ No plugin selected."
    if not audio_path:
        return None, "⚠️ Please upload audio first."

    # Build kwargs from sliders + JSON override
    kwargs: dict[str, Any] = {}
    info = plugin_manager.get_plugin_info(plugin_name)
    params_schema = info.get("parameters", []) if info else []

    for i, name in enumerate(pnames):
        if i < len(slider_values):
            ptype = "float"
            for p in params_schema:
                if p["name"] == name:
                    ptype = p.get("type", "float")
                    break
            val = slider_values[i]
            if ptype == "bool":
                kwargs[name] = bool(val >= 0.5)
            elif ptype == "int":
                kwargs[name] = int(val)
            else:
                kwargs[name] = float(val)

    if json_str and json_str.strip():
        try:
            extra = _json.loads(json_str)
            if isinstance(extra, dict):
                kwargs.update(extra)
        except _json.JSONDecodeError:
            pass

    # Load audio
    try:
        audio, sr = load_audio(audio_path)
    except Exception as exc:
        return None, f"❌ Audio load error: {exc}"

    # Apply
    try:
        result = plugin_manager.apply_plugin(plugin_name, audio, sr, **kwargs)
    except Exception as exc:
        logger.exception("Plugin execution failed")
        return None, f"❌ Plugin error: {exc}"

    # Save
    try:
        out_path = save_audio(result, sr)
    except Exception as exc:
        return None, f"❌ Save error: {exc}"

    return out_path, f"✅ Plugin '{plugin_name}' applied successfully."


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def create_tab() -> gr.Column:
    """Build and return the Plugins tab as a Gradio Column."""

    with gr.Column() as tab:
        # ---- State ----
        selected_plugin = gr.State("")
        param_names_state = gr.State([])

        # ---- Header ----
        gr.Markdown("## 🔌 Plugin Studio")
        gr.Markdown(
            "Browse and run third-party audio processing plugins. "
            "Select a plugin from the table, tweak parameters, and hit Run."
        )

        # ---- Plugin browser ----
        plugin_table = gr.Dataframe(
            headers=["Name", "Version", "Description", "Author"],
            value=_plugins_to_table_rows(),
            label="Available Plugins",
            interactive=False,
            row_count=10,
            col_count=4,
            wrap=True,
            elem_classes=["section-card"],
        )
        refresh_btn = gr.Button("🔄 Refresh Plugins")

        gr.Markdown("---")

        # ---- Plugin info ----
        plugin_info_md = gr.Markdown("_No plugin selected._")

        # ---- Parameter sliders (pre-allocated, updated on select) ----
        gr.Markdown("### ⚙️ Parameters")
        sliders: list[gr.Slider] = []
        with gr.Row():
            for col_idx in range(4):
                with gr.Column():
                    for row_idx in range(2):
                        s = gr.Slider(
                            0, 1, value=0, step=0.01,
                            label=f"Parameter {col_idx * 2 + row_idx + 1}",
                            visible=False,
                        )
                        sliders.append(s)

        # JSON override for non-slider types (choice, etc.)
        param_json = gr.Textbox(
            label="Advanced — JSON override for choice/complex parameters",
            placeholder='{"key": "value"}',
            lines=3,
            visible=False,
            elem_classes=["status-box"],
        )

        gr.Markdown("---")

        # ---- Audio I/O ----
        gr.Markdown("### 🎵 Audio")
        with gr.Row():
            with gr.Column(scale=1):
                input_audio = gr.Audio(
                    label="Input Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Output Audio", type="filepath")

        with gr.Row():
            run_btn = gr.Button("▶️ Run Plugin", variant="primary")

        status_text = gr.Textbox(label="Status", interactive=False)

        # ==============================================================
        # Wire callbacks
        # ==============================================================

        # Refresh
        refresh_btn.click(fn=_refresh_plugins, outputs=[plugin_table])

        # Table row select → populate info + sliders
        select_outputs = [
            plugin_info_md,
            selected_plugin,
            param_names_state,
            *sliders,
            param_json,
        ]
        plugin_table.select(fn=_on_plugin_select, outputs=select_outputs)

        # Run button
        run_btn.click(
            fn=_run_plugin,
            inputs=[
                selected_plugin,
                param_names_state,
                input_audio,
                param_json,
                *sliders,
            ],
            outputs=[output_audio, status_text],
        )

    return tab
