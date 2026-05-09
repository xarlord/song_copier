"""Batch Processing tab — separate multiple files with a single preset."""

import logging
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr

from config import load_config
from pipelines.stem_separator import separate_stems, mix_stems_from_files
from services.presets import list_presets, load_preset

logger = logging.getLogger(__name__)


def create_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown(
            "## 📦 Batch Stem Separator\n"
            "Upload multiple audio files, pick a preset, and separate them all at once. "
            "Downloads as a ZIP file with separated stems + mix per song."
        )

        with gr.Row():
            input_files = gr.File(
                label="Audio Files",
                file_count="multiple",
                file_types=[".mp3", ".wav", ".flac", ".ogg", ".m4a"],
            )

        with gr.Row():
            with gr.Column(scale=1):
                preset_names = [p["name"] for p in list_presets()]
                preset_dropdown = gr.Dropdown(
                    choices=preset_names,
                    value=preset_names[0] if preset_names else None,
                    label="Preset",
                    info="Volume and effect settings for the mix",
                )
                model_dropdown = gr.Dropdown(
                    choices=["htdemucs_6s", "htdemucs_ft", "htdemucs"],
                    value="htdemucs_6s",
                    label="Separation Model",
                )
                with gr.Row():
                    clean_checkbox = gr.Checkbox(value=True, label="Apply stem cleaning")
                    include_mix = gr.Checkbox(value=True, label="Include mix in output")
                run_btn = gr.Button("🚀 Process All Files", variant="primary", size="lg")

        status = gr.Textbox(label="Status", interactive=False, lines=4)
        output_file = gr.File(label="Download Results (ZIP)")

        def process_batch(files, preset_name, model_name, apply_cleaning, include_mix_flag, progress=gr.Progress()):
            if not files:
                return "Please upload at least one audio file.", None

            preset = load_preset(preset_name) if preset_name else None
            if not preset:
                return f"Preset '{preset_name}' not found.", None

            config = load_config()
            output_dir = config["paths"]["output"]
            temp_dir = config["paths"]["temp"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = temp_dir / f"batch_{timestamp}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            results = []
            total = len(files)

            for i, file_obj in enumerate(files):
                file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
                base_name = Path(file_path).stem

                try:
                    progress(i / total, desc=f"[{i+1}/{total}] Separating: {base_name}")

                    # Separate stems
                    sep_result = separate_stems(
                        input_path=file_path,
                        model_name=model_name,
                        output_format="wav",
                        apply_cleaning=apply_cleaning,
                    )

                    stem_files = sep_result["stem_files"]

                    # Build volumes and effects from preset
                    volumes = {}
                    effects = {}
                    preset_stems = preset.get("stems", {})
                    for stem_name, stem_file in stem_files.items():
                        stem_cfg = preset_stems.get(stem_name, {})
                        volumes[stem_name] = stem_cfg.get("volume", 1.0)
                        effects[stem_name] = stem_cfg.get("effects", {})

                    # Save mix
                    if include_mix_flag:
                        mix_path = batch_dir / f"{base_name}_mix.wav"
                        mix_stems_from_files(
                            stem_files=stem_files,
                            volumes=volumes,
                            output_path=str(mix_path),
                            effects=effects,
                        )

                    # Copy stem files to batch dir
                    for stem_name, stem_path in stem_files.items():
                        src = Path(stem_path)
                        dst = batch_dir / f"{base_name}_{stem_name}.wav"
                        if src.exists():
                            import shutil
                            shutil.copy2(src, dst)

                    results.append(f"✓ {base_name}")

                except Exception as e:
                    logger.error(f"Batch error on {base_name}: {e}", exc_info=True)
                    results.append(f"✗ {base_name}: {e}")

                progress((i + 1) / total, desc=f"[{i+1}/{total}] Done: {base_name}")

            # Create ZIP
            zip_path = output_dir / f"batch_results_{timestamp}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(batch_dir.iterdir()):
                    zf.write(f, f.name)

            # Cleanup temp
            import shutil
            shutil.rmtree(batch_dir, ignore_errors=True)

            success = sum(1 for r in results if r.startswith("✓"))
            failed = sum(1 for r in results if r.startswith("✗"))
            status_msg = f"Completed: {success}/{total} succeeded, {failed} failed.\n" + "\n".join(results)

            return status_msg, str(zip_path)

        run_btn.click(
            fn=process_batch,
            inputs=[input_files, preset_dropdown, model_dropdown, clean_checkbox, include_mix],
            outputs=[status, output_file],
        )

    return tab
