"""Instrumental pipeline — remove vocals from a song."""

import logging
from pathlib import Path
from datetime import datetime

from config import load_config, get_device
from models.demucs_wrapper import DemucsWrapper
from services.audio_io import save_audio
from services.mixing import mix_stems
from utils.gpu import register_model, unregister_model

logger = logging.getLogger(__name__)


def create_instrumental(
    input_path: str,
    model_name: str = "htdemucs_ft",
    segment: int = 10,
    output_format: str = "wav",
    progress_callback=None,
) -> dict:
    """Remove vocals from a song and return paths to output files.

    Args:
        input_path: Path to input audio file
        model_name: Demucs model variant
        segment: Segment length for VRAM control
        output_format: Output audio format
        progress_callback: Optional callback(progress_float, message_str)

    Returns:
        dict with keys: instrumental_path, vocals_path, stems
    """
    config = load_config()
    device = get_device(config)
    output_dir = config["paths"]["output"]

    if progress_callback:
        progress_callback(0.0, "Loading Demucs model...")

    demucs = DemucsWrapper(
        model_name=model_name,
        device=device,
        cache_dir=config["paths"]["cache"],
        segment=segment,
    )
    register_model("demucs", demucs)

    try:
        if progress_callback:
            progress_callback(0.1, "Separating stems...")

        stems, sr = demucs.separate(input_path)

        if progress_callback:
            progress_callback(0.7, "Mixing instrumental...")

        # Mix everything except vocals
        instrumental = mix_stems(stems, exclude=["vocals"])
        vocals = stems.get("vocals")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(input_path).stem

        instr_path = save_audio(
            instrumental,
            output_dir / f"{base_name}_instrumental_{timestamp}.{output_format}",
            sr,
            format=output_format,
        )

        vocals_path = None
        if vocals is not None:
            vocals_path = save_audio(
                vocals,
                output_dir / f"{base_name}_vocals_{timestamp}.{output_format}",
                sr,
                format=output_format,
            )

        if progress_callback:
            progress_callback(1.0, "Done!")

        return {
            "instrumental_path": str(instr_path),
            "vocals_path": str(vocals_path) if vocals_path else None,
            "stems": list(stems.keys()),
        }

    finally:
        unregister_model("demucs")
        demucs.unload_model()