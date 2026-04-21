"""Voice swap pipeline — replace vocals with a different voice."""

import logging
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

from config import load_config, get_device
from models.demucs_wrapper import DemucsWrapper
from models.rvc_wrapper import RVCWrapper, list_voice_models
from services.audio_io import load_audio, save_audio
from services.mixing import mix_stems
from utils.gpu import register_model, unregister_model

logger = logging.getLogger(__name__)


def swap_vocals(
    input_path: str,
    voice_model_path: str,
    model_name: str = "htdemucs_ft",
    segment: int = 10,
    f0method: str = "rmvpe",
    f0up_key: int = 0,
    index_rate: float = 0.5,
    protect: float = 0.33,
    output_format: str = "wav",
    progress_callback=None,
) -> dict:
    """Replace vocals in a song with a different voice using RVC.

    Returns:
        dict with keys: output_path, instrumental_path, converted_vocals_path
    """
    config = load_config()
    device = get_device(config)
    output_dir = config["paths"]["output"]
    temp_dir = config["paths"]["temp"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_path).stem

    demucs = DemucsWrapper(
        model_name=model_name, device=device,
        cache_dir=config["paths"]["cache"], segment=segment,
    )

    if progress_callback:
        progress_callback(0.0, "Loading Demucs...")

    register_model("demucs", demucs)
    demucs.load_model()

    try:
        if progress_callback:
            progress_callback(0.1, "Separating stems...")

        stems, sr = demucs.separate(input_path)
        demucs.unload_model()
        unregister_model("demucs")

        if progress_callback:
            progress_callback(0.4, "Extracting vocals...")

        vocals = stems.get("vocals")
        if vocals is None:
            raise ValueError("No vocals stem found in separation")

        # Save vocals to temp for RVC processing
        temp_vocals = temp_dir / f"{base_name}_vocals_{timestamp}.wav"
        save_audio(vocals, temp_vocals, sr)
        temp_converted = temp_dir / f"{base_name}_converted_{timestamp}.wav"

        if progress_callback:
            progress_callback(0.5, "Loading RVC voice model...")

        rvc = RVCWrapper(
            device=device, cache_dir=config["paths"]["cache"],
            f0method=f0method, f0up_key=f0up_key,
            index_rate=index_rate, protect=protect,
        )
        register_model("rvc", rvc)
        rvc.load_model()

        try:
            rvc.load_voice_model(voice_model_path)

            if progress_callback:
                progress_callback(0.6, "Converting vocals...")

            rvc.convert(str(temp_vocals), str(temp_converted))

        finally:
            rvc.unload_model()
            unregister_model("rvc")

        if progress_callback:
            progress_callback(0.8, "Mixing final output...")

        # Load converted vocals
        converted_vocals, conv_sr = load_audio(str(temp_converted), target_sr=sr)

        # Replace vocals stem with converted version
        stems["vocals"] = converted_vocals

        # Mix all stems back together
        final = mix_stems(stems)

        output_path = save_audio(
            final,
            output_dir / f"{base_name}_voice_swap_{timestamp}.{output_format}",
            sr,
            format=output_format,
        )

        # Save instrumental separately
        instrumental = mix_stems(stems, exclude=["vocals"])
        instr_path = save_audio(
            instrumental,
            output_dir / f"{base_name}_instrumental_{timestamp}.{output_format}",
            sr,
            format=output_format,
        )

        # Cleanup temp files
        temp_vocals.unlink(missing_ok=True)
        temp_converted.unlink(missing_ok=True)

        if progress_callback:
            progress_callback(1.0, "Done!")

        return {
            "output_path": str(output_path),
            "instrumental_path": str(instr_path),
            "converted_vocals_path": str(temp_converted) if temp_converted.exists() else None,
        }

    except Exception:
        demucs.unload_model()
        unregister_model("demucs")
        raise