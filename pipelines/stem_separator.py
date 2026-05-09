"""Stem separation pipeline with individual stem exports."""

import logging
from pathlib import Path
from datetime import datetime

from config import load_config, get_device
from models.demucs_wrapper import DemucsWrapper
from services.audio_io import save_audio
from services.mixing import mix_stems_with_volumes
from services.stem_cleaning import clean_stem
from utils.gpu import register_model, unregister_model

logger = logging.getLogger(__name__)


def separate_stems(
    input_path: str,
    model_name: str = "htdemucs_6s",
    segment: int = 0,
    output_format: str = "wav",
    apply_cleaning: bool = True,
    progress_callback=None,
) -> dict:
    """Separate audio file into stems and save each stem individually.

    Returns:
        dict with keys:
            - stems: list of stem names
            - stem_files: dict mapping stem name to file path
            - sample_rate: int
            - duration: float in seconds
    """
    config = load_config()
    device = get_device(config)
    temp_dir = config["paths"]["temp"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_path).stem

    if progress_callback:
        progress_callback(0.0, f"Loading {model_name} model...")

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
            progress_callback(0.6, "Applying stem cleaning...")

        # Apply per-stem cleaning to reduce artifacts
        if apply_cleaning:
            for name, audio in stems.items():
                stems[name] = clean_stem(name, audio, sr, enabled=True)

        if progress_callback:
            progress_callback(0.8, "Saving individual stems...")

        stem_files = {}
        for stem_name, stem_audio in stems.items():
            filename = f"{base_name}_{stem_name}_{timestamp}.{output_format}"
            filepath = save_audio(
                stem_audio,
                temp_dir / filename,
                sr,
                format=output_format,
            )
            stem_files[stem_name] = str(filepath)

        if progress_callback:
            progress_callback(1.0, "Done!")

        return {
            "stems": list(stems.keys()),
            "stem_files": stem_files,
            "sample_rate": sr,
            "duration": stems[list(stems.keys())[0]].shape[-1] / sr,
        }

    finally:
        unregister_model("demucs")
        demucs.unload_model()


def mix_stems_from_files(
    stem_files: dict[str, str],
    volumes: dict[str, float],
    output_path: str,
    output_format: str = "wav",
    effects: dict[str, dict] | None = None,
    progress_callback=None,
) -> str:
    """Load stem files, apply effects, mix with volumes, and save result.

    Args:
        stem_files: dict mapping stem name to file path
        volumes: dict mapping stem name to volume 0.0-1.0
        output_path: where to save the mixed file
        output_format: wav, mp3, flac
        effects: dict mapping stem name to effects_config dict.
            If None, no effects are applied.
        progress_callback: optional callback(pct, msg)

    Returns:
        Path to mixed audio file.
    """
    from services.audio_io import load_audio
    from services.effects import apply_effects

    if progress_callback:
        progress_callback(0.0, "Loading stems...")

    stems = {}
    sr = None
    for name, filepath in stem_files.items():
        audio, sample_rate = load_audio(filepath)
        if sr is None:
            sr = sample_rate
        stems[name] = audio

    if progress_callback:
        progress_callback(0.3, "Applying effects...")

    # Apply per-stem effects
    if effects:
        for name, audio in stems.items():
            stem_effects = effects.get(name)
            if stem_effects:
                stems[name] = apply_effects(audio, sr, stem_effects)

    if progress_callback:
        progress_callback(0.6, "Mixing...")

    mixed = mix_stems_with_volumes(stems, volumes, normalize=True)

    if progress_callback:
        progress_callback(0.8, "Saving mix...")

    result_path = save_audio(mixed, output_path, sr, format=output_format)

    if progress_callback:
        progress_callback(1.0, "Done!")

    return str(result_path)
