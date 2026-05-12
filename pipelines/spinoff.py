"""Spin-off pipeline — generate song variations using MusicGen."""

import logging
from pathlib import Path
from datetime import datetime

from config import load_config, get_device
from models.demucs_wrapper import DemucsWrapper
from models.musicgen_wrapper import MusicGenWrapper
from services.audio_io import load_audio, save_audio
from services.analysis import analyze_audio, build_prompt_from_analysis
from services.mixing import mix_stems, normalize_audio
from utils.gpu import register_model, unregister_model

logger = logging.getLogger(__name__)


def generate_spinoff(
    input_path: str,
    prompt: str = "",
    model_name: str = "facebook/musicgen-melody",
    duration: float = 15.0,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    use_melody_conditioning: bool = True,
    mix_with_original_stems: bool = False,
    demucs_segment: int = 10,
    output_format: str = "wav",
    progress_callback=None,
) -> dict:
    """Generate a spin-off/variation of the input song.

    Returns:
        dict with keys: output_path, raw_generated_path, analysis
    """
    config = load_config()
    device = get_device(config)
    output_dir = config["paths"]["output"]
    temp_dir = config["paths"]["temp"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_path).stem

    if progress_callback:
        progress_callback(0.0, "Separating stems...")

    # Step 1: Separate with Demucs to get instrumental for melody conditioning
    demucs = DemucsWrapper(
        model_name="htdemucs", device=device,
        cache_dir=config["paths"]["cache"], segment=demucs_segment,
    )
    register_model("demucs", demucs)
    demucs.load_model()

    try:
        stems, sr = demucs.separate(input_path)
    finally:
        demucs.unload_model()
        unregister_model("demucs")

    if progress_callback:
        progress_callback(0.2, "Analyzing audio...")

    # Step 2: Analyze original
    analysis = analyze_audio(input_path)

    # Save instrumental for melody conditioning
    instrumental = mix_stems(stems, exclude=["vocals"])
    melody_path = temp_dir / f"{base_name}_melody_{timestamp}.wav"
    save_audio(instrumental, melody_path, sr)

    if progress_callback:
        progress_callback(0.3, "Building prompt...")

    # Step 3: Build prompt
    full_prompt = build_prompt_from_analysis(analysis, prompt)

    if progress_callback:
        progress_callback(0.35, f"Generating music (prompt: '{full_prompt}')...")

    # Step 4: Generate with MusicGen
    musicgen = MusicGenWrapper(
        model_name=model_name, device=device,
        cache_dir=config["paths"]["cache"],
        duration=duration, temperature=temperature, cfg_coef=cfg_coef,
    )
    register_model("musicgen", musicgen)

    try:
        if use_melody_conditioning:
            generated, gen_sr = musicgen.generate(
                prompt=full_prompt,
                melody_path=str(melody_path),
                duration=duration,
                temperature=temperature,
                cfg_coef=cfg_coef,
            )
        else:
            generated, gen_sr = musicgen.generate(
                prompt=full_prompt,
                duration=duration,
                temperature=temperature,
                cfg_coef=cfg_coef,
            )
    finally:
        musicgen.unload_model()
        unregister_model("musicgen")

    if progress_callback:
        progress_callback(0.8, "Saving output...")

    # Save raw generated audio
    raw_path = save_audio(
        generated,
        output_dir / f"{base_name}_spinoff_raw_{timestamp}.{output_format}",
        gen_sr,
        format=output_format,
    )

    output = generated

    # Step 5: Optionally mix with original stems
    if mix_with_original_stems:
        import numpy as np
        import torch
        import torchaudio

        # Build stems to mix back in (drums + bass by default)
        mix_stem_names = ["drums", "bass"]
        stems_to_mix = {}
        for stem_name in mix_stem_names:
            if stem_name in stems:
                stem_audio = torch.from_numpy(stems[stem_name])
                # Resample to match generated sample rate
                if sr != gen_sr:
                    stem_audio = torchaudio.functional.resample(
                        stem_audio, sr, gen_sr
                    )
                stems_to_mix[stem_name] = stem_audio.numpy()

        if stems_to_mix:
            # Pad/trim stems and generated to same length, then mix
            mixed = output.copy()
            max_len = max(mixed.shape[-1], max(
                s.shape[-1] for s in stems_to_mix.values()
            ))
            # Pad generated if needed
            if mixed.shape[-1] < max_len:
                mixed = np.pad(
                    mixed, ((0, 0), (0, max_len - mixed.shape[-1]))
                )
            # Add each stem (padded to match length)
            for stem_name, stem_audio in stems_to_mix.items():
                if stem_audio.shape[-1] < max_len:
                    padded = np.zeros(
                        (stem_audio.shape[0], max_len), dtype=np.float32
                    )
                    padded[..., : stem_audio.shape[-1]] = stem_audio
                    stem_audio = padded
                mixed = mixed + stem_audio

            output = mixed
            if progress_callback:
                progress_callback(
                    0.85,
                    f"Mixed generated audio with: {', '.join(stems_to_mix.keys())}"
                )

    final_path = save_audio(
        normalize_audio(output),
        output_dir / f"{base_name}_spinoff_{timestamp}.{output_format}",
        gen_sr,
        format=output_format,
    )

    # Cleanup
    melody_path.unlink(missing_ok=True)

    if progress_callback:
        progress_callback(1.0, "Done!")

    return {
        "output_path": str(final_path),
        "raw_generated_path": str(raw_path),
        "analysis": analysis,
        "prompt_used": full_prompt,
    }