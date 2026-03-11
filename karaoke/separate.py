"""Vocal separation using Facebook's Demucs or SAM-Audio model."""

import os
import subprocess


def _separate_with_demucs(audio_path: str, output_dir: str) -> tuple[str, str]:
    """Separate vocals using Meta's Demucs (htdemucs model)."""
    print("Using Demucs (htdemucs) for vocal separation...")
    demucs_out = os.path.join(output_dir, "demucs_tmp")
    subprocess.run(
        [
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "-o", demucs_out,
            audio_path,
        ],
        check=True,
    )

    # Demucs outputs to <out>/<model>/<track_name>/vocals.wav and no_vocals.wav
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    model_dir = os.path.join(demucs_out, "htdemucs", basename)

    vocals_path = os.path.join(output_dir, "vocals.wav")
    instrumental_path = os.path.join(output_dir, "instrumental.wav")

    os.rename(os.path.join(model_dir, "vocals.wav"), vocals_path)
    os.rename(os.path.join(model_dir, "no_vocals.wav"), instrumental_path)

    print(f"Saved vocals to: {vocals_path}")
    print(f"Saved instrumental to: {instrumental_path}")
    return vocals_path, instrumental_path


def _separate_with_sam_audio(
    audio_path: str, output_dir: str, model_name: str,
) -> tuple[str, str]:
    """Separate vocals using Facebook's SAM-Audio model."""
    import torch
    import torchaudio
    from sam_audio import SAMAudio, SAMAudioProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM-Audio model '{model_name}' on {device}...")

    model = SAMAudio.from_pretrained(model_name)
    processor = SAMAudioProcessor.from_pretrained(model_name)
    model = model.eval().to(device)

    waveform, sr = torchaudio.load(audio_path)
    if sr != processor.audio_sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, processor.audio_sampling_rate)
        waveform = resampler(waveform)

    batch = processor(
        audios=[audio_path],
        descriptions=["vocals, singing voice"],
    ).to(device)

    print("Separating vocals from instrumental...")
    with torch.inference_mode():
        result = model.separate(batch)

    vocals_path = os.path.join(output_dir, "vocals.wav")
    instrumental_path = os.path.join(output_dir, "instrumental.wav")

    torchaudio.save(vocals_path, result.target.cpu(), processor.audio_sampling_rate)
    torchaudio.save(
        instrumental_path, result.residual.cpu(), processor.audio_sampling_rate
    )

    print(f"Saved vocals to: {vocals_path}")
    print(f"Saved instrumental to: {instrumental_path}")
    return vocals_path, instrumental_path


def separate_vocals(
    audio_path: str,
    output_dir: str,
    model_name: str = "facebook/sam-audio-base",
) -> tuple[str, str]:
    """Separate vocals from instrumental.

    Tries SAM-Audio first, falls back to Demucs if unavailable.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to save separated tracks.
        model_name: HuggingFace model identifier for SAM-Audio.

    Returns:
        Tuple of (vocals_path, instrumental_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from sam_audio import SAMAudio  # noqa: F401
        return _separate_with_sam_audio(audio_path, output_dir, model_name)
    except ImportError:
        print("SAM-Audio not available, falling back to Demucs...")
        return _separate_with_demucs(audio_path, output_dir)
