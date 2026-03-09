"""Vocal separation using Facebook's SAM-Audio model."""

import os

import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor


def separate_vocals(
    audio_path: str,
    output_dir: str,
    model_name: str = "facebook/sam-audio-base",
) -> tuple[str, str]:
    """Separate vocals from instrumental using SAM-Audio.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to save separated tracks.
        model_name: HuggingFace model identifier for SAM-Audio.

    Returns:
        Tuple of (vocals_path, instrumental_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM-Audio model '{model_name}' on {device}...")

    model = SAMAudio.from_pretrained(model_name)
    processor = SAMAudioProcessor.from_pretrained(model_name)
    model = model.eval().to(device)

    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed to match processor's expected sample rate
    if sr != processor.audio_sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, processor.audio_sampling_rate)
        waveform = resampler(waveform)

    # SAM-Audio can handle long audio by processing in chunks
    # Process the full audio with text prompt to isolate vocals
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
