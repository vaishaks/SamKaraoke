"""Download audio from YouTube using yt-dlp."""

import os
import re
import subprocess


def sanitize_filename(name: str) -> str:
    """Remove special characters from filename."""
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')


def download_youtube_audio(url: str, output_dir: str) -> tuple[str, str, str]:
    """Download audio from a YouTube URL.

    Returns:
        Tuple of (audio_path, title, artist).
    """
    os.makedirs(output_dir, exist_ok=True)

    # First, get video metadata
    result = subprocess.run(
        ["yt-dlp", "--print", "%(title)s\n%(artist)s", "--no-download", url],
        capture_output=True, text=True, check=True,
    )
    lines = result.stdout.strip().split('\n')
    raw_title = lines[0] if lines else "Unknown"
    artist = lines[1] if len(lines) > 1 and lines[1] != "NA" else ""

    # Try to parse "Artist - Title" from video title
    if not artist and " - " in raw_title:
        parts = raw_title.split(" - ", 1)
        artist = parts[0].strip()
        title = parts[1].strip()
    else:
        title = raw_title

    safe_name = sanitize_filename(raw_title)
    audio_path = os.path.join(output_dir, f"{safe_name}.wav")

    # Download as WAV at 44.1kHz mono (sam-audio expects this)
    subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 44100 -ac 1",
            "-o", audio_path,
            "--no-playlist",
            url,
        ],
        check=True,
    )

    return audio_path, title, artist
