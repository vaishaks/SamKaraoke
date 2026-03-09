# AI Karaoke Machine

Turn any YouTube video into a karaoke experience using Facebook's [SAM-Audio](https://github.com/facebookresearch/sam-audio) model for vocal separation.

## How It Works

1. **Download** - Extracts audio from a YouTube video using `yt-dlp`
2. **Separate** - Uses SAM-Audio to isolate vocals from instrumentals via text-prompted source separation
3. **Lyrics** - Fetches synced lyrics from online sources (LRCLib, Musixmatch, etc.) or falls back to Whisper-based alignment/transcription
4. **Video** - Generates a karaoke MP4 video with timed lyrics over the instrumental track

## Prerequisites

- Python >= 3.11
- CUDA-compatible GPU (recommended for SAM-Audio)
- `ffmpeg` installed on your system
- HuggingFace account with access to `facebook/sam-audio-base` or `facebook/sam-audio-large`

## Installation

```bash
# Clone the repository
git clone https://github.com/vaishaks/SamKaraoke.git
cd SamKaraoke

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install .
```

## Usage

```bash
# Basic usage - provide a YouTube URL
sam-karaoke https://www.youtube.com/watch?v=VIDEO_ID

# Specify output file
sam-karaoke https://youtu.be/VIDEO_ID -o my_karaoke.mp4

# Use the large SAM-Audio model for better separation
sam-karaoke URL --model facebook/sam-audio-large

# Use a larger Whisper model for better lyrics alignment
sam-karaoke URL --whisper-model medium

# Skip vocal separation (e.g., for testing lyrics/video pipeline)
sam-karaoke URL --skip-separation
```

### As a Python module

```python
from karaoke.download import download_youtube_audio
from karaoke.separate import separate_vocals
from karaoke.lyrics import get_lyrics
from karaoke.video import generate_karaoke_video

# Step 1: Download
audio_path, title, artist = download_youtube_audio(url, "output/")

# Step 2: Separate vocals
vocals_path, instrumental_path = separate_vocals(audio_path, "output/")

# Step 3: Get lyrics
lyrics = get_lyrics(title, artist, vocals_path=vocals_path)

# Step 4: Generate video
generate_karaoke_video(instrumental_path, lyrics, "karaoke.mp4",
                       title=title, artist=artist)
```

## Pipeline Architecture

```
YouTube URL
    │
    ▼
┌─────────────┐
│  yt-dlp     │  Download & extract audio (WAV 44.1kHz)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  SAM-Audio  │  Text-prompted separation: "vocals, singing voice"
└──┬──────┬───┘
   │      │
   ▼      ▼
vocals  instrumental
   │      │
   ▼      │
┌─────────────┐     │
│ syncedlyrics │     │  Fetch synced LRC lyrics from online APIs
│   or         │     │  Fallback: Whisper alignment / transcription
│ stable-ts    │     │
└─────┬───────┘     │
      │              │
      ▼              ▼
┌──────────────────────┐
│  moviepy + Pillow    │  Render timed lyrics over instrumental
└──────────┬───────────┘
           │
           ▼
     karaoke.mp4
```

## Lyrics Resolution Strategy

The system tries three approaches in order:

1. **Synced lyrics** - Pre-timed LRC lyrics from online databases (best quality)
2. **Aligned lyrics** - Plain lyrics from the internet, aligned to the vocal track using Whisper forced alignment
3. **Transcription** - Full vocal transcription using Whisper as a last resort

## Models

| Component | Model | Purpose |
|-----------|-------|---------|
| Vocal Separation | `facebook/sam-audio-base` | Isolate vocals from music |
| Lyrics Alignment | `openai/whisper` (via stable-ts) | Align/transcribe lyrics |

## License

This project uses Facebook's SAM-Audio model which has its own license terms. Please refer to the [SAM-Audio repository](https://github.com/facebookresearch/sam-audio) for model licensing details.
