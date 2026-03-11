# AI Karaoke Machine

Turn any YouTube video into a karaoke experience using AI-powered vocal separation and automatic lyrics sync.

## How It Works

1. **Download** - Extracts audio from a YouTube video using `yt-dlp`
2. **Separate** - Isolates vocals from instrumentals using [Demucs](https://github.com/facebookresearch/demucs) (or [SAM-Audio](https://github.com/facebookresearch/sam-audio) on Linux/CUDA)
3. **Lyrics** - Fetches synced lyrics from online sources (LRCLib, Musixmatch, etc.) or falls back to Whisper-based alignment/transcription
4. **Video** - Generates a karaoke MP4 video with timed lyrics over the instrumental track

## Prerequisites

- Python >= 3.11 (3.13 recommended; 3.14 is **not** supported due to missing dependency wheels)
- `ffmpeg` installed on your system
- ~1 GB disk space for model weights (downloaded on first run)

### Installing ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

## Installation

```bash
# Clone the repository
git clone https://github.com/vaishaks/SamKaraoke.git
cd SamKaraoke

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install .

# Install a vocal separation backend (at least one is required)
pip install demucs torchcodec   # Works on macOS and Linux
```

### Vocal Separation Backends

The tool supports two vocal separation backends. You need at least one installed:

| Backend | Install | Platform | Notes |
|---------|---------|----------|-------|
| **Demucs** (default fallback) | `pip install demucs torchcodec` | macOS, Linux, Windows | Meta's Hybrid Transformer model. Works everywhere. |
| **SAM-Audio** | `pip install -r requirements.txt` | Linux + CUDA GPU | Facebook's text-prompted separator. Requires `xformers` and CUDA. |

On **macOS** (Apple Silicon or Intel), use Demucs -- SAM-Audio's dependencies (`xformers`, `decord`) do not build on macOS.

On **Linux with CUDA**, you can install SAM-Audio for potentially better separation quality:
```bash
pip install -r requirements.txt
```
When both are installed, SAM-Audio is used first with Demucs as a fallback.

## Usage

```bash
# Basic usage - provide a YouTube URL
sam-karaoke https://www.youtube.com/watch?v=VIDEO_ID

# Specify output file
sam-karaoke https://youtu.be/VIDEO_ID -o my_karaoke.mp4

# Use a custom working directory for intermediate files
sam-karaoke URL --work-dir ./tmp

# Use the large SAM-Audio model (Linux/CUDA only)
sam-karaoke URL --model facebook/sam-audio-large

# Use a larger Whisper model for better lyrics alignment
sam-karaoke URL --whisper-model medium

# Skip vocal separation (uses original audio as-is)
sam-karaoke URL --skip-separation
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `url` | (required) | YouTube video URL |
| `-o`, `--output` | `output/<title>_karaoke.mp4` | Output video file path |
| `--work-dir` | `output` | Directory for intermediate files |
| `--model` | `facebook/sam-audio-base` | SAM-Audio model (ignored when using Demucs) |
| `--whisper-model` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `--skip-separation` | off | Skip vocal separation entirely |

### As a Python Module

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
    |
    v
+-------------+
|   yt-dlp    |  Download & extract audio (WAV 44.1kHz mono)
+------+------+
       |
       v
+-------------+
|   Demucs    |  Vocal/instrumental separation (htdemucs model)
|  or SAM-    |  SAM-Audio uses text prompt: "vocals, singing voice"
|   Audio     |
+--+------+---+
   |      |
   v      v
vocals  instrumental
   |      |
   v      |
+-------------+    |
| syncedlyrics |    |  Fetch synced LRC lyrics from online APIs
|   or         |    |  Fallback: Whisper alignment / transcription
| stable-ts    |    |
+------+------+    |
       |           |
       v           v
+---------------------+
|  moviepy + Pillow   |  Render timed lyrics over instrumental
+---------+-----------+
          |
          v
    karaoke.mp4
```

## Lyrics Resolution Strategy

The system tries three approaches in order:

1. **Synced lyrics** - Pre-timed LRC lyrics from online databases (best quality)
2. **Aligned lyrics** - Plain lyrics from the internet, aligned to the vocal track using Whisper forced alignment
3. **Transcription** - Full vocal transcription using Whisper as a last resort

## Dependencies

Core Python packages (installed automatically with `pip install .`):

| Package | Purpose |
|---------|---------|
| `yt-dlp` | YouTube audio download |
| `torch` + `torchaudio` | ML runtime, audio I/O |
| `syncedlyrics` | Fetching synced lyrics from online databases |
| `stable-ts` + `openai-whisper` | Lyrics alignment and transcription |
| `moviepy` + `Pillow` | Video rendering with text overlays |

Additional packages (install separately):

| Package | Purpose |
|---------|---------|
| `demucs` + `torchcodec` | Vocal separation (macOS/Linux/Windows) |
| `sam-audio` | Vocal separation (Linux + CUDA only) |

## Models

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Vocal Separation | Demucs `htdemucs` | ~80 MB | Isolate vocals from music |
| Vocal Separation | `facebook/sam-audio-base` | ~1 GB | Text-prompted source separation (CUDA only) |
| Lyrics Alignment | `openai/whisper` (via stable-ts) | 75-1550 MB | Align/transcribe lyrics |

Model weights are downloaded automatically on first run to `~/.cache/torch/hub/checkpoints/`.

## Troubleshooting

**`ffmpeg not found`** - Install ffmpeg (see [Prerequisites](#prerequisites)).

**`No supported JavaScript runtime`** - yt-dlp warns about missing JS runtime but still works. Install `deno` to suppress: `brew install deno` / `apt install deno`.

**`torchcodec` / `TorchCodec is required`** - Run `pip install torchcodec`. Required by newer versions of torchaudio for saving audio files.

**SAM-Audio import errors (`xformers`, `decord`)** - SAM-Audio doesn't work on macOS. Install Demucs instead: `pip install demucs torchcodec`.

**Python 3.14 compatibility** - Several dependencies (`decord`, some wheel builds) don't support Python 3.14 yet. Use Python 3.11-3.13.

## License

This project uses Facebook's [Demucs](https://github.com/facebookresearch/demucs) (MIT License) and optionally [SAM-Audio](https://github.com/facebookresearch/sam-audio). Please refer to their repositories for model licensing details.
