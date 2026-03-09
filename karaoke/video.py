"""Generate karaoke video with timed lyrics overlaid on the instrumental track."""

import os
import textwrap

import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    TextClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFont


# Video settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 24
BG_COLOR = (10, 0, 30)  # Dark purple-ish background
TEXT_COLOR = "white"
HIGHLIGHT_COLOR = "cyan"
TITLE_COLOR = "yellow"
FONT_SIZE = 48
TITLE_FONT_SIZE = 36
MAX_LINE_CHARS = 40


def create_text_frame(
    text: str,
    width: int,
    height: int,
    color: str = TEXT_COLOR,
    font_size: int = FONT_SIZE,
    bg_color: tuple = BG_COLOR,
    y_offset: int = 0,
) -> np.ndarray:
    """Render text onto an image frame using Pillow."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Wrap long lines
    wrapped = textwrap.fill(text, width=MAX_LINE_CHARS)
    lines = wrapped.split('\n')

    # Calculate total text height
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])

    total_height = sum(line_heights) + (len(lines) - 1) * 10
    start_y = (height // 2) - (total_height // 2) + y_offset

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = start_y + sum(line_heights[:i]) + i * 10
        draw.text((x, y), line, fill=color, font=font)

    return np.array(img)


def make_lyric_clip(
    text: str,
    start_time: float,
    duration: float,
    is_active: bool = True,
) -> ColorClip:
    """Create a video clip for a single lyric line."""
    color = HIGHLIGHT_COLOR if is_active else TEXT_COLOR
    frame = create_text_frame(text, VIDEO_WIDTH, VIDEO_HEIGHT, color=color)

    clip = (
        ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=BG_COLOR)
        .with_duration(duration)
        .with_start(start_time)
    )

    # Override the frame with our rendered text
    from moviepy import ImageClip
    text_clip = (
        ImageClip(frame)
        .with_duration(duration)
        .with_start(start_time)
    )

    return text_clip


def generate_karaoke_video(
    instrumental_path: str,
    lyrics: list[dict],
    output_path: str,
    title: str = "",
    artist: str = "",
) -> str:
    """Generate a karaoke video with timed lyrics over the instrumental.

    Args:
        instrumental_path: Path to the instrumental (no vocals) WAV file.
        lyrics: List of {"time": float, "text": str} entries.
        output_path: Where to save the final MP4 video.
        title: Song title for the intro screen.
        artist: Artist name for the intro screen.

    Returns:
        Path to the generated video file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    audio = AudioFileClip(instrumental_path)
    total_duration = audio.duration

    clips = []

    # Create title card (first 5 seconds)
    title_text = title
    if artist:
        title_text = f"{artist}\n{title}"
    title_frame = create_text_frame(
        title_text, VIDEO_WIDTH, VIDEO_HEIGHT,
        color=TITLE_COLOR, font_size=TITLE_FONT_SIZE, y_offset=-80,
    )
    # Add "KARAOKE" subtitle
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(title_frame)
    draw = ImageDraw.Draw(img)
    try:
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28
        )
    except (OSError, IOError):
        small_font = ImageFont.load_default()
    karaoke_text = "KARAOKE"
    bbox = draw.textbbox((0, 0), karaoke_text, font=small_font)
    kw = bbox[2] - bbox[0]
    draw.text(
        ((VIDEO_WIDTH - kw) // 2, VIDEO_HEIGHT // 2 + 40),
        karaoke_text, fill="white", font=small_font,
    )
    title_frame = np.array(img)

    from moviepy import ImageClip
    title_clip = ImageClip(title_frame).with_duration(5.0).with_start(0)
    clips.append(title_clip)

    # Create lyric clips
    for i, entry in enumerate(lyrics):
        start = entry["time"]
        # Determine duration: until next lyric or +4 seconds
        if "end" in entry:
            duration = entry["end"] - entry["time"]
        elif i + 1 < len(lyrics):
            duration = lyrics[i + 1]["time"] - entry["time"]
        else:
            duration = min(4.0, total_duration - start)

        if duration <= 0:
            duration = 0.5

        # Build frame showing current line highlighted + next line dimmed
        current_text = entry["text"]
        next_text = lyrics[i + 1]["text"] if i + 1 < len(lyrics) else ""

        # Create a composite frame with current + upcoming lyrics
        frame = create_text_frame(
            current_text, VIDEO_WIDTH, VIDEO_HEIGHT,
            color=HIGHLIGHT_COLOR, y_offset=-30,
        )
        if next_text:
            # Overlay the next line below
            img = Image.fromarray(frame)
            draw_obj = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32
                )
            except (OSError, IOError):
                font = ImageFont.load_default()
            bbox = draw_obj.textbbox((0, 0), next_text, font=font)
            tw = bbox[2] - bbox[0]
            draw_obj.text(
                ((VIDEO_WIDTH - tw) // 2, VIDEO_HEIGHT // 2 + 60),
                next_text, fill=(150, 150, 150), font=font,
            )
            frame = np.array(img)

        lyric_clip = ImageClip(frame).with_duration(duration).with_start(start)
        clips.append(lyric_clip)

    # Background clip for the full duration
    bg_clip = ColorClip(
        size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=BG_COLOR
    ).with_duration(total_duration)

    # Composite everything
    video = CompositeVideoClip([bg_clip] + clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
    video = video.with_audio(audio)

    print(f"Rendering karaoke video to: {output_path}")
    video.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        logger="bar",
    )

    print(f"Karaoke video saved: {output_path}")
    return output_path
