"""Fetch and align lyrics for karaoke timing."""

import re

import syncedlyrics
import stable_whisper


def fetch_synced_lyrics(title: str, artist: str) -> list[dict] | None:
    """Fetch time-synced lyrics from online sources.

    Returns:
        List of {"time": float_seconds, "text": str} or None if unavailable.
    """
    query = f"{artist} {title}".strip() if artist else title
    print(f"Searching for synced lyrics: '{query}'...")

    lrc_text = syncedlyrics.search(query, synced_only=True)
    if not lrc_text:
        print("No synced lyrics found online.")
        return None

    return parse_lrc(lrc_text)


def fetch_plain_lyrics(title: str, artist: str) -> str | None:
    """Fetch plain (unsynced) lyrics from online sources."""
    query = f"{artist} {title}".strip() if artist else title
    print(f"Searching for plain lyrics: '{query}'...")

    lyrics = syncedlyrics.search(query, synced_only=False)
    if not lyrics:
        print("No lyrics found online.")
        return None

    # Strip any LRC timestamps if present
    lines = []
    for line in lyrics.split('\n'):
        cleaned = re.sub(r'\[\d{2}:\d{2}[.\d]*\]', '', line).strip()
        if cleaned:
            lines.append(cleaned)

    return '\n'.join(lines)


def parse_lrc(lrc_text: str) -> list[dict]:
    """Parse LRC format into a list of timed lyrics entries."""
    entries = []
    pattern = re.compile(r'\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)')

    for line in lrc_text.split('\n'):
        match = pattern.match(line.strip())
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            centis = match.group(3)
            # Handle both 2-digit (centiseconds) and 3-digit (milliseconds)
            if len(centis) == 2:
                frac = int(centis) / 100.0
            else:
                frac = int(centis) / 1000.0
            time_sec = minutes * 60 + seconds + frac
            text = match.group(4).strip()
            if text:
                entries.append({"time": time_sec, "text": text})

    return entries if entries else []


def align_lyrics_with_audio(
    vocals_path: str,
    plain_lyrics: str,
    model_size: str = "base",
) -> list[dict]:
    """Use Whisper (stable-ts) to align plain lyrics with the vocal track.

    Args:
        vocals_path: Path to the isolated vocals WAV file.
        plain_lyrics: The full plain-text lyrics.
        model_size: Whisper model size to use for alignment.

    Returns:
        List of {"time": float_seconds, "text": str}.
    """
    print(f"Aligning lyrics with audio using Whisper ({model_size})...")

    model = stable_whisper.load_model(model_size)
    result = model.align(vocals_path, plain_lyrics, language="en")

    entries = []
    for segment in result.segments:
        entries.append({
            "time": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })

    print(f"Aligned {len(entries)} lyric segments.")
    return entries


def get_lyrics(
    title: str,
    artist: str,
    vocals_path: str | None = None,
    whisper_model: str = "base",
) -> list[dict]:
    """Get timed lyrics, trying synced sources first, then Whisper alignment.

    Returns:
        List of {"time": float, "text": str} entries sorted by time.
    """
    # Strategy 1: Try to get pre-synced lyrics from the internet
    synced = fetch_synced_lyrics(title, artist)
    if synced:
        print(f"Found {len(synced)} synced lyrics lines.")
        return synced

    # Strategy 2: Get plain lyrics and align with Whisper
    plain = fetch_plain_lyrics(title, artist)
    if plain and vocals_path:
        print("Using Whisper to align plain lyrics with audio...")
        return align_lyrics_with_audio(vocals_path, plain, whisper_model)

    # Strategy 3: Pure transcription from vocals as last resort
    if vocals_path:
        print("No lyrics found. Transcribing vocals with Whisper...")
        model = stable_whisper.load_model(whisper_model)
        result = model.transcribe(vocals_path)
        entries = []
        for segment in result.segments:
            entries.append({
                "time": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })
        return entries

    return []
