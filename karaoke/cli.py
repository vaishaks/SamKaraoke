"""CLI entry point for the AI Karaoke Machine."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="AI Karaoke Machine - Turn any YouTube video into karaoke!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ
  %(prog)s https://youtu.be/dQw4w9WgXcQ -o my_karaoke.mp4
  %(prog)s URL --model facebook/sam-audio-large --whisper-model medium
        """,
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output",
        help="Output video file path (default: output/<title>_karaoke.mp4)",
    )
    parser.add_argument(
        "--work-dir",
        default="output",
        help="Working directory for intermediate files (default: output)",
    )
    parser.add_argument(
        "--model",
        default="facebook/sam-audio-base",
        help="SAM-Audio model to use (default: facebook/sam-audio-base)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size for lyrics alignment fallback (default: base)",
    )
    parser.add_argument(
        "--skip-separation",
        action="store_true",
        help="Skip vocal separation (use original audio as instrumental)",
    )

    args = parser.parse_args()

    # Import here to allow --help without all dependencies loaded
    from karaoke.download import download_youtube_audio, sanitize_filename
    from karaoke.lyrics import get_lyrics
    from karaoke.separate import separate_vocals
    from karaoke.video import generate_karaoke_video

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Download audio from YouTube
    print("=" * 60)
    print("STEP 1: Downloading audio from YouTube")
    print("=" * 60)
    audio_path, title, artist = download_youtube_audio(args.url, work_dir)
    print(f"  Title:  {title}")
    print(f"  Artist: {artist or '(unknown)'}")
    print(f"  Audio:  {audio_path}")

    # Step 2: Separate vocals from instrumental
    vocals_path = None
    if args.skip_separation:
        print("\n" + "=" * 60)
        print("STEP 2: Skipping vocal separation (--skip-separation)")
        print("=" * 60)
        instrumental_path = audio_path
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Separating vocals using SAM-Audio")
        print("=" * 60)
        vocals_path, instrumental_path = separate_vocals(
            audio_path, work_dir, model_name=args.model,
        )

    # Step 3: Fetch and align lyrics
    print("\n" + "=" * 60)
    print("STEP 3: Fetching and aligning lyrics")
    print("=" * 60)
    lyrics = get_lyrics(
        title, artist,
        vocals_path=vocals_path or audio_path,
        whisper_model=args.whisper_model,
    )

    if not lyrics:
        print("WARNING: No lyrics could be found or generated.")
        print("The karaoke video will have no lyrics overlay.")
        lyrics = []

    print(f"  Got {len(lyrics)} lyric segments.")

    # Step 4: Generate karaoke video
    print("\n" + "=" * 60)
    print("STEP 4: Generating karaoke video")
    print("=" * 60)
    if args.output:
        output_path = args.output
    else:
        safe_name = sanitize_filename(f"{artist} - {title}" if artist else title)
        output_path = os.path.join(work_dir, f"{safe_name}_karaoke.mp4")

    video_path = generate_karaoke_video(
        instrumental_path, lyrics, output_path,
        title=title, artist=artist,
    )

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Karaoke video: {video_path}")
    print("=" * 60)

    return video_path


if __name__ == "__main__":
    main()
