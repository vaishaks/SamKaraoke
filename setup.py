from setuptools import setup, find_packages

setup(
    name="sam-karaoke",
    version="0.1.0",
    description="AI Karaoke Machine - Turn any YouTube video into karaoke using SAM-Audio",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "yt-dlp>=2024.1.0",
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "syncedlyrics>=0.10.0",
        "stable-ts>=2.16.0",
        "moviepy>=2.0.0",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "sam-karaoke=karaoke.cli:main",
        ],
    },
)
