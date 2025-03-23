# Long-form Video Subtitle Generator

A command-line tool that automatically generates subtitles for long-form videos by splitting them at natural silence points and using OpenAI's Whisper model for transcription and translation.

## Features

- Detects natural silence in videos to create optimal splitting points, ensuring that subtitles remain accurate for long-form content
- Uses OpenAI's Whisper model for high-quality transcription and translation
- Supports multiple languages
- Generates properly formatted SRT subtitle files
- GPU acceleration with fallback to CPU

## Installation

### Prerequisites

- Python 3.8 or higher
- A CUDA-compatible GPU (optional, but recommended for faster processing)
- FFmpeg (required for video processing)
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/pschua/generate-longform-video-subtitles.git
   cd video-subtitle-generator
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```bash
python subtitle_generator.py video_file.mp4
```

Advanced options:
```bash
python subtitle_generator.py video_file.mp4 --chunk-duration 100 --output custom_name.srt --language ja
```

### Options

- `video_file`: Path to the video file (required)
- `--chunk-duration`: Target duration in seconds for each chunk (default: 300)
- `--output`, `-o`: Custom output filename for the subtitle file
- `--device`: Device to use for transcription (default: `cuda` if available, otherwise `cpu`)
- `--language`: Language code for transcription (default: `ja` for Japanese) 
  - [language codes](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L38)

## Examples

Generate English subtitles using CPU:
```bash
python subtitle_generator.py my_video.mp4 --language en --device cpu
```

Generate subtitles from Japanese video with 10-minute chunks:
```bash
python subtitle_generator.py my_video.mp4 --chunk-duration 600 --language ja
```

![Japanese Example 1](jap-preview.gif)

![Japanese Example 2](jap-preview-2.gif)

![Korean Example](kor-preview.gif)

## License

MIT License - See LICENSE file for details.