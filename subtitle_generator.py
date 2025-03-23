import os
import sys
import argparse
from moviepy import VideoFileClip
import tempfile
from datetime import timedelta
import whisper
import torch
from pydub import AudioSegment
from pydub.silence import detect_silence


def detect_silence_in_audio(audio_filename, target_duration=300, silence_duration=1000, silence_threshold=-35) -> list:
    """
    Detect silent periods in audio file to use as natural splitting points.

    :param audio_filename: Path to the audio file
    :param target_duration: Target duration in seconds for each split
    :param silence_duration: Duration of silence in milliseconds to detect
    :param silence_threshold: Silence threshold in dB
    :return: List of split points in milliseconds
    """
    print(f"Detecting silence in {audio_filename}")
    audio = AudioSegment.from_file(audio_filename)
    silence_ranges = detect_silence(audio, min_silence_len=silence_duration, silence_thresh=silence_threshold)

    if not silence_ranges:
        print("No silence detected, using fixed duration")
        # Fall back to fixed duration if no silence is detected
        duration_ms = len(audio)
        return [i for i in range(0, duration_ms, target_duration * 1000)]

    # Convert target duration to milliseconds
    target_duration_ms = target_duration * 1000

    split_points = [0]
    last_position = 0

    for start, end in silence_ranges:
        # Use the middle of the silence range as the split point
        silence_middle = (start + end) / 2

        # If we've moved at least the target duration away from the last split point
        if silence_middle - last_position >= target_duration_ms * 0.8:
            split_points.append(silence_middle)
            last_position = silence_middle

    # Add the end point if it's not close to the last split point
    total_duration = len(audio)  # in milliseconds
    if total_duration - split_points[
        -1] > target_duration * 0.2 * 1000:  # If the last split point is not close to the end
        split_points.append(total_duration)

    print(f"Found {len(split_points)} split points")
    return split_points


def split_video_at_silence(filename, directory, duration=300):
    """
    Split video at detected silence points.

    :param filename: Path to the video file
    :param directory: Directory to save the split clips (temporary)
    :param duration: Target duration in seconds for each clip
    """
    full_video = VideoFileClip(filename)

    # Extract audio from the video
    temp_audio = os.path.join(directory, "temp_full_audio.wav")
    full_video.audio.write_audiofile(temp_audio)

    # Detect silence in the audio
    split_points = detect_silence_in_audio(temp_audio, target_duration=duration)

    for i in range(len(split_points) - 1):
        start_time = split_points[i] / 1000 if i != 0 else 0
        end_time = split_points[i + 1] / 1000

        print(f"Splitting {start_time:.2f}s to {end_time:.2f}s")
        clip_filename = os.path.join(directory, f"clip_{i}.mp3")
        clip = full_video.subclipped(start_time, min(end_time, full_video.duration))
        clip.audio.write_audiofile(clip_filename)

        print(f"Created clip {i}: {start_time:.2f}s to {end_time:.2f}s (duration: {end_time - start_time:.2f}s)")

    # Clean up temp audio file
    os.remove(temp_audio)

    return split_points


def process_segments_with_contextual_offset(segments, split_points, chunk_index, start_index=0) -> tuple[str, int]:
    """
    Process subtitle segments with proper time offsets.
    This function applies the time offset of the current chunk to the start and end times of each segment.

    :param segments: List of subtitle segments
    :param split_points: List of split points in milliseconds
    :param chunk_index: Index of the current chunk, used for determining the start time offset
    :param start_index: Index of the previous segment, used for incrementing IDs in srt file
    :return: Formatted subtitles and the number of segments processed
    """
    offset_seconds = split_points[chunk_index] / 1000  # Use the actual start time of this chunk
    formatted_subs = ""
    print(f"Processing segment with offset {offset_seconds:.2f}s")

    for segment in segments:
        # Apply offset to the start and end times
        start_time = segment['start'] + offset_seconds
        end_time = segment['end'] + offset_seconds

        # Convert seconds to timedelta, preserving milliseconds
        start_timedelta = timedelta(seconds=start_time)
        end_timedelta = timedelta(seconds=end_time)

        # Format with milliseconds (format: 00:00:00,000)
        # Extract hours, minutes, seconds and milliseconds
        start_hours, remainder = divmod(start_timedelta.total_seconds(), 3600)
        start_minutes, start_seconds = divmod(remainder, 60)
        start_milliseconds = int((start_seconds - int(start_seconds)) * 1000)
        start_seconds = int(start_seconds)

        end_hours, remainder = divmod(end_timedelta.total_seconds(), 3600)
        end_minutes, end_seconds = divmod(remainder, 60)
        end_milliseconds = int((end_seconds - int(end_seconds)) * 1000)
        end_seconds = int(end_seconds)

        # Format timestamps as required for SRT (00:00:00,000)
        start_time = f"{int(start_hours):02d}:{int(start_minutes):02d}:{start_seconds:02d},{start_milliseconds:03d}"
        end_time = f"{int(end_hours):02d}:{int(end_minutes):02d}:{end_seconds:02d},{end_milliseconds:03d}"

        text = segment['text']
        segment_id = segment['id'] + 1 + start_index

        formatted_segment = f"{segment_id}\n{start_time} --> {end_time}\n{text[1:] if text[0] == ' ' else text}\n\n"
        formatted_subs += formatted_segment

    return formatted_subs, len(segments)


def generate_subtitles(video_filename, chunk_duration=300, output_file=None, device="cuda", language="ja") -> str:
    """
    Generate subtitles for a given video file.

    :param video_filename: Path to the video file
    :param chunk_duration: Target duration in seconds for each chunk
    :param output_file: Output subtitle file name
    :param device: Device to run the model on (cuda or cpu), detected automatically
    :param language: Source language code for transcription
    :return: Path to the output subtitle file
    """
    # Set CUDA allocation config
    if device == "cuda":
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    video_shortname = os.path.splitext(os.path.basename(video_filename))[0]

    # Determine output filename
    if output_file is None:
        output_file = f"{video_shortname}.srt"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Split the video at natural silence points
        split_points = split_video_at_silence(video_filename, temp_dir, chunk_duration)

        # Load the model
        print("Loading Whisper model...")
        if device == "cuda":
            print("Running with CUDA")
            model = whisper.load_model("large-v3", device=device)
            print(f"Model loaded on {device}")
        elif device == "cpu":
            print("Running with CPU")
            model = whisper.load_model("large-v3", device="cpu")
            print("Model loaded on CPU")
        else:
            raise ValueError(f"Invalid device: {device}")


        # Generate subtitles
        options = dict(language=language, beam_size=5, best_of=5)
        translate_options = dict(task="translate", **options)
        files_to_translate = sorted(
            [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".mp3")],
            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        )

        translated_subs = ""
        previous_index = 0
        for i, file in enumerate(files_to_translate):
            print(f"Translating chunk {i + 1} of {len(files_to_translate)}")
            result = model.transcribe(file, **translate_options)

            if device == "cuda":
                torch.cuda.empty_cache()  # Clear GPU memory

            subtitle_segments = result['segments']

            # Process subtitle segments with offset
            formatted_subs, segment_count = process_segments_with_contextual_offset(
                subtitle_segments, split_points, i, previous_index
            )
            previous_index += segment_count
            translated_subs += formatted_subs

        print(f"Saving subtitles to {output_file}")

        # Delete model to free up memory
        del model

        # Save the subtitles to a file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_subs)
            print(f"Subtitles saved to {output_file}")

    return output_file


def main():
    """
    Main entry point for the script.

    :return: 0 if successful, 1 if an error occurred
    """
    parser = argparse.ArgumentParser(description="Generate subtitles for a video file using Whisper")
    parser.add_argument("video_file", help="Path to the video file")
    parser.add_argument("--chunk-duration", type=int, default=300,
                        help="Target duration in seconds for each chunk (default: 300)")
    parser.add_argument("--device", default="cuda", help="Device to run the model on (cuda or cpu)")
    parser.add_argument("--output", "-o", help="Output subtitle file (default: same as video name with .srt extension)")
    parser.add_argument("--language", default="ja",
                        help="Language code for transcription (default: ja for Japanese)")

    args = parser.parse_args()

    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA not available, using CPU")
        args.device = "cpu"

    # Validate that video file exists
    if not os.path.isfile(args.video_file):
        print(f"Error: Video file '{args.video_file}' not found")
        return 1

    try:
        output_file = generate_subtitles(
            args.video_file,
            chunk_duration=args.chunk_duration,
            output_file=args.output,
            device=args.device,
            language=args.language
        )
        print(f"Successfully generated subtitles at: {output_file}")
        return 0
    except Exception as e:
        print(f"Error generating subtitles: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())