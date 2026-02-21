import subprocess
import sys
from pathlib import Path

def mp3_to_wav_16k_mono(input_file, output_file=None):
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file is None:
        output_file = input_path.with_suffix(".wav")

    command = [
        "ffmpeg",
        "-y",                 # Overwrite output file if exists
        "-i", str(input_path),
        "-ac", "1",           # Mono
        "-ar", "16000",       # 16 kHz
        "-vn",                # No video
        str(output_file)
    ]

    subprocess.run(command, check=True)
    print(f"Converted: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py input.mp3 [output.wav]")
        sys.exit(1)

    input_mp3 = sys.argv[1]
    output_wav = sys.argv[2] if len(sys.argv) > 2 else None

    mp3_to_wav_16k_mono(input_mp3, output_wav)

