import os
import subprocess
from pathlib import Path
import argparse

"""
~~~ How to use this script ~~~
Run from terminal:
# Convert a single file
python convert_ts_to_mp4.py /path/to/video.ts

# Convert all .ts files in a folder
python convert_ts_to_mp4.py /path/to/folder

# Specify output file or folder
python convert_ts_to_mp4.py /path/to/video.ts --out /output/final.mp4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Requirements:
 - ffmpeg
Windows:
choco install ffmpeg

Ubuntu:
sudo apt update && sudo apt install ffmpeg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def convert_ts_to_mp4(input_path, output_path=None):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if not output_path:
        output_path = input_path.with_suffix('.mp4')
    else:
        output_path = Path(output_path)

    cmd = [
        'ffmpeg',
        '-y',                        # overwrite output without asking
        '-fflags', '+genpts',       # fix timestamp issues in .ts
        '-i', str(input_path),
        '-vf', 'yadif=0:-1:0',       # deinterlace if needed
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '22',                # controls quality: 0=best, 51=worst
        '-movflags', '+faststart',  # allow progressive playback
        str(output_path)
    ]

    print(f"Converting {input_path.name} → {output_path.name} ...")
    subprocess.run(cmd, check=True)
    print("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help=".ts file or folder containing .ts files")
    parser.add_argument("--out", help="Optional output .mp4 filename or folder")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else None

    if in_path.is_file() and in_path.suffix == '.ts':
        convert_ts_to_mp4(in_path, out_path)
    elif in_path.is_dir():
        ts_files = list(in_path.glob("*.ts"))
        if not ts_files:
            print("No .ts files found in the folder.")
        else:
            for ts_file in ts_files:
                out_file = out_path / ts_file.with_suffix(".mp4").name if out_path else None
                convert_ts_to_mp4(ts_file, out_file)
    else:
        print("Please provide a valid .ts file or folder.")
