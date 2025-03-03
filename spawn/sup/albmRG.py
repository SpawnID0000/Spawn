#!/usr/bin/env python3
import os
import sys
import subprocess
import re

def calculate_album_replaygain(folder_path):
    """
    Runs 'rsgain easy <folder_path>' to calculate ReplayGain values.
    It parses the output and returns the album gain and peak values.
    """
    cmd = ["rsgain", "easy", folder_path]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except FileNotFoundError:
        print("ERROR: rsgain not found in PATH. Please install it.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("ReplayGain calculation error:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    output = proc.stdout
    album_gain = None
    album_peak = None
    current_section = None  # Tracks which section we're in ("album" or something else)

    for line in output.splitlines():
        stripped_line = line.strip()
        lower_line = stripped_line.lower()

        # Look for the album section header.
        if lower_line.startswith("album:"):
            current_section = "album"
            continue

        # Only process lines after encountering the album section.
        if current_section == "album":
            if lower_line.startswith("gain:"):
                m = re.search(r"gain:\s*([-+]?\d+(?:\.\d+)?)\s*dB", stripped_line, re.IGNORECASE)
                if m:
                    album_gain = m.group(1) + " dB"
            elif lower_line.startswith("peak:"):
                m = re.search(r"peak:\s*([\d.]+)", stripped_line, re.IGNORECASE)
                if m:
                    album_peak = m.group(1)
    return album_gain, album_peak

def main():
    if len(sys.argv) != 2:
        print("Usage: python albmRG.py path/to/album")
        sys.exit(1)

    album_path = sys.argv[1]
    if not os.path.isdir(album_path):
        print(f"Error: '{album_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    album_gain, album_peak = calculate_album_replaygain(album_path)
    if album_gain is None or album_peak is None:
        print("Album ReplayGain values could not be determined.")
    else:
        print(f"Album Gain: {album_gain}")
        print(f"Album Peak: {album_peak}")

if __name__ == "__main__":
    main()
