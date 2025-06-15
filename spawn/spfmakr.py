#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

from mutagen.mp4 import MP4


def sanitize_filename(name: str) -> str:
    """Replace characters invalid in filenames with underscores."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def extract_track_info(file_path: Path):
    audio = MP4(str(file_path))
    tags = audio.tags or {}

    title = tags.get('\xa9nam', [""])[0]
    creator = tags.get('\xa9ART', [""])[0]
    album = tags.get('\xa9alb', [""])[0]
    tracknum = tags.get('trkn', [(0, 0)])[0][0]
    album_artist = tags.get('aART', [""])[0]

    raw_ids = tags.get('----:com.apple.iTunes:spawn_ID', [])
    if not raw_ids:
        raise ValueError(f"No spawn_ID tag found in {file_path}")
    raw_id = raw_ids[0]
    spawn_id = raw_id.decode('utf-8', errors='ignore') if isinstance(raw_id, bytes) else str(raw_id)

    duration_ms = int(audio.info.length * 1000)

    location = [f"../linx/{spawn_id}", f"../../aux/user/linx/{spawn_id}"]
    identifier = [f"urn:spawnid:{spawn_id}"]
    track_obj = {
        "location": location,
        "identifier": identifier,
        "title": title,
        "creator": creator,
        "annotation": "",
        "info": "",
        "image": "",
        "album": album,
        "trackNum": tracknum,
        "duration": duration_ms,
        "link": [{"urn:ardrive:data_tx_id": "https://arweave.net/insert_data_Tx_ID_here"}],
        "meta": [{"urn:ardrive:metadata_tx_id": "https://arweave.net/insert_metadata_Tx_ID_here"}],
        "extension": {
            "urn:spawnSPF:contract": ["https://arweave.net/tx/insert_contract_Tx_ID_here"],
            "urn:liquidsoap:cue_fade": [{"liq_cue_in": 0}, {"liq_fade_in": 0}, {"liq_fade_out": 0}]}
    }
    return tracknum, track_obj, album_artist


def main():
    parser = argparse.ArgumentParser(
        description="Generate a JSPF playlist from a directory of .m4a files."
    )
    parser.add_argument(
        "--type", required=True,
        choices=["s", "ep", "lp", "dlp", "pl"],
        help="urn:spawnSPF:type: s, ep, lp, dlp, or pl"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing .m4a files"
    )
    parser.add_argument(
        "--title", help="Playlist title (required if --type pl)"
    )
    parser.add_argument(
        "--creator", help="Playlist creator (optional; overrides for pl)"
    )
    parser.add_argument(
        "--output-dir", help="Directory to write the JSPF file (default: current working directory)"
    )
    args = parser.parse_args()

    if args.type == "pl" and not args.title:
        parser.error("--title is required when --type is 'pl'")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    raw = []
    for f in sorted(input_dir.rglob("*.m4a")):
        try:
            raw.append(extract_track_info(f))
        except Exception as e:
            print(f"Warning: {e}", file=sys.stderr)

    if not raw:
        print("Error: No valid .m4a files found.", file=sys.stderr)
        sys.exit(1)

    raw.sort(key=lambda x: x[0])
    tracks = [t[1] for t in raw]

    if args.type == "pl":
        title = args.title
        creator = args.creator or ""
    else:
        title = tracks[0]["album"]
        creator = raw[0][2]

    total_duration = sum(t["duration"] for t in tracks)
    total_tracks = len(tracks)

    playlist = {
        "title": title,
        "creator": creator,
        "annotation": "",
        "info": "",
        "location": "",
        "identifier": "",
        "image": "",
        "date": "",
        "license": "",
        "attribution": [{"identifier": ""}, {"location": ""}],
        "link": [{"urn:ardrive:data_tx_id": ""}],
        "meta": [{"urn:ardrive:metadata_tx_id": ""}],
        "extension": {
            "urn:spawnSPF:type": [args.type],
            "urn:spawnSPF:total_duration": [total_duration],
            "urn:spawnSPF:total_tracks": [total_tracks]
        },
        "track": tracks
    }

    json_str = json.dumps(
        {"playlist": playlist},
        indent=2,
        ensure_ascii=False,
        separators=(",", ": ")
    )

    # INLINE post-processing
    subs = [
        # attribution
        (
            r'"attribution":\s*\[\s*\{\s*"identifier":\s*""\s*\},\s*\{\s*"location":\s*""\s*\}\s*\]',
            r'"attribution": [{"identifier": ""}, {"location": ""}]'
        ),

        # playlist-level link
        (
            r'\[\s*\{\s*"urn:ardrive:data_tx_id":\s*"([^"]*)"\s*\}\s*\]',
            r'[{"urn:ardrive:data_tx_id": "\1"}]'
        ),
        # playlist-level meta
        (
            r'\[\s*\{\s*"urn:ardrive:metadata_tx_id":\s*"([^"]*)"\s*\}\s*\]',
            r'[{"urn:ardrive:metadata_tx_id": "\1"}]'
        ),

        # location & identifier
        (
            r'"location":\s*\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]',
            r'"location": ["\1", "\2"]'
        ),
        (
            r'"identifier":\s*\[\s*"([^"]+)"\s*\]',
            r'"identifier": ["\1"]'
        ),

        # extension arrays
        (
            r'"urn:spawnSPF:type":\s*\[\s*"([^"]+)"\s*\]',
            r'"urn:spawnSPF:type": ["\1"]'
        ),
        (
            r'"urn:spawnSPF:total_duration":\s*\[\s*(\d+)\s*\]',
            r'"urn:spawnSPF:total_duration": [\1]'
        ),
        (
            r'"urn:spawnSPF:total_tracks":\s*\[\s*(\d+)\s*\]',
            r'"urn:spawnSPF:total_tracks": [\1]'
        ),

        # contract
        (
            r'"urn:spawnSPF:contract":\s*\[\s*"([^"]*)"\s*\]',
            r'"urn:spawnSPF:contract": ["\1"]'
        ),

        # liquidsoap cue_fade
        (
            r'\[\s*\{\s*"liq_cue_in":\s*0\s*\},\s*\{\s*"liq_fade_in":\s*0\s*\},\s*\{\s*"liq_fade_out":\s*0\s*\}\s*\]',
            r'[{"liq_cue_in": 0}, {"liq_fade_in": 0}, {"liq_fade_out": 0}]'
        ),
    ]

    for pat, repl in subs:
        json_str = re.sub(pat, repl, json_str, flags=re.DOTALL)

    safe_t = sanitize_filename(title)
    safe_c = sanitize_filename(creator)
    filename = f"{(safe_c + ' - ') if safe_c else ''}{safe_t}.jspf"

    # Determine output path
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.exists():
            print(f"Error: output directory {out_dir} does not exist.", file=sys.stderr)
            sys.exit(1)
        out_path = out_dir / filename
    else:
        out_path = Path(filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    print(f"Wrote JSPF to {out_path}")


if __name__ == "__main__":
    main()