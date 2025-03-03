#!/usr/bin/env python3
"""
albmaker.py

Usage:
    python albmaker.py path/to/Spawn_library [-abs]

Description:
    This script scans the Spawn library’s Music directory (organized as:
      /Spawn/Music/Artist/Album/filename.m4a)
    and for each album, it reads the first track’s metadata (using the "trkn" tag)
    to determine the expected total number of tracks. If the album folder contains
    exactly that many tracks, the script generates an M3U playlist for the album in
    /Spawn/aux/user/albm/ named "Artist - Album.m3u".

    For each track, it uses the symlink created for the corresponding Spawn ID,
    where the symlink filename is "<spawn_id>.m4a" located under /Spawn/aux/user/linx/.
    Unless the optional "-abs" flag is provided (which causes the absolute file path
    to be used), the playlist will contain the relative symlink path.

    Each M3U file includes metadata comment lines:
      - #EXTART:<album artist>
      - #EXTALB:<album name>
      - Each track is preceded by a #EXTINF line that includes track duration and title.

    If a required symlink is missing or broken, the script pauses execution with a
    prompt warning the user.
"""

import os
import sys
import argparse
import logging
from mutagen import File as MutagenFile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s\n",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def extract_spawn_id(file_path: str) -> str:
    """
    Extract the spawn ID from the given audio file.
    This function first attempts to retrieve the tag "----:com.apple.iTunes:spawn_ID",
    and if not found, it will try "spawn_ID". Returns the spawn ID as a string,
    or an empty string if not found.
    """
    try:
        audio = MutagenFile(file_path)
        if audio is None or not audio.tags:
            return ""
        tag = audio.tags.get("----:com.apple.iTunes:spawn_ID")
        if not tag:
            tag = audio.tags.get("spawn_ID")
        if not tag:
            return ""
        if isinstance(tag, list):
            tag = tag[0]
        if isinstance(tag, bytes):
            tag = tag.decode("utf-8", errors="replace")
        return str(tag).strip()
    except Exception as e:
        logger.error("Error reading Spawn ID from %s: %s", file_path, e)
        return ""

def extract_album_info(file_path: str):
    """
    Extract album metadata from the given audio file.
    This function attempts to use the standard "trkn" tag (common in MP4/M4A files).
    The "trkn" tag is usually a list of tuples: [(track_number, total_tracks)].
    Optionally, it also attempts to read disc information from the "disk" tag.
    
    Returns a tuple (total_tracks, total_discs) if successful; otherwise, returns None.
    Note: For multi-disc albums, this function returns the info for the first track only.
    """
    try:
        audio = MutagenFile(file_path)
        if audio is None or not audio.tags:
            return None

        # Attempt to extract track info using the "trkn" tag
        trkn = audio.tags.get("trkn")
        if trkn:
            # Expecting something like: [(1, 13)]
            track_info = trkn[0]
            if isinstance(track_info, tuple) and len(track_info) >= 2:
                total_tracks = track_info[1]
            else:
                total_tracks = int(track_info)
        else:
            return None

        # Optionally, extract disc information using the "disk" tag.
        disk = audio.tags.get("disk")
        if disk:
            disc_info = disk[0]
            if isinstance(disc_info, tuple) and len(disc_info) >= 2:
                total_discs = disc_info[1]
            else:
                total_discs = int(disc_info)
        else:
            total_discs = 1

        return (total_tracks, total_discs)
    except Exception as e:
        logger.error("Error extracting album info from %s: %s", file_path, e)
        return None

def extract_track_info(file_path: str):
    """
    Extract track duration and title from the given audio file.
    Returns a tuple (duration, title) where duration is an integer number of seconds
    and title is a string. If extraction fails, returns (None, None).
    """
    try:
        audio = MutagenFile(file_path)
        if audio is None:
            return None, None
        duration = None
        title = ""
        # Get duration if available
        if hasattr(audio, 'info') and hasattr(audio.info, 'length'):
            duration = int(round(audio.info.length))
        # Try common tag keys for title (e.g. "©nam" for MP4 files)
        if audio.tags:
            if "©nam" in audio.tags:
                title = audio.tags["©nam"]
            elif "title" in audio.tags:
                title = audio.tags["title"]
            if isinstance(title, list):
                title = title[0]
            if isinstance(title, bytes):
                title = title.decode("utf-8", errors="replace")
            title = str(title).strip()
        return duration, title
    except Exception as e:
        logger.error("Error extracting track info from %s: %s", file_path, e)
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Generate M3U playlists for complete albums in the Spawn library."
    )
    parser.add_argument("spawn_path", help="Path to the Spawn library (e.g. /Volumes/Untitled/Spawn)")
    parser.add_argument("-abs", action="store_true", help="Use absolute file paths instead of relative symlink paths.")
    args = parser.parse_args()

    # Resolve key paths
    spawn_root = os.path.abspath(args.spawn_path)
    music_dir = os.path.join(spawn_root, "Music")
    user_aux_dir = os.path.join(spawn_root, "aux", "user")
    alb_dir = os.path.join(user_aux_dir, "albm")
    linx_dir = os.path.join(user_aux_dir, "linx")

    # Validate directories
    if not os.path.isdir(spawn_root):
        logger.error("Spawn library not found at %s", spawn_root)
        sys.exit(1)
    if not os.path.isdir(music_dir):
        logger.error("Music directory not found at %s", music_dir)
        sys.exit(1)
    if not os.path.isdir(user_aux_dir):
        logger.error("User aux directory not found at %s", user_aux_dir)
        sys.exit(1)
    if not os.path.isdir(linx_dir):
        logger.error("Linx directory not found at %s", linx_dir)
        sys.exit(1)

    # Create the 'albm' directory if it does not exist
    os.makedirs(alb_dir, exist_ok=True)
    logger.info("Using Spawn library at: %s", spawn_root)
    logger.info("Searching Music directory: %s", music_dir)
    logger.info("Playlists will be saved to: %s", alb_dir)

    # Lists to accumulate album logs
    extra_albums = []
    missing_albums = []

    # Iterate through artists (subdirectories of Music)
    for artist in os.listdir(music_dir):
        artist_dir = os.path.join(music_dir, artist)
        if not os.path.isdir(artist_dir):
            continue
        # Iterate through albums (subdirectories of each artist)
        for album in os.listdir(artist_dir):
            album_dir = os.path.join(artist_dir, album)
            if not os.path.isdir(album_dir):
                continue

            album_id = f"{artist} - {album}"

            # Collect .m4a files in the album folder (skipping hidden files)
            track_files = [f for f in os.listdir(album_dir)
                           if f.lower().endswith(".m4a") and not f.startswith(".")]
            if not track_files:
                missing_albums.append(f"{album_id}: No track files found")
                continue
            track_files.sort()  # Assume filenames sort in track order

            # Use the first track to extract initial album info.
            first_track_path = os.path.join(album_dir, track_files[0])
            album_info = extract_album_info(first_track_path)
            if album_info is None:
                logger.warning("No album info found for '%s'; skipping.", album_id)
                missing_albums.append(f"{album_id}: No album info found")
                continue
            expected_tracks, total_discs = album_info

            # Skip albums tagged as having only one track (singles)
            if expected_tracks == 1 and total_discs == 1:
                logger.info("Skipping single-track album '%s'", album_id)
                continue

            if total_discs > 1:
                # Multi-disc album: iterate through each track file to find the first track on each disc.
                disc_first_track_totals = {}
                for track in track_files:
                    track_path = os.path.join(album_dir, track)
                    try:
                        audio = MutagenFile(track_path)
                        if audio is None or not audio.tags:
                            continue

                        # Extract disc number from the "disk" tag (default to 1 if missing)
                        disk = audio.tags.get("disk")
                        if disk:
                            disc_info = disk[0]
                            if isinstance(disc_info, tuple) and len(disc_info) >= 2:
                                disc_number = disc_info[0]
                            else:
                                disc_number = int(disc_info)
                        else:
                            disc_number = 1

                        # Extract track info from the "trkn" tag
                        trkn = audio.tags.get("trkn")
                        if not trkn:
                            continue
                        track_info = trkn[0]
                        if isinstance(track_info, tuple) and len(track_info) >= 2:
                            track_number = track_info[0]
                            total_tracks_in_disc = track_info[1]
                        else:
                            track_number = int(track_info)
                            total_tracks_in_disc = int(track_info)

                        # If this is the first track on its disc, record the total track count for that disc.
                        if track_number == 1 and disc_number not in disc_first_track_totals:
                            disc_first_track_totals[disc_number] = total_tracks_in_disc
                    except Exception as e:
                        logger.error("Error processing %s: %s", track_path, e)
                        continue
                expected_tracks_total = sum(disc_first_track_totals.values())
            else:
                expected_tracks_total = expected_tracks

            actual_tracks = len(track_files)
            if actual_tracks != expected_tracks_total:
                logger.info("Album '%s' track count mismatch: expected %d tracks, found %d.",
                            album_id, expected_tracks_total, actual_tracks)
                if actual_tracks > expected_tracks_total:
                    extra_albums.append(f"{album_id}: Incomplete album (expected {expected_tracks_total}, found {actual_tracks})")
                else:
                    missing_albums.append(f"{album_id}: Incomplete album (expected {expected_tracks_total}, found {actual_tracks})")
                continue

            # Build the M3U playlist content
            m3u_filename = f"{artist} - {album}.m3u"
            m3u_path = os.path.join(alb_dir, m3u_filename)
            try:
                with open(m3u_path, "w", encoding="utf-8") as m3u_file:
                    m3u_file.write("#EXTM3U\n")
                    m3u_file.write(f"#EXTART:{artist}\n")
                    m3u_file.write(f"#EXTALB:{album}\n")
                    for track in sorted(track_files):
                        orig_track_path = os.path.join(album_dir, track)
                        spawn_id = extract_spawn_id(orig_track_path)
                        if not spawn_id:
                            logger.error("Missing spawn ID for track: %s", orig_track_path)
                            input(f"Missing spawn ID for track {orig_track_path}. Press Enter to continue...")
                            continue
                        # Construct expected symlink path from the linx directory
                        symlink_path = os.path.join(linx_dir, f"{spawn_id}.m4a")
                        if not os.path.lexists(symlink_path):
                            input(f"Warning: symlink {symlink_path} does not exist for track {orig_track_path}.\nPress Enter to continue...")
                        else:
                            try:
                                target = os.readlink(symlink_path)
                                if not os.path.exists(target):
                                    input(f"Warning: symlink {symlink_path} points to a non-existent file.\nPress Enter to continue...")
                            except Exception as e:
                                input(f"Warning: error reading symlink {symlink_path}: {e}\nPress Enter to continue...")

                        # Extract track duration and title from the original file.
                        duration, title = extract_track_info(orig_track_path)
                        if duration is None:
                            duration = 0
                        if not title:
                            title = os.path.splitext(track)[0]
                        m3u_file.write(f"#EXTINF:{duration},{artist} - {title}\n")

                        # Use absolute file path if -abs is provided, else use the relative symlink path
                        if args.abs:
                            entry_path = orig_track_path
                        else:
                            entry_path = os.path.relpath(symlink_path, start=alb_dir)
                            # entry_path = symlink_path # for absolute symlink path rather than relative symlink path

                        m3u_file.write(entry_path + "\n")
                logger.info("Created playlist: %s", m3u_path)
            except Exception as e:
                logger.error("Error writing playlist for '%s': %s", album_id, e)
                missing_albums.append(f"{album_id}: Error writing playlist: {e}")

    # # Write the incomplete album log to a file in the present working directory.
    # log_file_path = os.path.join(os.getcwd(), "incomplete_albums.txt")
    # try:
    #     with open(log_file_path, "w", encoding="utf-8") as log_file:
    #         if extra_albums:
    #             log_file.write("ALBUM FOLDERS WITH EXTRA TRACKS:\n")
    #             for entry in extra_albums:
    #                 log_file.write(entry + "\n")
    #             log_file.write("\n")
    #         if missing_albums:
    #             log_file.write("INCOMPLETE ALBUMS:\n")
    #             for entry in missing_albums:
    #                 log_file.write(entry + "\n")
    #     logger.info("Incomplete album log written to: %s", log_file_path)
    # except Exception as e:
    #     logger.error("Error writing incomplete album log: %s", e)

if __name__ == "__main__":
    main()
