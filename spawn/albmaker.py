#!/usr/bin/env python3
"""
albmaker.py

Usage (standalone):
    # Process the entire library:
    python albmaker.py LIB_PATH/Spawn/Music [-abs]

    # Process specific album folders:
    python albmaker.py LIB_PATH/Spawn/Music/Artist1/Album LIB_PATH/Spawn/Music/Artist2/Album [-abs]

Description:
    This module scans the Spawn library’s Music directory (organized as:
      /Spawn/Music/Artist/Album/filename.m4a)
    and for each album, it reads the first track’s metadata (using the "trkn" tag)
    to determine the expected total number of tracks. If the album folder contains
    exactly that many tracks (or extra files that can be interpreted as bonus tracks
    provided that all standard tracks are present), the module generates an M3U
    playlist for the album in /Spawn/aux/user/albm/ named "Artist - Album.m3u".

    When run with specific album folder paths, only those albums are processed.

    For each track, it uses the symlink created for the corresponding Spawn ID,
    where the symlink filename is "<spawn_id>.m4a" located under /Spawn/aux/user/linx/.
    Unless the optional "-abs" flag is provided (which causes the absolute file path
    to be used), the playlist will contain the relative symlink path.

    Each M3U file includes metadata comment lines:
      - #EXTART:<album artist>
      - #EXTALB:<album name>
      - Each track is preceded by a #EXTINF line that includes track duration and title.

    If a required symlink is missing or broken, the process will halt and prompt the user
    with options to "retry" or "skip & continue".

    At the end of processing the entire library, a summary of discrepancies is logged:
      - Albums with extra bonus tracks (but standard tracks present)
      - Incomplete albums (standard tracks missing)
"""

import os
import sys
import argparse
import logging
from mutagen import File as MutagenFile

# Setup logging
if __name__ == "__main__":
    # Running standalone: set up basic logging configuration.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
else:
    # When imported, adopt the centralized logger (already configured in track_importer.py)
    logger = logging.getLogger("spawn")


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


def check_symlink_interactively(symlink_path: str, orig_track_path: str) -> bool:
    """
    Check whether the symlink exists and points to a valid target.
    If not, prompt the user to retry or skip & continue.
    Returns True if the symlink is valid; False if the user chooses to skip.
    """
    while True:
        if not os.path.lexists(symlink_path):
            choice = input(
                f"Warning: symlink {symlink_path} does not exist for track {orig_track_path}.\n"
                "Enter R to retry or S to skip this track: "
            ).strip().lower()
            if choice == "r":
                continue
            elif choice == "s":
                return False
            else:
                print("Invalid input. Please enter 'R' or 'S'.")
                continue
        else:
            try:
                target = os.readlink(symlink_path)
                if not os.path.exists(target):
                    choice = input(
                        f"Warning: symlink {symlink_path} points to a non-existent file for track {orig_track_path}.\n"
                        "Enter R to retry or S to skip this track: "
                    ).strip().lower()
                    if choice == "r":
                        continue
                    elif choice == "s":
                        return False
                    else:
                        print("Invalid input. Please enter 'R' or 'S'.")
                        continue
                else:
                    return True
            except Exception as e:
                choice = input(
                    f"Warning: error reading symlink {symlink_path} for track {orig_track_path}: {e}\n"
                    "Enter R to retry or S to skip this track: "
                ).strip().lower()
                if choice == "r":
                    continue
                elif choice == "s":
                    return False
                else:
                    print("Invalid input. Please enter 'R' or 'S'.")
                    continue


def extract_track_numbers(file_path: str):
    """
    Extract disc number, track number, and total tracks in the disc from the given file.
    Returns a tuple: (disc_number, track_number, total_tracks_in_disc)
    or None if extraction fails.
    """
    try:
        audio = MutagenFile(file_path)
        if audio is None or not audio.tags:
            return None
        # Determine disc number (default to 1 if not present)
        disk = audio.tags.get("disk")
        if disk:
            disc_info = disk[0]
            if isinstance(disc_info, tuple) and len(disc_info) >= 2:
                disc_number = disc_info[0]
            else:
                disc_number = int(disc_info)
        else:
            disc_number = 1
        # Extract track number and total tracks from the "trkn" tag.
        trkn = audio.tags.get("trkn")
        if not trkn:
            return None
        track_info = trkn[0]
        if isinstance(track_info, tuple) and len(track_info) >= 2:
            track_number = track_info[0]
            total_tracks_in_disc = track_info[1]
        else:
            track_number = int(track_info)
            total_tracks_in_disc = int(track_info)
        return (disc_number, track_number, total_tracks_in_disc)
    except Exception as e:
        logger.error("Error extracting track numbers from %s: %s", file_path, e)
        return None


def generate_playlist_for_album(album_dir: str, linx_dir: str, alb_dir: str, use_absolute_paths: bool):
    """
    Process a single album folder and generate its M3U playlist.
    Updated logic:
      - If the number of files is less than the expected standard track count, the album
        is considered incomplete.
      - If extra files (bonus tracks) are present but all standard tracks are found,
        the playlist is generated using only the standard tracks.
    Returns a tuple of 4 values:
      (status, album_id, expected_tracks, found_files)
      For status "incomplete", expected_tracks and found_files are provided.
      For status "complete" or "extra", these values are None.
    """
    artist = os.path.basename(os.path.dirname(album_dir))
    album = os.path.basename(album_dir)
    album_id = f"{artist} - {album}"
    
    track_files = [f for f in os.listdir(album_dir)
                   if f.lower().endswith(".m4a") and not f.startswith(".")]
    if not track_files:
        logger.info("No track files found in album '%s'", album_id)
        return ("incomplete", album_id, 0, 0)
    track_files.sort()
    
    first_track_path = os.path.join(album_dir, track_files[0])
    album_info = extract_album_info(first_track_path)
    if album_info is None:
        logger.warning("No album info found for '%s'; skipping.", album_id)
        return ("incomplete", album_id, 0, len(track_files))
    expected_tracks, total_discs = album_info
    
    # Skip single-track albums.
    if expected_tracks == 1 and total_discs == 1:
        logger.info("Skipping single-track album '%s'", album_id)
        return ("complete", album_id, None, None)

    # For single-disc albums, if file count is less than expected, mark as incomplete.
    if total_discs == 1:
        if len(track_files) < expected_tracks:
            logger.info("Album '%s' incomplete: expected %d tracks, found %d.",
                        album_id, expected_tracks, len(track_files))
            return ("incomplete", album_id, expected_tracks, len(track_files))

    # Build a mapping: disc_number -> { track_number: full_file_path }
    mapping = {}
    for track in track_files:
        track_path = os.path.join(album_dir, track)
        result = extract_track_numbers(track_path)
        if result is None:
            continue
        disc_number, track_number, total_tracks_in_disc = result
        if disc_number not in mapping:
            mapping[disc_number] = {}
        mapping[disc_number][track_number] = track_path

    # Determine expected total tracks per disc.
    expected_total = {}
    if total_discs == 1:
        expected_total[1] = expected_tracks
    else:
        # For multi-disc albums, use track 1 from each disc to get the expected total.
        for disc, tracks in mapping.items():
            if 1 in tracks:
                result = extract_track_numbers(tracks[1])
                if result is not None:
                    _, _, total_tracks_in_disc = result
                    expected_total[disc] = total_tracks_in_disc
            else:
                logger.info("Album '%s' is missing track 1 on disc %s.", album_id, disc)
                return ("incomplete", album_id, 0, len(track_files))
        # Early check: if total files are less than the sum of expected tracks for all discs.
        expected_total_sum = sum(expected_total.values())
        if len(track_files) < expected_total_sum:
            logger.info("Album '%s' incomplete: expected %d tracks, found %d.",
                        album_id, expected_total_sum, len(track_files))
            return ("incomplete", album_id, expected_total_sum, len(track_files))

    # Verify that every standard track is present.
    expected_sum = sum(expected_total.values()) if total_discs > 1 else expected_total[1]
    for disc, exp_total in expected_total.items():
        if disc not in mapping:
            logger.info("Album '%s' is missing disc %s.", album_id, disc)
            return ("incomplete", album_id, expected_sum, len(track_files))
        for t in range(1, exp_total + 1):
            if t not in mapping[disc]:
                logger.info("Album '%s' is missing standard track %d on disc %s.", album_id, t, disc)
                return ("incomplete", album_id, expected_sum, len(track_files))
    
    # Build an ordered list of file paths for the standard tracks (ignoring bonus tracks).
    filtered_files = []
    for disc in sorted(expected_total.keys()):
        for t in range(1, expected_total[disc] + 1):
            filtered_files.append(mapping[disc][t])

    # If no tracks were found via track number extraction, fallback to sorted track_files
    if not filtered_files:
        logger.info("No standard tracks found via track number extraction; falling back to raw file order.")
        filtered_files = [os.path.join(album_dir, f) for f in track_files]

    # Determine status: if file count exactly matches expected, status is "complete";
    # if there are extra files, status is "extra".
    status = "complete" if len(track_files) == expected_sum else "extra"

    m3u_filename = f"{artist} - {album}.m3u"
    m3u_path = os.path.join(alb_dir, m3u_filename)
    try:
        with open(m3u_path, "w", encoding="utf-8") as m3u_file:
            m3u_file.write("#EXTM3U\n")
            m3u_file.write(f"#EXTART:{artist}\n")
            m3u_file.write(f"#EXTALB:{album}\n")
            for orig_track_path in filtered_files:
                spawn_id = extract_spawn_id(orig_track_path)
                if not spawn_id:
                    logger.error("Missing spawn ID for track: %s", orig_track_path)
                    user_choice = input(
                        f"Missing spawn ID for track {orig_track_path}. Enter S to skip this track or any key to continue: "
                    )
                    if user_choice.strip().lower() == "s":
                        continue
                symlink_path = os.path.join(linx_dir, f"{spawn_id}.m4a")
                if not check_symlink_interactively(symlink_path, orig_track_path):
                    logger.info("Skipping track %s due to missing/broken symlink.", orig_track_path)
                    continue
                duration, title = extract_track_info(orig_track_path)
                if duration is None:
                    duration = 0
                if not title:
                    title = os.path.splitext(os.path.basename(orig_track_path))[0]
                m3u_file.write(f"#EXTINF:{duration},{artist} - {title}\n")
                if use_absolute_paths:
                    entry_path = orig_track_path
                else:
                    entry_path = os.path.relpath(symlink_path, start=alb_dir)
                m3u_file.write(entry_path + "\n")
        logger.info("Created playlist: %s", m3u_path)
        return (status, album_id, None, None)
    except Exception as e:
        logger.error("Error writing playlist for '%s': %s", album_id, e)
        return ("incomplete", album_id, expected_sum, len(track_files))


def generate_album_playlists(spawn_root: str, use_absolute_paths: bool = False):
    """
    Generate M3U playlists for complete albums in the Spawn library.
    Expects spawn_root to be the full Spawn library.
    A summary is printed at the end listing:
      - Albums with bonus (extra) tracks.
      - Incomplete albums (with expected vs. found track counts).
    """
    spawn_root = os.path.abspath(spawn_root)
    # If the provided spawn_root does not end with 'Spawn', assume it's LIB_PATH and append 'Spawn'
    if os.path.basename(spawn_root).lower() != "spawn":
        spawn_root = os.path.join(spawn_root, "Spawn")
    
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

    # Lists to accumulate album logs for reporting.
    extra_albums = []
    missing_albums = []

    # Process each artist/album folder in Music
    for artist in os.listdir(music_dir):
        artist_dir = os.path.join(music_dir, artist)
        if not os.path.isdir(artist_dir):
            continue
        for album in os.listdir(artist_dir):
            album_dir = os.path.join(artist_dir, album)
            if not os.path.isdir(album_dir):
                continue
            
            result = generate_playlist_for_album(album_dir, linx_dir, alb_dir, use_absolute_paths)
            if result:
                status, album_id, expected, found = result
                if status == "incomplete":
                    missing_albums.append((album_id, expected, found))
                elif status == "extra":
                    extra_albums.append(album_id)

    if extra_albums or missing_albums:
        logger.info("\nAlbum M3U processing complete.\n")
        if extra_albums:
            logger.info("************** Albums with bonus (extra) tracks **************\n")
            for entry in extra_albums:
                logger.info("  %s", entry)
        if missing_albums:
            logger.info("\n******************** Incomplete albums ********************\n")
            for album_id, expected, found in missing_albums:
                logger.info("  %s (expected %s tracks, found %s)", album_id, expected, found)
    else:
        logger.info("\nAlbum M3U processing complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate M3U playlists for complete albums in the Spawn library."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=("Path(s) to process. "
              "For a full library, supply the Music directory (e.g. /Volumes/Untitled/Spawn/Music). "
              "For a single album, supply its folder (e.g. /Volumes/Untitled/Spawn/Music/Artist/Album).")
    )
    parser.add_argument("-abs", action="store_true",
                        help="Use absolute file paths instead of relative symlink paths.")
    args = parser.parse_args()

    for path in args.paths:
        abs_path = os.path.abspath(path)
        # If the basename is "Music", process the entire library.
        if os.path.basename(abs_path).lower() == "music":
            # Derive spawn_root as the parent of Music.
            spawn_root = os.path.dirname(abs_path)
            generate_album_playlists(spawn_root, use_absolute_paths=args.abs)
        else:
            # Assume it is an album folder.
            # Walk up the directory tree until we find the folder named "Music".
            cur = abs_path
            while os.path.basename(cur).lower() != "music":
                parent = os.path.dirname(cur)
                if parent == cur:
                    logger.error("Provided path %s is not inside a Music folder.", abs_path)
                    break
                cur = parent
            else:
                # 'cur' is now the Music folder; spawn_root is its parent.
                spawn_root = os.path.dirname(cur)
                # Set up auxiliary directories.
                user_aux_dir = os.path.join(spawn_root, "aux", "user")
                alb_dir = os.path.join(user_aux_dir, "albm")
                linx_dir = os.path.join(user_aux_dir, "linx")
                if not os.path.isdir(user_aux_dir) or not os.path.isdir(linx_dir):
                    logger.error("Auxiliary directories not found under spawn_root %s", spawn_root)
                    continue
                os.makedirs(alb_dir, exist_ok=True)
                result = generate_playlist_for_album(abs_path, linx_dir, alb_dir, use_absolute_paths=args.abs)
                if result:
                    status, album_id, expected, found = result
                    if status == "incomplete":
                        logger.info("Album '%s' incomplete: expected %s tracks, found %s.", album_id, expected, found)
                    elif status == "extra":
                        logger.info("Album '%s' contained bonus tracks.", album_id)

if __name__ == "__main__":
    main()
