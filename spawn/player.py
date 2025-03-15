# player.py

import os
import subprocess
import shlex
import re
import logging

from typing import Optional
from logging.handlers import TimedRotatingFileHandler

# Module-level logger
logger = logging.getLogger(__name__)

def setup_player_logging(debug_to_console: bool, spawn_root: str):
    """
    Sets up a logger so that:
      - All DEBUG+ messages go to 'player_log.txt' (rotated daily).
      - Console messages are INFO+ unless debug_to_console=True 
        (which sets console to DEBUG).
      - Python warnings are also captured to the log file, 
        but only CRITICAL warnings go to console.
    """
    # Where we want to store the logs (same path as import_log.txt)
    log_dir = os.path.join(spawn_root, "Spawn", "aux", "temp")
    os.makedirs(log_dir, exist_ok=True)

    # TimedRotatingFileHandler for daily rotation
    player_log_path = os.path.join(log_dir, "player_log.txt")

    # Remove any existing handlers to avoid duplicates
    logger.handlers = []

    file_handler = TimedRotatingFileHandler(
        filename=player_log_path,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.suffix = "%Y-%m-%d.txt"
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}\.txt$")
    file_handler.setLevel(logging.DEBUG)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug_to_console else logging.INFO)

    # Common formatter for both
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Attach both handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.debug(f"Player logging initialized. Logging to: {player_log_path}")

    # Capture Python warnings into logging
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.propagate = False  # don't double-log warnings

    # Add file handler to warnings
    warn_logger.addHandler(file_handler)

    # A console handler that only shows CRITICAL warnings
    warn_console_handler = logging.StreamHandler()
    warn_console_handler.setLevel(logging.CRITICAL)
    warn_console_handler.setFormatter(formatter)
    warn_logger.addHandler(warn_console_handler)


def play_m3u_menu(spawn_root: str, debug_to_console: bool = False):
    """
    Called from main.py => choice #4 => "Play M3U playlist".
    
    1) Asks user for the path to an .m3u file or offers a small menu
       of discovered .m3u files in the 'Spawn/Playlists' subfolders.
    2) Checks if mpv is installed. If not found, prompts user to install
       or provide the path.
    3) If everything is okay, we launch mpv with the selected playlist.
    """

    # Set up logging for player
    setup_player_logging(debug_to_console=debug_to_console, spawn_root=spawn_root)
    logger.info("Launching M3U playback menu...")

    # 1) Look for .m3u files under Spawn/Playlists
    playlists_dir = os.path.join(spawn_root, "Spawn", "Playlists")
    if not os.path.isdir(playlists_dir):
        msg = f"The playlists folder does not exist: {playlists_dir}"
        print(f"[ERROR] {msg}")
        logger.error(msg)
        return

    # Gather .m3u files recursively, ignoring hidden files
    all_m3u_files = []
    for root, dirs, files in os.walk(playlists_dir):
        # Exclude hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.lower().endswith(".m3u") and not f.startswith("."):
                full_path = os.path.join(root, f)
                all_m3u_files.append(full_path)

    if not all_m3u_files:
        print("[INFO] No .m3u files found in your Playlists folder.")
        logger.info("No .m3u files found under '%s'.", playlists_dir)
        return

    # 2) Let user pick which .m3u to play
    print("\nChoose one of these M3U playlists to play:\n")
    for i, m3u_path in enumerate(all_m3u_files, start=1):
        relative_path = os.path.relpath(m3u_path, start=playlists_dir)
        print(f"  {i}) {relative_path}")

    print("\n  Or enter another path to an .m3u file not listed.")

    choice = input("\nEnter number or path (or press ENTER to cancel): ").strip()
    if not choice:
        print("[INFO] Cancelled.")
        logger.info("User canceled M3U playback menu.")
        return

    selected_path = None

    # If user typed a digit that matches one of the above
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(all_m3u_files):
            selected_path = all_m3u_files[idx - 1]
        else:
            print("[WARN] Invalid selection index. Aborting.")
            logger.warning("User entered invalid playlist index: %s", idx)
            return
    else:
        # user typed a path
        potential_path = os.path.expanduser(choice)
        if os.path.isfile(potential_path) and not os.path.basename(potential_path).startswith("."):
            selected_path = potential_path
        else:
            print(f"[WARN] The file '{potential_path}' does not exist. Aborting.")
            logger.warning("User entered invalid path: %s", potential_path)
            return

    # 3) Confirm mpv is installed or discoverable
    mpv_path = find_mpv_binary()
    if not mpv_path:
        print("[ERROR] Could not locate 'mpv' in PATH. Please install mpv or specify path.")
        logger.error("mpv not found in PATH.")
        return

    # 4) Launch mpv with the selected playlist
    logger.info("Launching mpv for playlist: %s", selected_path)
    launch_player(mpv_path, selected_path, spawn_root)


def find_mpv_binary() -> Optional[str]:
    """
    Attempt to find 'mpv' in the user's PATH. If found, return the path.
    If not found, return None. On Windows, you might adapt the logic to
    check typical install paths, etc.
    """
    from shutil import which
    mpv_path = which("mpv")
    return mpv_path  # None if not found


def launch_player(mpv_executable: str, playlist_file: str, spawn_root: str):
    """
    Launch mpv with the given .m3u playlist file. 
    Add extra arguments as desired (shuffle, no-video, etc.).
    """

    log_dir = os.path.join(spawn_root, "Spawn", "aux", "temp")
    os.makedirs(log_dir, exist_ok=True)
    mpv_log_path = os.path.join(log_dir, "player_log.txt")

    cmd = [
        mpv_executable,

        "--really-quiet",
        f"--log-file={mpv_log_path}",
        "--msg-level=term=fatal",
        "--msg-level=file=info",
        "--msg-level=player=info",
        #"--msg-level=all=warn",
        #"--term-status-msg=Playing: ${filename}\\nA: ${=time-pos}/${=duration}",
        "--msg-level=ffmpeg=error",
        
        f"--playlist={playlist_file}",

        # Add more flags as needed
        # e.g. "--shuffle"
        # e.g. "--loop=playlist"
        #"--no-video",
        #"--audio-display=no",
    ]


    print(f"[INFO] Launching mpv: {' '.join(shlex.quote(c) for c in cmd)}")

    try:
        subprocess.run(cmd, check=True)
        logger.info("mpv completed successfully for playlist: %s", playlist_file)
    except subprocess.CalledProcessError as cpe:
        print(f"[ERROR] mpv exited with error: {cpe}")
        logger.error("mpv exited with error: %s", cpe)
    except FileNotFoundError:
        print("[ERROR] mpv not found. Aborting player.")
        logger.error("FileNotFoundError: mpv not found.")
