# dumb.py

import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple

# Initialize module-specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def convert_size_to_gb(size_in_bytes: int) -> float:
    """Convert size from bytes to gigabytes using SI standard (1 GB = 1,000,000,000 bytes)."""
    return size_in_bytes / 1_000_000_000
    #return size_in_bytes / (1024 ** 3)   # for GiB (gibibytes)


def sanitize_path(path: str) -> Path:
    """
    Sanitize the input path by removing backslashes before spaces and normalizing the path.

    Args:
        path (str): The original path string input by the user.

    Returns:
        Path: The sanitized Path object.
    """
    try:
        # Replace escaped spaces (\ ) with regular spaces
        sanitized_str = path.replace('\\ ', ' ')
        # Replace double backslashes with single backslash
        sanitized_str = sanitized_str.replace('\\\\', '\\')
        sanitized_path = Path(sanitized_str).expanduser()
        return sanitized_path
    except Exception as e:
        logger.error(f"Exception in sanitize_path: {e}")
        raise

def validate_path(path: Path, description: str) -> bool:
    """
    Validate that the provided path exists and is of the expected type.

    Args:
        path (Path): The Path object to validate.
        description (str): Description of the path for logging purposes.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not path.exists():
        logger.error(f"The {description} '{path}' does not exist.")
        return False
    if description == "M3U playlist file" and not path.is_file():
        logger.error(f"The {description} '{path}' is not a file.")
        return False
    if description == "music directory" and not path.is_dir():
        logger.error(f"The {description} '{path}' is not a directory.")
        return False
    if description == "output directory" and not path.is_dir():
        # For output directories, we can create them later if needed.
        logger.error(f"The {description} '{path}' does not exist.")
        return False
    return True

def copy_tracks_with_sequence(
    m3u_file: str,
    music_dir: str,
    output_folder: str,
    max_size_gb: Optional[float] = None,
    dry_run: bool = False,
    base_path: Optional[str] = None
) -> Tuple[int, int]:
    """
    Copy tracks listed in an M3U file from the music directory to the output folder,
    renaming them with a six-digit sequence number.

    Args:
        m3u_file (str): Path to the M3U playlist file.
        music_dir (str): Path to the source music directory.
        output_folder (str): Path to the destination folder where tracks will be copied.
        max_size_gb (float, optional): Maximum cumulative size in GB for the copied tracks.
                                        Defaults to None (no limit).
        dry_run (bool, optional): If True, simulates the copying process without making changes.
        base_path (str, optional): Base path to resolve relative tracks. If None, defaults to
                                   the M3U file's directory.

    Returns:
        tuple: (number_of_successful_copies, number_of_failures)
    """
    try:
        # Sanitize input paths
        m3u_path = sanitize_path(m3u_file)
        music_directory = sanitize_path(music_dir)
        output_dir = sanitize_path(output_folder)

        if not validate_path(m3u_path, "M3U playlist file"):
            return (0, 0)
        if not validate_path(music_directory, "music directory"):
            return (0, 0)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        # Determine the base path for resolving relative paths
        if base_path:
            base_path_obj = sanitize_path(base_path).resolve()
            if not base_path_obj.is_dir():
                logger.error(f"The specified base path '{base_path_obj}' is not a directory.")
                return (0, 0)
            logger.info(f"Using specified base path for resolving relative tracks: {base_path_obj}")
        else:
            base_path_obj = m3u_path.parent.resolve()
            logger.info(f"Using M3U file's directory as base path: {base_path_obj}")

        # Ensure the output "Music" subfolder exists
        music_folder = output_dir / 'Music'
        if not dry_run:
            music_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured existence of output subfolder: {music_folder}")
        else:
            logger.info(f"[Dry Run] Would ensure existence of output subfolder: {music_folder}")

        # Convert max size to bytes if specified
        max_size_bytes = max_size_gb * (1024 ** 3) if max_size_gb else None
        total_copied_size = 0

        # Initialize counters
        success_count = 0
        failure_count = 0

        # Read the M3U file and get the track paths (skipping comments)
        with m3u_path.open('r', encoding='utf-8') as file:
            tracks = [line.strip() for line in file if line.strip() and not line.startswith('#')]

        logger.info(f"Total tracks to process from M3U: {len(tracks)}")

        # Copy each track with a six-digit sequence number as prefix
        for idx, relative_track in enumerate(tracks):
            relative_track_path = sanitize_path(relative_track)
            track_path = (base_path_obj / relative_track_path).resolve()

            # Ensure the resolved track is within the music directory
            try:
                track_path.relative_to(music_directory.resolve())
            except ValueError:
                logger.warning(f"Track '{track_path}' is outside the music directory '{music_directory.resolve()}'. Skipping.")
                failure_count += 1
                continue

            if not track_path.is_file():
                logger.warning(f"Track not found: {track_path}")
                failure_count += 1
                continue

            original_size = track_path.stat().st_size

            # Check if adding this track exceeds the max size limit
            if max_size_bytes and (total_copied_size + original_size) > max_size_bytes:
                logger.info(f"Max size limit of {max_size_gb} GB reached. Stopping execution.")
                break

            sequence_num = f"{idx + 1:06d}"
            new_filename = f"{sequence_num} - {track_path.name}"
            new_filepath = music_folder / new_filename

            if new_filepath.exists():
                logger.warning(f"File already exists and will be skipped: {new_filepath}")
                failure_count += 1
                continue

            if dry_run:
                logger.info(f"[Dry Run] Would copy: {track_path} -> {new_filepath} (Size: {original_size} bytes)")
                success_count += 1
                total_copied_size += original_size
                continue

            try:
                shutil.copy2(track_path, new_filepath)
                copied_size = new_filepath.stat().st_size
                if copied_size != original_size:
                    raise IOError(f"File size mismatch after copying {track_path} -> {new_filepath}")
                total_copied_size += copied_size
                success_count += 1
                logger.info(f"Copied: {track_path} -> {new_filepath} (Size: {copied_size} bytes)")
                cumulative_size_gb = convert_size_to_gb(total_copied_size)
                logger.info(f"Cumulative size of copied files: {total_copied_size} bytes ({cumulative_size_gb:.2f} GB)")
                logger.info("")
            except Exception as e:
                logger.error(f"Error copying {track_path}: {e}")
                failure_count += 1
                continue

        logger.info("File copying process complete.")
        logger.info(f"Total successful copies: {success_count}")
        logger.info(f"Total failures: {failure_count}")
        logger.info(f"Total size copied: {total_copied_size} bytes ({convert_size_to_gb(total_copied_size):.2f} GB)")
        return (success_count, failure_count)

    except Exception as e:
         logger.error(f"An error occurred: {e}")
         return (0, 0)

def copy_all_tracks_without_sequence(
    music_dir: str,
    output_folder: str,
    max_size_gb: Optional[float] = None,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Copy all tracks from the music directory to the output folder without renaming them.
    """
    try:
        music_directory = sanitize_path(music_dir)
        output_dir = sanitize_path(output_folder)

        if not validate_path(music_directory, "music directory"):
            return (0, 0)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        music_folder = output_dir / 'Music'
        if not dry_run:
            music_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured existence of output subfolder: {music_folder}")
        else:
            logger.info(f"[Dry Run] Would ensure existence of output subfolder: {music_folder}")

        tracks = [f for f in music_directory.rglob('*') if f.is_file()]
        logger.info(f"Total tracks to copy: {len(tracks)}")

        max_size_bytes = max_size_gb * (1024 ** 3) if max_size_gb else None
        total_copied_size = 0
        success_count = 0
        failure_count = 0

        for idx, track_path in enumerate(tracks):
            original_size = track_path.stat().st_size
            if max_size_bytes and (total_copied_size + original_size) > max_size_bytes:
                logger.info(f"Max size limit of {max_size_gb} GB reached. Stopping execution.")
                break

            new_filepath = music_folder / track_path.name

            if new_filepath.exists():
                logger.warning(f"File already exists and will be skipped: {new_filepath}")
                failure_count += 1
                continue

            if dry_run:
                logger.info(f"[Dry Run] Would copy: {track_path} -> {new_filepath} (Size: {original_size} bytes)")
                success_count += 1
                total_copied_size += original_size
                continue

            try:
                shutil.copy2(track_path, new_filepath)
                copied_size = new_filepath.stat().st_size
                if copied_size != original_size:
                    raise IOError(f"File size mismatch after copying {track_path} -> {new_filepath}")
                total_copied_size += copied_size
                success_count += 1
                logger.info(f"Copied: {track_path} -> {new_filepath} (Size: {copied_size} bytes)")
                cumulative_size_gb = convert_size_to_gb(total_copied_size)
                logger.info(f"Cumulative size of copied files: {total_copied_size} bytes ({cumulative_size_gb:.2f} GB)")
                logger.info("")
            except Exception as e:
                logger.error(f"Error copying {track_path}: {e}")
                failure_count += 1
                continue

        logger.info("File copying process complete.")
        logger.info(f"Total successful copies: {success_count}")
        logger.info(f"Total failures: {failure_count}")
        logger.info(f"Total size copied: {total_copied_size} bytes ({convert_size_to_gb(total_copied_size):.2f} GB)")
        return (success_count, failure_count)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return (0, 0)

def copy_all_tracks_with_sequence(
    music_dir: str,
    output_folder: str,
    max_size_gb: Optional[float] = None,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Copy all tracks from the music directory to the output folder,
    renaming them with a six-digit sequence number.
    """
    try:
        music_directory = sanitize_path(music_dir)
        output_dir = sanitize_path(output_folder)

        if not validate_path(music_directory, "music directory"):
            return (0, 0)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        music_folder = output_dir / 'Music'
        if not dry_run:
            music_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured existence of output subfolder: {music_folder}")
        else:
            logger.info(f"[Dry Run] Would ensure existence of output subfolder: {music_folder}")

        tracks = [f for f in music_directory.rglob('*') if f.is_file()]
        logger.info(f"Total tracks to copy: {len(tracks)}")

        max_size_bytes = max_size_gb * (1024 ** 3) if max_size_gb else None
        total_copied_size = 0
        success_count = 0
        failure_count = 0

        for idx, track_path in enumerate(tracks):
            original_size = track_path.stat().st_size
            if max_size_bytes and (total_copied_size + original_size) > max_size_bytes:
                logger.info(f"Max size limit of {max_size_gb} GB reached. Stopping execution.")
                break

            sequence_num = f"{idx + 1:06d}"
            new_filename = f"{sequence_num} - {track_path.name}"
            new_filepath = music_folder / new_filename

            if new_filepath.exists():
                logger.warning(f"File already exists and will be skipped: {new_filepath}")
                failure_count += 1
                continue

            if dry_run:
                logger.info(f"[Dry Run] Would copy: {track_path} -> {new_filepath} (Size: {original_size} bytes)")
                success_count += 1
                total_copied_size += original_size
                continue

            try:
                shutil.copy2(track_path, new_filepath)
                copied_size = new_filepath.stat().st_size
                if copied_size != original_size:
                    raise IOError(f"File size mismatch after copying {track_path} -> {new_filepath}")
                total_copied_size += copied_size
                success_count += 1
                logger.info(f"Copied: {track_path} -> {new_filepath} (Size: {copied_size} bytes)")
                cumulative_size_gb = convert_size_to_gb(total_copied_size)
                logger.info(f"Cumulative size of copied files: {total_copied_size} bytes ({cumulative_size_gb:.2f} GB)")
                logger.info("")
            except Exception as e:
                logger.error(f"Error copying {track_path}: {e}")
                failure_count += 1
                continue

        logger.info("File copying process complete.")
        logger.info(f"Total successful copies: {success_count}")
        logger.info(f"Total failures: {failure_count}")
        logger.info(f"Total size copied: {total_copied_size} bytes ({convert_size_to_gb(total_copied_size):.2f} GB)")
        return (success_count, failure_count)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return (0, 0)

def convert_m3u_to_file_sequence(lib_path: str) -> None:
    """
    Interactive function to convert an M3U playlist to a file sequence for dumb players.
    It defaults the source music directory to lib_path/Spawn/Music and always prompts
    the user to enter the output folder. The base path for resolving relative track paths
    is automatically set to the music directory.
    """
    from pathlib import Path
    #print("\n=== Convert M3U to File Sequence ===\n")
    
    # Prompt for the M3U playlist file
    m3u_file = input("Enter the path to the M3U playlist file: ").strip()
    if not m3u_file:
        print("No M3U file path provided. Exiting conversion.")
        return

    # Automatically set the music directory to lib_path/Spawn/Music
    music_dir = str((Path(lib_path) / "Spawn" / "Music").resolve())
    #print(f"Using default music directory (source): {music_dir}")

    # Prompt for the output folder (no default option)
    output_folder = input("Enter the path to the output folder: ").strip()
    if not output_folder:
        print("No output folder provided. Exiting conversion.")
        return

    # Optionally, ask for a maximum cumulative size in GB
    max_size_input = input("Enter maximum cumulative size in GB (press Enter for no limit): ").strip()
    max_size_gb = float(max_size_input) if max_size_input else None

    # Optional dry run option
    dry_run = False

    # Automatically set the base path for resolving relative track paths to the music directory
    base_path = music_dir

    print("\nStarting conversion process...\n")
    success_count, failure_count = copy_tracks_with_sequence(
        m3u_file=m3u_file,
        music_dir=music_dir,
        output_folder=output_folder,
        max_size_gb=max_size_gb,
        dry_run=dry_run,
        base_path=base_path
    )
    print("\nConversion process complete.")
