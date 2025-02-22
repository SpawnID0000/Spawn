import os
import sys
import sqlite3
import pickle
import argparse
import logging
import mutagen.mp4

logger = logging.getLogger(__name__)

# List all the tags you want to preserve *exactly* in their canonical spelling/case.
DESIRED_TAGS = [
    # Common iTunes/MP4 fields:
    "©nam",  # Track Title
    "©alb",  # Album
    "©ART",  # Artist
    "aART",  # Album Artist
    "©day",  # Year
    "©gen",  # Genre
    "trkn",  # Track Number
    "disk",  # Disc Number
    "©com",  # Comment
    "©wrt",  # Composer
    "©lyr",  # Lyrics
    "covr",  # Album art

    # MusicBrainz:
    "----:com.apple.iTunes:MusicBrainz Artist Id",
    "----:com.apple.iTunes:MusicBrainz Track Id",
    "----:com.apple.iTunes:MusicBrainz Release Group Id",
    # "----:com.apple.iTunes:MusicBrainz Release Track Id",
    # "----:com.apple.iTunes:MusicBrainz Album Id",
    # "----:com.apple.iTunes:MusicBrainz Album Artist Id",
    # "----:com.apple.iTunes:MusicBrainz Album Status",
    # "----:com.apple.iTunes:MusicBrainz Album Type",
    # "----:com.apple.iTunes:MusicBrainz Album Release Country",

    # Apple/iTunes:
    "----:com.apple.iTunes:iTunSMPB",
    "----:com.apple.iTunes:iTunNORM",

    # Original date fields that you want to keep:
    # "originalyear",
    # "originaldate",
    # "----:com.apple.iTunes:originalyear",
    # "----:com.apple.iTunes:originaldate",

    # Some extra MB or random ID tags
    # "----:com.apple.iTunes:ISRC",
    # "----:com.apple.iTunes:ASIN",
    # "----:com.apple.iTunes:ARTISTS",
    # "----:com.apple.iTunes:LABEL",
    # "----:com.apple.iTunes:MEDIA",

    # ReplayGain
    "----:com.apple.iTunes:replaygain_track_peak",
    "----:com.apple.iTunes:replaygain_track_gain",
    "----:com.apple.iTunes:replaygain_album_peak",
    "----:com.apple.iTunes:replaygain_album_gain",

    # AcoustID
    "----:com.apple.iTunes:Acoustid Fingerprint",
    "----:com.apple.iTunes:Acoustid Id",

    # Spotify
    "----:com.apple.iTunes:spotify_track_ID",
    "----:com.apple.iTunes:spotify_artist_ID",

    # Audio Features
    "----:com.apple.iTunes:feature_valence",
    "----:com.apple.iTunes:feature_time_signature",
    "----:com.apple.iTunes:feature_tempo",
    "----:com.apple.iTunes:feature_speechiness",
    "----:com.apple.iTunes:feature_mode",
    "----:com.apple.iTunes:feature_loudness",
    "----:com.apple.iTunes:feature_liveness",
    "----:com.apple.iTunes:feature_key",
    "----:com.apple.iTunes:feature_instrumentalness",
    "----:com.apple.iTunes:feature_energy",
    "----:com.apple.iTunes:feature_danceability",
    "----:com.apple.iTunes:feature_acousticness",

    # Spawn
    "----:com.apple.iTunes:spawnre",
    "----:com.apple.iTunes:spawnre_hex",
    "----:com.apple.iTunes:spawn_ID",
    "----:com.apple.iTunes:metadata_rev",

    # Script, etc.
    "----:com.apple.iTunes:SCRIPT",
]

# Create a mapping from a lowercase version of each desired tag to its canonical form.
CANONICAL_TAG_MAP = {tag.lower(): tag for tag in DESIRED_TAGS}

# Hard-coded variable to select whether to update tags in the source files.
#case = ignore
case = "change"  # Set to "ignore" to leave files unchanged or "change" to update them.

def main():
    parser = argparse.ArgumentParser(
        description="Check DB spawn_ids against mp4tovec embeddings and validate m4a metadata."
    )
    parser.add_argument("--spawn-root", required=True,
                        help="Path to the Spawn project root.")
    args = parser.parse_args()

    log_level = logging.DEBUG if os.getenv('DEBUG', '0') == '1' else logging.INFO
    logging.basicConfig(level=log_level,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Derive paths
    spawn_root_abs = os.path.abspath(args.spawn_root)
    db_path = os.path.join(spawn_root_abs, "aux", "glob", "spawn_catalog.db")
    pickle_path = os.path.join(spawn_root_abs, "aux", "glob", "mp4tovec.p")
    music_dir = os.path.join(spawn_root_abs, "Music")

    # Check existence
    if not os.path.isfile(db_path):
        logger.error(f"spawn_catalog.db not found at: {db_path}")
        sys.exit(1)
    if not os.path.isfile(pickle_path):
        logger.error(f"mp4tovec.p not found at: {pickle_path}")
        sys.exit(1)
    if not os.path.isdir(music_dir):
        logger.error(f"Music directory not found at: {music_dir}")
        sys.exit(1)

    # Load DB spawn_ids
    db_spawn_ids = load_spawn_ids_from_db(db_path)
    db_spawn_ids_set = set(db_spawn_ids)
    logger.info(f"Loaded {len(db_spawn_ids)} spawn_ids from DB.")

    # Load embeddings dict
    emb_dict = load_embeddings_dict(pickle_path)
    emb_spawn_ids_set = set(emb_dict.keys())
    logger.info(f"Loaded {len(emb_dict)} embeddings from {pickle_path}.")

    # Check database vs embeddings
    check_spawn_ids(db_spawn_ids_set, emb_spawn_ids_set, emb_dict, pickle_path)

    # Check .m4a files for valid spawn_id and normalize tag key cases
    check_m4a_files(music_dir, db_spawn_ids_set, emb_spawn_ids_set)

def load_spawn_ids_from_db(db_path):
    """
    Returns a list of all spawn_id values from the 'tracks' table of spawn_catalog.db.
    """
    spawn_ids = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT spawn_id FROM tracks")
        rows = cur.fetchall()
        spawn_ids = [row[0] for row in rows]
    except Exception as e:
        logger.error(f"Failed to load spawn_ids from DB: {e}")
    finally:
        if conn:
            conn.close()
    return spawn_ids

def load_embeddings_dict(pickle_path):
    """Loads and returns the embeddings dictionary from the given pickle file."""
    try:
        with open(pickle_path, "rb") as f:
            emb_dict = pickle.load(f)
        if not isinstance(emb_dict, dict):
            logger.error(f"Unexpected data type in {pickle_path}, expected dict.")
            sys.exit(1)
        return emb_dict
    except Exception as e:
        logger.error(f"Failed to load embeddings from {pickle_path}: {e}")
        sys.exit(1)

def check_spawn_ids(db_spawn_ids_set, emb_spawn_ids_set, emb_dict, pickle_path):
    """Compares DB spawn_ids with embedding spawn_ids and reports discrepancies."""
    missing_in_pickle = db_spawn_ids_set - emb_spawn_ids_set
    extra_in_pickle = emb_spawn_ids_set - db_spawn_ids_set

    if missing_in_pickle:
        logger.warning("The following spawn_ids are in DB but MISSING in mp4tovec.p:")
        for sid in missing_in_pickle:
            logger.warning(f"  {sid}")
        logger.info(f"Total missing spawn_ids: {len(missing_in_pickle)}")
    else:
        logger.info("No missing spawn_ids. All DB IDs are present in mp4tovec.p!")

    if extra_in_pickle:
        logger.warning("The following spawn_ids are in mp4tovec.p but NOT in the DB:")
        for sid in extra_in_pickle:
            logger.warning(f"  {sid}")
        logger.info(f"Total extra spawn_ids: {len(extra_in_pickle)}")
        user_input = input("Do you want to remove these extra embeddings and save a new pickle? (y/n): ").strip().lower()
        if user_input == "y":
            for sid in extra_in_pickle:
                del emb_dict[sid]
            new_pickle_path = os.path.join(os.path.dirname(pickle_path), "mp4tovec_new.p")
            try:
                with open(new_pickle_path, "wb") as f:
                    pickle.dump(emb_dict, f)
                logger.info(f"New pickle saved at: {new_pickle_path}")
            except Exception as e:
                logger.error(f"Failed to save new pickle: {e}")
        else:
            logger.info("No changes made to the pickle file.")

def check_m4a_files(music_dir, db_spawn_ids_set, emb_spawn_ids_set):
    """Scans .m4a files in the music directory and checks for valid spawn_id tags.
       Also normalizes tag key cases based on DESIRED_TAGS.
       If 'case' is set to "change", only files with detected changes are updated.
    """
    logger.info("Scanning .m4a files for spawn_id verification...")
    logger.debug(f"Checking directory: {music_dir}")
    untagged_files = []
    spawn_id_counts = {}
    checked_files = 0
    unmatched_files = []

    for root, dirs, files in os.walk(music_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        logger.debug(f"Scanning directory: {root}, found {len(files)} files")
        for file in files:
            # Process only .m4a files and skip hidden files
            if not file.lower().endswith(".m4a") or file.startswith("."):
                continue
            checked_files += 1
            logger.debug(f"Found file: {file}")
            file_path = os.path.join(root, file)
            try:
                # Load the audio file so we can later update it if needed.
                audio = mutagen.mp4.MP4(file_path)
                tags = audio.tags
                logger.debug(f"Reading metadata for: {file_path}")

                # Normalize the tag keys to their canonical form.
                changed = False
                if tags:
                    for key in list(tags.keys()):
                        key_lower = key.lower()
                        if key_lower in CANONICAL_TAG_MAP:
                            canonical = CANONICAL_TAG_MAP[key_lower]
                            if key != canonical:
                                changed = True
                                # If a canonical version already exists, merge the values.
                                if canonical in tags:
                                    tags[canonical] = tags[canonical] + tags[key]
                                    del tags[key]
                                else:
                                    tags[canonical] = tags.pop(key)
                    # Uncomment the next line to log the normalized tags for debugging.
                    # logger.debug(f"Normalized tags for {file_path}: {list(tags.keys())}")

                # If set to "change" and normalization changed any tags, update the file.
                if case == "change" and changed:
                    try:
                        audio.save()
                        logger.info(f"Updated tags for file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to save updated tags for {file_path}: {e}")

                # Check for the spawn_ID tag (using the canonical name).
                if tags and "----:com.apple.iTunes:spawn_ID" in tags:
                    logger.debug(f"Found spawn_ID tag in: {file_path}")
                    # Assuming the tag's value is a list with a byte string.
                    spawn_id = tags["----:com.apple.iTunes:spawn_ID"][0].decode("utf-8")
                    spawn_id_counts[spawn_id] = spawn_id_counts.get(spawn_id, 0) + 1
                    if spawn_id not in db_spawn_ids_set:
                        unmatched_files.append((file_path, spawn_id))
                else:
                    untagged_files.append(file_path)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    if untagged_files:
        logger.warning("The following .m4a files are missing a spawn_id tag:")
        for file in untagged_files:
            logger.warning(f"  {file}")

    if unmatched_files:
        logger.warning("The following .m4a files have spawn_id values that do not match any database entry:")
        for file, spawn_id in unmatched_files:
            logger.warning(f"  {file} (spawn_id: {spawn_id})")

    duplicates = {sid: count for sid, count in spawn_id_counts.items() if count > 1}
    if duplicates:
        logger.warning(f"The following {len(duplicates)} spawn_ids are duplicated across multiple .m4a files:")
        for sid, count in duplicates.items():
            logger.warning(f"  spawn_id: {sid} appears {count} times")
    logger.info(f"Checked {checked_files} .m4a files.")

if __name__ == "__main__":
    main()
