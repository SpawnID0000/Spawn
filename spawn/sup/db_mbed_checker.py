#!/usr/bin/env python3
"""
db_mbed_checker.py

This auxiliary script compares Spawn IDs stored in the database (spawn_catalog.db)
against embeddings stored in a pickle file. It also validates that the .m4a audio files
in the Music directory have valid spawn_ID tags.

New optional flag:
    --tfidf
If provided, the script uses the TF‑IDF embeddings file (mp4tovecTFIDF.p) instead of the standard one.
If the TF‑IDF file is not found, the script will attempt to generate it by calling calc_tfidf.py.

Usage example:
    python db_mbed_checker.py --spawn-root /path/to/Spawn --tfidf
"""

import os
import sys
import sqlite3
import pickle
import argparse
import logging
import mutagen.mp4
import subprocess
from typing import List
from time import sleep

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

def build_spawnid_to_filepath(music_dir: str) -> dict:
    """
    Walks the Music directory and returns a dict mapping each spawn_ID or local_ID
    (extracted from m4a file tags) to its file path.
    Skips hidden files.
    """
    logger.debug("Building mapping of Spawn IDs to file paths in %s", music_dir)
    mapping = {}
    for root, dirs, files in os.walk(music_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file.lower().endswith(".m4a") and not file.startswith("."):
                file_path = os.path.join(root, file)
                try:
                    audio = mutagen.mp4.MP4(file_path)
                    tags = audio.tags
                    sid = None
                    if tags:
                        if "----:com.apple.iTunes:spawn_ID" in tags:
                            raw = tags["----:com.apple.iTunes:spawn_ID"][0]
                        elif "----:com.apple.iTunes:local_ID" in tags:
                            raw = tags["----:com.apple.iTunes:local_ID"][0]
                        else:
                            continue
                        sid = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                    if sid:
                        mapping[sid] = file_path
                        logger.debug(f"Mapping ID {sid} to {file_path}")
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
    return mapping

def load_spawn_ids_from_db(db_path: str, table_name: str = "tracks") -> List[str]:
    """
    Returns a list of all spawn_id values from the given table in the database.
    """
    logger.debug(f"Loading Spawn IDs from table '{table_name}' in database {db_path}")
    spawn_ids = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT spawn_id FROM {table_name}")
        rows = cur.fetchall()
        spawn_ids = [row[0] for row in rows if row[0]]
        logger.debug("Loaded %d Spawn IDs from DB", len(spawn_ids))
    except Exception as e:
        logger.error("Failed to load Spawn IDs from DB: %s", e)
    finally:
        if conn:
            conn.close()
    return spawn_ids

def load_embeddings_dict(pickle_path: str) -> dict:
    """Loads and returns the embeddings dictionary from the given pickle file."""
    logger.debug("Loading embeddings from pickle file %s", pickle_path)
    try:
        with open(pickle_path, "rb") as f:
            emb_dict = pickle.load(f)
        if not isinstance(emb_dict, dict):
            logger.error("Data in %s is not a dict.", pickle_path)
            sys.exit(1)
        return emb_dict
    except Exception as e:
        logger.error("Failed to load embeddings from %s: %s", pickle_path, e)
        sys.exit(1)

def check_spawn_ids(db_spawn_ids_set: set, emb_spawn_ids_set: set, emb_dict: dict, pickle_path: str) -> None:
    """Compares DB spawn_ids with embedding spawn_ids and reports discrepancies."""
    missing_in_pickle = db_spawn_ids_set - emb_spawn_ids_set
    extra_in_pickle = emb_spawn_ids_set - db_spawn_ids_set

    if missing_in_pickle:
        logger.warning("The following Spawn IDs are in the database but MISSING in the embeddings pickle:")
        for sid in missing_in_pickle:
            logger.warning(f"  {sid}")
        logger.info(f"Total missing Spawn IDs: {len(missing_in_pickle)}")
    else:
        logger.info("No missing Spawn IDs. All database IDs are present in the embeddings pickle!")

    if extra_in_pickle:
        logger.warning("The following Spawn IDs are in the embeddings pickle but NOT in the database:")
        for sid in extra_in_pickle:
            logger.warning(f"  {sid}")
        logger.info(f"Total extra Spawn IDs: {len(extra_in_pickle)}")
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

            # Update the embedding spawn IDs set after adding new embeddings
            emb_spawn_ids_set = set(emb_dict.keys())

        else:
            logger.info("No changes made to the pickle file.")

def check_m4a_files(music_dir: str, db_spawn_ids_set: set, emb_spawn_ids_set: set) -> None:
    """Scans .m4a files in the music directory and checks for valid spawn_id tags.
       Also normalizes tag key cases based on DESIRED_TAGS.
       If 'case' is set to "change", only files with detected changes are updated.
    """
    logger.info("Scanning .m4a files for Spawn ID verification...")
    logger.debug(f"Checking music directory: {music_dir}")
    untagged_files = []
    spawn_id_counts = {}
    checked_files = 0
    unmatched_files = []

    for root, dirs, files in os.walk(music_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        logger.debug("Scanning %s; %d files found", root, len(files))
        for file in files:
            # Process only .m4a files and skip hidden files
            if not file.lower().endswith(".m4a") or file.startswith("."):
                continue
            checked_files += 1
            logger.debug(f"Found file: {file}")
            file_path = os.path.join(root, file)
            logger.debug("Processing file: %s", file_path)
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

                # Try spawn_ID, then fallback to local_ID
                track_id = None
                if tags:
                    if "----:com.apple.iTunes:spawn_ID" in tags:
                        raw = tags["----:com.apple.iTunes:spawn_ID"][0]
                    elif "----:com.apple.iTunes:local_ID" in tags:
                        raw = tags["----:com.apple.iTunes:local_ID"][0]
                    else:
                        untagged_files.append(file_path)
                        continue
                    track_id = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    track_id = track_id.strip()

                if track_id:
                    spawn_id_counts[track_id] = spawn_id_counts.get(track_id, 0) + 1
                    if track_id not in db_spawn_ids_set:
                        unmatched_files.append((file_path, track_id))
                else:
                    untagged_files.append(file_path)

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    if untagged_files:
        logger.warning("The following .m4a files are missing a spawn_ID or local_ID tag:")
        for file in untagged_files:
            logger.warning(f"  {file}")

    if unmatched_files:
        logger.warning("The following .m4a files have spawn_ID/local_ID values not found in the database:")
        for file, tid in unmatched_files:
            logger.warning(f"  {file} (ID: {tid})")

    duplicates = {sid: count for sid, count in spawn_id_counts.items() if count > 1}
    if duplicates:
        logger.warning(f"The following {len(duplicates)} IDs are duplicated across multiple .m4a files:")
        for sid, count in duplicates.items():
            logger.warning(f"  ID: {sid} appears {count} times")

    logger.info(f"Checked {checked_files} .m4a files.")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check DB spawn_ids against mp4tovec embeddings and validate m4a metadata."
    )
    parser.add_argument("--spawn-root", required=True,
                        help="Path to the Spawn project root (e.g. LIB_PATH/Spawn).")
    parser.add_argument("--tfidf", action="store_true",
                        help="Use TF-IDF embeddings (mp4tovecTFIDF.p) instead of standard embeddings.")
    parser.add_argument("--local", action="store_true",
                        help="Use local (user-mode) embeddings from mp4tovec_local.p instead of catalog.")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Maximum number of worker processes (optional).")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for TF-IDF calculation (passed to calc_tfidf.py).")
    args = parser.parse_args()

    log_level = logging.DEBUG if os.getenv('DEBUG', '0') == '1' else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.debug("Starting db_mbed_checker...")
    spawn_root_abs = os.path.abspath(args.spawn_root)
    if args.local:
        db_path = os.path.join(spawn_root_abs, "aux", "user", "spawn_library.db")
        table_name = "lib_tracks"
    else:
        db_path = os.path.join(spawn_root_abs, "aux", "glob", "spawn_catalog.db")
        table_name = "tracks"
    # Choose pickle file based on flag (none, --tfidf, or --local).
    if args.local:
        pickle_file_name = "mp4tovec_local.p"
        pickle_path = os.path.join(spawn_root_abs, "aux", "user", pickle_file_name)
    elif args.tfidf:
        pickle_file_name = "mp4tovecTFIDF.p"
        pickle_path = os.path.join(spawn_root_abs, "aux", "glob", pickle_file_name)
    else:
        pickle_file_name = "mp4tovec.p"
        pickle_path = os.path.join(spawn_root_abs, "aux", "glob", pickle_file_name)
    music_dir = os.path.join(spawn_root_abs, "Music")

    # Check existence of required files/directories.
    if not os.path.isfile(db_path):
        logger.error(f"spawn_catalog.db not found at: {db_path}")
        sys.exit(1)
    if not os.path.isdir(music_dir):
        logger.error(f"Music directory not found at: {music_dir}")
        sys.exit(1)

    # If using TF-IDF mode and pickle file not found, try to generate it.
    if args.tfidf and not os.path.isfile(pickle_path):
        logger.info(f"{pickle_file_name} not found at: {pickle_path}. Attempting to generate it using calc_tfidf.py")
        try:
            import subprocess
            calc_tfidf_path = os.path.join(os.path.dirname(__file__), "..", "calc_tfidf.py")
            cmd = [sys.executable, calc_tfidf_path,
                   "--mp3tovecs_file", os.path.join("aux", "glob", "mp4tovec_raw.p"),
                   "--mp3tovec_file", os.path.join("aux", "glob", pickle_file_name),
                   "--batch_size", str(args.batch_size)]
            if args.spawn_root:
                cmd.extend(["--spawn_root", spawn_root_abs])
            logger.debug(f"Executing command: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            logger.info(f"Generated {pickle_file_name} using calc_tfidf.py")
        except Exception as e:
            logger.error(f"Failed to generate {pickle_file_name} using calc_tfidf.py: {e}")
            sys.exit(1)

    # Load DB spawn_ids.
    db_spawn_ids = load_spawn_ids_from_db(db_path, table_name=table_name)
    db_spawn_ids_set = set(db_spawn_ids)
    logger.info(f"Loaded {len(db_spawn_ids)} Spawn IDs from database.")

    # Load embeddings dictionary.
    emb_dict = load_embeddings_dict(pickle_path)
    emb_spawn_ids_set = set(emb_dict.keys())
    logger.info(f"Loaded {len(emb_dict)} embeddings from {pickle_path}.")

    # If in TF-IDF mode, check for missing embeddings and try to generate them.
    if args.tfidf:
        missing_in_pickle = db_spawn_ids_set - emb_spawn_ids_set
        if missing_in_pickle:
            logger.info(f"[TF-IDF Mode] {len(missing_in_pickle)} spawn_ids are missing from the pickle file.")
            spawnid_to_file = build_spawnid_to_filepath(music_dir)
            try:
                from spawn import calc_tfidf  # import from the spawn package
            except Exception as e:
                logger.error(f"Failed to import calc_tfidf module: {e}")
                sys.exit(1)
            for sid in missing_in_pickle:
                if sid in spawnid_to_file:
                    file_path = spawnid_to_file[sid]
                    logger.info(f"Generating TF-IDF embedding for spawn_id {sid} from file {file_path}")

                    # Generate raw snippet vectors for the track
                    try:
                        # Generate the TF-IDF embedding for this file.
                        tfidf_embedding = calc_tfidf.generate_tfidf_embedding(file_path)
                        if tfidf_embedding is not None:
                            emb_dict[sid] = tfidf_embedding
                            logger.info(f"Generated and added TF-IDF embedding for {sid}")
                        else:
                            logger.warning(f"No valid embedding found for {sid} from file {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to process {sid} from file {file_path}: {e}")

                else:
                    logger.warning(f"Could not find audio file for spawn_id {sid} in Music directory")

            # Save updated pickle.
            try:
                with open(pickle_path, "wb") as f:
                    pickle.dump(emb_dict, f)
                logger.info(f"Updated {pickle_file_name} with generated embeddings for missing spawn_ids")
            except Exception as e:
                logger.error(f"Failed to update {pickle_file_name}: {e}")
        else:
            logger.info("No missing spawn_ids in TF-IDF embeddings.")
    else:
        # For non-TFIDF modes (--local or no flag), auto-generate missing embeddings using MP4ToVec.
        missing_in_pickle = db_spawn_ids_set - emb_spawn_ids_set
        if missing_in_pickle:
            logger.info(f"{len(missing_in_pickle)} spawn_ids are missing from the pickle file in non-TFIDF mode.")
            spawnid_to_file = build_spawnid_to_filepath(music_dir)
            try:
                from spawn.MP4ToVec import load_mp4tovec_model_diffusion, generate_embedding
                model = load_mp4tovec_model_diffusion()
            except Exception as e:
                logger.error(f"Failed to load MP4ToVec model: {e}")
                model = None
            if model is None:
                logger.info("MP4ToVec model not available; cannot generate missing embeddings.")
            else:
                for sid in missing_in_pickle:
                    if sid in spawnid_to_file:
                        file_path = spawnid_to_file[sid]
                        logger.info(f"Generating embedding for spawn_id {sid} from file {file_path}")
                        try:
                            embedding = generate_embedding(file_path, model)
                            if embedding is not None:
                                emb_dict[sid] = embedding
                                logger.info(f"Generated and added embedding for {sid}")
                            else:
                                logger.warning(f"No valid embedding found for {sid} from file {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to process {sid} from file {file_path}: {e}")
                    else:
                        logger.warning(f"Could not find audio file for spawn_id {sid} in Music directory")
                try:
                    with open(pickle_path, "wb") as f:
                        pickle.dump(emb_dict, f)
                    logger.info(f"Updated {pickle_file_name} with generated embeddings for missing spawn_ids")
                except Exception as e:
                    logger.error(f"Failed to update {pickle_file_name}: {e}")
        else:
            logger.info("No missing spawn_ids in embeddings.")

    # Update emb_spawn_ids_set after potential modifications
    emb_spawn_ids_set = set(emb_dict.keys())

    # Compare DB spawn_ids with embedding spawn_ids.
    check_spawn_ids(db_spawn_ids_set, emb_spawn_ids_set, emb_dict, pickle_path)

    # Check .m4a files.
    check_m4a_files(music_dir, db_spawn_ids_set, emb_spawn_ids_set)
    logger.debug("db_mbed_checker script completed.")

if __name__ == "__main__":
    main()
