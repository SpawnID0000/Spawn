#!/usr/bin/env python3
"""
symlinker.py

Usage:
    Normal mode (create symlinks for each track file that has a Spawn ID):
        python symlinker.py SPAWN_PATH
    Database comparison mode (report differences between DB and files):
        python symlinker.py SPAWN_PATH -all user
        python symlinker.py SPAWN_PATH -all admin

Where SPAWN_PATH is the full path to the Spawn directory (e.g. /Volumes/Untitled/Spawn).

In normal mode, the script will search under SPAWN_PATH/Music for track files (skipping hidden files)
and create symlinks in SPAWN_PATH/aux/user/linx/ named "<SpawnID>.m4a".

In “-all” mode, the script will load spawn IDs from the specified database:
  - For “admin”, from SPAWN_PATH/aux/glob/spawn_catalog.db (table “tracks”).
  - For “user”, from SPAWN_PATH/aux/user/spawn_library.db (table “cat_tracks”).
Then it will compare the set of spawn IDs found in the database with those extracted from the Music folder
and report any IDs that exist in one set but not the other.
"""

import os
import sys
import argparse
import logging
import sqlite3

# Setup basic logging:
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def extract_spawn_id(file_path: str) -> str:
    """
    Open the file using mutagen and try to extract the spawn ID from the
    "----:com.apple.iTunes:spawn_ID" tag. Returns the spawn ID as a string
    (or an empty string if not found).
    """
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(file_path)
        if audio is None or not audio.tags:
            return ""
        tag = audio.tags.get("----:com.apple.iTunes:spawn_ID")
        if not tag:
            return ""
        if isinstance(tag, list):
            tag = tag[0]
        if isinstance(tag, bytes):
            tag = tag.decode("utf-8", errors="replace")
        return str(tag).strip()
    except Exception as e:
        logger.error(f"Error reading Spawn ID from {file_path}: {e}")
        return ""

def create_symlink_for_track(track_path: str, spawn_root: str, spawn_id: str) -> None:
    """
    Create a symbolic link for the track file in spawn_root/aux/user/linx/ named
    "<spawn_id>.m4a" that points to the absolute path of track_path.
    """
    linx_dir = os.path.join(spawn_root, "aux", "user", "linx")
    os.makedirs(linx_dir, exist_ok=True)
    symlink_path = os.path.join(linx_dir, f"{spawn_id}.m4a")
    try:
        if os.path.lexists(symlink_path):
            logger.info(f"Symlink already exists for spawn_id {spawn_id}: {symlink_path}")
        else:
            os.symlink(os.path.abspath(track_path), symlink_path)
            logger.info(f"Created symlink: {symlink_path} -> {os.path.abspath(track_path)}")
    except Exception as e:
        logger.error(f"Error creating symlink for {track_path} (spawn_id {spawn_id}): {e}")

def load_spawn_ids_from_db(db_path: str, table_name: str) -> set:
    """
    Connect to the SQLite database at db_path and return a set of Spawn IDs
    from the specified table.
    """
    spawn_ids = set()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT spawn_id FROM {table_name}")
        rows = cursor.fetchall()
        conn.close()
        for (spid,) in rows:
            if spid:
                spawn_ids.add(str(spid).strip())
    except Exception as e:
        logger.error(f"Error loading Spawn IDs from database at {db_path}: {e}")
    return spawn_ids

def collect_spawn_ids_from_files(music_dir: str) -> set:
    """
    Walk through the Music directory and extract Spawn IDs from all .m4a files (skipping hidden files).
    Returns a set of Spawn IDs.
    """
    file_spawn_ids = set()
    for root, dirs, files in os.walk(music_dir):
        for file in files:
            if file.startswith("."):
                continue
            if file.lower().endswith(".m4a"):
                track_path = os.path.join(root, file)
                spid = extract_spawn_id(track_path)
                if spid:
                    file_spawn_ids.add(spid)
    return file_spawn_ids

def main():
    parser = argparse.ArgumentParser(
        description="Create symlinks for all track files in the Spawn library that have a unique Spawn ID."
    )
    parser.add_argument("spawn_path", help="Path to the Spawn directory (e.g. /Volumes/Untitled/Spawn)")
    parser.add_argument("-all", choices=["user", "admin"], help="If provided, compare spawn IDs from the DB and files. Use '-all user' for spawn_library.db or '-all admin' for spawn_catalog.db.")
    args = parser.parse_args()

    spawn_root = os.path.abspath(args.spawn_path)
    music_dir = os.path.join(spawn_root, "Music")

    if not os.path.isdir(spawn_root):
        logger.error(f"Spawn directory not found at {spawn_root}")
        sys.exit(1)
    if not os.path.isdir(music_dir):
        logger.error(f"Music directory not found at {music_dir}")
        sys.exit(1)

    # If the -all option is provided, perform database comparison mode.
    if args.all:
        mode = args.all.lower()
        if mode == "admin":
            db_path = os.path.join(spawn_root, "aux", "glob", "spawn_catalog.db")
            table_name = "tracks"
        else:  # user mode
            db_path = os.path.join(spawn_root, "aux", "user", "spawn_library.db")
            # For user mode, we assume the curated tracks are in table "cat_tracks"
            table_name = "cat_tracks"

        db_spawn_ids = load_spawn_ids_from_db(db_path, table_name)
        file_spawn_ids = collect_spawn_ids_from_files(music_dir)

        missing_in_files = db_spawn_ids - file_spawn_ids
        missing_in_db = file_spawn_ids - db_spawn_ids

        print("\n=== Spawn ID Comparison Report ===")
        print(f"Total spawn IDs in database ({table_name}): {len(db_spawn_ids)}")
        print(f"Total spawn IDs found in Music folder: {len(file_spawn_ids)}")
        if missing_in_files:
            print("\nSpawn IDs present in DB but not found in files:")
            for sid in sorted(missing_in_files):
                print(f"  {sid}")
        else:
            print("\nAll Spawn IDs from database were found in files.")

        if missing_in_db:
            print("\nSpawn IDs found in files but not present in DB:")
            for sid in sorted(missing_in_db):
                print(f"  {sid}")
        else:
            print("\nAll Spawn IDs from files were found in the database.\n")

        sys.exit(0)

    # Otherwise, run the normal mode: create symlinks from Music folder.
    logger.info(f"Using Spawn library at: {spawn_root}")
    logger.info(f"Looking for tracks under: {music_dir}")

    processed_spawn_ids = set()

    for root, dirs, files in os.walk(music_dir):
        for file in files:
            if file.startswith("."):
                continue
            if file.lower().endswith(".m4a"):
                track_path = os.path.join(root, file)
                spid = extract_spawn_id(track_path)
                if spid:
                    if spid not in processed_spawn_ids:
                        create_symlink_for_track(track_path, spawn_root, spid)
                        processed_spawn_ids.add(spid)
                    else:
                        logger.info(f"Spawn ID {spid} already processed; skipping {track_path}")
                else:
                    logger.info(f"No spawn ID found in {track_path}; skipping.")

    logger.info("Symlink creation complete.")

if __name__ == "__main__":
    main()