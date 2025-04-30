#!/usr/bin/env python3

"""
track_importer.py

Purpose:
    1. Checks for desired tags, canonicalizes names.
    2. Cleans and repackages AAC/ALAC files (removing unwanted metadata).
    3. Re-inserts only desired tags (MusicBrainz, AcoustID, etc.).
    4. Groups files by parent folder. For each folder:
       - Runs normal pipeline on each track (MBIDs, AcoustID).
       - Then calculates ReplayGain (track + album) across all tracks
         in the folder, ensuring album-level RG values are consistent.
       - Updates each file's tags with the track-level gain/peak plus
         the folder's album-level gain/peak.
    5. Confirms or updates MusicBrainz & Spotify tags using API calls.
    6. Determines AcoustID Fingerprint and writes value and ID to tags.
    7. Confirms or updates disc & track number based on MusicBrainz data.
    8. Writes all final tags back to disk.
    9. Assigns each track a unique 8-digit hex "spawn_id" if it doesn't already have one,
       stores it as an embedded tag, and records all final tags in a SQLite database.
    10. Each batch import triggers a new 'db_rev' reflecting total track count.
    11. Deej-AI / MP4ToVec: Generates advanced embeddings for each track 
        via a loaded MP4ToVec model and pickles them in a dictionary keyed by spawn_id.

Dependencies:
    - MP4Box (for AAC cleaning) => brew install gpac
    - ffmpeg (for ALAC repackage)
    - chromaprint / fpcalc (for fingerprinting) => e.g. brew install chromaprint
    - pip install mutagen pyacoustid musicbrainzngs spotipy requests librosa numpy
    - ACOUSTID_API_KEY env var or hardcode in script
    - For ReplayGain: bs1770gain
    - For database: sqlite3
    - For Deej-AI embedding: MP4ToVec.py (user-provided), plus any TensorFlow libs used by MP4ToVec
        - https://github.com/teticio/Deej-AI

Usage (as a standalone script):
    python3 track_importer.py [-acu] [-sp_id] [-sp_sec] [-last] [-skippy] <output_path> <input_path>

Usage (from another script):
    from track_importer import run_import
    run_import(
        output_path="/path/to/out",
        music_path="/path/to/music",
        acoustid_key="abc",
        spotify_client_id="xyz",
        spotify_client_secret="123",
        lastfm_key="456",
        skip_prompts=False
    )
"""

import os
import sys
import argparse
import subprocess
import shlex
import re
import musicbrainzngs
import acoustid
import spotipy
import requests
import logging
import warnings
import shutil
import time
import random
import json
import sqlite3
import difflib
import mutagen
import io
import datetime
import librosa
import pickle
import numpy as np
import builtins
import unicodedata

from mutagen import File as MutagenFile
from mutagen.mp4 import MP4, MP4FreeForm, MP4Cover, MP4Tags
from acoustid import WebServiceError, NoBackendError
from logging.handlers import TimedRotatingFileHandler
from collections import defaultdict, Counter
from collections.abc import Mapping, Sequence
from spotipy.oauth2 import SpotifyClientCredentials
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from difflib import SequenceMatcher
from requests.exceptions import SSLError
from scipy.spatial.distance import cosine

from spawn.dic_spawnre import genre_mapping, genre_synonyms, subgenre_to_parent

from spawn.MP4ToVec import load_mp4tovec_model_diffusion, generate_embedding
#from spawn.MP4ToVec import load_mp4tovec_model_torch, generate_embedding
#from spawn.MP4ToVec import load_mp4tovec_model_tf, generate_embedding

from spawn.albmaker import generate_album_playlists
from spawn.albmaker import generate_playlist_for_album

# Use Hugging Face's built-in logging controls:
try:
    import huggingface_hub.utils.logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

try:
    from diffusers.utils import logging as d_logging
    d_logging.set_verbosity_error()
except ImportError:
    pass

###############################################################################
# Global Settings, API Authorizations, and Logging Setup
###############################################################################

# These are defined as None so Python knows about them at module scope,
# but they will be assigned in main() after argument parsing:
OUTPUT_PARENT_DIR = None
LOG_DIR = None
LOG_FILE = None
DB_PATH = None
PLAYLISTS_DIR = None

# Toggle whether debug messages should display to console (True => console shows debug logs)
DEBUG_MODE = False
#DEBUG_MODE = True

# Will be set by CLI argument "-skippy"
SKIP_PROMPTS = False

# Create a global logger object at the module level
logger = logging.getLogger("spawn")
#logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture everything in the logger

spawn_id_to_embeds = {}
local_user_embeddings = {}
MP4TOVEC_MODEL = None
MP4TOVEC_AVAILABLE = False

EMBEDDING_DUPLICATE_THRESHOLD = 0.99999

try:
    from spawn.MP4ToVec import load_mp4tovec_model_diffusion, generate_embedding

    # Attempt to load the Hugging Face-based encoder
    MP4TOVEC_MODEL = load_mp4tovec_model_diffusion()  # or pass local dir
    if MP4TOVEC_MODEL is not None:
        MP4TOVEC_AVAILABLE = True
    else:
        logger.warning("AudioEncoder returned None; embeddings will be skipped.")

except ImportError as e:
    logger.warning(f"Failed to import or load AudioEncoder: {e}")


def store_key_in_env_file(env_path, key, value):
    """
    Appends or updates a key=value in the .env file at env_path.
    If the key already exists, it's overwritten in the file.
    """
    # 1) Read all lines from the existing .env
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    # 2) Check if the key is already in lines
    found = False
    for i, line in enumerate(lines):
        line_strip = line.strip()
        # e.g. "ACOUSTID_API_KEY=something"
        if line_strip.startswith(key + "="):
            lines[i] = f"{key}={value}\n"
            found = True
            break

    # 3) If not found, append new line
    if not found:
        lines.append(f"{key}={value}\n")

    # 4) Write back the updated lines
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"Stored '{key}' in {env_path}.")


# Save the original input function
_original_input = builtins.input

def send_pushover_alert(prompt_text):
    """
    Sends a Pushover notification with the prompt text.
    It checks that both PUSHOVER_USER_KEY and PUSHOVER_API_KEY are set.
    """
    user_key = os.environ.get("PUSHOVER_USER_KEY", "").strip()
    api_token = os.environ.get("PUSHOVER_API_KEY", "").strip()
    if not user_key or not api_token:
        return  # Do nothing if either key is missing

    try:
        response = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": api_token,
            "user": user_key,
            "message": f"User prompt triggered:\n{prompt_text}",
            "title": "Spawn Prompt Alert"
        })
        # Optionally, log or print if response.status_code != 200
    except Exception as e:
        print(f"Error sending Pushover notification: {e}")

def wake_screen():
    """Wake up the screen on macOS by simulating user activity."""
    subprocess.run(["caffeinate", "-u", "-t", "2"])  # Wakes the display for 2 seconds

def custom_input(prompt):
    """
    Overrides built-in input() to send a Pushover notification before prompting.
    """
    wake_screen()
    send_pushover_alert(prompt)
    return _original_input(prompt)

# Override built-in input globally in this module
builtins.input = custom_input


def setup_logging(debug_to_console, log_file):
    """
    Sets up a logger so that:
      - All debug messages go to a file at 'log_file'
      - Console prints debug messages only if debug_to_console=True,
        otherwise logs console messages at INFO level and above.
      - Additionally captures Python warnings into 'py.warnings' logger,
        and routes them only to the log file (not console).
    """
    if not log_file:
        # Fallback if log_file is None or empty
        log_file = "import_temp_log.txt"

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Remove any existing handlers to avoid duplicates
    logger.handlers = []

    # 1) File handler => all messages at DEBUG+ go to log file
    #file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    # Rotate daily at midnight, keep 7 backups
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, "import_log.txt"),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.suffix = "%Y-%m-%d.txt"
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}\.txt$")
    file_handler.setLevel(logging.DEBUG)

    # 2) Console handler => DEBUG if debug_to_console else INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug_to_console else logging.INFO)

    # Common formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Attach both handlers to our module-level logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.debug(f"Logging initialized. log_file='{log_file}', debug_to_console={debug_to_console}")

    # Capture Python warnings into logging
    logging.captureWarnings(True)
    # By default, captured warnings go to 'py.warnings' at level WARNING

    warn_logger = logging.getLogger("py.warnings")
    warn_logger.propagate = False  # don't let them also appear in the root logger

    warn_logger.addHandler(file_handler)

    # And add a console handler that only shows CRITICAL warnings
    # so normal library warnings won't clutter the console
    warn_console_handler = logging.StreamHandler()
    warn_console_handler.setLevel(logging.CRITICAL)
    warn_console_handler.setFormatter(formatter)
    warn_logger.addHandler(warn_console_handler)


def get_user_input(prompt, default="y"):
    """
    If SKIP_PROMPTS is True, automatically use the `default`.
    Otherwise, prompt the user, and if they press enter, return `default`.
    """
    if SKIP_PROMPTS:
        logger.info(f"[skippy] Using default '{default}' for prompt: {prompt}")
        return default

    user_val = input(prompt).strip().lower()
    if user_val == "":
        return default
    return user_val


# Initialize MusicBrainz
musicbrainzngs.set_useragent(
    "Spawn",
    "0.0.0",
    "spawn.id.0000@gmail.com"
)

# AcoustID Authorization
ACOUSTID_API_KEY = ""  # optionally enter API key here
if not ACOUSTID_API_KEY:
    logger.info("WARNING: No AcoustID API Key found (ACOUSTID_API_KEY). Lookups may fail.\n")

# Spotify Authorization
SPOTIFY_CLIENT_ID = ""  # optionally enter client ID here
SPOTIFY_CLIENT_SECRET = ""  # optionally enter client secret here

# Last.FM Authorization
DEFAULT_LASTFM_API_KEY = ""  # optionally enter API key here


def build_output_subpath(input_file, levels=2):
    """
    Extract the last `levels` directories plus the base filename
    to mirror the sub-directory structure in the output folder.

    levels=2 => keep the last 2 directories plus filename.
    """
    parts = input_file.split(os.sep)
    # Use the last `levels` directories plus the filename => total `levels + 1` from the end:
    subpath_parts = parts[-(levels+1):]
    # Join them
    return os.path.join(*subpath_parts)


def fetch_with_retry(url, max_retries=3):
    """Attempt to fetch the URL, prompt user on SSL error."""
    attempt = 1
    while attempt <= max_retries:
        try:
            print(f"\nAttempt {attempt} of {max_retries} for URL: {url}")
            response = requests.get(url)
            return response

        except SSLError as ssl_err:
            print(f"\nEncountered an SSL error: {ssl_err}")
            choice = input("SSL error encountered. Retry (r) or abort (a)? [r/a] ").strip().lower()
            if choice == "a":
                print("Aborting fetch.")
                return None
            elif choice == "r":
                attempt += 1
                continue
            else:
                print("Unrecognized option, aborting fetch.")
                return None

        except Exception as e:
            # Optionally handle other exceptions here
            print(f"\nEncountered an unexpected error: {e}")
            return None

    print("\nMax retries exceeded, aborting.")
    return None


###############################################################################
# Spawn ID + Database Functions
###############################################################################

def generate_spawn_id():
    """Return an 8-digit hexadecimal string, e.g. 'AB12CD34'."""
    return ''.join(random.choices('0123456789ABCDEF', k=8))

def generate_local_id():
    """
    Generate a local ID for unmatched user-mode tracks.
    Returns:
        A string in the format 'xxxXXXXX', where 'xxx' is fixed and 'XXXXX' is a 5-digit hexadecimal number.
    """
    return "xxx" + ''.join(random.choices("0123456789ABCDEF", k=5))

def init_db(db_path):
    """
    Creates a SQLite database with a 'tracks' table for storing all DESIRED_TAGS as JSON,
    keyed by 'spawn_id'. Used for the main catalog (admin mode).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            spawn_id TEXT PRIMARY KEY,
            tag_data TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def init_db_revisions(db_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS db_revisions (
            revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            db_rev TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def store_db_revision(db_path, db_rev_val):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find the current highest remaining revision_id
    cursor.execute("SELECT MAX(revision_id) FROM db_revisions;")
    max_rev = cursor.fetchone()[0] or 0  # Default to 0 if table is empty

    next_rev = max_rev + 1

    sql = """
        INSERT INTO db_revisions (revision_id, db_rev)
        VALUES (?, ?)
    """
    cursor.execute(sql, (next_rev, db_rev_val))
    conn.commit()
    conn.close()

def get_latest_db_rev(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT db_rev
        FROM db_revisions
        ORDER BY revision_id DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def init_user_library_db(lib_db_path):
    """
    Creates a SQLite database with two tables:
      - 'lib_tracks' for brand-new user imports (not in the catalog)
      - 'cat_tracks' for user imports that do match an existing catalog entry.
    Note that this connects to the user library DB at lib_db_path, not the main DB_PATH.
    """

    if not lib_db_path:
        raise ValueError("Must specify user library DB path.")

    conn = sqlite3.connect(lib_db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lib_tracks (
            spawn_id TEXT PRIMARY KEY,
            tag_data TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cat_tracks (
            spawn_id TEXT PRIMARY KEY,
            tag_data TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def fallback_json_serializer(obj):
    """
    Custom fallback for objects that are not JSON-serializable by default.
    """
    # 1) If it's MP4FreeForm, decode its .data
    if isinstance(obj, MP4FreeForm):
        try:
            return obj.data.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)

    # 2) If it's MP4Cover, omit or store a placeholder
    if 'MP4Cover' in globals() and isinstance(obj, MP4Cover):
        return "<MP4Cover omitted>"

    # 3) If it's anything else we didn't catch, just use str()
    return str(obj)


def normalize_fields_to_lists(tag_dict, fields_to_listify=None):
    """
    Ensure specified fields in tag_dict are lists, even if currently strings.
    """
    default_fields = [
        "artist", "albumartist", "genre",
        "metadata_rev",
        "----:com.apple.iTunes:metadata_rev",
        "----:com.apple.iTunes:Acoustid Id",
        "----:com.apple.iTunes:Acoustid Fingerprint",
        "----:com.apple.iTunes:replaygain_album_gain",
        "----:com.apple.iTunes:replaygain_album_peak",
        "----:com.apple.iTunes:replaygain_track_gain",
        "----:com.apple.iTunes:replaygain_track_peak",
        "----:com.apple.iTunes:spawnre_hex",
    ]
    if fields_to_listify is None:
        fields_to_listify = default_fields

    for field in fields_to_listify:
        if field in tag_dict and not isinstance(tag_dict[field], list):
            tag_dict[field] = [tag_dict[field]]
    return tag_dict


def store_tags_in_user_db(spawn_id, tag_dict, metadata_rev=None, table="lib_tracks", lib_db_path=None):
    """
    Insert or replace a row in either 'lib_tracks' or 'cat_tracks' table
    in the user's spawn_library.db, storing final DESIRED_TAGS as JSON.
    If metadata_rev is provided, it's injected into the JSON.
    """
    if not lib_db_path:
        raise ValueError("Must specify lib_db_path for user library DB.")

    # Only skip if this is truly a catalog track with a real spawn_id (8-char hex)
    if table == "lib_tracks" and spawn_id and not spawn_id.startswith("xxx"):
        logger.debug(f"[store_tags_in_user_db] spawn_id='{spawn_id}' => skipping insertion into 'lib_tracks'.")
        return

    # Only clear spawn_id if it's not a local_id
    if table == "lib_tracks" and (not spawn_id or not spawn_id.startswith("xxx")):
        spawn_id = None

    logger.debug(f"[store_tags_in_user_db] Inserting spawn_id='{spawn_id}' into table='{table}' at '{lib_db_path}'")

    # A small helper: strip the literal b'...' prefix from strings
    def strip_b_prefix(s: str) -> str:
        if len(s) > 2:
            if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
                return s[2:-1]
        return s

    # Fallback serializer
    def fallback_json_serializer(obj):
        # Can keep or expand any needed logic for MP4FreeForm, MP4Cover, etc.
        return str(obj)

    # This recursive function converts MP4FreeForm, MP4Cover, bytes, etc.
    # into normal JSON-friendly data (strings, lists, dicts).
    def universal_decode(obj):
        """
        Recursively convert `obj` into standard Python types (str, list, dict)
        so json.dumps() won't choke on custom objects like MP4FreeForm, MP4Cover, etc.
        Also skip 'covr' at any nesting level, convert non-string dict keys, etc.
        """

        # 1) Basic scalar
        if isinstance(obj, (str, int, float, bool, type(None))):
            # If it's a string, pass through strip_b_prefix.
            if isinstance(obj, str):
                return strip_b_prefix(obj)
            return obj

        # 2) MP4FreeForm => decode .data to string
        if isinstance(obj, MP4FreeForm):
            try:
                dec_str = obj.data.decode("utf-8", errors="replace")
            except:
                dec_str = str(obj)
            return strip_b_prefix(dec_str)

        # 3) MP4Cover => skip or flatten
        if 'MP4Cover' in globals() and isinstance(obj, MP4Cover):
            return "<MP4Cover omitted>"

        # 4) If it’s bytes
        if isinstance(obj, bytes):
            dec_str = obj.decode("utf-8", errors="replace")
            return strip_b_prefix(dec_str)

        # 5) If it’s a list/tuple => decode each element
        if isinstance(obj, (list, tuple)):
            return [universal_decode(x) for x in obj]

        # 6) If it’s a dict => recursively convert each key/value
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                # optionally skip 'covr' or check k.lower()
                if k.lower() == 'covr':
                    continue
                decoded_key = universal_decode(k)
                decoded_val = universal_decode(v)
                new_dict[decoded_key] = decoded_val
            return new_dict

        # 7) Fallback => string-ify
        return strip_b_prefix(str(obj))

    # 1) Convert the entire dictionary
    decoded_dict = universal_decode(tag_dict)
    decoded_dict = normalize_fields_to_lists(decoded_dict)

    # Ensure the final result is a dict
    if not isinstance(decoded_dict, dict):
        logger.warning("[store_tags_in_user_db] universal_decode() returned "
                       f"{type(decoded_dict).__name__} instead of dict. "
                       "Wrapping it in a dict under key 'original_value'.")
        decoded_dict = {"original_value": decoded_dict}

    # 2) Ensure JSON metadata_rev matches embedded tag
    if "----:com.apple.iTunes:metadata_rev" in decoded_dict:
        decoded_dict["metadata_rev"] = decoded_dict["----:com.apple.iTunes:metadata_rev"]
    # # 2) Inject metadata_rev if provided
    # if metadata_rev:
    #     clean_rev = universal_decode(metadata_rev)
    #     decoded_dict["metadata_rev"] = clean_rev

    # 3) JSON serialize the cleaned data
    tag_data_json = json.dumps(
        decoded_dict,
        ensure_ascii=False,
        sort_keys=True,
        default=fallback_json_serializer
    )

    # 4) Connect to the user DB (not the main DB) and use the requested table
    conn = sqlite3.connect(lib_db_path)
    cursor = conn.cursor()
    sql = f"INSERT OR REPLACE INTO {table} (spawn_id, tag_data) VALUES (?, ?)"
    cursor.execute(sql, (spawn_id, tag_data_json))
    conn.commit()
    conn.close()


def fetch_tags_from_user_db(spawn_id_val, table="lib_tracks", lib_db_path=None):
    """
    Returns the stored tag_data as a dict for the given spawn_id from 'lib_tracks'
    or 'cat_tracks' in the user's DB, or None if not found or JSON is invalid.
    """
    if not lib_db_path:
        raise ValueError("Must specify lib_db_path for user library DB.")

    conn = sqlite3.connect(lib_db_path)
    cursor = conn.cursor()
    sql = f"SELECT tag_data FROM {table} WHERE spawn_id = ?"
    cursor.execute(sql, (spawn_id_val,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None

def store_tags_in_db(DB_PATH, spawn_id, tag_dict, metadata_rev=None):
    """
    Insert or replace a row in the 'tracks' table in the main spawn_catalog.db,
    storing the final DESIRED_TAGS as JSON form. Skips 'covr' or other
    binary fields that can't be JSON-serialized by default.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # A small helper: strip the literal b'...' prefix from strings
    def strip_b_prefix(s: str) -> str:
        if len(s) > 2:
            if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
                return s[2:-1]
        return s

    def universal_decode(obj):
        """
        Recursively convert MP4FreeForm, MP4Cover, etc., to plain
        strings or placeholders so we can safely do `json.dumps()`.
        Skips 'covr' at any level.
        """
        # 1) Basic scalar
        if isinstance(obj, (str, int, float, bool, type(None))):
            # If it's a string, pass through strip_b_prefix.
            if isinstance(obj, str):
                return strip_b_prefix(obj)
            return obj

        # 2) MP4FreeForm => decode .data
        if isinstance(obj, MP4FreeForm):
            try:
                dec_str = obj.data.decode("utf-8", errors="replace")
            except:
                dec_str = str(obj)
            return strip_b_prefix(dec_str)

        # 3) MP4Cover => skip/omit
        if 'MP4Cover' in globals() and isinstance(obj, MP4Cover):
            return "<MP4Cover omitted>"

        # 4) Raw bytes => decode
        if isinstance(obj, bytes):
            dec_str = obj.decode("utf-8", errors="replace")
            return strip_b_prefix(dec_str)

        # 5) If it’s a dict => handle each key/value
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                dec_k = universal_decode(k)
                if not isinstance(dec_k, str):
                    dec_k = str(dec_k)
                if dec_k.lower() == 'covr':
                    continue
                new_dict[dec_k] = universal_decode(v)
            return new_dict

        # 6) If it’s a list/tuple => decode each element
        if isinstance(obj, (list, tuple)):
            return [universal_decode(x) for x in obj]

        # 7) Fallback => string-ify
        return strip_b_prefix(str(obj))

    # 1) Convert the entire dictionary
    decoded_dict = universal_decode(tag_dict)
    decoded_dict = normalize_fields_to_lists(decoded_dict)

    # Ensure the final result is a dict
    if not isinstance(decoded_dict, dict):
        logger.warning(
            f"[store_tags_in_user_db] universal_decode() returned {type(decoded_dict).__name__} instead of dict. "
            "Wrapping it in a dict under key 'original_value'."
        )
        decoded_dict = {"original_value": decoded_dict}

    # 2) Ensure JSON metadata_rev matches embedded tag
    if "----:com.apple.iTunes:metadata_rev" in decoded_dict:
        decoded_dict["metadata_rev"] = decoded_dict["----:com.apple.iTunes:metadata_rev"]
    # # 2) Inject metadata_rev if provided
    # if metadata_rev:
    #     clean_rev = universal_decode(metadata_rev)
    #     decoded_dict["metadata_rev"] = clean_rev

    # 3) JSON serialize the cleaned data
    tag_data_json = json.dumps(decoded_dict, ensure_ascii=False, sort_keys=True, default=fallback_json_serializer)

    # 4) Insert or replace in the specified table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    table = "tracks"
    sql = f"INSERT OR REPLACE INTO {table} (spawn_id, tag_data) VALUES (?, ?)"
    cursor.execute(sql, (spawn_id, tag_data_json))
    conn.commit()
    conn.close()


def check_for_potential_match_in_db(conn, current_tags, current_spawn_id=None, similarity_threshold=0.90):
    """
    Checks if the given track (via current_tags) is "close enough"
    to any existing track in the database, excluding any row with the same spawn_id.
    Returns True if a match is found, else False.

    similarity_threshold = 0.90 means 90% string similarity is considered a match.
    """

    # 1) Extract local artist + track title
    def _tag_to_str(val):
        """
        Convert various tag formats (bytes, MP4FreeForm, list) to a clean string.
        """
        if isinstance(val, list) and val:  # If it's a list, take the first item
            val = val[0]
        if isinstance(val, MP4FreeForm):  # If it's an MP4FreeForm object, decode its data
            try:
                val = val.data.decode("utf-8", errors="replace")
            except AttributeError:  # Handle cases where .data might not exist
                val = str(val)
        elif isinstance(val, bytes):  # Decode bytes
            val = val.decode("utf-8", errors="replace")
        return str(val).strip() if val else None

    local_artist = _tag_to_str(current_tags.get("©ART"))
    local_title  = _tag_to_str(current_tags.get("©nam"))

    if not local_artist or not local_title:
        logger.debug("[check_db] Not enough info (artist/title) to check for duplicates. Skipping match check.")
        return False

    # 2) Query database for all existing rows
    cursor = conn.cursor()
    cursor.execute("SELECT spawn_id, tag_data FROM tracks")

    rows = cursor.fetchall()

    # If no rows, obviously no match
    if not rows:
        logger.debug("[check_db] Database is empty, so no possible matches.")
        return False

    logger.debug(f"[check_db] Checking for potential match => local: '{local_artist}' - '{local_title}' ...")
    found_any_match = False

    # 3) Compare with each row
    found_any_match = False
    for (db_spawn_id, tag_data_json) in rows:
        # Optionally skip if spawn_id is the same as current_spawn_id
        if current_spawn_id and db_spawn_id == current_spawn_id:
            continue

        # Parse JSON
        try:
            db_tags = json.loads(tag_data_json)
        except json.JSONDecodeError:
            logger.debug("[check_db] Could not parse JSON from database row, skipping.")
            continue

        db_artist = _tag_to_str(db_tags.get("©ART"))
        db_title  = _tag_to_str(db_tags.get("©nam"))
        if not db_artist or not db_title:
            continue

        # 4) Compute difflib ratio
        artist_ratio = difflib.SequenceMatcher(None, local_artist.lower(), db_artist.lower()).ratio()
        title_ratio  = difflib.SequenceMatcher(None, local_title.lower(),  db_title.lower()).ratio()
        avg_ratio    = (artist_ratio + title_ratio) / 2

        #logger.debug(f"[check_db]   => Compare to DB: '{db_artist}' - '{db_title}' | ratio={avg_ratio:.3f}")

        # If ratio >= threshold => match
        if avg_ratio >= similarity_threshold:
            logger.info(f"[check_db]   => Found match with database row spawn_id={db_spawn_id}; ratio={avg_ratio:.3f} >= {similarity_threshold}")
            logger.info(f"                matching track name = '{db_title}'; matching artist name = '{db_artist}'")
            found_any_match = True
            break

    if not found_any_match:
        logger.debug("[check_db] No matches found above threshold.")

    return found_any_match


def check_for_potential_match_in_mbed(
    current_tags_map: dict,
    new_file_paths_map: dict,
    new_embeddings: dict,
    existing_embeddings: dict,
    threshold: float = EMBEDDING_DUPLICATE_THRESHOLD
):
    """
    After generating embeddings for new imports, compare each new track
    against existing embeddings to catch likely duplicates.

    current_tags_map: {spawn_id: tags_dict} for each new track
    new_embeddings:    {spawn_id: numpy.ndarray} for new tracks
    existing_embeddings: {spawn_id: numpy.ndarray} already in EMBED_PATH
    threshold: cosine similarity above which we warn the user
    """
    removed = []

    for new_id, new_vec in list(new_embeddings.items()):
        # skip if this ID was already in the existing set
        if new_id in existing_embeddings:
            continue

        # find closest existing embedding
        best_sim = -1.0
        best_id  = None
        for exist_id, exist_vec in existing_embeddings.items():
            sim = 1.0 - cosine(new_vec, exist_vec)
            if sim > best_sim:
                best_sim = sim
                best_id  = exist_id

        if best_sim >= threshold:
            print(f"\n⚠️  Potential duplicate by embedding: "
                  f"new track {new_id} ≈ {best_id} (similarity {best_sim:.3f})")
            resp = input("Treat as duplicate? ([y]/n): ").strip().lower() or "y"
            if resp == "y":

                # 1) Temporarily force a metadata‐difference so handle_existing_spawn_id will prompt
                _orig_md = metadata_differs
                try:
                    globals()['metadata_differs'] = lambda a, b: True
                    db_tags       = fetch_tags_from_db(best_id)
                    incoming_tags = current_tags_map[new_id]
                    handle_existing_spawn_id(
                        best_id,
                        db_tags,
                        incoming_tags,
                        [False],                        # any_db_changes_ref
                        new_file_paths_map.get(new_id),# new file path
                        set(),                          # overridden_spawn_ids
                        set()                           # donotupdate_spawn_ids
                    )
                finally:
                    globals()['metadata_differs'] = _orig_md

                # 2) Log that we’re dropping the duplicate
                logger.info(f"[EMBEDDING] Duplicate match: not importing {new_id}, matches existing {best_id}")

                # 3) Write blank .txt marker + remove the .m4a
                new_file_path = new_file_paths_map.get(new_id)
                if new_file_path:
                    rel = os.path.relpath(new_file_path, start=OUTPUT_PARENT_DIR)
                    base, _ = os.path.splitext(rel)
                    licn_root = os.path.join(os.path.dirname(OUTPUT_PARENT_DIR),
                                             "aux", "user", "licn")
                    txt_path = os.path.join(licn_root, base + ".txt")
                    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                    with open(txt_path, "w", encoding="utf-8"):
                        pass
                    logger.info(f"Blank txt file created => {txt_path}")
                    try:
                        os.remove(new_file_path)
                        logger.info(f"Removed embedding‐duplicate file: {new_file_path}")
                    except FileNotFoundError:
                        logger.warning(f"File not found when removing duplicate: {new_file_path}")
                    except Exception as e:
                        logger.warning(f"Error removing file {new_file_path}: {e}")

                # 4) Pop its embedding & mark it removed
                new_embeddings.pop(new_id, None)
                removed.append(new_id)

            else:
                print(f"✔ Keeping {new_id} as a unique import.\n")
    return removed


def fetch_matching_spawn_id_from_db(incoming_tags):
    """
    Finds the catalog row that matches these tags (via artist/title partial match)
    and returns the spawn_id from that row.
    If no single match, returns None.
    """
    def _tag_to_str(val):
        # same logic used in check_for_potential_match_in_db
        if isinstance(val, list) and val:
            val = val[0]
        if isinstance(val, MP4FreeForm):
            try:
                val = val.data.decode("utf-8", errors="replace")
            except:
                val = str(val)
        elif isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        return str(val).strip() if val else ""

    local_artist = _tag_to_str(incoming_tags.get("©ART", ""))
    local_title  = _tag_to_str(incoming_tags.get("©nam", ""))

    if not local_artist or not local_title:
        return None

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT spawn_id, tag_data FROM tracks")
    possible_rows = c.fetchall()
    conn.close()

    best_match_id = None
    best_ratio = 0.0
    from difflib import SequenceMatcher

    for (row_id, row_data_json) in possible_rows:
        try:
            row_data = json.loads(row_data_json)
        except:
            continue
        
        row_artist = _tag_to_str(row_data.get("©ART", ""))
        row_title  = _tag_to_str(row_data.get("©nam", ""))
        if row_artist and row_title:
            art_ratio = SequenceMatcher(None, local_artist.lower(), row_artist.lower()).ratio()
            tit_ratio = SequenceMatcher(None, local_title.lower(),  row_title.lower()).ratio()
            avg = (art_ratio + tit_ratio) / 2
            if avg > best_ratio and avg >= 0.90:
                best_ratio = avg
                best_match_id = row_id
    
    return best_match_id


def fetch_tags_from_db(spawn_id_val: str) -> dict:
    """
    Returns the stored tag_data as a dict for the given spawn_id from the DB,
    or None if not found or JSON is invalid.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT tag_data FROM tracks WHERE spawn_id = ?", (spawn_id_val,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def metadata_differs(db_tags: dict, incoming_tags: dict) -> bool:
    """
    Return True if db_tags vs incoming_tags differ in any key or value,
    ignoring 'covr' (album art). Also rename metadata_rev key for easy comparison.
    """

    def universal_decode_for_compare(obj):
        # Similar to your universal_decode, but let's omit the
        # skip-'covr' logic or rename logic if you want. 
        # We'll still handle MP4FreeForm, bytes, nested dict, etc.
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, MP4FreeForm):
            try:
                return obj.data.decode("utf-8", errors="replace")
            except:
                return str(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, Mapping):
            # Recursively decode
            return {universal_decode_for_compare(k): universal_decode_for_compare(v) for k, v in obj.items()}
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            return [universal_decode_for_compare(x) for x in obj]
        return str(obj)

    def normalize_dict(tag_dict: dict) -> dict:
        result = {}
        for key, val in tag_dict.items():
            # Skip 'covr'
            if key.lower() == 'covr':
                continue
            # If the key is the special metadata_rev => rename it
            if key == "----:com.apple.iTunes:metadata_rev":
                norm_key = "metadata_rev"
            else:
                norm_key = key
            result[norm_key] = universal_decode_for_compare(val)
        return result

    db_sanitized = normalize_dict(db_tags)
    incoming_sanitized = normalize_dict(incoming_tags)

    return db_sanitized != incoming_sanitized


def handle_existing_spawn_id(
    spawn_id_str: str,
    db_tags: dict,
    incoming_tags: dict,
    any_db_changes_ref: list,
    new_file_path: str,
    overridden_spawn_ids: set,
    donotupdate_spawn_ids: set
):
    """
    1) Compare db_tags with incoming_tags.
    2) If they differ, prompt user:
       - 'override' => generate a new Spawn ID (do NOT update the old ID).
       - 'y' => update the old ID's metadata in the DB.
       - 'n' => keep DB as-is, revert the new file’s tags to match DB.
    """

    if not metadata_differs(db_tags, incoming_tags):
        logger.info(f"No metadata differences for spawn_id={spawn_id_str}. No update needed.")
        return None  # or return the same spawn_id_str to indicate no change

    user_prompt = (
        "Existing Spawn ID found with conflicting metadata.\n"
        "Enter 'override' if the match is inaccurate.\n"
        "Otherwise, would you like to update the catalog database? (y/n): "
    )
    valid_responses = {"override", "y", "n"}

    while True:
        user_in = get_user_input(user_prompt, default="override").strip().lower()

        if user_in in valid_responses:
            break
        logger.info("Invalid input. Please enter 'override', 'y', or 'n'.")

    user_in = get_user_input(user_prompt, default="override").strip().lower()

    if user_in == "override":
        # 1) Generate a new ID and assign it to incoming_tags
        new_spawn_id = generate_spawn_id()
        incoming_tags["----:com.apple.iTunes:spawn_ID"] = new_spawn_id
        # Reset metadata_rev
        incoming_tags["----:com.apple.iTunes:metadata_rev"] = "AAA"

        # Track overridden spawn_id
        overridden_spawn_ids.add(new_spawn_id)
        overridden_spawn_ids.add(spawn_id_str)

        logger.info(
            f"Spawn ID {spawn_id_str} was overridden. "
            f"New Spawn ID assigned: {new_spawn_id}."
        )
        return new_spawn_id  # The calling code MUST handle this new ID

    elif user_in == "y":
        # Overwrite DB with incoming_tags, increment metadata_rev
        old_rev = db_tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
        new_rev = "AAA"  # Ensure new_rev is always initialized

        try:
            new_rev = bump_metadata_rev(old_rev)  # Increment revision
        except Exception as e:
            logger.warning(f"Error bumping metadata_rev: {e}")

        incoming_tags["----:com.apple.iTunes:metadata_rev"] = new_rev
        store_tags_in_db(DB_PATH, spawn_id_str, incoming_tags, metadata_rev=new_rev)

        # Also rewrite tags so disk matches updated DB
        rewrite_tags(new_file_path, incoming_tags)

        logger.info(
            f"DB updated for existing spawn_id={spawn_id_str} "
            f"with new metadata_rev='{new_rev}'."
        )
        any_db_changes_ref[0] = True
        # Return None to indicate we kept the same spawn ID
        return None

    elif user_in == "n":
        # user_in == "n" => revert the new file's tags to match DB
        logger.info(
            f"User chose NOT to update DB for spawn_id={spawn_id_str}. "
            "Reverting new file to DB tags."
        )
        # Track overridden spawn_id
        donotupdate_spawn_ids.add(spawn_id_str)

        rewrite_tags(new_file_path, db_tags)
        return None


def bump_metadata_rev(old_val: str) -> str:
    """
    Example: "AAA" -> "AAB", "AAB" -> "AAC". If it's not in that style, fallback.
    """
    if not old_val or len(old_val) < 3:
        return "AAA"
    prefix = old_val[:-1]  # e.g. "AA"
    last_char = old_val[-1]  # e.g. "A"
    next_char = chr(ord(last_char) + 1)  # "B"
    return prefix + next_char


def update_embedded_metadata_rev(file_path, new_rev):
    """
    Opens the MP4 file at file_path, updates its embedded metadata_rev tag
    (under "----:com.apple.iTunes:metadata_rev") with new_rev (stored as bytes via MP4FreeForm),
    saves the file, then re-reads the file to verify the update.
    """
    try:
        audio = MP4(file_path)

        # Get the current metadata_rev; default to bytes for consistency.
        current_rev = audio.tags.get("----:com.apple.iTunes:metadata_rev", [b"AAA"])
        if isinstance(current_rev, list) and len(current_rev) > 0:
            # Convert current_rev to a string for logging, if it's bytes.
            if isinstance(current_rev[0], bytes):
                current_rev_str = current_rev[0].decode("utf-8", errors="replace")
            else:
                current_rev_str = str(current_rev[0])
        else:
            current_rev_str = "AAA"
        logger.debug(f"Current embedded metadata_rev for {file_path}: {current_rev_str}")

        # Update the tag: encode the new_rev as UTF-8 bytes and wrap it in MP4FreeForm.
        new_rev_bytes = new_rev.encode("utf-8")
        audio.tags["----:com.apple.iTunes:metadata_rev"] = [MP4FreeForm(new_rev_bytes)]
        audio.save()

        # Re-read the file to verify the update.
        updated_audio = MP4(file_path)
        updated_rev = updated_audio.tags.get("----:com.apple.iTunes:metadata_rev", [b"UNKNOWN"])
        # Convert updated_rev for logging.
        if isinstance(updated_rev, list) and len(updated_rev) > 0 and isinstance(updated_rev[0], bytes):
            updated_rev_str = updated_rev[0].decode("utf-8", errors="replace")
        else:
            updated_rev_str = str(updated_rev[0]) if updated_rev else "UNKNOWN"
        logger.info(f"Updated embedded metadata_rev to {updated_rev_str} for file {file_path}\n")

    except Exception as e:
        logger.error(f"Error updating embedded metadata_rev for file {file_path}: {e}")


def bump_db_alpha_part(alpha: str) -> str:
    """
    'AA' -> 'AB', 'AB' -> 'AC', etc.
    If alpha is not 2 letters, fallback to 'AA'.
    """
    if not alpha or len(alpha) != 2:
        return "AA"
    prefix = alpha[0]   # e.g. 'A'
    last_char = alpha[1]  # e.g. 'A'
    next_char = chr(ord(last_char) + 1)  # => 'B'
    return prefix + next_char

def next_db_revision(old_db_rev: str, old_track_count: int, new_track_count: int) -> str:
    """
    If the track count changed => numeric portion becomes new_track_count, alpha resets to 'AA'.
    Else => keep numeric portion, alpha++.
    Timestamp can be updated to the current time in both cases.
    """
    # Parse old_db_rev => "YYYY-MM-DDTHH:MM:SSZ.000000002.AA"
    parts = old_db_rev.split('.')
    # Typically => [ "2025-01-25T19:36:25Z", "000000002", "AA" ]
    if len(parts) < 3:
        # Fallback: generate a brand new style from scratch.
        return generate_db_rev_with_count(new_track_count)  # your existing function

    old_timestamp = parts[0]  
    old_numeric   = parts[1]  # e.g. "000000002"
    old_alpha     = parts[2]  # e.g. "AA"

    # Re-check old_track_count vs new_track_count
    if new_track_count != old_track_count:
        # => numeric portion changes to new_track_count, alpha resets
        numeric_str = str(new_track_count).rjust(9, "0")
        alpha_str = "AA"
    else:
        # => same numeric portion, alpha++
        numeric_str = old_numeric
        alpha_str = bump_db_alpha_part(old_alpha)

    # Possibly refresh the date/time:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(tz=ZoneInfo("UTC"))
    new_timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    return f"{new_timestamp}.{numeric_str}.{alpha_str}"


###############################################################################
# Desired Tags (Canonical Version)
###############################################################################

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
    "rtng",  # Explicit/Clean advisory rating

    # MusicBrainz:
    "----:com.apple.iTunes:MusicBrainz Artist Id",
    "----:com.apple.iTunes:MusicBrainz Track Id",
    "----:com.apple.iTunes:MusicBrainz Release Group Id",
    #"----:com.apple.iTunes:MusicBrainz Release Track Id",
    #"----:com.apple.iTunes:MusicBrainz Album Id",
    #"----:com.apple.iTunes:MusicBrainz Album Artist Id",
    #"----:com.apple.iTunes:MusicBrainz Album Status",
    #"----:com.apple.iTunes:MusicBrainz Album Type",
    #"----:com.apple.iTunes:MusicBrainz Album Release Country",

    # Apple/iTunes:
    "----:com.apple.iTunes:iTunSMPB",
    "----:com.apple.iTunes:iTunNORM",

    # Original date fields that you want to keep:
    # "originalyear",
    # "originaldate",
    # "----:com.apple.iTunes:originalyear",
    # "----:com.apple.iTunes:originaldate",

    # Some extra MB or random ID tags
    #"----:com.apple.iTunes:ISRC",
    #"----:com.apple.iTunes:ASIN",
    #"----:com.apple.iTunes:ARTISTS",
    #"----:com.apple.iTunes:LABEL",
    #"----:com.apple.iTunes:MEDIA",

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

# Map lower -> canonical
desired_tags_map = {t.lower(): t for t in DESIRED_TAGS}

# Define FLAC -> MP4 tag mapping:
FLAC_TO_MP4_TAG_MAP = {
    "artist": "©ART",
    "album": "©alb",
    "title": "©nam",
    "tracknumber": "trkn",
    "tracknumbers": "trkn",
    "discnumber": "disk",
    "date": "©day",
    "year": "©day",
    # etc.
}


###############################################################################
# MusicBrainz, AcoustID, and Spotify Helper Functions
###############################################################################

def get_releases_from_release_group(release_group_mbid):
    """
    Fetch releases for the given release_group_mbid with 'releases' included.
    """
    try:
        result = musicbrainzngs.get_release_group_by_id(
            release_group_mbid,
            includes=["releases"]
        )
        return result["release-group"]["release-list"]
    except musicbrainzngs.WebServiceError as e:
        logger.info(f"  Error fetching releases for release group {release_group_mbid}: {e}")
        return []

def get_recordings_from_release(release_mbid):
    """
    Fetch tracks/recordings within a given release.
    """
    try:
        result = musicbrainzngs.get_release_by_id(
            release_mbid,
            includes=["recordings"]
        )
        recordings = []
        for medium in result["release"]["medium-list"]:
            for track in medium.get("track-list", []):
                recordings.append(track)
        return recordings
    except musicbrainzngs.WebServiceError as e:
        logger.info(f"  Error fetching recordings for release {release_mbid}: {e}")
        return []
    except IndexError:
        logger.info(f"  No recordings found for release {release_mbid}.")
        return []

def find_recording_mbid(track_title, recordings):
    """
    Finds the MBID of a recording matching track_title in a list of recordings.
    """
    for recording in recordings:
        title = recording.get("recording", {}).get("title", "").lower()
        if title == track_title.lower():
            return recording["recording"]["id"]
    return None

def get_spotify_access_token():
    """Obtain a Spotify access token via Client Credentials Flow with retry logic."""
    logger.debug("Entering get_spotify_access_token()")
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.info("WARNING: No SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET set.")
        return None

    url = "https://accounts.spotify.com/api/token"
    auth = (SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}

    wait_times = [60, 300, 900]  # in seconds: 1 min, 5 min, 15 min
    fallback_wait = 1800       # 30 minutes for subsequent attempts
    attempt = 0

    while True:
        attempt += 1
        try:
            resp = requests.post(url, data=data, auth=auth, timeout=10)
            logger.debug(f"Spotify token request => status code {resp.status_code}")
            if resp.status_code == 200:
                token_data = resp.json()
                logger.debug("Successfully obtained Spotify token.")
                return token_data["access_token"]
            else:
                logger.info(f"Spotify token request failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.info(f"Error obtaining Spotify token: {e}")

        if attempt <= len(wait_times):
            wait_time = wait_times[attempt - 1]
        else:
            wait_time = fallback_wait
        logger.warning(f"Retrying Spotify token request in {wait_time/60:.1f} minutes (attempt {attempt})...")
        time.sleep(wait_time)


###############################################################################
# Librosa-based Feature Extraction
###############################################################################

def extract_librosa_features(audio_path: str) -> dict:
    """
    Analyze `audio_path` using librosa. Returns a dict of
    { "feature_valence": float, "feature_tempo": float, ... }.

    **Heads up**: Librosa doesn't directly provide some
    Spotify-like features (valence, danceability, etc.),
    so we do either naive or placeholder approaches:

    - tempo: from `librosa.beat.beat_track`.
    - time_signature: we'll assume 4 or do a naive guess from the beat pattern.
    - loudness: approximate from RMS in dB.
    - key, mode: naive approach from chroma or HPCP. We'll do a
      simplified guess for demonstration.
    - speechiness, acousticness, instrumentalness, valence, liveness, energy:
      placeholders or naive calculations for demonstration.

    Future refinements can replace placeholders with more robust analysis.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # (A) Tempo
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # (B) Key + mode
        # A naive approach is to use `librosa.feature.chroma_stft` and pick the peak.
        # Then guess major/minor from some simple ratio. This is not exact, but a demonstration.
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        avg_chroma = np.mean(chroma, axis=1)  # average across time
        key_idx = int(np.argmax(avg_chroma))  # 0..11
        # We'll guess "major" if the chroma ratio is > 0.5, else minor,
        # purely as a placeholder:
        major_score = np.mean(chroma[chroma > 0.7]) if np.any(chroma > 0.7) else 0.0
        mode_flag = 1 if major_score > 0.5 else 0

        # (C) Loudness approx from RMS
        # We'll do an overall root-mean-square, then convert to dB
        rms = np.sqrt(np.mean(y**2))
        loudness_db = 20 * np.log10(rms + 1e-9)  # avoid log(0)

        # (D) "Danceability" placeholder
        # There's no direct Spotify-like danceability measure in librosa,
        # so let's do a naive approach: ratio of strong beat intervals
        # to total, or just random for a placeholder:
        danceability = random.random()

        # (E) The other fields: speechiness, acousticness, instrumentalness, valence, liveness, energy
        # We'll do naive placeholders or short computations:

        # energy: let's approximate with the RMS again, but scaled 0..1
        # If we define "max" ~ -6 dB, "min" ~ -60 dB:
        normal_loudness = (loudness_db + 60) / 54  # so -60 dB => 0, -6 dB => ~1
        energy_val = np.clip(normal_loudness, 0.0, 1.0)

        # placeholders:
        time_signature = 0.5
        speechiness_val = 0.5
        acousticness_val = 0.5
        instrumentalness_val = 0.5
        valence_val = 0.5
        liveness_val = 0.5

        return {
            "feature_tempo": float(tempo),
            "feature_time_signature": float(time_signature),
            "feature_loudness": float(loudness_db),
            "feature_danceability": float(danceability),
            "feature_mode": float(mode_flag),
            "feature_key": float(key_idx),
            "feature_energy": float(energy_val),
            "feature_speechiness": float(speechiness_val),
            "feature_acousticness": float(acousticness_val),
            "feature_instrumentalness": float(instrumentalness_val),
            "feature_valence": float(valence_val),
            "feature_liveness": float(liveness_val),
        }

    except Exception as e:
        logger.warning(f"[librosa] Failed to extract features for '{audio_path}': {e}")
        return {}


###############################################################################
# Spawnre (Genre) Logic
###############################################################################

# 1) genre_mapping & genre_synonyms (from dic_spawnre.py)

# This dictionary will track how many times each artist has used each genre,
# so a single "spawnre_tag" can be picked for the artist at the end.
artist_genre_count = defaultdict(lambda: defaultdict(int))

# This dictionary will store the final spawnre_tag for each artist
artist_spawnre_tags = {}


# 2) Normalization & combination logic

def normalize_genre(
    genre: str,
    genre_mapping: dict,
    subgenre_to_parent: dict,
    genre_synonyms: dict
) -> str:
    """
    1. Apply synonyms (e.g. 'hip hop' -> 'hip-hop').
    2. Find matching code by exact name in genre_mapping.
    3. Stop. (No roll-up to Parent, no usage of Related).
    4. Return the subgenre's name, or '' if not recognized.
    """

    # Trim & lower
    g = genre.strip().lower()

    # (A) Apply synonyms
    if g in genre_synonyms:
        g = genre_synonyms[g]

    # (B) Find code by name
    matched_code = None
    for code, details in genre_mapping.items():
        if details["Genre"].lower() == g:
            matched_code = code
            break

    # If not found, return empty => unrecognized
    if not matched_code:
        return ""

    final_genre_name = genre_mapping[matched_code]["Genre"]
    return final_genre_name.lower()


def normalize_spawnre(tags):
    """
    Ensures that the '----:com.apple.iTunes:spawnre' field in the tags dictionary
    is stored as a list of strings. If it is a comma‐separated string or bytes,
    it will split and strip it.
    """
    key = "----:com.apple.iTunes:spawnre"
    if key in tags:
        value = tags[key]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            # Split by commas and strip each element; filter out any empty strings
            tags[key] = [s.strip() for s in value.split(",") if s.strip()]
        elif isinstance(value, list):
            # Ensure each element is a string (decode bytes if necessary)
            new_list = []
            for item in value:
                if isinstance(item, bytes):
                    new_list.append(item.decode("utf-8", errors="replace").strip())
                else:
                    new_list.append(str(item).strip())
            tags[key] = new_list


def combine_and_prioritize_genres_refined(
    embedded_genre: str,
    last_fm_genres: list,
    spotify_genres: list,
    musicbrainz_genres: list,
    genre_mapping: dict,
    subgenre_to_parent: dict,
    genre_synonyms: dict,
    artist_name: str
) -> list:
    """
    Combine multiple sources of genres (embedded, Last.fm, Spotify, MusicBrainz),
    normalize them, count them, return up to 5 recognized genres.
    """
    genre_count = defaultdict(int)

    # Build one combined list
    all_genres = []
    if embedded_genre:
        all_genres.append(embedded_genre)
    all_genres.extend(last_fm_genres)
    all_genres.extend(spotify_genres)
    all_genres.extend(musicbrainz_genres)

    for raw_g in all_genres:
        # Use our updated normalize_genre
        norm = normalize_genre(raw_g, genre_mapping, subgenre_to_parent, genre_synonyms)
        if norm:
            genre_count[norm] += 1
            # Also update the artist-based counter for spawnre_tag
            artist_genre_count[artist_name.lower()][norm] += 1

    # Sort by frequency (descending)
    sorted_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
    final_genres = [g for (g, cnt) in sorted_genres][:5]
    return final_genres


def find_closest_genre_matches(genres: list, genre_mapping: dict) -> (list, str):
    """
    For each final genre, find its code's Hex and build spawnre_hex (max 10 chars).
    """
    spawnre_hex = "x"
    matched_genres = []

    for genre in genres:
        # Find the code with this final name
        # Note: the final name is assumed to definitely be in genre_mapping
        # because it has been normalized already. So just do a quick match:
        for code, details in genre_mapping.items():
            if details["Genre"].lower() == genre.lower():
                hex_str = details["Hex"].replace("0x", "")
                spawnre_hex += hex_str
                matched_genres.append(details["Genre"])  # official name
                break

        if len(spawnre_hex) >= 11:
            break

    spawnre_hex = spawnre_hex[:11]      # 11 digits for "x##########", i.e. 5 genres
    return matched_genres, spawnre_hex


# 3) Process Spawnre

def process_spawnre(
    file_path: str,
    artist_name: str,
    track_title: str,
    embedded_genre: str,
    last_fm_genres: list,
    spotify_genres: list,
    musicbrainz_genres: list,
    temp_tags: dict
):
    """
    1) Combine & refine genres from multiple sources.
    2) Compute spawnre (multi-genre, up to 5), spawnre_hex (coded version),
       and then determine the top genre.
    3) Update temp_tags with these values.
    """
    combined = combine_and_prioritize_genres_refined(
        embedded_genre=embedded_genre,
        last_fm_genres=last_fm_genres,
        spotify_genres=spotify_genres,
        musicbrainz_genres=musicbrainz_genres,
        genre_mapping=genre_mapping,
        subgenre_to_parent=subgenre_to_parent,
        genre_synonyms=genre_synonyms,
        artist_name=artist_name
    )
    # combined is a list of up to 5 normalized genre strings.
    final_genres, spawnre_hex = find_closest_genre_matches(combined, genre_mapping)
    
    # Keep the multi-genre assignment as comma-separated values:
    spawnre = ", ".join(final_genres)
    # And then choose a single top genre (for example, the first one)
    top_genre = final_genres[0] if final_genres else ""

    logger.info("")
    logger.info(f"  Artist: {artist_name}  |  Track: {track_title}")
    logger.info(f"  => Combined genres: {combined}")
    logger.info(f"  => spawnre (multi): '{spawnre}'")
    logger.info(f"  => spawnre_hex: '{spawnre_hex}'")
    logger.info(f"  => Top genre (©gen): '{top_genre}'\n")

    # Update custom tags with the multi-genre assignment:
    temp_tags["----:com.apple.iTunes:spawnre"] = spawnre.encode("utf-8")
    temp_tags["----:com.apple.iTunes:spawnre_hex"] = spawnre_hex.encode("utf-8")
    temp_tags["©gen"] = top_genre

# 4) Artist-level spawnre_tag
#    Once all tracks have been processed, call this function to pick
#    each artist’s single “most frequent” genre overall.

def finalize_spawnre_tags():
    """
    After processing all tracks, pick each artist's single top genre overall.
    Print them out for demonstration.
    """
    for artist_lower, genres_dict in artist_genre_count.items():
        if not genres_dict:
            artist_spawnre_tags[artist_lower] = ""
            continue
        # e.g., { "house": 3, "rock": 1 }
        most_frequent = max(genres_dict, key=genres_dict.get)
        artist_spawnre_tags[artist_lower] = most_frequent

    logger.info("=> spawnre_tag:")
    for artist_lower, tag in artist_spawnre_tags.items():
        logger.info(f"  {artist_lower} -> {tag}")
    logger.info("")


###################################
# Multi-source genre calls
###################################

# Optional caches to avoid repeated requests for the same artist
spotify_genre_cache = {}
musicbrainz_genre_cache = {}

def fetch_genre_lastfm(artist: str, track: str, api_key: str, retries: int = 3, delay: int = 5, timeout: int = 10) -> list:
    """
    Fetch Last.fm genres (tags) for a given artist and track.
    Similar to your anal_M3U.py snippet, with retries + delay.
    Returns a list of lower-case genre strings or [] if none found.
    """
    if not api_key:
        logger.warning("Last.fm API key not provided.")
        return []

    url = (
        "https://ws.audioscrobbler.com/2.0/"
        f"?method=track.getInfo&api_key={api_key}"
        f"&artist={requests.utils.quote(artist)}"
        f"&track={requests.utils.quote(track)}"
        "&format=json"
    )

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Fetching Last.fm genres for '{artist} - {track}' (Attempt {attempt})")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            # e.g. data["track"]["toptags"]["tag"] = [ {"name": "pop"}, {"name": "dance"} ]
            toptags = data.get("track", {}).get("toptags", {})
            tag_list = toptags.get("tag", [])
            if tag_list:
                genres = [t["name"].lower() for t in tag_list]
                logger.debug(f"Last.FM genres extracted: {genres}")
                return genres
            else:
                logger.warning(f"No genres found for {artist} - {track} on Last.fm.")
                return []

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error while fetching Last.fm genres: {e}")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout occurred. Retrying ({attempt}/{retries})...")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            break

        time.sleep(delay)

    logger.error(f"Failed to fetch Last.fm genres for '{artist} - {track}' after {retries} attempts.")
    return []


def get_spotify_genres(artist_name: str, sp: spotipy.Spotify, retries: int = 3, delay: int = 5) -> list:
    """
    Fetch up to 5 Spotify genres for the given artist name.
    Uses sp (a spotipy.Spotify instance) to do a search, then returns the artist's 'genres' field.
    Caches results in spotify_genre_cache to avoid repeated calls.
    """
    # If fetched before, return cached result
    if artist_name in spotify_genre_cache:
        logger.debug(f"Spotify genres for '{artist_name}' fetched from cache.")
        return spotify_genre_cache[artist_name]

    if not sp:
        return []

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Fetching Spotify genres for '{artist_name}' (Attempt {attempt})")
            results = sp.search(q=f"artist:{artist_name}", type='artist', limit=1)
            items = results.get("artists", {}).get("items", [])

            if items:
                artist_obj = items[0]  # first match
                genres = [g.lower() for g in artist_obj.get("genres", [])[:5]]
                spotify_genre_cache[artist_name] = genres
                logger.debug(f"Spotify genres extracted: {genres}")
                return genres
            else:
                logger.warning(f"No Spotify genres found for artist: {artist_name}")
                return []

        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                # Rate limit
                retry_after = int(e.headers.get('Retry-After', 60))
                jitter = random.uniform(0, 1)
                wait_time = retry_after + jitter
                logger.warning(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
                time.sleep(wait_time)
            else:
                logger.error(f"Spotify API error: {e}. Retrying ({attempt}/{retries})...")
                time.sleep(delay)
        except requests.exceptions.RequestException as e:
            logger.error(f"Spotify request failed: {e}. Retrying ({attempt}/{retries})...")
            time.sleep(delay)

    logger.error(f"Failed to fetch Spotify genres for '{artist_name}' after {retries} attempts.")
    return []


def get_musicbrainz_genres(artist_name: str) -> list:
    """
    Fetch up to 5 MusicBrainz genres (tags) for the given artist name.
    If found, caches them in musicbrainz_genre_cache.
    """
    if artist_name in musicbrainz_genre_cache:
        logger.debug(f"MusicBrainz genres for '{artist_name}' fetched from cache.")
        return musicbrainz_genre_cache[artist_name]

    logger.info(f"Fetching MusicBrainz genres for '{artist_name}'")

    try:
        # e.g. musicbrainzngs.search_artists(artist=..., limit=1)
        result = musicbrainzngs.search_artists(artist=artist_name, limit=1)
        if 'artist-list' in result and result['artist-list']:
            artist_data = result['artist-list'][0]
            # e.g. "tags": { "tag": [ {"name": "pop"}, {"name":"dance"} ] }
            tags = artist_data.get('tag-list', [])
            genre_names = [t["name"].lower() for t in tags]
            final = genre_names[:5]
            musicbrainz_genre_cache[artist_name] = final
            logger.debug(f"MusicBrainz genres extracted: {final}")
            return final
        else:
            logger.warning(f"No genre tags found for artist: {artist_name} on MusicBrainz.")
    except musicbrainzngs.WebServiceError as e:
        logger.error(f"MusicBrainz API request failed: {e}")

    return []


###############################################################################
# MusicBrainz Search Utility
###############################################################################

def search_and_update_mbid(
    tag_name,
    query,
    search_function,
    mbid_field,
    temp_tags,
    entity_type=None
):
    """
    Searches for an MBID using the provided MusicBrainz search function and updates temp_tags[tag_name] if found.

    Priority order for release-group selection:
      1) correct ARTIST match
      2) highest ext:score
      3) album > ep > single
      4) earliest date
    while excluding 'live','remix','compilation','video' secondaries.

    Args:
        tag_name (str): The tag name to update with the MBID (e.g. '----:com.apple.iTunes:MusicBrainz Artist Id')
        query (dict): The search query parameters (e.g. {"artist": "ABBA"}).
        search_function (function): The MusicBrainz search function, e.g. musicbrainzngs.search_artists
        mbid_field (str): The entity being searched, e.g. 'artist', 'release-group', etc.
        temp_tags (dict): The dictionary of temporary tags to update.
        entity_type (str, optional): e.g. 'release-group' for special filtering. Defaults to None.

    Returns:
        bool: True if MBID was found and updated, False otherwise.
    """
    try:
        logger.info(f"Searching for missing {mbid_field} MBID using query: {query}")
        results = search_function(**query, limit=10)
        candidates = results.get(f"{mbid_field}-list", [])

        logger.debug(f"  Found {len(candidates)} initial {mbid_field} candidates.")
        logger.debug(f"  Candidates: {candidates}")

        if entity_type != "release-group":
            # If not searching release-groups, just pick the first result if any.
            # e.g. for artist MBID searches.
            if candidates:
                best_candidate = candidates[0]
                mbid = best_candidate["id"]
                logger.info(f"  Selected MBID: {mbid}\n")
                temp_tags[tag_name] = mbid
                return True
            else:
                logger.info(f"  No results found for {mbid_field}.")
                return False

        # If entity_type == "release-group", refine logic
        # Filter out non-album/EP/single and undesired secondaries,
        # then prefer the correct ARTIST, highest ext:score, album>ep>single, earliest date.

        # The user’s typed artist in the query: e.g. query["artist"]
        # Use it to confirm a correct artist-credit match
        typed_artist = (query.get("artist") or "").strip().lower()

        valid_primary_types = ["album", "ep", "single"]  # in ascending order of priority
        # Define a small map to assign type-priority, e.g. album=0, ep=1, single=2, else=9
        def get_type_priority(prim_type):
            prim_type_lower = prim_type.lower().strip() if prim_type else ""
            if prim_type_lower == "album":
                return 0
            elif prim_type_lower == "ep":
                return 1
            elif prim_type_lower == "single":
                return 2
            return 9  # if we want to allow fallback beyond single

        filtered_candidates = []

        for c in candidates:
            prim_type = (c.get("primary-type") or "").lower().strip()
            # This might be e.g. "album","ep","single", or something else like "other","broadcast"
            secondaries = [s.lower().strip() for s in c.get("secondary-type-list", [])]
            ext_score_str = c.get("ext:score", "0")
            try:
                ext_score = int(ext_score_str)
            except ValueError:
                ext_score = 0

            # Exclude if secondaries contain live,remix,compilation,video
            if any(s in ["live","remix","compilation","video"] for s in secondaries):
                logger.debug(f"Excluding {c.get('id')} => secondary-type in {secondaries}")
                continue

            # Must be in album/ep/single to pass
            if prim_type not in valid_primary_types:
                logger.debug(f"Excluding {c.get('id')} => primary-type={prim_type} not in {valid_primary_types}")
                continue

            # Ensure the artist-credit matches typed_artist
            # Can do a straightforward check if typed_artist is in the artist-credit phrase.
            # For more robust matching, do a difflib ratio or remove diacritics.
            # For now, just do a simple case-insensitive containment check:
            ac_phrase = (c.get("artist-credit-phrase") or "").lower()
            if typed_artist and typed_artist not in ac_phrase:
                logger.debug(f"Excluding {c.get('id')} => artist-credit-phrase='{ac_phrase}' does not match typed='{typed_artist}'")
                continue

            # If it passed all filters, keep it
            c["_ext_score"] = ext_score
            c["_type_priority"] = get_type_priority(prim_type)

            filtered_candidates.append(c)

        # Now sort them:
        #   1) descending ext_score
        #   2) ascending type_priority
        #   3) ascending first-release-date
        # For a stable sort with ext_score descending, use -c["_ext_score"] as the first key
        def sort_key(cand):
            neg_score = -cand["_ext_score"]
            t_priority = cand["_type_priority"]
            date_str = cand.get("first-release-date","9999-99-99")
            return (neg_score, t_priority, date_str)

        filtered_candidates.sort(key=sort_key)

        if not filtered_candidates:
            logger.info(f"  No suitable release-group result after filtering for correct artist & type.")
            return False

        best_candidate = filtered_candidates[0]
        mbid = best_candidate["id"]
        logger.info(f"  => Selected MBID: {mbid}")
        logger.info(f"     primary-type: {best_candidate.get('primary-type')}, "
                    f"release-date={best_candidate.get('first-release-date')}, "
                    f"ext:score={best_candidate.get('ext:score')}, "
                    f"artist-credit-phrase={best_candidate.get('artist-credit-phrase')}\n")

        temp_tags[tag_name] = mbid
        return True

    except Exception as e:
        logger.info(f"Error searching for {mbid_field}: {e}")
        return False

###############################################################################
# Format Detection & Repackaging
###############################################################################

def detect_audio_format(ffprobe_output):
    """
    Search for 'Audio: <codec>' in ffprobe output.
    """
    match = re.search(r"Audio:\s+([^\s,]+)", ffprobe_output, re.IGNORECASE)
    if match:
        codec = match.group(1).lower()
        if "alac" in codec:
            return "ALAC"
        if "aac" in codec or "mp4a" in codec:
            return "AAC"
        if "flac" in codec:
            return "FLAC"
    return "Unknown"

def convert_flac_to_alac(input_audio, target_file):
    """
    Convert a FLAC file to ALAC (M4A container) using ffmpeg.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_audio,
            "-vn",
            "-map_metadata", "0",
            "-c:a", "alac",
            "-movflags", "+faststart",
            "-brand", "M4A ",
            target_file
        ]
        logger.debug("Running FLAC->ALAC: %s", " ".join(shlex.quote(c) for c in cmd))
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        if proc.returncode == 0:
            logger.info(f"Successfully converted FLAC to ALAC: {target_file}")
        else:
            logger.info(f"Error converting FLAC to ALAC. ffmpeg stderr:\n{proc.stderr}")
            proc.check_returncode()

    except subprocess.CalledProcessError as e:
        logger.info(f"Error converting FLAC to ALAC: {e}")
        raise
    except Exception as e:
        logger.info(f"Unexpected error converting FLAC to ALAC: {e}")
        raise

def repackage_alac_file(input_audio, target_file):
    """
    Repackage an ALAC file using ffmpeg without re-encoding.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_audio,
            "-map", "0:a",    # only audio
            "-c", "copy",
            "-movflags", "+faststart",
            "-brand", "M4A ",
            target_file
        ]
        logger.debug("Running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        if proc.returncode == 0:
            logger.info(f"Successfully repackaged ALAC file: {target_file}")
        else:
            logger.info(f"Error during ALAC repackaging. ffmpeg stderr:\n{proc.stderr}")
            proc.check_returncode()
    except subprocess.CalledProcessError as e:
        logger.info(f"Error during ALAC repackaging: {e}")
        raise
    except Exception as e:
        logger.info(f"Unexpected error during ALAC repackaging: {e}")
        raise


def clean_or_repackage_aac(input_audio, target_file):
    """
    Clean/repackage an AAC file using MP4Box
    """
    tmp_aac = "audio.aac"
    try:
        cmd_raw = ["MP4Box", "-raw", "1", input_audio, "-out", tmp_aac]
        logger.debug("Running: %s", " ".join(cmd_raw))
        subprocess.run(cmd_raw, check=True)

        cmd_add = ["MP4Box", "-add", tmp_aac, "-new", target_file]
        logger.debug("Running: %s", " ".join(cmd_add))
        subprocess.run(cmd_add, check=True)

        os.remove(tmp_aac)
        logger.info(f"Successfully cleaned/repackaged AAC file: {target_file}")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error during AAC repackaging: {e}")
        if os.path.exists(tmp_aac):
            os.remove(tmp_aac)
        raise
    except Exception as e:
        logger.info(f"Unexpected error during AAC repackaging: {e}")
        if os.path.exists(tmp_aac):
            os.remove(tmp_aac)
        raise

def sanitize_for_directory(name: str, max_len: int = 50) -> str:
    """
    Replaces or removes problematic characters (and truncates if needed) so 'name' can be used as a folder.
    """
    # Normalize the string to NFKD and then encode to ASCII, ignoring non-ASCII characters.
    normalized = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    # Replace problematic punctuation with underscores, using the normalized text.
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', normalized.strip())
    # Truncate if too long
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].rstrip("_- ")
    return sanitized or "Unknown"


###############################################################################
# Tag Extraction Helper Functions
###############################################################################

def extract_year_from_tag_value(raw_value):
    """
    Return the first 4-digit year if found, else None.
    """
    if isinstance(raw_value, list) and raw_value:
        raw_value = raw_value[0]
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="replace")
    elif isinstance(raw_value, MP4FreeForm):
        try:
            raw_value = raw_value.data.decode("utf-8", errors="replace")
        except:
            raw_value = ""

    full_str = str(raw_value).strip()
    match = re.match(r"(\d{4})", full_str)
    return match.group(1) if match else None


def fill_missing_basic_tags(mp4_file, file_path):
    """Fill missing tags ©nam, ©alb, and ©ART/aART from file path context."""
    tags = mp4_file.tags

    # Get path components
    filename = os.path.splitext(os.path.basename(file_path))[0]
    album_folder = os.path.basename(os.path.dirname(file_path))
    artist_folder = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

    if '©nam' not in tags:
        tags['©nam'] = [filename]

    if '©alb' not in tags:
        tags['©alb'] = [album_folder]

    if '©ART' not in tags:
        tags['©ART'] = [artist_folder]
    if 'aART' not in tags:
        tags['aART'] = [artist_folder]


def extract_desired_tags(file_path):
    """
    1. Read all tags from `file_path`.
    2. If tag.lower() is in desired_tags_map, store it under canonical form.
    3. Regardless of whether it's in DESIRED_TAGS, if the tag name includes
       'date', 'day', or 'year', parse out the first 4-digit year (if any).
    4. After reading all tags, pick the oldest year from all matched date-like
       tags, set it in '©day', and remove other date-like tags from final output.
    """

    # Update any missing basic tags using the MP4 object.
    try:
        mp4 = MP4(file_path)
        fill_missing_basic_tags(mp4, file_path)
        mp4.save()
    except Exception as e:
        logger.warning(f"  WARNING: Could not process MP4 tags for fallback in '{file_path}': {e}")

    # Re-read the file so that the updated tags are loaded.
    audio = MutagenFile(file_path)
    if not audio or not audio.tags:
        logger.info(f"  WARNING: No tags found in '{file_path}'")
        return {}

    logger.debug("All tags found by Mutagen:")
    for k, v in audio.tags.items():
        # Skip logging 'covr' key
        if k.lower() == 'covr':
            continue
        logger.debug(f"  {repr(k)}: {v}")
    logger.debug("")

    temp_tags = {}
    all_years_found = []

    date_tags_to_remove = {
        "date",
        "year",
        "originalyear",
        "originaldate",
        "----:com.apple.itunes:originalyear",
        "----:com.apple.itunes:originaldate",
        "----:com.apple.iTunes:originalyear",
        "----:com.apple.iTunes:originaldate",
    }

    for file_tag, raw_value in audio.tags.items():
        tag_lower = file_tag.lower()

        # 1) If it's a known FLAC tag => map it
        if tag_lower in FLAC_TO_MP4_TAG_MAP:
            mapped_tag = FLAC_TO_MP4_TAG_MAP[tag_lower]

            # Special handling for tracknumber/discnumber
            if mapped_tag in ["trkn", "disk"]:
                # parse out main vs total
                raw_str = raw_value[0] if isinstance(raw_value, list) else str(raw_value)
                main, total = 0, 0
                if "/" in raw_str:
                    left, right = raw_str.split("/", 1)
                    if left.isdigit():  main = int(left)
                    if right.isdigit(): total = int(right)
                else:
                    # e.g. '01'
                    if raw_str.isdigit():
                        main = int(raw_str)
                # Store as the standard MP4 (main, total) in a list
                temp_tags[mapped_tag] = [(main, total)]
            else:
                temp_tags[mapped_tag] = raw_value

        # 2) If it’s already an iTunes MP4 tag
        elif tag_lower in desired_tags_map:
            canonical = desired_tags_map[tag_lower]
            temp_tags[canonical] = raw_value

        # 3) Collect possible date/year values for picking the oldest
        if any(x in tag_lower for x in ["date", "year", "day"]):
            # parse out e.g. '1999'
            parsed_year = extract_year_from_tag_value(raw_value)
            if parsed_year:
                all_years_found.append(parsed_year)

    # 4) If we found multiple years, keep oldest
    if all_years_found:
        oldest = min(all_years_found)
        temp_tags["©day"] = oldest

    return temp_tags


###############################################################################
# Tag Rewriting (Using Canonical Names)
###############################################################################

MULTI_VALUE_FREEFORM = {
    "----:com.apple.iTunes:spawnre",
    # You can add more tags here if you want them stored as multi-value.
}

def rewrite_tags(file_path, tags):
    """
    Clear existing tags, then rewrite.
    Certain freeform tags (in MULTI_VALUE_FREEFORM) get stored as multi-value lists,
    but most are forced into a single string.
    """
    audio = MutagenFile(file_path)
    if not audio:
        logger.info(f"  WARNING: Unable to load file for tagging: '{file_path}'")
        return

    # Clear all existing tags to rewrite from scratch
    audio.tags.clear()


    for tag, value in tags.items():
        if tag.startswith("----:com.apple.iTunes:"):
            # Special handling for MusicBrainz freeform tags:
            if "MusicBrainz" in tag:
                # Get a single string value (similar to branch B below)
                if isinstance(value, list):
                    val = value[0] if value else ""
                else:
                    val = value

                if isinstance(val, bytes):
                    val_str = val.decode("utf-8", errors="replace")
                else:
                    val_str = str(val)

                # Create a freeform atom with the proper structure.
                freeform = MP4FreeForm(val_str.encode("utf-8", errors="replace"), dataformat=0)
                freeform.mean = "com.apple.iTunes"
                # Set the freeform name to the part after "----:com.apple.iTunes:"
                freeform.name = tag.replace("----:com.apple.iTunes:", "")
                audio.tags[tag] = [freeform]

            # (A) Multi-value freeform tags:
            elif tag in MULTI_VALUE_FREEFORM:
                # Convert `value` to a list of strings
                if isinstance(value, list):

                    # We might already have something like [b'pop', b'rock', b'electronic']
                    # Decode them into Python strings
                    str_list = []
                    for item in value:
                        if isinstance(item, bytes):
                            str_list.append(item.decode("utf-8", errors="replace"))
                        else:
                            str_list.append(str(item))
                else:
                    # It's a single string/bytes; decode if needed
                    if isinstance(value, bytes):
                        decoded = value.decode("utf-8", errors="replace")
                    else:
                        decoded = str(value)

                    # Split by comma if desired
                    str_list = [v.strip() for v in decoded.split(",")]

                # Step 2: Store it as a list of bytes
                # e.g. ["pop", "rock"] => [b"pop", b"rock"]
                byte_list = [s.encode("utf-8", errors="replace") for s in str_list]

                audio.tags[tag] = byte_list

            # (B) Otherwise, store as a single string
            else:
                if isinstance(value, list):
                    val = value[0] if value else ""
                else:
                    val = value

                if isinstance(val, bytes):
                    val_str = val.decode("utf-8", errors="replace")
                else:
                    val_str = str(val)

                # Force single-element list of bytes
                audio.tags[tag] = [val_str.encode("utf-8", errors="replace")]

        else:
            # For non-freeform tags like "©day", "©nam", etc.
            audio.tags[tag] = value

    audio.save()
    logger.info(f"Updated tags successfully written to: {file_path}")


###############################################################################
# Helper for D-TT-title.m4a
###############################################################################
def sanitize_title_for_filename(title: str, max_len: int = 60) -> str:
    """
    Replaces or removes problematic characters in the title so it can be used in filenames.
    Also truncates if the resulting name is still too long.
    """
    # Replace / \ : * ? " < > | with underscores
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', title)

    # If it's extremely long, truncate
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].rstrip("_- ")  # remove trailing underscores/spaces
    return sanitized


def build_d_tt_title_filename(
    disc_num: int,
    track_num: int,
    track_title: str,
    spawn_id_str: str = None
) -> str:
    """
    Returns a string with the format:
      If spawn_id_str is present => "D-TT [SPAWN_ID] - title.m4a"
      Otherwise => "D-TT - title.m4a"

    Disc number "D" can be 1,2, etc.
    Track number "TT" is zero-padded to 2 digits (e.g. 00, 01, 02, ...).
    spawn_id_str is optional; if provided, it's included in the filename.
    """
    # default disc to 1 if None or invalid
    if not disc_num or disc_num < 1:
        disc_num = 1

    # default track to 0 if None or invalid
    if not track_num or track_num < 1:
        track_num = 0

    # build zero-padded track number
    track_str = f"{track_num:02d}"

    # sanitize title
    safe_title = sanitize_title_for_filename(track_title or "untitled")

    if spawn_id_str:
        return f"{disc_num}-{track_str} [{spawn_id_str}] - {safe_title}.m4a"
    else:
        return f"{disc_num}-{track_str} - {safe_title}.m4a"


def get_track_duration_seconds(path):
    """
    Returns the integer number of seconds for the track at `path`.
    If the file is unreadable or doesn't have a known length, returns 0.
    """
    try:
        audio = mutagen.File(path)
        if audio and audio.info:
            return int(round(audio.info.length))
    except Exception as e:
        logger.debug(f"Could not determine duration for '{path}': {e}")
    return 0


###############################################################################
# Helper for symlink creation
###############################################################################

def create_symlink_for_track(track_path: str, lib_base: str, spawn_id: str) -> None:
    """
    Create a symbolic link in the directory LIB_PATH/Spawn/aux/user/linx/ named "<spawn_id>.m4a"
    that points to the absolute path of track_path.
    """
    # Build the target directory for symlinks.
    linx_dir = os.path.join(lib_base, "Spawn", "aux", "user", "linx")
    os.makedirs(linx_dir, exist_ok=True)

    # The symlink filename is the spawn ID plus .m4a.
    symlink_path = os.path.join(linx_dir, f"{spawn_id}.m4a")

    try:
        # Remove any existing symlink/file with that name.
        if os.path.lexists(symlink_path):
            os.remove(symlink_path)
        os.symlink(os.path.abspath(track_path), symlink_path)
        logger.info(f"Created symlink: {symlink_path} -> {os.path.abspath(track_path)}")
    except Exception as e:
        logger.error(f"Error creating symlink for {track_path}: {e}")


###############################################################################
# AcoustID Using fpcalc
###############################################################################

def fingerprint_via_fpcalc(file_path):
    """
    Use external 'fpcalc' command to generate an AcoustID fingerprint + duration.
    Returns (fingerprint, duration) or (None, None) if error.

    Example 'fpcalc' output lines:
        FILE="somefile.wav"
        DURATION=123
        FINGERPRINT=abcdefghijk...
    """
    try:
        # call fpcalc
        cmd = ["fpcalc", file_path]
        out_bytes = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        out_str = out_bytes.decode("utf-8", errors="replace").strip()
        # parse lines
        fingerprint = None
        duration = None
        for line in out_str.splitlines():
            if line.startswith("FINGERPRINT="):
                fingerprint = line.split("=", 1)[1].strip()
            elif line.startswith("DURATION="):
                duration = line.split("=", 1)[1].strip()

        if fingerprint and duration:
            return (fingerprint, duration)
        return (None, None)

    except subprocess.CalledProcessError as e:
        logger.info(f"fpcalc error: {e}")
        return (None, None)
    except FileNotFoundError:
        logger.info("ERROR: fpcalc not found in PATH. Install 'chromaprint' or put it in PATH.")
        return (None, None)
    except Exception as e:
        logger.info(f"Unexpected error calling fpcalc: {e}")
        return (None, None)

def generate_and_update_acoustid(file_path, temp_tags):
    """
    1. Shell out to fpcalc => get fingerprint, duration
    2. Overwrite "----:com.apple.iTunes:Acoustid Fingerprint"
    3. Use acoustid.lookup() => find best ID => store in "----:com.apple.iTunes:Acoustid Id"
    """
    if not ACOUSTID_API_KEY:
        logger.info("Skipping AcoustID because no ACOUSTID_API_KEY is set.\n")
        return

    logger.info("Generating AcoustID fingerprint and ID via fpcalc...")

    fingerprint, duration = fingerprint_via_fpcalc(file_path)
    if not fingerprint or not duration:
        logger.info("  Could not generate fingerprint via fpcalc. Skipping AcoustID lookup.")
        return

    # Overwrite the fingerprint tag
    temp_tags["----:com.apple.iTunes:Acoustid Fingerprint"] = fingerprint

    # Now do the lookup using acoustid.lookup()
    try:
        # Cast duration to int if needed
        dur_int = int(float(duration))

        resp = acoustid.lookup(
            apikey=ACOUSTID_API_KEY,
            fingerprint=fingerprint,
            duration=dur_int,
            meta=["recordings"]
        )
        if resp["status"] == "ok" and "results" in resp and resp["results"]:
            best_score = 0.0
            best_id = None
            for r in resp["results"]:
                score = r.get("score", 0.0)
                rid = r.get("id")
                logger.debug(f"  Candidate => score={score}, id={rid}")
                if rid and score > best_score:
                    best_score = score
                    best_id = rid

            if best_id:
                temp_tags["----:com.apple.iTunes:Acoustid Id"] = best_id
                logger.info(f"  AcoustID => best match rid='{best_id}', score={best_score}\n")
            else:
                logger.info("  No suitable AcoustID candidate found in response.")
        else:
            logger.info("  No results returned from AcoustID lookup.")
    except WebServiceError as wse:
        logger.info(f"  AcoustID web service error: {wse}")
    except Exception as e:
        logger.info(f"  Unexpected error while calling acoustid.lookup: {e}")


###############################################################################
# MusicBrainz MBID Validation & Automatic Search
###############################################################################

def confirm_or_update_tags(temp_tags, file_path):
    """
    1. Show existing tags (artist, track, album, year, MBIDs).
    2. Validate existing MBIDs by comparing names if you want to keep them accurate.
    3. If any MBIDs are missing, do a search with the logic:
       - Artist MBID => search by artist name
       - Release Group MBID => search by album + artist
       - Track MBID => search within the chosen release group or do a separate search
    """
    def normalize_value(value):
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, MP4FreeForm):
            try:
                value = value.data.decode("utf-8", errors="replace")
            except:
                value = ""
        return str(value).strip() if value else "N/A"

    # Print a summary
    artist_name = normalize_value(temp_tags.get("©ART"))
    track_title = normalize_value(temp_tags.get("©nam"))
    album_title = normalize_value(temp_tags.get("©alb"))
    year_str    = normalize_value(temp_tags.get("©day"))
    artist_mbid = normalize_value(temp_tags.get("----:com.apple.iTunes:MusicBrainz Artist Id"))
    track_mbid  = normalize_value(temp_tags.get("----:com.apple.iTunes:MusicBrainz Track Id"))
    rg_mbid     = normalize_value(temp_tags.get("----:com.apple.iTunes:MusicBrainz Release Group Id"))

    logger.info("Current tags in memory:\n")
    logger.info(f"  Artist:             {artist_name}")
    logger.info(f"  Track:              {track_title}")
    logger.info(f"  Album:              {album_title}")
    logger.info(f"  Year:               {year_str}\n")

    logger.info(f"  Artist MBID:        {artist_mbid}")
    logger.info(f"  Recording MBID:     {track_mbid}")
    logger.info(f"  Release Group MBID: {rg_mbid}\n")

    # If a Spawn ID is present and a database entry exists, use its metadata
    spawn_id = temp_tags.get("----:com.apple.iTunes:spawn_ID")
    if spawn_id:
        if isinstance(spawn_id, list) and spawn_id:
            spawn_id = spawn_id[0]
        if isinstance(spawn_id, bytes):
            spawn_id = spawn_id.decode("utf-8", errors="replace")
        spawn_id = str(spawn_id).strip()
        db_entry = fetch_tags_from_db(spawn_id)
        # Define the MBID keys we care about.
        mb_keys = [
            "----:com.apple.iTunes:MusicBrainz Artist Id",
            "----:com.apple.iTunes:MusicBrainz Track Id",
            "----:com.apple.iTunes:MusicBrainz Release Group Id"
        ]
        # Only override if all MBID keys are present and valid.
        if db_entry and all(db_entry.get(key) not in [None, "", "N/A"] for key in mb_keys):
            logger.info(f"[confirm_or_update_tags] Found complete MBID data in DB for spawn_id {spawn_id}; using catalog metadata for MusicBrainz fields.")
            for key in mb_keys + [
                        "----:com.apple.iTunes:spawnre",
                        "----:com.apple.iTunes:spawnre_hex",
                        "©gen"]:
                if key in db_entry:
                    temp_tags[key] = db_entry[key]
                    logger.info(f"[confirm_or_update_tags] Set {key} to: {db_entry[key]}")
            # Proceed with AcoustID confirmation or update
            generate_and_update_acoustid(file_path, temp_tags)
            # Proceed with Spotify confirmation or update
            confirm_or_update_spotify(temp_tags, file_path)
            logger.info("\n\n                           Completed check of MBID, AcoustID, & Spotify ID. Rewriting updated tags...\n")
            rewrite_tags(file_path, temp_tags)
            return

    # (A) If artist MBID is missing, search by artist name
    if artist_mbid == "N/A" and artist_name.lower() not in ["", "n/a"]:
        search_and_update_mbid(
            "----:com.apple.iTunes:MusicBrainz Artist Id",
            {"artist": artist_name},
            musicbrainzngs.search_artists,
            "artist",
            temp_tags
        )

    # (B) If release-group MBID is missing, search by album + artist name
    rg_mbid = normalize_value(temp_tags.get("----:com.apple.iTunes:MusicBrainz Release Group Id"))
    if rg_mbid == "N/A" and album_title.lower() not in ["", "n/a"] and artist_name.lower() not in ["", "n/a"]:
        search_and_update_mbid(
            "----:com.apple.iTunes:MusicBrainz Release Group Id",
            {"releasegroup": album_title, "artist": artist_name},
            musicbrainzngs.search_release_groups,
            "release-group",
            temp_tags,
            entity_type="release-group"
        )

    # (C) If track MBID is missing, use the release-group if found
    track_mbid = normalize_value(temp_tags.get("----:com.apple.iTunes:MusicBrainz Track Id"))
    rg_mbid = normalize_value(temp_tags.get("----:com.apple.iTunes:MusicBrainz Release Group Id"))
    if track_mbid == "N/A" and rg_mbid != "N/A":
        # Attempt to pick the earliest release from the RG, then find the matching track
        recordings = []
        releases = get_releases_from_release_group(rg_mbid)
        if releases:
            # Sort by earliest date
            releases_sorted = sorted(
                releases,
                key=lambda r: r.get("date", "9999-99-99")
            )
            # Pick the earliest
            selected_release = releases_sorted[0]
            release_mbid = selected_release["id"]
            # Now fetch the recordings
            recordings = get_recordings_from_release(release_mbid)
            if recordings:
                found_mbid = find_recording_mbid(track_title, recordings)
                if found_mbid:
                    logger.info(f"  Found track MBID for '{track_title}': {found_mbid}\n")
                    temp_tags["----:com.apple.iTunes:MusicBrainz Track Id"] = found_mbid
                else:
                    logger.info(f"  Could not find a track matching '{track_title}' in earliest release.")
        else:
            logger.info(f"  No releases found within release group {rg_mbid}.")

    # Finally rewrite tags in case anything was updated
    generate_and_update_acoustid(file_path, temp_tags)

    confirm_or_update_spotify(temp_tags, file_path)

    logger.info("\n\n                           Completed check of MBID, AcoustID, & Spotify ID. Rewriting updated tags...\n")
    rewrite_tags(file_path, temp_tags)


###############################################################################
# Spotify ID Validation & Automatic Search
###############################################################################

def spotify_search_track(artist_name, track_title, token):
    """Simple search on Spotify. Returns the best track object if found, else None."""
    logger.debug(f"Entering spotify_search_track(artist='{artist_name}', track='{track_title}')")
    if not token:
        logger.debug("No token available in spotify_search_track, returning None")
        return None

    query = f"track:{track_title} artist:{artist_name}"

    # Truncate if the full query is too long
    MAX_Q_LENGTH = 250
    if len(query) > MAX_Q_LENGTH:
        # You might add an ellipsis or just slice; here's an example with "..."
        query = query[: (MAX_Q_LENGTH - 3)] + "..."

    url = "https://api.spotify.com/v1/search"
    params = {"q": query, "type": "track", "limit": 5}
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        logger.debug(f"Spotify search => status code {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            tracks = data.get("tracks", {}).get("items", [])
            logger.debug(f"Found {len(tracks)} track(s) in Spotify search.")
            if not tracks:
                return None
            return tracks[0]
        else:
            logger.info(f"Spotify track search failed: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        logger.info(f"Error searching track on Spotify: {e}")
        return None

def confirm_or_update_spotify(temp_tags, file_path):
    """
    If either 'spotify_track_ID' or 'spotify_artist_ID' is present, attempt to
    validate them individually. If either is obviously mismatched, prompt or skip.

    If there's only one ID (e.g., track ID but no artist ID), still confirm that one.

    If user chooses to override, or if both IDs are missing, do a normal Spotify search.

    Also print out the existing track/artist ID before prompting.
    """

    logger.debug("Confirming or updating Spotify IDs")

    # Check database for valid Spotify IDs before doing API lookups
    spawn_id = temp_tags.get("----:com.apple.iTunes:spawn_ID")
    if spawn_id:
        if isinstance(spawn_id, list) and spawn_id:
            spawn_id = spawn_id[0]
        if isinstance(spawn_id, bytes):
            spawn_id = spawn_id.decode("utf-8", errors="replace")
        spawn_id = str(spawn_id).strip()
        db_entry = fetch_tags_from_db(spawn_id)
        spotify_keys = [
            "----:com.apple.iTunes:spotify_artist_ID",
            "----:com.apple.iTunes:spotify_track_ID"
        ]
        # Only use database Spotify data if both keys exist and are not empty
        if db_entry and all(db_entry.get(key) not in [None, "", "N/A"] for key in spotify_keys):
            logger.info(f"[confirm_or_update_spotify] Found complete Spotify metadata in DB for spawn_id {spawn_id}; using catalog metadata.")
            for key in spotify_keys:
                temp_tags[key] = db_entry[key]
                logger.info(f"[confirm_or_update_spotify] Set {key} to: {db_entry[key]}")
            # Since database values are valid, skip further API confirmation.
            return

    track_id_key = "----:com.apple.iTunes:spotify_track_ID"
    artist_id_key = "----:com.apple.iTunes:spotify_artist_ID"

    # Extract existing IDs (may be None or partial)
    track_id = temp_tags.get(track_id_key)
    artist_id = temp_tags.get(artist_id_key)

    # Helper to unify any potential MP4FreeForm, list, or bytes => str
    def _tag_to_str(v):
        if isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        return str(v).strip() if v else ""

    artist_name  = _tag_to_str(temp_tags.get("©ART"))
    track_title  = _tag_to_str(temp_tags.get("©nam"))
    embedded_gen = _tag_to_str(temp_tags.get("©gen"))

    logger.debug(f"  Artist name = '{artist_name}', Track title = '{track_title}'")

    track_id = temp_tags.get(track_id_key)
    artist_id = temp_tags.get(artist_id_key)

    # Convert track_id/artist_id to strings if needed
    if isinstance(track_id, list) and track_id:
        track_id = track_id[0]
    if isinstance(track_id, bytes):
        track_id = track_id.decode("utf-8", errors="replace")
    track_id = str(track_id).strip() if track_id else None

    if isinstance(artist_id, list) and artist_id:
        artist_id = artist_id[0]
    if isinstance(artist_id, bytes):
        artist_id = artist_id.decode("utf-8", errors="replace")
    artist_id = str(artist_id).strip() if artist_id else None

    if not artist_name or not track_title:
        logger.info("No artist or track title found, skipping Spotify search.")
        return

    # Acquire a Spotify token
    logger.debug("Attempting to acquire Spotify access token...")
    token = get_spotify_access_token()
    if not token:
        logger.info("No Spotify token, skipping.")
        return

    # ----------------------------------------------------------------------------
    # 1) If there are values for track_id or artist_id, attempt to validate them
    #    and store a boolean if a mismatch (or partial mismatch) is found.
    # ----------------------------------------------------------------------------
    mismatch_found = False

    # (A) If track_id is present, validate that track => see if it references the correct track + (optionally) artist
    if track_id:
        logger.debug(f"validate track_id='{track_id}' vs local '{track_title}' / '{artist_name}'")
        track_valid = validate_spotify_track_id(artist_name, track_title, track_id, artist_id, token)
        if not track_valid:
            logger.info(f"Existing Spotify track ID mismatch for this track.\n"
                        f"  => Existing track_id: {track_id}\n"
                        f"  => New search will be attempted.")
            mismatch_found = True

    # (B) If there is an artist_id but no track_id, do a simpler check => fetch artist, compare name
    if artist_id and not track_id:
        logger.debug(f"validate artist_id='{artist_id}' vs local artist '{artist_name}'")
        artist_valid = validate_spotify_artist_id(artist_name, artist_id, token)
        if not artist_valid:
            logger.info(f"Existing Spotify artist ID mismatch for this artist.\n"
                        f"  => Existing artist_id: {artist_id}\n"
                        f"  => New search will be attempted.")
            mismatch_found = True

    if mismatch_found == False and (track_id or artist_id):
        logger.info("Spotify ID(s) appear correct => skipping search.")
        return

    # If there are both track & artist ID, rely primarily on track_id validation above
    # because validate_spotify_track_id can check the first artist. 
    # Optionally also do a separate artist check if you want.

    # ----------------------------------------------------------------------------
    # 2) If mismatch is found, prompt user: keep existing or re-search?
    # ----------------------------------------------------------------------------
    if mismatch_found:
        ans = get_user_input("Overwrite existing Spotify ID(s) with new search results? ([y]/n): ", default="y")
        if ans == "n":
            logger.info("User chose to keep existing (possibly mismatched) Spotify ID(s). Skipping search.")
            return
        # If user says 'y', do normal search below
    else:
        # If there is at least one ID, but no mismatch
        # skip searching if they are considered valid
        if track_id or artist_id:
            logger.info("Spotify ID(s) appear correct => skipping search.")
            return
        # Otherwise, if no IDs, continue to normal search below.

    # ----------------------------------------------------------------------------
    # 3) If this part is reached, either:
    #    - No ID was present,
    #    - OR user wants to re-search
    # => do normal search
    # ----------------------------------------------------------------------------
    logger.info(f"Searching Spotify for '{track_title}' by '{artist_name}'...")
    best_track = spotify_search_track(artist_name, track_title, token)
    if not best_track:
        logger.info("No suitable Spotify track found from search.")
        return

    found_track_id = best_track.get("id")
    primary_artist_obj = best_track["artists"][0] if best_track.get("artists") else None
    found_artist_id = primary_artist_obj["id"] if primary_artist_obj else None

    # Show user what was found by the new search
    logger.info(f"  => Proposed new Spotify Track ID => {found_track_id}")
    logger.info(f"  => Proposed new Spotify Artist ID => {found_artist_id}")

    if found_track_id:
        temp_tags[track_id_key] = found_track_id
        logger.info(f"  => Overwriting track ID with => {found_track_id}")
    if found_artist_id:
        temp_tags[artist_id_key] = found_artist_id
        logger.info(f"  => Overwriting artist ID with => {found_artist_id}")


def validate_spotify_track_id(artist_name, track_title, track_id, artist_id, token):
    """
    Attempt to fetch track_id from Spotify => confirm local track_title
    If artist_id is also present, confirm first artist matches that ID (and optionally name).
    Return True if it matches local tags, else False.
    """
    try:
        url = f"https://api.spotify.com/v1/tracks/{track_id}"
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.debug(f"track fetch error {resp.status_code}")
            return False

        data = resp.json()
        remote_title = (data.get("name") or "").lower().strip()
        remote_artists = data.get("artists", [])

        local_title = track_title.lower().strip()

        # Instead of strict equality, do a partial ratio with difflib
        ratio = SequenceMatcher(None, local_title, remote_title).ratio()
        logger.debug(f"Comparing local='{local_title}' to remote='{remote_title}' => ratio={ratio:.3f}")

        # If ratio is below some threshold (e.g. 0.80), treat it as a mismatch
        if ratio < 0.80:
            logger.debug("track mismatch => ratio < 0.80")
            return False

        # if remote_title != local_title:
        #     logger.debug(f"track mismatch => local='{local_title}', remote='{remote_title}'")
        #     return False

        # If there is an artist_id, check if remote track's first artist matches that ID
        if artist_id and remote_artists:
            local_artist_id = str(artist_id).strip()
            first_artist_id = remote_artists[0].get("id", "")
            if first_artist_id != local_artist_id:
                logger.debug(f"first artist ID mismatch => local='{local_artist_id}', remote='{first_artist_id}'")
                return False

        # Optionally also compare local artist name if you want
        # e.g. if remote_artists[0].name != artist_name => mismatch

        logger.debug("validate_spotify_track_id => track ID seems correct.")
        return True
    except Exception as e:
        logger.debug(f"validate_spotify_track_id => error {e}")
        return False


def validate_spotify_artist_id(artist_name, artist_id, token):
    """
    If there is only an artist_id, can check if the local artist_name matches
    the name from Spotify's /v1/artists/{artist_id}.
    """
    try:
        url = f"https://api.spotify.com/v1/artists/{artist_id}"
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.debug(f"artist fetch error {resp.status_code}")
            return False

        data = resp.json()
        remote_name = (data.get("name") or "").lower().strip()
        local_name = artist_name.lower().strip()
        if remote_name != local_name:
            logger.debug(f"artist mismatch => local='{local_name}', remote='{remote_name}'")
            return False

        logger.debug("validate_spotify_artist_id => artist ID matches local tags.")
        return True
    except Exception as e:
        logger.debug(f"validate_spotify_artist_id => error {e}")
        return False


###############################################################################
# ReplayGain Calculation (Album-Based)
###############################################################################

def get_embedded_tags(file_path):
    """
    Reads the embedded tags from the given MP4 file using mutagen.
    Returns a dictionary of tags.
    """
    try:
        audio = MP4(file_path)
        tags = {}
        if audio.tags:
            for key, value in audio.tags.items():
                # Normalize list values to a single value if appropriate
                if isinstance(value, list) and len(value) == 1:
                    tags[key] = value[0]
                else:
                    tags[key] = value
        # Ensure we store the file path in the tags (normalized)
        tags["file_path"] = os.path.normpath(file_path)
        logger.debug(f"Embedded tags for {file_path}: {tags}")
        return tags
    except Exception as e:
        logger.info(f"Error reading embedded tags from {file_path}: {e}")
        return {}


def update_album_rg_for_spawn_id(DB_PATH, spawn_id, album_gain, album_peak):
    """
    Searches for a record in the tracks table where the spawn_id column matches the given spawn_id.
    If found, it updates the JSON stored in tag_data by replacing the keys
      "----:com.apple.iTunes:replaygain_album_gain" and "----:com.apple.iTunes:replaygain_album_peak"
    with the new values, and bumps the metadata revision.
    """
    # Convert bytes to string if necessary.
    if isinstance(spawn_id, bytes):
        spawn_id = spawn_id.decode('utf-8')
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Look up the record by spawn_id (the database column).
        cursor.execute("SELECT tag_data FROM tracks WHERE spawn_id = ?", (spawn_id,))
        row = cursor.fetchone()
        if row:
            tag_data_json = row[0]
            try:
                tags = json.loads(tag_data_json)
            except Exception as e:
                logger.info(f"Error parsing JSON for spawn_id {spawn_id}: {e}")
                conn.close()
                return
            # Update only the album ReplayGain values (stored as lists)
            tags["----:com.apple.iTunes:replaygain_album_gain"] = [album_gain]
            tags["----:com.apple.iTunes:replaygain_album_peak"] = [album_peak]

            # Retrieve the current metadata_rev.
            old_rev = tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
            if isinstance(old_rev, list):
                old_rev = old_rev[0]
            try:
                new_rev = bump_metadata_rev(old_rev)
            except Exception as e:
                logger.warning(f"Error bumping metadata_rev: {e}")
                new_rev = old_rev
            tags["----:com.apple.iTunes:metadata_rev"] = [new_rev]
            tags["metadata_rev"] = [new_rev]

            new_json = json.dumps(tags, ensure_ascii=False, sort_keys=True)
            cursor.execute("UPDATE tracks SET tag_data = ? WHERE spawn_id = ?", (new_json, spawn_id))
            conn.commit()
            logger.info(f"Updated album ReplayGain and bumped metadata_rev for spawn_id {spawn_id} in DB.")
            conn.close()
            return new_rev
        else:
            logger.info(f"No record found in DB for spawn_id {spawn_id}.")
        conn.close()
    except Exception as e:
        logger.info(f"Error updating DB for spawn_id {spawn_id}: {e}")


def is_catalog_match(spawn_id):
    """
    Helper: returns True if the spawn_id indicates a matched catalog track.
    For example, if spawn_id does not start with "xxx", it is considered a catalog match.
    Adjust this logic as needed.
    """
    return isinstance(spawn_id, str) and not spawn_id.startswith("xxx")


def run_replaygain_on_folder(cleaned_files_map, DB_PATH, mode="admin", lib_db_path=None):
    """
    Uses `rsgain easy <folder>` to do a multi-track album scan on all cleaned files
    in that folder, ensuring previously existing tracks are also included.
    Parses the output lines to extract track-level and album-level gain/peak values,
    then updates each file's temp_tags.

    In admin mode, for each track it calls update_album_rg_for_spawn_id() to update the
    spawn_catalog.db entry. In user mode, it updates the appropriate table in the
    spawn_library.db (either 'cat_tracks' or 'lib_tracks') and does not modify spawn_catalog.db.

    'cleaned_files_map' is:
        {
          "/path/to/cleaned_Track1.m4a": temp_tags_for_track1,
          "/path/to/cleaned_Track2.m4a": temp_tags_for_track2,
          ...
        }

    Note: In user mode, lib_db_path must be provided.
    """
    if not cleaned_files_map:
        return

    # All files are presumably in the same folder; assume cleaned_files_map keys share the same parent.
    file_list = list(cleaned_files_map.keys())
    any_file_path = file_list[0]
    folder_path = os.path.dirname(any_file_path)

    # Gather all .m4a files in the folder, ignoring hidden files.
    existing_tracks = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".m4a") and not f.startswith('.') and not f.startswith("._")
    ]
    file_count = len(existing_tracks)

    logger.info(f"Calculating album ReplayGain via 'rsgain easy \"{folder_path}\"' for folder with {file_count} tracks...\n")

    cmd = ["rsgain", "easy", folder_path]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = proc.stdout
        logger.debug("rsgain output:\n" + output)

        album_gain = None
        album_peak = None

        # Dictionary to store track gain/peak:
        # track_rg_map[filepath] = [gain_str, peak_str]
        track_rg_map = {}
        current_file = None

        # Parse output lines with simple state-based logic / regex:
        for line in output.splitlines():
            line_str = line.strip()
            line_lower = line_str.lower()

            # Detect "Track: /path/to/file.m4a"
            if line_lower.startswith("track:"):
                track_path = line_str[6:].strip()
                track_path_norm = os.path.normpath(track_path)
                track_rg_map[track_path_norm] = [None, None]  # placeholders
                current_file = track_path_norm
                logger.debug(f"Found track: {track_path_norm}")
                continue

            # Detect "Album:" line
            if line_lower.startswith("album:"):
                current_file = "ALBUM"
                logger.debug("Entering album section")
                continue

            # Within a track section
            if current_file and current_file != "ALBUM":
                if line_lower.startswith("peak:"):
                    mg = re.search(r"peak:\s*([\d.]+)", line_str, re.IGNORECASE)
                    if mg:
                        track_peak = mg.group(1)
                        track_rg_map[current_file][1] = track_peak
                        logger.debug(f"Track {current_file} peak: {track_peak}")
                elif line_lower.startswith("gain:"):
                    mg = re.search(r"gain:\s*([-+]?\d+(?:\.\d+)?)\s*dB", line_str, re.IGNORECASE)
                    if mg:
                        track_gain = mg.group(1) + " dB"
                        track_rg_map[current_file][0] = track_gain
                        logger.debug(f"Track {current_file} gain: {track_gain}")

            # In the album section
            elif current_file == "ALBUM":
                if line_lower.startswith("peak:"):
                    mg = re.search(r"peak:\s*([\d.]+)", line_str, re.IGNORECASE)
                    if mg:
                        album_peak = mg.group(1)
                        logger.debug(f"Album peak: {album_peak}")
                elif line_lower.startswith("gain:"):
                    mg = re.search(r"gain:\s*([-+]?\d+(?:\.\d+)?)\s*dB", line_str, re.IGNORECASE)
                    if mg:
                        album_gain = mg.group(1) + " dB"
                        logger.debug(f"Album gain: {album_gain}")

        # Apply ReplayGain values to the in-memory tags
        for path_key, tags in cleaned_files_map.items():
            norm_path = os.path.normpath(path_key)
            # Track-level
            if norm_path in track_rg_map:
                track_gain_val, track_peak_val = track_rg_map[norm_path]
                if track_gain_val:
                    tags["----:com.apple.iTunes:replaygain_track_gain"] = track_gain_val
                if track_peak_val:
                    tags["----:com.apple.iTunes:replaygain_track_peak"] = track_peak_val
            # Album-level
            if album_gain:
                tags["----:com.apple.iTunes:replaygain_album_gain"] = album_gain
            if album_peak:
                tags["----:com.apple.iTunes:replaygain_album_peak"] = album_peak
            logger.debug(f"Updated cleaned_files_map for {norm_path}: {tags}")

        # Now update the database for every track in the folder.
        for file_path in existing_tracks:
            norm_path = os.path.normpath(file_path)
            base_name = os.path.basename(norm_path)
            if base_name.startswith("temp_"):
                continue

            logger.info(f"Processing file for database update: {norm_path}")

            # Use tags from cleaned_files_map if available; otherwise, read embedded tags from the file.
            if norm_path in cleaned_files_map:
                tags = cleaned_files_map[norm_path]
                logger.debug(f"Using tags from cleaned_files_map for {norm_path}: {tags}")
            else:
                tags = get_embedded_tags(norm_path)
                logger.debug(f"Read tags from file for {norm_path}: {tags}")

            # Ensure album ReplayGain values are set in tags.
            if album_gain:
                tags["----:com.apple.iTunes:replaygain_album_gain"] = album_gain
            if album_peak:
                tags["----:com.apple.iTunes:replaygain_album_peak"] = album_peak

            # Extract the spawn ID from embedded tags.
            spawn_id = tags.get("----:com.apple.iTunes:spawn_ID")
            if spawn_id and isinstance(spawn_id, bytes):
                spawn_id = spawn_id.decode('utf-8')
            logger.info(f"For file {norm_path}, found spawn_ID: {spawn_id}")

            if spawn_id:
                if mode == "admin":
                    # In admin mode, update the catalog database.
                    new_rev = update_album_rg_for_spawn_id(DB_PATH, spawn_id, album_gain, album_peak)
                    if new_rev:
                        update_embedded_metadata_rev(norm_path, new_rev)
                else:
                    # In user mode, we do NOT update spawn_catalog.db.
                    # Instead, update the user library DB.
                    # Bump the metadata revision using the current value from embedded tags.
                    old_rev = tags.get("----:com.apple.iTunes:metadata_rev", ["AAA"])[0]
                    new_rev = bump_metadata_rev(old_rev)
                    # Update both embedded and JSON metadata_rev.
                    tags["----:com.apple.iTunes:metadata_rev"] = [new_rev]
                    tags["metadata_rev"] = [new_rev]
                    # Determine target table: 'cat_tracks' if a catalog match, otherwise 'lib_tracks'.
                    table = "cat_tracks" if is_catalog_match(spawn_id) else "lib_tracks"
                    if lib_db_path is None:
                        logger.warning(f"User mode selected but no lib_db_path provided; skipping DB update for spawn_id {spawn_id}.")
                    else:
                        store_tags_in_user_db(spawn_id, tags, table=table, lib_db_path=lib_db_path)
                    # Optionally, update the embedded file with the bumped metadata_rev.
                    update_embedded_metadata_rev(norm_path, new_rev)
            else:
                logger.info(f"Skipping DB update for {norm_path} as no spawn_ID found in embedded tags.")

    except FileNotFoundError:
        logger.info("ERROR: rsgain not found in PATH. Please install or adapt to another tool.")
    except subprocess.CalledProcessError as e:
        logger.info(f"ReplayGain calculation error: {e}")
        logger.info(f"=== rsgain stderr ===\n{e.stderr}\n")
    except Exception as e:
        logger.info(f"Unexpected error during ReplayGain calc: {e}")


###############################################################################
# Update Disc & Track Numbers from MusicBrainz
###############################################################################

# When set to True, mismatch always overrides embedded values with MusicBrainz values
ALWAYS_OVERRIDE_WITH_MBID = False

# When set to True, mismatch always retains embedded values and discards MusicBrainz values
ALWAYS_KEEP_EXISTING = False

# If both are False, prompt the user if there's a mismatch
PROMPT_USER_FOR_TRACK_DISK_MISMATCH = True

def get_year_from_date(date_str: str) -> int:
    """
    Given a string like '1999-10-12' or '2001' or '2012',
    return the integer year (e.g. 1999, 2001, 2012).
    If no valid year is found, return 9999 as a fallback.
    """
    match = re.match(r'(\d{4})', date_str)
    if match:
        return int(match.group(1))
    return 9999

def sort_by_year_only(release_dict):
    """
    The release dict typically has a 'date' field like '1999-10-12'.
    We'll parse out only the year and sort by that integer.
    """
    date_str = release_dict.get("date", "9999-99-99")
    return get_year_from_date(date_str)

def update_disc_and_track_numbers_from_mbz(temp_tags, any_db_changes_ref, file_path, spawn_id_str):
    """
    If "MusicBrainz Track Id" is present, fetch a release from MusicBrainz, filter by official/promo, then prefer:
       (A) The earliest release year among those whose mediums have "Digital Media" or "CD",
       (B) Otherwise fallback to earliest official release year among all.
    Then extract disc/track from that chosen release.

    If there's a mismatch in main track/disc #, or an existing non-zero total that conflicts,
    we prompt the user. But if the old totals are zero/missing, we fill them in automatically.

    """
    mbid_field = "----:com.apple.iTunes:MusicBrainz Track Id"
    track_mbid = temp_tags.get(mbid_field)

    # 1) Check if there's actually a track MBID
    if not track_mbid:
        logger.debug("No MusicBrainz Track Id found, skipping disc/track number update.")
        return

    # Convert track_mbid to string if needed
    if isinstance(track_mbid, list) and track_mbid:
        track_mbid = track_mbid[0]
    if isinstance(track_mbid, bytes):
        track_mbid = track_mbid.decode("utf-8", errors="replace")

    track_mbid = str(track_mbid).strip()
    if not track_mbid or track_mbid.lower() == "n/a":
        logger.debug(f"track_mbid = '{track_mbid}' is empty, skipping disc/track update.")
        return

    logger.info(f"Updating disc/track numbers from MusicBrainz for MBID={track_mbid}...")

    try:
        # 2) Get the recording, including its release-list

        # Get local album title for matching
        local_album_title = ""
        if "©alb" in temp_tags:
            val = temp_tags["©alb"]
            if isinstance(val, list) and val:
                val = val[0]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            local_album_title = str(val).strip().lower()

        # Query the track MBID
        rec_data = musicbrainzngs.get_recording_by_id(track_mbid, includes=["releases"])
        release_list = rec_data["recording"].get("release-list", [])
        if not release_list:
            logger.info("  No releases found for this recording on MusicBrainz, skipping.")
            return

        logger.debug(f"Found {len(release_list)} releases in the recording’s release-list.")
        for idx, rl in enumerate(release_list, start=1):
            logger.debug(f" {idx}. ID={rl.get('id')} status={rl.get('status')} "
                        f"title={rl.get('title')} date={rl.get('date')}")


        # 3) is_preferred_release => Must be official, not a compilation/promo, 
        #    and must match local album title (if available).
        def is_preferred_release(r, local_title):
            status = (r.get("status") or "").lower().strip()
            title  = (r.get("title")  or "").lower().strip()

            # Must be official
            if status != "official":
                logger.debug(f"  => skipping because status != official (status={status})")
                return False
            # If local_album_title is not empty, skip if release.title != local_album_title
            if local_title and (title != local_title):
                logger.debug(f"  => skipping because release title '{title}' != local album '{local_title}'")
                return False
            # Optionally skip if “compilation” in title or "promo" in status
            if "compilation" in title or "promo" in status:
                logger.debug(f"  => skipping because it's compilation/promo (status={status}, title={title})")
                return False
            return True

        # 4) First pass => filter by official + album title match
        filtered_releases = [
            r for r in release_list
            if is_preferred_release(r, local_album_title)
        ]

        if filtered_releases:
            logger.debug(f"Found {len(filtered_releases)} releases that match local album='{local_album_title}' (plus official).")
            candidates = filtered_releases
        else:
            logger.debug(f"No official release found matching local album='{local_album_title}', fallback to earliest official or entire list.")
            # fallback to original official logic
            def fallback_is_preferred(r):
                # official & not compilation/promo
                s = (r.get("status") or "").lower().strip()
                t = (r.get("title")  or "").lower().strip()
                if s != "official":
                    return False
                if "compilation" in t or "promo" in s:
                    return False
                return True

            fallback_filtered = [r for r in release_list if fallback_is_preferred(r)]
            if fallback_filtered:
                candidates = fallback_filtered
                logger.debug(f"Found {len(candidates)} suitable official releases (fallback).")
            else:
                logger.debug("No official (non-compilation/promo) release found, fallback to entire release_list.")
                candidates = release_list

        # Sort by earliest release date
        candidates.sort(key=sort_by_year_only)


        # 5) Among these `candidates`, prioritize mediums with 'Digital Media' or 'CD'.
        def has_preferred_format(release_dict):
            # We'll fetch mediums from get_release_by_id
            test_mbid = release_dict["id"]
            try:
                test_data = musicbrainzngs.get_release_by_id(test_mbid, includes=["recordings"])
                mediums_ = test_data["release"].get("medium-list", [])
                for med in mediums_:
                    fm = (med.get("format") or "").lower().strip()
                    if fm in ["digital media", "cd"]:
                        return True
            except Exception as e:
                # If there's an error, skip
                logger.debug(f"Error while checking format for {test_mbid}: {e}")
            return False

        preferred_format_releases = [r for r in candidates if has_preferred_format(r)]
        if preferred_format_releases:
            # Now these are the subset that have "Digital Media" or "CD" => already sorted
            best_release = preferred_format_releases[0]
            logger.info(f"  Found release with 'Digital Media' or 'CD'")
        else:
            # fallback to the earliest year among the entire sorted `candidates`
            best_release = candidates[0]

        release_mbid = best_release["id"]
        release_title = best_release.get("title", "(no title)")
        release_status = best_release.get("status", "")
        release_date = best_release.get("date", "")

        release_data = musicbrainzngs.get_release_by_id(release_mbid, includes=["recordings"])
        mediums = release_data["release"].get("medium-list", [])
        if not mediums:
            logger.info("  This release has no mediums => skipping track numbering.")
            return

        format_set = set()
        for m in mediums:
            fmt = m.get("format")
            if fmt:
                format_set.add(fmt)

        if format_set:
            format_str = ", ".join(sorted(format_set))
        else:
            format_str = "N/A"
        logger.info(f"  Using release '{release_title}' (MBID={release_mbid}), status={release_status}, date={release_date}, format={format_str}")

        # 6) get_release_by_id => includes=["recordings"] => mediums/tracks
        # release_data = musicbrainzngs.get_release_by_id(release_mbid, includes=["recordings"])
        # mediums = release_data["release"].get("medium-list", [])
        if not mediums:
            logger.info("  This release has no mediums, skipping track numbering.")
            return

        found_tracknum = None
        found_discnum = None
        found_totaltracks = None
        found_totaldiscs = len(mediums)

        # 7) Locate the track referencing our same MBID
        for disc_index, medium in enumerate(mediums, start=1):
            track_list = medium.get("track-list", [])
            for t in track_list:
                rec_id = t.get("recording", {}).get("id", "")
                if rec_id == track_mbid:
                    found_tracknum = t.get("position")
                    found_discnum  = disc_index
                    found_totaltracks = medium.get("track-count")
                    break
            if found_tracknum:
                break

        if not found_tracknum:
            logger.info("  Could not find a matching track in the release mediums, skipping disc/track.")
            return

        # Convert them to integers
        try:
            tracknum_int = int(found_tracknum)
        except:
            tracknum_int = 0
        try:
            totaltracks_int = int(found_totaltracks) if found_totaltracks else 0
        except:
            totaltracks_int = 0
        try:
            discnum_int = int(found_discnum)
        except:
            discnum_int = 0

        if not found_totaldiscs:
            found_totaldiscs = 0

        new_trkn = (tracknum_int, totaltracks_int)
        new_disk = (discnum_int, found_totaldiscs)

        old_trkn = temp_tags.get("trkn")
        old_disk = temp_tags.get("disk")

        if old_trkn:
            old_trkn_value = old_trkn[0] if isinstance(old_trkn, list) else old_trkn
        else:
            old_trkn_value = (0, 0)  # treat as none => (track_main=0, track_total=0)

        if old_disk:
            old_disk_value = old_disk[0] if isinstance(old_disk, list) else old_disk
        else:
            old_disk_value = (0, 0)

        (old_track_main, old_track_total) = old_trkn_value
        (old_disk_main, old_disk_total)   = old_disk_value

        # If there is no existing track/disc data (both main and total are zero),
        # then simply assign the new values without prompting or bumping metadata_rev.
        if (old_track_main == 0 and old_track_total == 0) and (old_disk_main == 0 and old_disk_total == 0):
            temp_tags["trkn"] = [new_trkn]
            temp_tags["disk"] = [new_disk]
            logger.info(f"  => No existing track/disc data; setting 'trkn'={new_trkn}, 'disk'={new_disk} without incrementing metadata_rev.\n")
            return

        # Check mismatch on main track/disc numbers
        # "Full mismatch" => track_main or disc_main differ
        # "Partial mismatch" => main is same but total is different
        mismatch_type = None

        # Compare main track #
        if old_track_main != tracknum_int:
            mismatch_type = "full"
        # Compare main disc #
        if old_disk_main != discnum_int:
            mismatch_type = "full" if mismatch_type == "full" else "full"

        # If we haven't flagged full mismatch yet, check total differences
        # partial mismatch if total track differs or total disc differs
        # but main numbers are the same
        if mismatch_type is None:
            if old_track_total != totaltracks_int or old_disk_total != found_totaldiscs:
                mismatch_type = "partial"

        # -- If no mismatch => just overwrite
        if mismatch_type is None:
            logger.debug("No mismatch (or old track/disc not set). Overwriting with MB values.")
            temp_tags["trkn"] = [new_trkn]
            temp_tags["disk"] = [new_disk]
            logger.info(f"  => Setting 'trkn'={new_trkn}, 'disk'={new_disk}\n")
            return

        # For partial mismatch => maybe some fields are missing on old
        if mismatch_type == "partial":
            logger.info("Found partial mismatch: same track/disc main #, different or missing totals.")
            logger.info(f"Existing => trkn={old_trkn}, disk={old_disk}")
            logger.info(f"MusicBrainz => trkn={new_trkn}, disk={new_disk}")

            # Decide if we can fill missing automatically
            auto_filled_something = False

            # 1) If old_track_total==0 but new total !=0 => fill it
            final_track_total = old_track_total
            if old_track_total == 0 and totaltracks_int != 0:
                final_track_total = totaltracks_int
                auto_filled_something = True

            # 2) If old_disk_total==0 but MB says e.g. 1 or 2
            final_disc_total = old_disk_total
            if old_disk_total == 0 and found_totaldiscs != 0:
                final_disc_total = found_totaldiscs
                auto_filled_something = True

            # Build final updated pairs for potential no-prompt scenario
            final_trkn = (old_track_main, final_track_total)
            final_disk = (old_disk_main, final_disc_total)

            # Check if there's still a genuine mismatch => e.g. old_total was non-zero but different
            # or old_disk_total was non-zero but different
            # If so, we prompt
            needs_prompt = False

            if (old_track_total != 0) and (old_track_total != totaltracks_int):
                # We do have a conflict
                if old_track_total != final_track_total:
                    needs_prompt = True

            if (old_disk_total != 0) and (old_disk_total != found_totaldiscs):
                if old_disk_total != final_disc_total:
                    needs_prompt = True

            if needs_prompt:
                logger.info("We have a real mismatch in track/disc totals (non-zero conflict). Prompting user.")
                user_in = get_user_input(
                    "\nAdd missing/corrected data from MusicBrainz for track/disc total? ([y]/n): ",
                    default="y"
                )
                logger.info(f"User input => '{user_in}'")

                if user_in.lower() != "n":
                    temp_tags["trkn"] = [new_trkn]
                    temp_tags["disk"] = [new_disk]
                    old_rev = temp_tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
                    try:
                        new_rev = bump_metadata_rev(old_rev)
                    except Exception as e:
                        logger.warning(f"Error bumping metadata_rev: {e}")
                    temp_tags["----:com.apple.iTunes:metadata_rev"] = new_rev
                    any_db_changes_ref[0] = True
                    rewrite_tags(file_path, temp_tags)
                    store_tags_in_db(DB_PATH, spawn_id_str, temp_tags, metadata_rev=new_rev)
                    logger.info(f"  => Overwriting partial => trkn={new_trkn}, disk={new_disk}\n")
                    logger.info(f"  => Also incrementing track's metadata_rev to {new_rev}")

                else:
                    # same fallback logic: keep existing or let them type custom
                    keep_ans = get_user_input("\nWould you like to keep the existing values? ([y]/n): ", default="y")
                    if keep_ans.lower() != "n":
                        logger.debug("Keeping existing track/disc => do nothing.")
                        return
                    else:
                        new_vals = input("\nEnter new values (Track_num/Track_total, Disc_num/Disc_total): ").strip()
                        logger.info(f"User new track/disc => '{new_vals}'")
                        try:
                            trk_part, dsk_part = new_vals.split(",", 1)
                            t_main, t_tot = trk_part.split("/", 1)
                            d_main, d_tot = dsk_part.split("/", 1)

                            t_main_int = int(t_main)
                            t_tot_int  = int(t_tot)
                            d_main_int = int(d_main)
                            d_tot_int  = int(d_tot)

                            temp_tags["trkn"] = [(t_main_int, t_tot_int)]
                            temp_tags["disk"] = [(d_main_int, d_tot_int)]
                            logger.info(f"  => user-chosen => trkn=({t_main_int},{t_tot_int}), disk=({d_main_int},{d_tot_int})")
                            # Instead of bumping the revision, keep the current value:
                            old_rev = temp_tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
                            any_db_changes_ref[0] = True
                            rewrite_tags(file_path, temp_tags)
                            store_tags_in_db(DB_PATH, spawn_id_str, temp_tags, metadata_rev=old_rev)
                            
                        except Exception as e:
                            logger.warning(f"Could not parse => {e}, keeping existing.")
                        return
            else:
                # No genuine mismatch => we can quietly fill in what's missing
                temp_tags["trkn"] = [final_trkn]
                temp_tags["disk"] = [final_disk]
                logger.info(f"  => Filled missing totals => trkn={final_trkn}, disk={final_disk}")
                #old_rev = temp_tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
                #new_rev = bump_metadata_rev(old_rev)
                #temp_tags["----:com.apple.iTunes:metadata_rev"] = new_rev
                any_db_changes_ref[0] = True
                rewrite_tags(file_path, temp_tags)
                #store_tags_in_db(DB_PATH, spawn_id_str, temp_tags, metadata_rev=new_rev)
                store_tags_in_db(DB_PATH, spawn_id_str, temp_tags)
                #logger.info(f"  => Incrementing track's metadata_rev to {new_rev}")
            return

        # (G) Full mismatch
        else:
            # mismatch_type == "full"
            logger.info("Found full mismatch in track/disc => Overwrite existing or keep old?")
            logger.info(f"Existing => trkn={old_trkn}, disk={old_disk}")
            logger.info(f"MusicBrainz => trkn={new_trkn}, disk={new_disk}")

            user_in = get_user_input(
                "\nOverwrite existing track/disc with MB values? (y/[n]): ",
                default="n"
            )
            logger.info(f"User input => '{user_in}'")

            if user_in.lower() == "y":
                temp_tags["trkn"] = [new_trkn]
                temp_tags["disk"] = [new_disk]
                logger.info(f"  => Overwriting with => trkn={new_trkn}, disk={new_disk}.")
                old_rev = temp_tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
                try:
                    new_rev = bump_metadata_rev(old_rev)
                except Exception as e:
                    logger.warning(f"Error bumping metadata_rev: {e}")
                temp_tags["----:com.apple.iTunes:metadata_rev"] = new_rev
                any_db_changes_ref[0] = True
                rewrite_tags(file_path, temp_tags)
                store_tags_in_db(DB_PATH, spawn_id_str, temp_tags, metadata_rev=new_rev)
                #logger.info(f"  => Incrementing track's metadata_rev to {new_rev}")
            else:
                logger.debug("User chose not to overwrite => ask if they want new custom values?")

                keep_ans = get_user_input("\nWould you like to keep the existing values? ([y]/n): ", default="y")
                if keep_ans.lower() != "n":
                    logger.debug("Keeping existing => do nothing.")
                else:
                    new_vals = input("\nEnter new values (Track_num/Track_total, Disc_num/Disc_total): ").strip()
                    logger.info(f"User entered new track/disc => '{new_vals}'")

                    try:
                        trk_part, dsk_part = new_vals.split(",", 1)
                        t_main, t_tot = trk_part.split("/", 1)
                        d_main, d_tot = dsk_part.split("/", 1)

                        t_main_int = int(t_main)
                        t_tot_int = int(t_tot)
                        d_main_int = int(d_main)
                        d_tot_int = int(d_tot)

                        temp_tags["trkn"] = [(t_main_int, t_tot_int)]
                        temp_tags["disk"] = [(d_main_int, d_tot_int)]
                        logger.info(f"  => user-chosen => trkn=({t_main_int},{t_tot_int}), disk=({d_main_int},{d_tot_int})")
                        # Instead of bumping the revision, keep the current value:
                        old_rev = temp_tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
                        any_db_changes_ref[0] = True
                        rewrite_tags(file_path, temp_tags)
                        store_tags_in_db(DB_PATH, spawn_id_str, temp_tags, metadata_rev=old_rev)

                    except Exception as e:
                        logger.warning(f"Could not parse user input => {e}, keeping existing track/disc.")
                    return

    except musicbrainzngs.WebServiceError as e:
        logger.info(f"[MBID] MusicBrainz WebServiceError: {e}")
    except KeyError as ke:
        logger.info(f"[MBID] Unexpected KeyError reading MB data: {ke}")
    except Exception as ex:
        logger.info(f"[MBID] Unexpected error: {ex}")


###############################################################################
# M3U Helper Functions
###############################################################################

def parse_txt_artist_title_from_path(full_path, base_music_dir):
    """
    Given a .txt file path like:
      ../Music/<Artist>/<Album>/D-TT - <title>.txt
    returns (artist, title).

    We assume the path segments are standardized:
      -3 => Artist name
      -2 => Album name
      -1 => "D-TT - Title.txt"
    """

    # 1) Convert to absolute path or ensure consistent separators
    p = os.path.normpath(full_path)

    # 2) Split into components
    # e.g. ["..","Music","Britney Spears","In the Zone","1-06 - Toxic.txt"]
    parts = p.split(os.sep)

    # Quick fallback
    if len(parts) < 3:
        return ("Unknown", os.path.splitext(os.path.basename(full_path))[0])

    # 3) Artist is parts[-3] if the path truly has 
    #    ../Music/<Artist>/<Album>/<D-TT - Title>.txt
    artist = parts[-3]

    # 4) The last part is "D-TT - Title.txt"
    last_part = parts[-1]  # e.g. "1-06 - Toxic.txt"
    base_name = os.path.splitext(last_part)[0]  # "1-06 - Toxic"

    # 5) Split by " - " => left = "1-06", right = "Toxic"
    #    If no " - " found, fallback
    title = base_name
    if " - " in base_name:
        # e.g. "1-06 - Toxic"
        disc_track_str, title_str = base_name.split(" - ", 1)
        title = title_str.strip()

    # Artist might have spaces, e.g. "Britney Spears". That’s fine.
    artist = artist.strip()
    title = title.strip() or base_name

    return (artist, title)


def generate_import_playlist(all_tracks, base_music_dir, playlists_dir):
    """
    Generates an M3U file named e.g. 'import_YYYY-MM-DD_HHMM.m3u' 
    in 'Spawn/Playlists/Imported' with two sections (both sorted alphabetically):

    1) .txt files (commented out path, #EXTXT:)
    2) .m4a files (#EXTINF:)

    base_music_dir is the path segment to remove & replace with '../../Music/'
    so that the playlist references files relatively (two levels up).
    """
    # 1) Build a timestamped filename
    timestamp_str = time.strftime("%Y-%m-%d_%H%M")
    m3u_filename = f"import_{timestamp_str}.m3u"
    m3u_path = os.path.join(playlists_dir, m3u_filename)

    # 2) Create the Imported subfolder if it doesn't exist
    imported_dir = os.path.join(playlists_dir, "Imported")
    os.makedirs(imported_dir, exist_ok=True)

    # 3) The final .m3u path is inside "Imported/"
    m3u_path = os.path.join(imported_dir, m3u_filename)

    # 4) Separate .txt vs .m4a, store tuples => (track_path, track_tags)
    txt_entries = []
    m4a_entries = []

    for (track_path, track_tags, id_str) in all_tracks:
        lower_path = track_path.lower()
        if lower_path.endswith(".txt"):
            txt_entries.append((track_path, track_tags))
        elif lower_path.endswith(".m4a"):
            m4a_entries.append((track_path, track_tags))

    # Sort each list by filename (case-insensitive)
    txt_entries.sort(key=lambda x: os.path.basename(x[0]).lower())
    m4a_entries.sort(key=lambda x: os.path.basename(x[0]).lower())

    # 5) Helper function => replace base_music_dir with '../../Music'
    def build_relative_path(full_path):
        """
        If 'full_path' starts with base_music_dir, replace that portion
        with '../../Music'. Otherwise return only the filename.
        """

        # Ensure consistent case
        lower_full = full_path.lower()
        lower_base = base_music_dir.lower()
        # if the file is inside base_music_dir
        if lower_full.startswith(lower_base):
            return "../../Music" + full_path[len(base_music_dir):]
        else:
            return os.path.basename(full_path)

    def _decode(v):
        if isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        return str(v).strip() if v else ""

    lines = ["#EXTM3U", ""]

    # (A) .txt files => #EXTINF lines with real duration
    for (txt_file_path, txt_tags) in txt_entries:
        # Parse from path (=> )e.g. "../Music/Artist/Album/D-TT - Title.txt")
        artist, title = parse_txt_artist_title_from_path(txt_file_path, base_music_dir)
        relative_path = build_relative_path(txt_file_path)
        lines.append(f"#EXTXT:0,{title} - {artist}")
        lines.append(f"#{relative_path}")
        lines.append("")  # blank line

    # (B) .m4a files => #EXTINF lines
    for (m4a_file_path, m4a_tags) in m4a_entries:
        # Default placeholders
        artist = "Unknown"
        title  = "Unknown"

        # If we have tags, decode them
        if m4a_tags:
            artist_val = m4a_tags.get("©ART")
            title_val  = m4a_tags.get("©nam")

            artist = _decode(artist_val) or "Unknown"
            title  = _decode(title_val)  or "Unknown"

        # Get track duration in seconds
        duration_seconds = get_track_duration_seconds(m4a_file_path)

        relative_path = build_relative_path(m4a_file_path)

        # Normal EXTINF line
        lines.append(f"#EXTINF:{duration_seconds},{title} - {artist}")
        lines.append(f"{relative_path}")
        lines.append("")

    # 6) Write out to the .m3u file
    with open(m3u_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    logger.info(f"Playlist created: {m3u_path}\n")


###############################################################################
# Album Art Helper Functions
###############################################################################

def fetch_caa_art(mbid):
    # Skip lookup if MBID is missing
    if not mbid:
        logger.info("Skipping Cover Art Archive lookup because MBID is missing.")
        return None

    # Construct the Cover Art Archive URL
    url = f"https://coverartarchive.org/release-group/{mbid}/front"
    
    # Query CAA with retry logic
    response = fetch_with_retry(url)
    if not response:
        logger.info(f"Aborted or failed to fetch album art for MBID: {mbid}")
        return None
    if response.status_code == 200:
        return url  # Return URL of the front cover
    elif response.status_code == 404:
        logger.info(f"No album art found in Cover Art Archive for MBID: {mbid}")
    else:
        logger.error(f"Error querying Cover Art Archive: {response.status_code}")
    return None


def fetch_spotify_art(track_id, client_id, client_secret):
    # Skip lookup if track_id is missing
    if not track_id:
        logger.info("Skipping Spotify album art lookup because track_id is missing.")
        return None

    # Step 1: Get access token using the updated function (with retry)
    access_token = get_spotify_access_token()
    if not access_token:
        logger.error("Error fetching Spotify access token after retries.")
        return None

    # Step 2: Fetch track metadata with retry logic.
    track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    wait_times = [60, 300, 900]  # in seconds
    fallback_wait = 1800
    attempt = 0

    while True:
        attempt += 1
        try:
            track_response = requests.get(track_url, headers=headers, timeout=10)
            if track_response.status_code == 200:
                break  # Successful response; exit loop
            else:
                logger.error(f"Error fetching Spotify track metadata for {track_id}: {track_response.status_code} {track_response.text}")
        except Exception as e:
            logger.error(f"Error fetching Spotify track metadata for {track_id}: {e}")
        if attempt <= len(wait_times):
            wait_time = wait_times[attempt - 1]
        else:
            wait_time = fallback_wait
        logger.warning(f"Retrying Spotify track metadata request in {wait_time/60:.1f} minutes (attempt {attempt})...")
        time.sleep(wait_time)

    track_data = track_response.json()
    album_images = track_data.get('album', {}).get('images', [])
    if album_images:
        # Return the largest image (typically the first in the list)
        return album_images[0]['url']
    return None


def get_image_dimensions(image_url):
    """
    Download the image from 'image_url' in memory and return (width, height).
    Returns None if we can't fetch or parse as an image.
    """
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        # Use Pillow to read the in-memory bytes
        with Image.open(io.BytesIO(resp.content)) as im:
            return (im.width, im.height)
    except Exception as e:
        logger.error(f"Error fetching or parsing image '{image_url}': {e}")
        return None


def get_album_art(track_id, mbid, client_id, client_secret, album_folder, min_dimension=640):
    """
    1) If album_folder contains a cover.jpg file, return its path immediately.
    2) Otherwise, attempt to fetch album art from Cover Art Archive (CAA) and Spotify.
    
    - For CAA: if found and at least min_dimension in width & height, return its URL.
    - Otherwise, fall back to Spotify and pick whichever image is larger.
    """
    # 0) Check for local album art first.
    local_cover = os.path.join(album_folder, "cover.jpg")
    if os.path.exists(local_cover):
        logger.info(f"Using existing album art from {local_cover}.")
        return local_cover

    # 1) Try Cover Art Archive (CAA)
    logger.debug(f"Fetching CAA album art for MBID: {mbid}")
    caa_art_url = fetch_caa_art(mbid)   # e.g. "https://coverartarchive.org/release-group/..."
    caa_dims = None
    if caa_art_url:
        # Determine how large it is
        caa_dims = get_image_dimensions(caa_art_url)
        if caa_dims:
            logger.info(f"CAA art = {caa_art_url} => {caa_dims[0]}x{caa_dims[1]}")

    # 2) Fallback to Spotify
    logger.debug(f"Fetching Spotify album art for track ID: {track_id}")
    spotify_art_url = fetch_spotify_art(track_id, client_id, client_secret)
    sp_dims = None
    if spotify_art_url:
        sp_dims = get_image_dimensions(spotify_art_url)
        if sp_dims:
            logger.info(f"Spotify art = {spotify_art_url} => {sp_dims[0]}x{sp_dims[1]}")

    # Comparisons for the following scenarios:
    #  (A) only CAA
    #  (B) only Spotify
    #  (C) both, so pick bigger
    #  (D) neither
    if caa_art_url and not spotify_art_url:
        logger.debug("Using CAA because there's no Spotify art at all.")
        return caa_art_url

    if spotify_art_url and not caa_art_url:
        logger.debug("Using Spotify because there's no CAA art at all.")
        return spotify_art_url

    if not caa_art_url and not spotify_art_url:
        logger.error("No album art found (both CAA and Spotify missing).")
        return None

    # Now we have both => pick whichever is bigger dimension
    if caa_dims and sp_dims:
        # Compare minimum dimension or total area, your choice
        caa_min_side = min(caa_dims)
        sp_min_side  = min(sp_dims)
        if caa_min_side >= sp_min_side:
            logger.debug(f"Using CAA (bigger or equal) => {caa_dims[0]}x{caa_dims[1]}")
            return caa_art_url
        else:
            logger.debug(f"Using Spotify (bigger) => {sp_dims[0]}x{sp_dims[1]}")
            return spotify_art_url

    # If for some reason one has dimensions but the other is None, decide
    if caa_dims and not sp_dims:
        logger.debug("Spotify image couldn't be retrieved or parsed, so using CAA.")
        return caa_art_url
    if sp_dims and not caa_dims:
        logger.debug("CAA image couldn't be retrieved or parsed, so using Spotify.")
        return spotify_art_url

    # Fallback
    logger.debug("One or both have no known dimensions, defaulting to CAA if present.")
    return caa_art_url or spotify_art_url

# Album art dimensions & filesize settings
MAX_DIMENSION = 1000
MAX_FILESIZE = 350 * 1024  # 350 KB in bytes
QUALITY_STEPS = [90, 80, 70, 60, 50, 40, 30, 20, 10]

def save_album_art(image_url, save_path):
    """
    Downloads 'image_url' into memory using requests.
    If the original image is larger than MAX_DIMENSION on any side:
      - Save an exact copy to 'cover_hi-res.jpg' in a separate folder under 'Spawn/aux/user/hart/...'.
      - Then resize to <= MAX_DIMENSION in dimension for 'cover.jpg' in the usual folder.
      - If 'cover.jpg' ends up > MAX_FILESIZE, tries lower quality in increments of 10.
    Otherwise, if no hi-res is needed, only 'cover.jpg' is created in the normal folder.
    """

    # 1) Download the raw bytes
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.info(f"Error downloading album art from {image_url}: {e}")
        return

    # 2) Save cover.jpg to target directory
    base_dir = os.path.dirname(save_path)

    # 3) Open the downloaded bytes with Pillow to get dimensions
    try:
        with Image.open(io.BytesIO(resp.content)) as original_img:
            w, h = original_img.size
            logger.info(f"Original album art size: {w}x{h}")

            # Decide if we need hi-res raw copy (dimensions > MAX_DIMENSION)
            needs_hi_res = (w > MAX_DIMENSION or h > MAX_DIMENSION)

            # (A) If bigger than 1000 => cover_hi-res.jpg = raw bytes
            if needs_hi_res:

                # Compute parallel path for 'cover_hi-res.jpg' under Spawn/aux/user/hart/...
                rel_subpath = os.path.relpath(base_dir, start=OUTPUT_PARENT_DIR)
                hart_root = os.path.join(os.path.dirname(OUTPUT_PARENT_DIR), "aux", "user", "hart")
                hi_res_base = os.path.join(hart_root, rel_subpath)

                # Create the folder only now, since it is truly needed to store hi-res
                os.makedirs(hi_res_base, exist_ok=True)
                hi_res_path = os.path.join(hi_res_base, "cover_hi-res.jpg")

                # Save raw bytes
                with open(hi_res_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Saved hi-res album art (raw) => {hi_res_path}, {w}x{h}")

            # (B) Now create the "cover.jpg" (resized if dimension > 1000)
            #     Step 1: resize if needed
            if needs_hi_res:
                scale_factor = MAX_DIMENSION / w
                new_w = MAX_DIMENSION
                #scale_factor = min(MAX_DIMENSION / w, MAX_DIMENSION / h)
                #new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
            else:
                # already <= MAX_DIMENSION => no resizing
                new_w, new_h = w, h

            # Step 2: create a PIL image with the correct dimension
            if new_w != w or new_h != h:
                logger.info(f"Resizing => {new_w}x{new_h}")
                resized_img = original_img.resize((new_w, new_h), Image.LANCZOS)
            else:
                resized_img = original_img  # not resized

            # If RGBA, convert to RGB
            if resized_img.mode == "RGBA":
                resized_img = resized_img.convert("RGB")
                logger.debug(f"Converted RGBA to RBG => {hi_res_path}")

            # Step 3: Attempt to save "cover.jpg" with decreasing quality
            #         until we get a file <= 350 MB or run out of attempts
            cover_bytes = try_saving_within_max_size(
                resized_img,
                MAX_FILESIZE,
                save_path,
                hi_res_exists=needs_hi_res or None,  # boolean if hi-res is known or not
                original_bytes=resp.content
            )
            if cover_bytes is None:
                # means we never managed to get under 350 MB
                logger.info("Could not reduce the file size below 350 KB even at lowest quality.")
            else:
                final_size = len(cover_bytes)
                mb = final_size / (1024 * 1024)
                logger.info(f"Final cover.jpg => {mb:.2f} MB")
    except Exception as e:
        logger.info(f"Error parsing or saving album art: {e}")


def try_saving_within_max_size(
    pil_img,
    max_size_bytes,
    final_path,
    hi_res_exists=False,
    original_bytes=None
):
    """
    Saves 'pil_img' (Pillow image) as JPEG with decreasing quality in increments of 10
    until file size is <= max_size_bytes.

    If final_path would exceed max_size_bytes at all possible quality settings,
    returns None. Otherwise, returns the final bytes used to create 'final_path'.

    If hi_res_exists is False, but the file is > max_size_bytes even at the first attempt,
    then we create 'cover_hi-res.jpg' from raw original_bytes. 
    """

    # If no hi_res_exists, but we might need to create it if the file is too large at the first try
    hi_res_path = os.path.join(os.path.dirname(final_path), "cover_hi-res.jpg")

    for q in QUALITY_STEPS:
        # We'll encode in memory first
        temp_buffer = io.BytesIO()
        pil_img.save(
            temp_buffer,
            format="JPEG",
            quality=q,
            optimize=True
        )
        size_now = temp_buffer.getbuffer().nbytes

        if size_now <= max_size_bytes:
            # Good => write final_path from this buffer
            with open(final_path, "wb") as out_f:
                out_f.write(temp_buffer.getvalue())
            logger.info(f"Saved cover.jpg at quality={q}, size={size_now} bytes")
            return temp_buffer.getvalue()
        else:
            logger.info(f"Quality={q} => {size_now} bytes (over {max_size_bytes}). Retrying...")

            # If we do not have hi_res_exists, but the file is huge => create hi-res now
            # (only do this once => after the first check at Q=90 fails, e.g.)
            if not hi_res_exists:
                # Create cover_hi-res.jpg from original_bytes
                if original_bytes:
                    with open(hi_res_path, "wb") as f:
                        f.write(original_bytes)
                    logger.info(f"Saved hi-res album art (raw) => {hi_res_path} due to large file at Q={q}")
                hi_res_exists = True

    # If we exhaust the loop with no success => return None
    return None

def embed_cover_art_into_file(audio_path, cover_jpeg_path, tags_dict):
    """
    Embeds the cover art, and also puts 'covr' into tags_dict so rewrite_tags() won't remove it.
    """
    audio = MP4(audio_path)
    #audio = MutagenFile(audio_path)
    if not audio:
        logger.info(f"Unable to load file for embedding: {audio_path}")
        return

    # Remove any existing covr
    if "covr" in audio.tags:
        del audio.tags["covr"]

    # Read the JPEG
    try:
        with open(cover_jpeg_path, "rb") as f:
            jpg_data = f.read()
    except IOError as e:
        logger.info(f"Cannot read cover.jpg for embedding: {cover_jpeg_path} => {e}")
        return

    # For MP4 files
    from mutagen.mp4 import MP4Cover
    mp4c = MP4Cover(jpg_data, imageformat=MP4Cover.FORMAT_JPEG)
    audio.tags["covr"] = [mp4c]
    audio.save()

    logger.info(f"Embedded new cover.jpg into: {audio_path}")

    #  Add the covr entry to `tags_dict` so the next rewrite_tags() call
    #  re-inserts it (since rewrite_tags() forcibly clears everything first).
    tags_dict["covr"] = [mp4c]


###############################################################################
# Helper Functions for db_rev
###############################################################################

def get_total_track_count(db_path):
    """
    Returns the current total number of tracks in the 'tracks' table.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM tracks")
    (count,) = c.fetchone()
    conn.close()
    return count

def generate_db_rev_with_count(new_total):
    """
    Revision forat: 'YYYY-MM-DDTHH:MM:SSZ.000000010.AA'
    where YYYY-MM-DDTHH:MM:SSZ is UTC timestamp (ISO 8601 format),
    where 000000010 represents the total # of tracks in the database (zero-padded),
    and AA indicates revision tracking for edits of existing track sin the database.
    """
    now = datetime.now(tz=ZoneInfo("UTC"))
    timestamp_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    numeric_part = str(new_total).rjust(9, "0")
    alpha_part = "AA"
    return f"{timestamp_str}.{numeric_part}.{alpha_part}"


###############################################################################
# Helper Function for Clearing Emptied Music Directories
###############################################################################

def remove_empty_music_dirs(music_root):
    """
    Recursively walk the 'music_root' directory bottom-up and remove any subfolders
    that have zero audio files (and presumably only leftover .jpg, .txt, etc.).
    Also removes leftover non-audio files before deleting the folder.
    """
    for dirpath, dirnames, filenames in os.walk(music_root, topdown=False):
        # 1) Check if this directory has any .m4a or .mp4
        has_audio = any(f.lower().endswith(('.m4a', '.mp4')) for f in filenames)

        if not has_audio:
            # 2) Remove leftover files (cover.jpg, cover_hi-res.jpg, etc.)
            for fname in filenames:
                file_to_delete = os.path.join(dirpath, fname)
                try:
                    os.remove(file_to_delete)
                except OSError as e:
                    print(f"Warning: Could not remove file {file_to_delete}: {e}")

            # 3) Now see if the directory is really empty (no subdirs, no leftover files)
            #    If so, remove it. Otherwise skip it quietly.
            try:
                # If dirpath is truly empty, os.rmdir() will succeed
                os.rmdir(dirpath)
                #print(f"Removed empty folder: {dirpath}")
            except OSError as e:
                # Typically OSError 66 = Directory not empty
                # We'll just ignore that, so there's no noisy warning
                if e.errno != 66:
                    print(f"Warning: Could not remove directory {dirpath}: {e}")


###############################################################################
# Deej-AI / MP4ToVec Embedding Flow
###############################################################################

def generate_deejai_embedding_for_track(track_path, spawn_id_str):
    """
    If MP4ToVec is available and a global model is loaded, generate an embedding
    for the track and store it in 'spawn_id_to_embeds'.
    """
    if not MP4TOVEC_AVAILABLE or (MP4TOVEC_MODEL is None):
        # If there's no model or the import failed, skip
        logger.info(f"[MP4ToVec] Model not loaded or not found; skipping embedding for {track_path}")
        return

    try:
        logger.info(f"[MP4ToVec] Generating embedding for track: {track_path}")
        emb = generate_embedding(track_path, MP4TOVEC_MODEL)
        if emb is not None:
            spawn_id_to_embeds[spawn_id_str] = emb
            logger.info(f"[MP4ToVec] Stored embedding for spawn_id='{spawn_id_str}', shape={emb.shape}")
        else:
            logger.warning(f"[MP4ToVec] Returned None for spawn_id='{spawn_id_str}'")
    except Exception as e:
        logger.warning(f"[MP4ToVec] Error generating embedding for '{track_path}': {e}")


def save_combined_embeddings(EMBED_PATH, spawn_id_to_embeds):
    """
    1) Load existing embeddings from EMBED_PATH if present.
    2) Merge with spawn_id_to_embeds.
    3) Write the combined dictionary back to EMBED_PATH.
    """
    if not spawn_id_to_embeds:
        logger.info("[MP4ToVec] No newly generated embeddings to save. Skipping.")
        return

    # 1) Load existing dictionary
    existing_dict = {}
    if os.path.isfile(EMBED_PATH):
        try:
            with open(EMBED_PATH, "rb") as f:
                existing_dict = pickle.load(f)
        except Exception as e:
            logger.warning(f"[MP4ToVec] Could not load existing embeddings from {EMBED_PATH}: {e}")

    # 2) Merge
    for sid, emb in spawn_id_to_embeds.items():
        existing_dict[sid] = emb  # Overwrite or add

    # 3) Write combined dictionary
    try:
        with open(EMBED_PATH, "wb") as f:
            pickle.dump(existing_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[MP4ToVec] Appended {len(spawn_id_to_embeds)} embeddings; total now {len(existing_dict)}. Saved => {EMBED_PATH}\n")
    except Exception as e:
        logger.error(f"[MP4ToVec] Failed to write updated embeddings => {e}")



###############################################################################
# Helper: Process a Single Album Folder
###############################################################################
def process_album_folder(folder, files_in_folder, keep_matched, lastfm_api_key, sp, is_admin, USER_DB_PATH):
    logger.info(
        "\n================================================================================================================\n"
        f"Processing album folder: {folder}\n"
    )
    cleaned_files_map = {}  # final_file_path -> final_tags
    all_tracks = []

    # --- Gather minimal tags from each file ---
    all_tags_by_file = {}
    year_by_file = {}
    artist_by_file = {}
    album_by_file = {}

    for file_path in files_in_folder:
        tmp_tags = extract_desired_tags(file_path)
        if not tmp_tags:
            logger.info(f"    No tags found for {file_path}, skipping.")
            continue
        all_tags_by_file[file_path] = tmp_tags

        # Extract year from ©day tag
        day_val = tmp_tags.get("©day", "")
        if isinstance(day_val, list) and day_val:
            day_val = day_val[0]
        if isinstance(day_val, bytes):
            day_val = day_val.decode("utf-8", errors="replace")
        year_str = str(day_val).strip()
        if len(year_str) >= 4 and year_str[:4].isdigit():
            year_str = year_str[:4]
        year_by_file[file_path] = year_str

        # Extract artist from ©ART tag
        art_val = tmp_tags.get("©ART", "")
        if isinstance(art_val, list) and art_val:
            art_val = art_val[0]
        if isinstance(art_val, bytes):
            art_val = art_val.decode("utf-8", errors="replace")
        artist_by_file[file_path] = (art_val or "Unknown").strip()

        # Extract album from ©alb tag
        alb_val = tmp_tags.get("©alb", "")
        if isinstance(alb_val, list) and alb_val:
            alb_val = alb_val[0]
        if isinstance(alb_val, bytes):
            alb_val = alb_val.decode("utf-8", errors="replace")
        album_by_file[file_path] = (alb_val or "Unknown").strip()

    # --- Determine unified year if needed ---
    unified_year = None
    if len(all_tags_by_file) > 1:
        unique_years = Counter(year_by_file[f] for f in all_tags_by_file)
        if len(unique_years) > 1:
            logger.info(f"[ALBUM YEAR WARNING] Folder '{folder}' has mismatched years: {unique_years}")
            most_common_year, _ = unique_years.most_common(1)[0]
            unified_year = most_common_year
        else:
            unified_year = next(iter(unique_years))
    else:
        if all_tags_by_file:
            unified_year = next(iter(year_by_file.values()))

    # --- If album folder is consistent, perform single MB/Spotify lookup ---
    unique_artists = set(artist_by_file[f] for f in all_tags_by_file)
    unique_albums = set(album_by_file[f] for f in all_tags_by_file)
    if len(unique_artists) == 1 and len(unique_albums) == 1:
        final_artist = list(unique_artists)[0]
        final_album = list(unique_albums)[0]
        logger.info(f"[ALBUM FOLDER] Consistent artist='{final_artist}' and album='{final_album}'; doing single MB/Spotify lookup.")
        artist_mbid = None
        rg_mbid = None
        try:
            artist_mbid = find_musicbrainz_artist_mbid(final_artist)
        except Exception:
            pass
        try:
            rg_mbid = find_musicbrainz_rg_mbid(final_album, final_artist)
        except Exception:
            pass
        spotify_artist_id = None
        spotify_album_id = None
        try:
            if sp:
                spotify_artist_id = find_spotify_artist_id(final_artist, sp)
        except Exception:
            pass
        try:
            if sp:
                spotify_album_id = find_spotify_album_id(final_album, final_artist, sp)
        except Exception:
            pass
        for fpath in all_tags_by_file:
            tags = all_tags_by_file[fpath]
            if artist_mbid:
                tags["----:com.apple.iTunes:MusicBrainz Artist Id"] = artist_mbid
            if rg_mbid:
                tags["----:com.apple.iTunes:MusicBrainz Release Group Id"] = rg_mbid
            if spotify_artist_id:
                tags["----:com.apple.iTunes:spotify_artist_ID"] = spotify_artist_id
            if spotify_album_id:
                tags["----:com.apple.iTunes:spotify_album_ID"] = spotify_album_id
            rewrite_tags(fpath, tags)

    # --- Process each track in the album folder ---
    for i, file_path in enumerate(files_in_folder, start=1):
        logger.info(f"Processing track {i}/{len(files_in_folder)}: {file_path}")
        temp_tags = extract_desired_tags(file_path)
        if not temp_tags:
            logger.info("    No desired tags found, skipping track.")
            continue
        if unified_year:
            temp_tags["©day"] = unified_year

        # B. Determine spawn_id (admin vs. user mode)
        spawn_id_tag = "----:com.apple.iTunes:spawn_ID"
        spawn_id_val = None
        if spawn_id_tag in temp_tags:
            val_raw = temp_tags[spawn_id_tag]
            if isinstance(val_raw, list) and val_raw:
                val_raw = val_raw[0]
            if isinstance(val_raw, bytes):
                val_raw = val_raw.decode("utf-8", errors="replace")
            spawn_id_val = str(val_raw).strip()
        else:
            if is_admin:
                with sqlite3.connect(DB_PATH) as temp_conn:
                    found_match = check_for_potential_match_in_db(
                        temp_conn, temp_tags, current_spawn_id=None, similarity_threshold=0.90
                    )
                if found_match:
                    logger.info("Admin mode: Found partial match in DB.")
                    matched_id = fetch_matching_spawn_id_from_db(temp_tags)
                    if matched_id:
                        spawn_id_val = matched_id
                    else:
                        spawn_id_val = generate_spawn_id()
                else:
                    spawn_id_val = generate_spawn_id()
                temp_tags[spawn_id_tag] = spawn_id_val
            else:
                # In user mode, do not generate a new spawn_id.
                spawn_id_val = None

        # C. Detect audio format and build target filename
        safe_file_path = shlex.quote(file_path)
        ffprobe_output = subprocess.getoutput(f"ffprobe -i {safe_file_path} 2>&1")
        format_type = detect_audio_format(ffprobe_output)
        def _decode_str(v):
            if isinstance(v, list) and v:
                v = v[0]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            return str(v).strip() if v else ""
        artist_name = _decode_str(temp_tags.get("©ART")) or "Unknown"
        album_name  = _decode_str(temp_tags.get("©alb")) or "Unknown"
        artist_dir = sanitize_for_directory(artist_name)
        album_dir  = sanitize_for_directory(album_name)
        subpath    = os.path.join(artist_dir, album_dir)
        base = os.path.splitext(os.path.basename(file_path))[0]
        temp_filename = f"temp_{base}.m4a"
        target_file   = os.path.join(OUTPUT_PARENT_DIR, subpath, temp_filename)
        os.makedirs(os.path.dirname(target_file), exist_ok=True)

        # D. Convert/repackage based on format type
        if format_type == "FLAC":
            logger.info("    FLAC detected: converting to ALAC (M4A)...")
            convert_flac_to_alac(file_path, target_file)
        elif format_type == "ALAC":
            logger.info("    ALAC detected: repackaging...")
            repackage_alac_file(file_path, target_file)
        elif format_type == "AAC":
            logger.info("    AAC detected: repackaging via MP4Box...")
            clean_or_repackage_aac(file_path, target_file)
        else:
            logger.info("    Unknown format: attempting ALAC repackaging...")
            repackage_alac_file(file_path, target_file)
        rewrite_tags(target_file, temp_tags)
        temp_tags = extract_desired_tags(target_file)

        # E. Database updates (spawn_id assignment and duplicate-check logic)
        if spawn_id_val:
            logger.info(f"[DB Update] spawn_id_val is: {spawn_id_val}")
            db_tags = fetch_tags_from_db(spawn_id_val)
            if db_tags is not None:
                if is_admin:
                    new_id = handle_existing_spawn_id(
                        spawn_id_val, db_tags, temp_tags, [False], target_file, set(), set()
                    )
                    if new_id and new_id != spawn_id_val:
                        logger.info(f"Track overridden; using new spawn_id: {new_id}")
                        spawn_id_val = new_id
                        temp_tags[spawn_id_tag] = new_id
                else:
                    # For user mode: merge catalog metadata from spawn_catalog.db into temp_tags
                    # Exclude only fields that are file-specific (e.g., Acoustid Fingerprint and ReplayGain fields)
                    exclude_keys = {
                        "----:com.apple.iTunes:Acoustid Fingerprint",
                        "----:com.apple.iTunes:Acoustid Id",
                        "----:com.apple.iTunes:replaygain_album_gain",
                        "----:com.apple.iTunes:replaygain_album_peak",
                        "----:com.apple.iTunes:replaygain_track_gain",
                        "----:com.apple.iTunes:replaygain_track_peak",
                    }
                    for key, value in db_tags.items():
                        if key not in exclude_keys:
                            temp_tags[key] = value

                    # Normalize catalog spawnre to derive the top genre
                    spawnre_raw = db_tags.get("----:com.apple.iTunes:spawnre", [])
                    if isinstance(spawnre_raw, list) and spawnre_raw:
                        top_genre = spawnre_raw[0]
                        temp_tags["©gen"] = [top_genre]
                        logger.info(f"[User Mode] Set ©gen tag to top spawnre genre: {top_genre}")
                    else:
                        logger.warning(f"[User Mode] No spawnre genre list found in DB for spawn_id: {spawn_id_val}")

                    # Also copy spawnre_hex if present
                    if "----:com.apple.iTunes:spawnre_hex" in db_tags:
                        temp_tags["----:com.apple.iTunes:spawnre_hex"] = db_tags["----:com.apple.iTunes:spawnre_hex"]

                    # Remove unwanted extra keys, e.g. '©wrt'
                    if "©wrt" in temp_tags:
                        del temp_tags["©wrt"]

                    # Ensure metadata_rev is set from catalog metadata
                    metadata_rev_val = db_tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
                    temp_tags["----:com.apple.iTunes:metadata_rev"] = metadata_rev_val

                    logger.info(f"[User Mode] Merged temp_tags before storing: {temp_tags}")
                    store_tags_in_user_db(
                        spawn_id_val,
                        temp_tags,
                        metadata_rev=metadata_rev_val,
                        table="cat_tracks",
                        lib_db_path=USER_DB_PATH
                    )
                    logger.info(f"[User Mode] Stored catalog metadata for spawn_id {spawn_id_val} with top genre '{top_genre}'")
            else:
                logger.info("[DB Update] No catalog metadata found for spawn_id.")
                temp_tags["----:com.apple.iTunes:metadata_rev"] = "AAA"
                if is_admin:
                    store_tags_in_db(DB_PATH, spawn_id_val, temp_tags, metadata_rev="AAA")
                    new_admin_spawn_ids.add(spawn_id_val)
                else:
                    logger.info("[user] No match in catalog DB; deferring insertion.")
        else:
            # For user mode, if no spawn_id is present, generate a local_ID.
            if is_admin:
                new_id = generate_spawn_id()
                logger.info(f"[admin] Assigning new spawn_id: {new_id}")
                temp_tags[spawn_id_tag] = new_id
                temp_tags["----:com.apple.iTunes:metadata_rev"] = "AAA"
                store_tags_in_db(DB_PATH, new_id, temp_tags, metadata_rev="AAA")
                spawn_id_val = new_id
            else:
                with sqlite3.connect(DB_PATH) as temp_conn:
                    found_match = check_for_potential_match_in_db(temp_conn, temp_tags, current_spawn_id=None)
                if found_match:
                    matched_id = fetch_matching_spawn_id_from_db(temp_tags)
                    if matched_id:
                        logger.info(f"[user] Found match in catalog DB: spawn_id={matched_id}")
                        temp_tags[spawn_id_tag] = matched_id
                        matched_db_tags = fetch_tags_from_db(matched_id)
                        existing_rev = matched_db_tags.get("----:com.apple.iTunes:metadata_rev", "AAA") if matched_db_tags else "AAA"
                        temp_tags["----:com.apple.iTunes:metadata_rev"] = existing_rev
                        store_tags_in_user_db(
                            matched_id, temp_tags, metadata_rev=existing_rev,
                            table="cat_tracks", lib_db_path=USER_DB_PATH
                        )
                    else:
                        logger.info("[user] Match found but no spawn_id retrieved; deferring insertion.")
                else:
                    logger.info("[user] No match found in catalog DB; deferring insertion.")

        # F. Update additional tags (e.g., AcoustID, MBIDs, Spotify info)
        confirm_or_update_tags(temp_tags, target_file)
        if is_admin:
            update_disc_and_track_numbers_from_mbz(temp_tags, [False], target_file, spawn_id_val)
            def _tag_to_str(val):
                if isinstance(val, list) and val:
                    val = val[0]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                return str(val).strip() if val else ""
            artist_name = _tag_to_str(temp_tags.get("©ART"))
            track_title = _tag_to_str(temp_tags.get("©nam"))
            embedded_gen = _tag_to_str(temp_tags.get("©gen"))
            last_fm_genres = fetch_genre_lastfm(artist_name, track_title, api_key=lastfm_api_key)
            spotify_genres = get_spotify_genres(artist_name, sp)
            mb_genres = get_musicbrainz_genres(artist_name)
            if artist_name or track_title:
                # In admin mode, compute spawnre if needed
                process_spawnre(
                    file_path=target_file,
                    artist_name=artist_name,
                    track_title=track_title,
                    embedded_genre=embedded_gen,
                    last_fm_genres=last_fm_genres,
                    spotify_genres=spotify_genres,
                    musicbrainz_genres=mb_genres,
                    temp_tags=temp_tags
                )
                normalize_spawnre(temp_tags)
        else:
            # User Mode: Retrieve spawn_id and local_id as usual
            spawn_id_val = temp_tags.get("----:com.apple.iTunes:spawn_ID")
            if isinstance(spawn_id_val, list) and spawn_id_val:
                spawn_id_val = spawn_id_val[0]
            if isinstance(spawn_id_val, bytes):
                spawn_id_val = spawn_id_val.decode("utf-8", errors="replace")
            spawn_id_val = str(spawn_id_val).strip() if spawn_id_val else None
            local_id = temp_tags.get("----:com.apple.iTunes:local_ID")
            if isinstance(local_id, list) and local_id:
                local_id = local_id[0]
            if isinstance(local_id, bytes):
                local_id = local_id.decode("utf-8", errors="replace")
            local_id = str(local_id).strip() if local_id else None
            if not spawn_id_val and not local_id:
                local_id = generate_local_id()
                temp_tags["----:com.apple.iTunes:local_ID"] = [local_id.encode("utf-8")]
                rewrite_tags(target_file, temp_tags)
            id_str = spawn_id_val or local_id
            if not id_str:
                raise ValueError(f"No spawn_id or local_id available for file: {target_file}")
            logger.info(f"[User Mode] Retrieved spawnre info for spawn_id: {spawn_id_val}")
            # Try fetching genre info from the catalog database:
            db_tags = fetch_tags_from_db(spawn_id_val) if spawn_id_val else {}
            if db_tags:
                for key in ["----:com.apple.iTunes:spawnre", "----:com.apple.iTunes:spawnre_hex"]:
                    if key in db_tags:
                        temp_tags[key] = db_tags[key]
                        logger.info(f"[User Mode] Set {key} to: {db_tags[key]}")
                spawnre_tag = db_tags.get("----:com.apple.iTunes:spawnre", [])
                if isinstance(spawnre_tag, bytes):
                    spawnre_tag = spawnre_tag.decode("utf-8", errors="replace")
                if isinstance(spawnre_tag, list) and spawnre_tag:
                    top_genre = spawnre_tag[0]
                    temp_tags["©gen"] = [top_genre]
                    logger.info(f"[User Mode] Set ©gen tag to top spawnre genre: {top_genre}")
                else:
                    logger.info("[User Mode] DB entry exists but no spawnre tag found; computing spawnre.")
                    # Fall back to computing spawnre for local tracks:
                    def _tag_to_str(val):
                        if isinstance(val, list) and val:
                            val = val[0]
                        if isinstance(val, bytes):
                            val = val.decode("utf-8", errors="replace")
                        return str(val).strip() if val else ""
                    artist_name = _tag_to_str(temp_tags.get("©ART"))
                    track_title = _tag_to_str(temp_tags.get("©nam"))
                    embedded_gen = _tag_to_str(temp_tags.get("©gen"))
                    last_fm_genres = fetch_genre_lastfm(artist_name, track_title, api_key=lastfm_api_key)
                    spotify_genres = get_spotify_genres(artist_name, sp)
                    mb_genres = get_musicbrainz_genres(artist_name)
                    process_spawnre(
                        file_path=target_file,
                        artist_name=artist_name,
                        track_title=track_title,
                        embedded_genre=embedded_gen,
                        last_fm_genres=last_fm_genres,
                        spotify_genres=spotify_genres,
                        musicbrainz_genres=mb_genres,
                        temp_tags=temp_tags
                    )
                    # After computing, use the computed spawnre to update ©gen.
                    spawnre_tag = temp_tags.get("----:com.apple.iTunes:spawnre", [])
                    if isinstance(spawnre_tag, list) and spawnre_tag:
                        top_genre = spawnre_tag[0]
                    elif isinstance(spawnre_tag, bytes):
                        top_genre = spawnre_tag.decode("utf-8", errors="replace").split(',')[0].strip()
                    elif isinstance(spawnre_tag, str):
                        top_genre = spawnre_tag.split(',')[0].strip()
                    else:
                        top_genre = ""

                    if top_genre:
                        temp_tags["©gen"] = [top_genre]
                        logger.info(f"[User Mode] Computed and set ©gen tag to spawnre_tag: {top_genre}")
                    else:
                        logger.warning("[User Mode] Unable to compute spawnre tag for local track.")
            else:
                logger.info(f"[User Mode] No DB entry found for spawn_id: {spawn_id_val}; computing spawnre for local track.")
                def _tag_to_str(val):
                    if isinstance(val, list) and val:
                        val = val[0]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    return str(val).strip() if val else ""
                artist_name = _tag_to_str(temp_tags.get("©ART"))
                track_title = _tag_to_str(temp_tags.get("©nam"))
                embedded_gen = _tag_to_str(temp_tags.get("©gen"))
                last_fm_genres = fetch_genre_lastfm(artist_name, track_title, api_key=lastfm_api_key)
                spotify_genres = get_spotify_genres(artist_name, sp)
                mb_genres = get_musicbrainz_genres(artist_name)
                process_spawnre(
                    file_path=target_file,
                    artist_name=artist_name,
                    track_title=track_title,
                    embedded_genre=embedded_gen,
                    last_fm_genres=last_fm_genres,
                    spotify_genres=spotify_genres,
                    musicbrainz_genres=mb_genres,
                    temp_tags=temp_tags
                )
                # Ensure metadata_rev is preserved
                metadata_rev_val = temp_tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
                temp_tags["----:com.apple.iTunes:metadata_rev"] = metadata_rev_val
                # After computing, update ©gen to hold only the top genre:
                spawnre_tag = temp_tags.get("----:com.apple.iTunes:spawnre", b"")
                if isinstance(spawnre_tag, bytes):
                    spawnre_tag = spawnre_tag.decode("utf-8", errors="replace")
                if spawnre_tag:
                    top_genre = spawnre_tag.split(',')[0].strip().lower()
                    temp_tags["©gen"] = [top_genre]
                    logger.info(f"[User Mode] Computed and set ©gen tag to top spawnre genre: {top_genre}")
                else:
                    logger.error(f"[User Mode] Unable to compute spawnre tag for track: {target_file}")

        # G. Save final state for this track
        def _decode_id_field(raw):
            if isinstance(raw, list) and raw:
                raw = raw[0]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            return str(raw).strip() if raw else ""
        spawn_id_val = _decode_id_field(temp_tags.get("----:com.apple.iTunes:spawn_ID"))
        local_id = _decode_id_field(temp_tags.get("----:com.apple.iTunes:local_ID"))
        id_str = spawn_id_val or local_id
        if not id_str:
            logger.error(f"[ID ERROR] Could not resolve spawn_id or local_id in tags:\n{temp_tags}")
            raise ValueError(f"No spawn_id or local_id for file: {target_file}")
        cleaned_files_map[target_file] = temp_tags
        all_tracks.append((target_file, temp_tags, id_str))

    # --- Album-level post-processing: ReplayGain & album art ---
    if cleaned_files_map:
        if is_admin:
            run_replaygain_on_folder(cleaned_files_map, DB_PATH, mode="admin")
        else:
            run_replaygain_on_folder(cleaned_files_map, DB_PATH, mode="user", lib_db_path=USER_DB_PATH)

        # Use one track's tags as representative for album art lookup
        temp_tags = list(cleaned_files_map.values())[-1]
        def _tag_to_str(val):
            if isinstance(val, list) and val:
                val = val[0]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            return str(val).strip() if val else ""
        spotify_track_id = _tag_to_str(temp_tags.get("----:com.apple.iTunes:spotify_track_ID"))
        musicbrainz_mbid = _tag_to_str(temp_tags.get("----:com.apple.iTunes:MusicBrainz Release Group Id"))
        artist_name = _tag_to_str(temp_tags.get("©ART")) or "Unknown"
        album_name  = _tag_to_str(temp_tags.get("©alb")) or "Unknown"
        artist_dir = sanitize_for_directory(artist_name)
        album_dir  = sanitize_for_directory(album_name)
        # Create the final output album directory:
        final_album_dir = os.path.join(OUTPUT_PARENT_DIR, artist_dir, album_dir)
        os.makedirs(final_album_dir, exist_ok=True)
        album_art_path = os.path.join(final_album_dir, "cover.jpg")
        
        # 1. First check: is cover.jpg present in the output folder?
        if os.path.exists(album_art_path):
            logger.info(f"Found existing album art at {album_art_path}.")
            logger.info(f"Embedding into tracks.")
            for final_path, final_tags in cleaned_files_map.items():
                embed_cover_art_into_file(final_path, album_art_path, final_tags)
        else:
            # 2. Next check: is cover.jpg present in the input album folder?
            input_album_art_path = os.path.join(folder, "cover.jpg")
            if os.path.exists(input_album_art_path):
                logger.info(f"Found existing album art in input folder: {input_album_art_path}.")
                logger.info(f"Copying to final album directory and embedding into tracks.")
                shutil.copy2(input_album_art_path, album_art_path)
                for final_path, final_tags in cleaned_files_map.items():
                    embed_cover_art_into_file(final_path, album_art_path, final_tags)
            else:
                # 3. Otherwise, perform album art lookup.
                logger.info(f"No cover.jpg found in output or input folder; performing album art lookup.")
                album_art_url = get_album_art(
                    track_id=spotify_track_id,
                    mbid=musicbrainz_mbid,
                    client_id=SPOTIFY_CLIENT_ID,
                    client_secret=SPOTIFY_CLIENT_SECRET,
                    album_folder=final_album_dir
                )
                if album_art_url:
                    save_album_art(album_art_url, album_art_path)
                    for final_path, final_tags in cleaned_files_map.items():
                        embed_cover_art_into_file(final_path, album_art_path, final_tags)
                else:
                    logger.info(f"No album art found for '{artist_name} - {album_name}'.")
        for final_path, final_tags in cleaned_files_map.items():
            rewrite_tags(final_path, final_tags)

    album_data = {
        "folder": folder,
        "cleaned_files_map": cleaned_files_map,
        "all_tracks": all_tracks
    }
    return album_data



###############################################################################
# Process Audio Files in Mini-Batches (and Finalize Each Batch)
###############################################################################
def process_audio_files_in_batches(input_dir, batch_threshold=10, keep_matched=False,
                                   lastfm_api_key="", sp=None, is_admin=True,
                                   USER_DB_PATH=None, EMBED_PATH=None):
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.startswith('.') or f.startswith('._'):
                continue
            if f.lower().endswith(('.m4a', '.mp4', '.flac')):
                full_path = os.path.join(root, f)
                audio_files.append(full_path)
    if not audio_files:
        logger.info(f"No audio files found in '{input_dir}'")
        return
    folder_map = defaultdict(list)
    for f in audio_files:
        parent_folder = os.path.abspath(os.path.dirname(f))
        folder_map[parent_folder].append(f)
    album_folders = list(folder_map.items())
    total_albums = len(album_folders)
    logger.info(f"Found {total_albums} album folders to process.")
    # Process each mini-batch fully before moving on
    for i in range(0, total_albums, batch_threshold):
        mini_batch = album_folders[i : i + batch_threshold]
        logger.info(f"\n--- Processing mini-batch of album folders {i} to {i + len(mini_batch) - 1} ---")
        mini_batch_album_data = []
        for folder, files_in_folder in mini_batch:
            album_data = process_album_folder(folder, files_in_folder, keep_matched,
                                              lastfm_api_key, sp, is_admin, USER_DB_PATH)
            if album_data:
                mini_batch_album_data.append(album_data)
        if is_admin:
            finalize_spawnre_tags()
        # Finalize this mini-batch completely
        post_process_imported_tracks(mini_batch_album_data, is_admin, keep_matched, lastfm_api_key, sp)
        logger.info(f"Completed mini-batch of {len(mini_batch)} album folders.")

###############################################################################
# Post-Process Imported Tracks for a Batch
###############################################################################
def post_process_imported_tracks(all_album_data, is_admin, keep_matched, lastfm_api_key, sp):

    global EMBED_PATH, USER_EMBED_PATH

    # Aggregate tracks from the current mini-batch
    all_tracks = []

    for album_data in all_album_data:
        all_tracks.extend(album_data.get("all_tracks", []))
    # Normalize all_tracks to 3-tuple form: (track_path, track_tags, id_str)
    all_tracks = [(i[0], i[1], i[2] if len(i) > 2 else None) for i in all_tracks]
    if not all_tracks:
        logger.info("No tracks processed in this mini-batch for final post-processing.")
        return

    logger.info("")
    logger.info("=== Final post-processing for current mini-batch: Rewriting tags ===")
    for (track_path, _, _) in all_tracks:
        final_tags = extract_desired_tags(track_path)
        if not final_tags:
            logger.info(f"No desired tags found in final pass: {track_path}")
            continue
        rewrite_tags(track_path, final_tags)
        logger.debug(f"Final tags for {track_path}: {final_tags}")

    print()
    logger.info("=== Renaming files to D-TT [spawn_id] - title.m4a ===")
    for idx, (old_path, track_tags, id_str) in enumerate(all_tracks):
        disc_tag = track_tags.get("disk")
        track_tag = track_tags.get("trkn")
        disc_main = disc_tag[0][0] if (disc_tag and isinstance(disc_tag, list) and disc_tag) else 1
        track_main = track_tag[0][0] if (track_tag and isinstance(track_tag, list) and track_tag) else 0
        title_raw = track_tags.get("©nam")
        if isinstance(title_raw, list) and title_raw:
            title_raw = title_raw[0]
        if isinstance(title_raw, bytes):
            title_raw = title_raw.decode("utf-8", errors="replace")
        track_title_str = str(title_raw).strip() if title_raw else "untitled"
        spawn_id_str = id_str
        logger.info(f"Resolved spawn_id_str for {old_path}: {spawn_id_str}")
        new_filename = build_d_tt_title_filename(disc_main, track_main, track_title_str, spawn_id_str=spawn_id_str)
        old_dir = os.path.dirname(old_path)
        new_path = os.path.join(old_dir, new_filename)
        if os.path.exists(new_path):
            # Library already has this file—skip overwriting it.
            logger.info(f"Target exists, skipping rename of duplicate import: {new_path}")
            # Keep old_path (the temp import) around so it can be deleted below
            all_tracks[idx] = (old_path, track_tags, id_str)
        else:
            try:
                os.rename(old_path, new_path)
                logger.info(f"Renamed => {new_path}")
                all_tracks[idx] = (new_path, track_tags, id_str)
            except OSError as e:
                logger.warning(f"Unable to rename file: {e}")
        try:
            audio = MP4(new_path)
            audio["----:com.apple.iTunes:spawn_ID"] = [MP4FreeForm(spawn_id_str.encode("utf-8"))]
            if spawn_id_str.startswith("xxx"):
                audio["----:com.apple.iTunes:local_ID"] = [MP4FreeForm(spawn_id_str.encode("utf-8"))]
            audio.save()
            logger.info(f"Embedded spawn_ID ({spawn_id_str}) into tags for: {new_path}")
        except Exception as e:
            logger.warning(f"Failed to write spawn_ID to {new_path}: {e}")

    logger.info("All files renamed to D-TT [spawn_id] - title.m4a format.\n")

    # ─── Load existing embeddings ───
    existing_embeddings = {}
    if os.path.exists(EMBED_PATH):
        with open(EMBED_PATH, "rb") as f:
            existing_embeddings = pickle.load(f)

    # collect any spawn_ids the user confirms as duplicates
    removed_ids = set()

    # Build maps from all_tracks (which is in scope) for the embedding check
    # only include entries where track_tags isn’t None
    current_tags_map   = { id_str: tags  for (path,tags,id_str) in all_tracks if tags }
    new_file_paths_map = { id_str: path  for (path,tags,id_str) in all_tracks if tags }

    # ─── Embeddings Generation & Duplicate Check ───
    if is_admin:
        for (track_path, track_tags, id_str) in all_tracks:
            if not track_tags:
                continue
            # extract spawn_id_str as before...
            spawn_vec_id = id_str
            generate_deejai_embedding_for_track(track_path, spawn_vec_id)
        if spawn_id_to_embeds:
            # run duplicate check, remove confirmed duplicates from the embed dict
            newly_removed = check_for_potential_match_in_mbed(
                current_tags_map,
                new_file_paths_map,
                spawn_id_to_embeds,
                existing_embeddings,
                threshold=EMBEDDING_DUPLICATE_THRESHOLD
            )
            removed_ids.update(newly_removed)
            # drop those from all_tracks so DB‐insertion skips them
            for idx, (p, tags, sid) in enumerate(all_tracks):
                if sid in removed_ids:
                    all_tracks[idx] = (p, None, sid)
            # now save only the survivors
            existing_embeddings.update(spawn_id_to_embeds)
            save_combined_embeddings(EMBED_PATH, existing_embeddings)
            spawn_id_to_embeds.clear()
    else:
        for (track_path, track_tags, id_str) in all_tracks:
            if not track_tags:
                continue
            generate_deejai_embedding_for_track(track_path, id_str)
            if id_str in spawn_id_to_embeds:
                local_user_embeddings[id_str] = spawn_id_to_embeds.pop(id_str)
        if local_user_embeddings:
            newly_removed = check_for_potential_match_in_mbed(
                current_tags_map,
                new_file_paths_map,
                local_user_embeddings,
                existing_embeddings,
                threshold=EMBEDDING_DUPLICATE_THRESHOLD
            )
            removed_ids.update(newly_removed)
            for idx, (p, tags, sid) in enumerate(all_tracks):
                if sid in removed_ids:
                    all_tracks[idx] = (p, None, sid)
            save_combined_embeddings(USER_EMBED_PATH, local_user_embeddings)
            local_user_embeddings.clear()

    logger.info("=== Duplicate-checking & Database Update for current mini-batch ===")

    if is_admin:
        latest_rev = get_latest_db_rev(DB_PATH)
        logger.info(f"Current DB revision: {latest_rev}\n")
        conn = sqlite3.connect(DB_PATH)
        old_count = get_total_track_count(DB_PATH)
    else:
        conn = sqlite3.connect(DB_PATH)
        old_count = get_total_track_count(DB_PATH)
        latest_rev = None

    newly_imported_tracks = []
    for idx, (track_path, track_tags, id_str) in enumerate(all_tracks):
        # skip anything the user marked as an embedding‐duplicate
        if id_str in removed_ids:
            continue
        final_temp_tags = extract_desired_tags(track_path)
        if not final_temp_tags:
            logger.info(f"No tags found for DB insertion: {track_path}")
            continue
        # Extract spawn_id from tags
        spawn_id_data = final_temp_tags.get("----:com.apple.iTunes:spawn_ID")
        if isinstance(spawn_id_data, list) and spawn_id_data:
            spawn_id_data = spawn_id_data[0]
        if isinstance(spawn_id_data, bytes):
            spawn_id_data = spawn_id_data.decode("utf-8", errors="replace")
        spawn_id_str = str(spawn_id_data).strip() if spawn_id_data else None

        if spawn_id_str:
            # Check if this spawn_id is already in the catalog DB.
            db_tags = fetch_tags_from_db(spawn_id_str)
            if db_tags is not None:
                # For admin mode: if this spawn_id was just inserted, do not treat it as a duplicate.
                if is_admin and spawn_id_str in new_admin_spawn_ids:
                    newly_imported_tracks.append((spawn_id_str, final_temp_tags, track_path))
                else:
                    if not keep_matched:
                        if is_admin:
                            logger.info(f"[ADMIN] Duplicate found for spawn_id '{spawn_id_str}'. Not importing duplicate track: {track_path}.")
                        else:
                            store_tags_in_user_db( ... )
                        # Remove duplicate file
                        relative_path = os.path.relpath(track_path, start=OUTPUT_PARENT_DIR)
                        base_no_ext, _ = os.path.splitext(relative_path)
                        licn_root = os.path.join(os.path.dirname(OUTPUT_PARENT_DIR), "aux", "user", "licn")
                        new_txt_path = os.path.join(licn_root, base_no_ext + ".txt")
                        os.makedirs(os.path.dirname(new_txt_path), exist_ok=True)
                        try:
                            with open(new_txt_path, "w", encoding="utf-8") as f:
                                f.write("")
                            logger.info(f"Blank txt file created => {new_txt_path}")
                            os.remove(track_path)
                            logger.info(f"Removed file: {track_path}")
                        except Exception as e:
                            logger.warning(f"Error handling file {track_path}: {e}")
                        all_tracks[idx] = (new_txt_path, None, spawn_id_str)
                        continue
                    else:
                        newly_imported_tracks.append((spawn_id_str, final_temp_tags, track_path))
            else:
                # No existing DB entry; treat as a new track.
                newly_imported_tracks.append((spawn_id_str, final_temp_tags, track_path))
        else:
            # No spawn_id found: fallback to local_id.
            local_id = final_temp_tags.get("----:com.apple.iTunes:local_ID")
            if isinstance(local_id, list) and local_id:
                local_id = local_id[0]
            if isinstance(local_id, bytes):
                local_id = local_id.decode("utf-8", errors="replace")
            local_id = str(local_id).strip() if local_id else ""
            if not local_id:
                local_id = generate_local_id()
                final_temp_tags["----:com.apple.iTunes:local_ID"] = [local_id.encode("utf-8")]
                rewrite_tags(track_path, final_temp_tags)
                final_temp_tags = extract_desired_tags(track_path)
            logger.info(f"No spawn_id found for {track_path}. Using local_id: {local_id}")
            newly_imported_tracks.append((local_id, final_temp_tags, track_path))
            all_tracks[idx] = (track_path, final_temp_tags, local_id)
    conn.close()

    logger.info(f"Newly imported tracks count: {len(newly_imported_tracks)}")
    if not is_admin:
        for (id_str, final_temp_tags, track_path) in newly_imported_tracks:
            store_tags_in_user_db(
                spawn_id=id_str,
                tag_dict=final_temp_tags,
                metadata_rev=final_temp_tags.get("----:com.apple.iTunes:metadata_rev", "AAA"),
                table="lib_tracks",
                lib_db_path=USER_DB_PATH
            )
            rewrite_tags(track_path, final_temp_tags)
        logger.info("Final user database update complete (user mode).")
    else:
        conn = sqlite3.connect(DB_PATH)
        for (id_str, final_temp_tags, track_path) in newly_imported_tracks:
            db_tags = fetch_tags_from_db(id_str)
            if db_tags is None:
                if "----:com.apple.iTunes:metadata_rev" not in final_temp_tags:
                    final_temp_tags["----:com.apple.iTunes:metadata_rev"] = "AAA"
                store_tags_in_db(DB_PATH, id_str, final_temp_tags,
                                 metadata_rev=final_temp_tags["----:com.apple.iTunes:metadata_rev"])
                rewrite_tags(track_path, final_temp_tags)
            else:
                old_rev = db_tags.get("----:com.apple.iTunes:metadata_rev", "AAA")
                new_rev = final_temp_tags.get("----:com.apple.iTunes:metadata_rev", old_rev)
                final_temp_tags["----:com.apple.iTunes:metadata_rev"] = new_rev
                store_tags_in_db(DB_PATH, id_str, final_temp_tags, metadata_rev=new_rev)
                rewrite_tags(track_path, final_temp_tags)
        conn.close()
        logger.info("Final database update complete (admin mode).")
        if is_admin:
            if not latest_rev:
                old_count_for_rev = 0
            else:
                parts = latest_rev.split(".")
                old_count_for_rev = int(parts[1]) if len(parts) > 1 else 0
            new_count = get_total_track_count(DB_PATH)
            # only bump the revision if new tracks were actually imported during this run
            if newly_imported_tracks and new_count > old_count_for_rev:
                db_rev_val = next_db_revision(latest_rev or "", old_count_for_rev, new_count)
                store_db_revision(DB_PATH, db_rev_val)
                logger.info(f"Updated db_rev='{db_rev_val}' (new tracks imported).")
            else:
                logger.info("No new tracks imported; db_rev not incremented.")
        # Cleanup any stray catalog rows for embedding‐duplicates
        if is_admin and removed_ids:
            conn = sqlite3.connect(DB_PATH)
            cur  = conn.cursor()
            for rid in removed_ids:
                try:
                    cur.execute("DELETE FROM tracks WHERE spawn_id = ?", (rid,))
                    logger.info(f"Purged stray duplicate catalog record for spawn_id {rid}")
                except Exception as e:
                    logger.warning(f"Failed to purge catalog record for {rid}: {e}")
            conn.commit()
            conn.close()
        else:
            logger.info("No database changes; db_rev not incremented.")

    # --- Assign spawnre (genre) info to each track and update database entries ---
    logger.info("Assigning spawnre_tag to each track and updating database entries\n")
    for idx, (track_path, track_tags, id_str) in enumerate(all_tracks):
        if track_tags is None:
            continue

        def _tag_to_str(val):
            if isinstance(val, list) and val:
                val = val[0]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            return str(val).strip() if val else ""

        if is_admin:
            # For admin mode, use the finalized per-artist top genre from artist_spawnre_tags.
            artist_name = _tag_to_str(track_tags.get("©ART"))
            artist_lower = artist_name.lower() if artist_name else ""
            top_genre = artist_spawnre_tags.get(artist_lower, "")
            if not top_genre:
                logger.warning(f"No finalized top genre found for artist {artist_name} (track: {track_path}).")
            # Store the top genre (as a single-item list) in the standard tag.
            track_tags["©gen"] = [top_genre]
            logger.info(f"Track: {track_path} | Artist: {artist_name} => Set ©gen to finalized top genre: '{top_genre}'")
            rewrite_tags(track_path, track_tags)
            store_tags_in_db(DB_PATH, id_str, track_tags,
                             metadata_rev=track_tags.get("----:com.apple.iTunes:metadata_rev", "AAA"))
        else:
            # For user mode, re-fetch the spawnre data from the catalog DB and update ©gen.
            spawn_id_val = _tag_to_str(track_tags.get("----:com.apple.iTunes:spawn_ID"))
            if spawn_id_val:
                db_tags = fetch_tags_from_db(spawn_id_val)
                if db_tags and "----:com.apple.iTunes:spawnre" in db_tags:
                    spawnre_value = db_tags.get("----:com.apple.iTunes:spawnre")
                    if isinstance(spawnre_value, list) and spawnre_value:
                        top_genre = spawnre_value[0]
                    elif isinstance(spawnre_value, bytes):
                        top_genre = spawnre_value.decode("utf-8", errors="replace").split(',')[0].strip()
                    else:
                        top_genre = str(spawnre_value).split(',')[0].strip()
                    # Update the standard genre tag with only the top genre (as a single-item list)
                    track_tags["©gen"] = [top_genre]
                    logger.info(f"[User Mode] Updated ©gen tag to top spawnre genre: {top_genre} for track: {track_path}")
                    rewrite_tags(track_path, track_tags)
                    store_tags_in_db(DB_PATH, id_str, track_tags,
                                     metadata_rev=track_tags.get("----:com.apple.iTunes:metadata_rev", "AAA"))
                else:
                    logger.info(f"[User Mode] No catalog spawnre found for spawn_id: {spawn_id_val} for track: {track_path}. Leaving tags unchanged.")
                    rewrite_tags(track_path, track_tags)
            else:
                logger.info(f"[User Mode] No spawn_id available for track: {track_path}. Leaving tags unchanged.")
                rewrite_tags(track_path, track_tags)
    lib_base = os.path.dirname(os.path.dirname(OUTPUT_PARENT_DIR))
    print()
    logger.info("Creating symlinks for imported tracks...")
    linx_dir = os.path.join(lib_base, "Spawn", "aux", "user", "linx")
    os.makedirs(linx_dir, exist_ok=True)
    for (track_path, track_tags, id_str) in all_tracks:
        if track_tags is None:
            continue
        if id_str:
            create_symlink_for_track(track_path, lib_base, id_str)
        else:
            logger.warning(f"No spawn_id or local_id for track {track_path}; skipping symlink.")
    print()
    logger.info("Generating album M3U playlists for newly imported albums...")
    album_folders_set = set()
    for (track_path, track_tags, id_str) in all_tracks:
        if track_tags is None:
            continue
        album_folder = os.path.dirname(track_path)
        album_folders_set.add(album_folder)
    for album_folder in album_folders_set:
        logger.info(f"Generating album playlist for: {album_folder}")
        generate_playlist_for_album(album_folder, linx_dir, os.path.join(os.path.dirname(OUTPUT_PARENT_DIR), "aux", "user", "albm"), use_absolute_paths=False)
    logger.info("=== Creating import M3U playlist for tracks not in catalog ===")
    base_music_dir = os.path.abspath(OUTPUT_PARENT_DIR)
    generate_import_playlist(all_tracks, base_music_dir, PLAYLISTS_DIR)
    music_root = os.path.abspath(OUTPUT_PARENT_DIR)
    remove_empty_music_dirs(music_root)
    logger.info("Post-processing complete for current mini-batch.")
    
###############################################################################
# Refactored run_import Function
###############################################################################
def run_import(output_path, music_path, skip_prompts=False, keep_matched=False,
               acoustid_key=None, spotify_client_id=None, spotify_client_secret=None,
               lastfm_key=None, is_admin=False):
    # Caffeinate to prevent sleep
    try:
        caffeinate_process = subprocess.Popen(['caffeinate', '-i'])
        logger.debug("Caffeinate started to prevent sleep.")
    except Exception as e:
        logger.warning("Failed to start caffeinate: " + str(e))
        caffeinate_process = None

    try:
        global SKIP_PROMPTS, KEEP_MATCHED, OUTPUT_PARENT_DIR, LOG_DIR, LOG_FILE, DB_PATH, USER_DB_PATH, PLAYLISTS_DIR, EMBED_PATH, USER_EMBED_PATH
        global ACOUSTID_API_KEY, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, DEFAULT_LASTFM_API_KEY
        global DEBUG_MODE, MP4TOVEC_MODEL, spawn_id_to_embeds, local_user_embeddings

        global new_admin_spawn_ids
        new_admin_spawn_ids = set()

        # keep track of embeddings generated this session
        spawn_id_to_embeds = {}
        local_user_embeddings = {}


        SKIP_PROMPTS = skip_prompts
        KEEP_MATCHED = keep_matched

        expanded_out = os.path.expanduser(output_path)
        OUTPUT_PARENT_DIR = os.path.join(expanded_out, "Spawn", "Music")
        LOG_DIR = os.path.join(expanded_out, "Spawn", "aux", "temp")
        LOG_FILE = os.path.join(LOG_DIR, "log.txt")
        PLAYLISTS_DIR = os.path.join(expanded_out, "Spawn", "Playlists")
        GLOB_DIR = os.path.join(expanded_out, "Spawn", "aux", "glob")
        USER_DIR = os.path.join(expanded_out, "Spawn", "aux", "user")
        EMBED_PATH = os.path.join(GLOB_DIR, "mp4tovec.p")
        USER_EMBED_PATH = os.path.join(USER_DIR, "mp4tovec_local.p")

        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(PLAYLISTS_DIR, exist_ok=True)
        os.makedirs(GLOB_DIR, exist_ok=True)
        os.makedirs(USER_DIR, exist_ok=True)

        DB_PATH = os.path.join(GLOB_DIR, "spawn_catalog.db")
        USER_DB_PATH = os.path.join(USER_DIR, "spawn_library.db")

        setup_logging(debug_to_console=DEBUG_MODE, log_file=LOG_FILE)

        if not os.path.isdir(music_path):
            logger.info(f"Error: {music_path} is not a valid directory.")
            sys.exit(1)

        env_path = os.path.join(os.path.dirname(__file__), "APId.env")
        load_dotenv(env_path)

        # Handle API credentials (AcoustID, Last.FM, Spotify)
        just_entered_acoustid = False
        just_entered_lastfm = False
        just_entered_spotify = False

        if acoustid_key:
            ACOUSTID_API_KEY = acoustid_key
            just_entered_acoustid = True
        else:
            ACOUSTID_API_KEY = os.environ.get("ACOUSTID_API_KEY", "")
        if not ACOUSTID_API_KEY and not skip_prompts:
            ACOUSTID_API_KEY = input("Enter AcoustID API Key: ").strip()
            if ACOUSTID_API_KEY:
                just_entered_acoustid = True

        if lastfm_key:
            DEFAULT_LASTFM_API_KEY = lastfm_key
            just_entered_lastfm = True
        else:
            DEFAULT_LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "")
        if not DEFAULT_LASTFM_API_KEY and not skip_prompts:
            DEFAULT_LASTFM_API_KEY = input("Enter Last.FM API Key: ").strip()
            if DEFAULT_LASTFM_API_KEY:
                just_entered_lastfm = True

        if spotify_client_id:
            SPOTIFY_CLIENT_ID = spotify_client_id
            just_entered_spotify = True
        else:
            SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
        if spotify_client_secret:
            SPOTIFY_CLIENT_SECRET = spotify_client_secret
            just_entered_spotify = True
        else:
            SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
        if not SPOTIFY_CLIENT_ID and not skip_prompts:
            SPOTIFY_CLIENT_ID = input("Enter Spotify Client ID: ").strip()
            if SPOTIFY_CLIENT_ID:
                just_entered_spotify = True
        if not SPOTIFY_CLIENT_SECRET and not skip_prompts:
            SPOTIFY_CLIENT_SECRET = input("Enter Spotify Client Secret: ").strip()
            if SPOTIFY_CLIENT_SECRET:
                just_entered_spotify = True

        if not skip_prompts:
            if just_entered_acoustid and ACOUSTID_API_KEY:
                ans = input("Do you want to save ACOUSTID_API_KEY to APId.env? ([y]/n): ").strip() or "y"
                if ans.lower() != "n":
                    store_key_in_env_file(env_path, "ACOUSTID_API_KEY", ACOUSTID_API_KEY)
            if just_entered_lastfm and DEFAULT_LASTFM_API_KEY:
                ans = input("Do you want to save LASTFM_API_KEY to APId.env? ([y]/n): ").strip() or "y"
                if ans.lower() != "n":
                    store_key_in_env_file(env_path, "LASTFM_API_KEY", DEFAULT_LASTFM_API_KEY)
            if just_entered_spotify and SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
                ans = input("Do you want to save Spotify credentials to APId.env? ([y]/n): ").strip() or "y"
                if ans.lower() != "n":
                    store_key_in_env_file(env_path, "SPOTIFY_CLIENT_ID", SPOTIFY_CLIENT_ID)
                    store_key_in_env_file(env_path, "SPOTIFY_CLIENT_SECRET", SPOTIFY_CLIENT_SECRET)

        init_db(DB_PATH)
        init_db_revisions(DB_PATH)
        if not is_admin:
            init_user_library_db(USER_DB_PATH)

        sp = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET
            )
        )

        global MP4TOVEC_AVAILABLE
        if MP4TOVEC_AVAILABLE:
            try:
                MP4TOVEC_MODEL = load_mp4tovec_model_diffusion()
                if (callable(MP4TOVEC_MODEL) and getattr(MP4TOVEC_MODEL, "__name__", "") == "<lambda>"):
                    logger.warning("[MP4ToVec] Placeholder model in use (no real load code).")
                elif MP4TOVEC_MODEL is None:
                    logger.warning("[MP4ToVec] Model is None; using placeholder or disabled.")
                else:
                    logger.info("[MP4ToVec] Model loaded successfully.")
            except Exception as e:
                MP4TOVEC_MODEL = None
                MP4TOVEC_AVAILABLE = False
                logger.warning(f"[MP4ToVec] Could not load model: {e}")
        else:
            logger.warning("[MP4ToVec] MP4ToVec not imported; embeddings won't be generated.")

        # Process input audio files in mini-batches (each mini-batch is fully finalized before next begins)
        process_audio_files_in_batches(
            input_dir=music_path,
            batch_threshold=10,
            keep_matched=KEEP_MATCHED,
            lastfm_api_key=DEFAULT_LASTFM_API_KEY,
            sp=sp,
            is_admin=is_admin,
            USER_DB_PATH=USER_DB_PATH,
            EMBED_PATH=EMBED_PATH
        )
    finally:
        # Terminate caffeinate to allow sleep
        if caffeinate_process is not None:
            caffeinate_process.terminate()
            logger.debug("Caffeinate terminated, allowing system to sleep.")



###############################################################################
# Argparse Wrapper
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Process audio files to standardize tags, MBIDs, and ReplayGain."
    )
    parser.add_argument("-acu", "--acoustid", help="ACOUSTID_API_KEY")
    parser.add_argument("-sp_id", "--spotify_id", help="SPOTIFY_CLIENT_ID")
    parser.add_argument("-sp_sec", "--spotify_secret", help="SPOTIFY_CLIENT_SECRET")
    parser.add_argument("-last", "--lastfm", help="DEFAULT_LASTFM_API_KEY")
    parser.add_argument("-skippy", "--skip-prompts", action="store_true", help="Skip user prompts.")
    parser.add_argument("output_path", help="Where 'Spawn' folder will be placed.")
    parser.add_argument("music_path", help="Folder containing input music.")
    args = parser.parse_args()

    # Just call run_import with the parsed args
    run_import(
        output_path=args.output_path,
        music_path=args.music_path,
        acoustid_key=args.acoustid,
        spotify_client_id=args.spotify_id,
        spotify_client_secret=args.spotify_secret,
        lastfm_key=args.lastfm,
        skip_prompts=args.skip_prompts
    )


if __name__ == "__main__":
    main()
