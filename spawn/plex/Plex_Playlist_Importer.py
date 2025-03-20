#!/usr/bin/env python3
"""
Plex_Playlist_Importer.py

Imports an M3U playlist into Plex by grouping tracks by artist to reduce API calls.
If a direct lookup fails for a track, a fallback mechanism attempts to match the track
using multiple strategies (exact, partial, and fuzzy matching).
"""

import os
import argparse
import re
import unicodedata
import json
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from plexapi.server import PlexServer
from plexapi.exceptions import NotFound
from fuzzywuzzy import fuzz

# --- Global Setup ---

lib_path = os.environ.get("LIB_PATH", "").strip()


# --- Helper Functions for GUID & ratingKey Lookup ---

def load_guid_data():
    """Load the plex_guid.json data as a list of mappings."""
    guid_data = []
    if lib_path and os.path.isdir(lib_path):
        guid_json_path = os.path.join(lib_path, "Spawn", "aux", "user", "plex_guid.json")
        if os.path.exists(guid_json_path):
            try:
                with open(guid_json_path, "r", encoding="utf-8") as f:
                    guid_data = json.load(f)
            except UnicodeDecodeError as e:
                print(f"⚠️ Unicode decode error loading plex_guid.json: {e}")
                print("⚠️ Attempting to load plex_guid.json with errors replaced and cleaning control characters.")
                try:
                    with open(guid_json_path, "r", encoding="utf-8", errors="replace") as f:
                        data = f.read()
                        # Remove control characters (aggressively remove all ASCII control characters)
                        cleaned_data = re.sub(r'[\x00-\x1F\x7F]', '', data)
                        # Set strict=False to allow control characters that remain
                        guid_data = json.loads(cleaned_data, strict=False)
                except Exception as e:
                    print(f"❌ Failed to load plex_guid.json even after cleaning: {e}")
            except Exception as e:
                print(f"⚠️ Error loading plex_guid.json: {e}")
    return guid_data

def save_guid_data(guid_data):
    """Save the updated plex_guid.json mapping to disk."""
    if lib_path and os.path.isdir(lib_path):
        out_dir = os.path.join(lib_path, "Spawn", "aux", "user")
        os.makedirs(out_dir, exist_ok=True)
        guid_json_path = os.path.join(out_dir, "plex_guid.json")
        try:
            with open(guid_json_path, "w", encoding="utf-8") as f:
                json.dump(guid_data, f, indent=2)
            print(f"✅ Updated plex_guid.json saved to {guid_json_path}")
        except Exception as e:
            print(f"❌ Error saving plex_guid.json: {e}")

def lookup_in_guid_json(guid_data, spawn_id):
    """Return a mapping entry matching the given spawn_id, if any."""
    for entry in guid_data:
        if entry.get("spawn_id") == spawn_id:
            return entry
    return None

def extract_spawn_id_from_file(file_path):
    """
    Extract spawn_id from an audio file.
    Currently implemented for MP4/M4A files.
    """
    try:
        ext = file_path.lower().split('.')[-1]
        spawn_id = None
        if ext in ["m4a", "mp4"]:
            audio = MP4(file_path)
            spawn_ids = audio.tags.get("----:com.apple.iTunes:spawn_ID", [])
            if spawn_ids:
                spawn_id = spawn_ids[0]
                if isinstance(spawn_id, bytes):
                    spawn_id = spawn_id.decode("utf-8", errors="ignore")
        return spawn_id
    except Exception as e:
        print(f"⚠️ Error extracting spawn_id from {file_path}: {e}")
        return None


# --- Other Helper Functions ---

def normalize_text(text):
    """Normalize text by removing special characters, fixing Unicode, and standardizing numbers."""
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)  # Normalize Unicode variations
    text = re.sub(r'number\s+(\d+)', r'#\1', text)  # Convert "Number 1" → "#1"
    text = re.sub(r'⁄', '/', text)  # Fix alternative slashes
    text = re.sub(r'∶', ':', text)  # Fix colons
    text = re.sub(r'․․․', '...', text)  # Fix ellipses
    text = re.sub(r'’', "'", text)  # Replace curly apostrophes
    text = re.sub(r'“|”', '"', text)  # Replace curly quotes
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'[^\w\s#/:.-]', '', text)  # Allow only useful special characters
    return text

def extract_metadata_from_file(file_path):
    """Extracts metadata (artist, album, title) from an audio file.
       If the file is not found, logs the error and returns None values."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None, None
    try:
        ext = file_path.lower().split('.')[-1]
        if ext in ["mp3"]:
            audio = MP3(file_path, ID3=EasyID3)
            artist = audio.get("artist", [None])[0]
            album = audio.get("album", [None])[0]
            title = audio.get("title", [None])[0]
        elif ext in ["flac"]:
            audio = FLAC(file_path)
            artist = audio.get("artist", [None])[0]
            album = audio.get("album", [None])[0]
            title = audio.get("title", [None])[0]
        elif ext in ["m4a", "mp4"]:
            audio = MP4(file_path)
            artist = audio.get("\xa9ART", [None])[0]
            album = audio.get("\xa9alb", [None])[0]
            title = audio.get("\xa9nam", [None])[0]
        else:
            print(f"⚠️ Unsupported file type: {file_path}")
            return None, None, None

        if artist and album and title:
            print(f"📀 Metadata found: {title} by {artist} ({album})")
            return artist, album, title
        else:
            print(f"⚠️ Missing metadata for: {file_path}")
            return None, None, None

    except Exception as e:
        print(f"⚠️ Error reading metadata for {file_path}: {e}")
        return None, None, None

def find_track_in_plex(plex, artist, album, track_title):
    """
    Tries multiple ways to find a track in Plex, including:
      1. Exact match: title, artist, and album.
      2. Title + artist match (ignoring album).
      3. Fuzzy matching (≥85% similarity).
    """
    try:
        print(f"🔍 Expanding search for track in Plex: '{track_title}' by {artist} from {album}")
        results = plex.search(track_title)
        norm_title = normalize_text(track_title)
        norm_album = normalize_text(album)

        # 1️⃣ First attempt: Exact match (Title + Artist + Album)
        for result in results:
            if result.TYPE == 'track' and result.grandparentTitle == artist and result.parentTitle == album:
                print(f"✅ Found track: {result.title} (Exact match)")
                return result

        # 2️⃣ Second attempt: Title + Artist only (Ignore Album)
        print("⚠️ No match found, retrying with only title and artist...")
        for result in results:
            if result.TYPE == 'track' and result.grandparentTitle == artist:
                print(f"✅ Found track (no album match): {result.title} (Album in Plex: {result.parentTitle})")
                return result

        # 3️⃣ Third attempt: Fuzzy matching (≥85% similarity)
        print("⚠️ No match found, trying fuzzy matching...")
        best_match = None
        highest_score = 0
        for result in results:
            fuzzy_title = normalize_text(result.title)
            fuzzy_album = normalize_text(result.parentTitle)
            title_score = fuzz.ratio(fuzzy_title, norm_title)
            album_score = fuzz.ratio(fuzzy_album, norm_album)
            combined_score = (title_score * 0.7) + (album_score * 0.3)  # Weight title higher than album
            if combined_score > 85 and combined_score > highest_score:
                best_match = result
                highest_score = combined_score

        if best_match:
            print(f"✅ Found track via fuzzy matching: {best_match.title} (Plex Album: {best_match.parentTitle}) - Score: {highest_score}")
            return best_match

        print(f"❌ Track not found after multiple attempts: {track_title} by {artist} ({album})")
        return None

    except Exception as e:
        print(f"⚠️ Error finding track in Plex: {e}")
        return None

def create_playlist(plex, playlist_name, tracks):
    """Creates a playlist in Plex."""
    try:
        if not tracks:
            print("⚠️ No tracks found. Skipping playlist creation.")
            return
        print(f"\nCreating playlist '{playlist_name}' in Plex.")
        playlist = plex.createPlaylist(playlist_name, items=tracks)
        print(f"✅ Playlist '{playlist_name}' created successfully with {len(tracks)} tracks.")
    except Exception as e:
        print(f"❌ Error creating playlist in Plex: {e}")

# --- Main Import Function ---

def import_m3u_to_plex(plex, m3u_path, music_dir):
    """Imports an M3U file into Plex using batched artist queries."""
    if not os.path.exists(m3u_path):
        print(f"❌ The file {m3u_path} does not exist.")
        return

    print(f"📂 Reading M3U file: {m3u_path}")
    track_paths = []
    with open(m3u_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith("../../Music/"):
                    # Remove the "../../Music/" part and use the provided music_dir
                    rel_path = line[len("../../Music/"):]
                    full_path = os.path.abspath(os.path.join(music_dir, rel_path))
                else:
                    # For other cases, process as before
                    full_path = os.path.abspath(os.path.join(music_dir, os.path.normpath(line)))
                track_paths.append(full_path)

    print("🔍 Matching tracks in Plex.")
    matched_tracks = []
    missing_tracks = []

    # Load existing mapping from plex_guid.json
    guid_data = load_guid_data()
    guid_data_updated = False

    # Group tracks by artist for batched API calls.
    tracks_by_artist = {}

    for track_path in track_paths:
        artist, album, track_title = extract_metadata_from_file(track_path)
        if not artist or not album or not track_title:
            print(f"⚠️ Could not extract metadata from: {track_path}")
            missing_tracks.append(track_path)
            continue

        # Attempt to extract spawn_id for direct lookup
        spawn_id = extract_spawn_id_from_file(track_path)
        track = None

        if spawn_id:
            entry = lookup_in_guid_json(guid_data, spawn_id)
            if entry:

                rating_key = entry.get("ratingKey")
                plex_key = f"/library/metadata/{rating_key}"
                print(f"Looking for match using spawn_id {spawn_id} and ratingKey {rating_key}")

                try:
                    track = plex.fetchItem(plex_key)
                    print(f"✅ Found track via ratingKey lookup: {track.title}")
                except Exception as e:
                    print(f"⚠️ Direct lookup via plex_guid.json failed for {track_title}: {e}")
                    track = None

        # Fallback to existing search if direct lookup failed or spawn_id not available
        if not track:
            track = find_track_in_plex(plex, artist, album, track_title)
            # If fallback succeeded and a spawn_id is available, add to mapping with both ratingKey and GUID
            if track and spawn_id:
                if not lookup_in_guid_json(guid_data, spawn_id):
                    # Format the GUID by stripping the "plex://track/" prefix if needed
                    guid_value = track.guid
                    if guid_value.startswith("plex://track/"):
                        guid_value = guid_value.replace("plex://track/", "")
                    new_entry = {
                        "artist": artist,
                        "album": album,
                        "track": track_title,
                        "plex_guid": guid_value,
                        "ratingKey": str(track.ratingKey),
                        "spawn_id": spawn_id,
                        "media type": "track"
                    }
                    guid_data.append(new_entry)
                    guid_data_updated = True

        if track:
            matched_tracks.append(track)
        else:
            # Group the track by its artist.
            if artist not in tracks_by_artist:
                tracks_by_artist[artist] = []
            tracks_by_artist[artist].append({
                "track_path": track_path,
                "artist": artist,
                "album": album,
                "track_title": track_title,
                "spawn_id": spawn_id
            })

    # Process each artist group with a single API call.
    for artist, track_infos in tracks_by_artist.items():
        print(f"🔍 Searching Plex for artist: {artist}")
        try:
            # Perform a single API call filtered by artist.
            # Adjust the parameters as needed based on your Plex setup.
            results = plex.search("", mediatype="track", artist=artist)
        except Exception as e:
            print(f"⚠️ Error searching for artist {artist}: {e}")
            results = []

        for info in track_infos:
            album = info["album"]
            track_title = info["track_title"]
            found = None

            # First attempt: exact matching based on album and title.
            for result in results:
                if (result.TYPE == 'track' and result.grandparentTitle == artist and 
                    result.parentTitle == album and result.title.lower() == track_title.lower()):
                    print(f"✅ Exact match found: {result.title} by {artist} ({album})")
                    found = result
                    break

            # Second attempt: fuzzy matching if no exact match.
            if not found:
                norm_title = normalize_text(track_title)
                norm_album = normalize_text(album)
                best_match = None
                highest_score = 0
                for result in results:
                    fuzzy_title = normalize_text(result.title)
                    fuzzy_album = normalize_text(result.parentTitle)
                    title_score = fuzz.ratio(fuzzy_title, norm_title)
                    album_score = fuzz.ratio(fuzzy_album, norm_album)
                    combined_score = (title_score * 0.7) + (album_score * 0.3)
                    if combined_score > 85 and combined_score > highest_score:
                        best_match = result
                        highest_score = combined_score
                if best_match:
                    print(f"✅ Fuzzy match found: {best_match.title} by {artist} ({album}) - Score: {highest_score}")
                    found = best_match

            if found:
                matched_tracks.append(found)
                # Update the GUID mapping if a spawn_id is present.
                if info["spawn_id"]:
                    if not lookup_in_guid_json(guid_data, info["spawn_id"]):
                        guid_value = found.guid
                        if guid_value.startswith("plex://track/"):
                            guid_value = guid_value.replace("plex://track/", "")
                        new_entry = {
                            "artist": artist,
                            "album": album,
                            "track": track_title,
                            "plex_guid": guid_value,
                            "ratingKey": str(found.ratingKey),
                            "spawn_id": info["spawn_id"],
                            "media type": "track"
                        }
                        guid_data.append(new_entry)
                        guid_data_updated = True
            else:
                print(f"❌ Track not found: {track_title} by {artist} ({album})")
                missing_tracks.append(info["track_path"])

    # Print summary
    print("\n==== Import Summary ====")
    print(f"📀 Total tracks in M3U: {len(track_paths)}")
    print(f"✅ Successfully matched: {len(matched_tracks)}")
    print(f"❌ Missing tracks: {len(missing_tracks)}")

    if missing_tracks:
        if lib_path and os.path.isdir(lib_path):
            temp_dir = os.path.join(lib_path, "Spawn", "aux", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            log_file = os.path.join(temp_dir, "tracks_missing_from_Plex.log")
            with open(log_file, 'w') as f:
                for track in missing_tracks:
                    f.write(track + "\n")
            print(f"⚠️ Log of tracks missing from Plex saved to '{log_file}' for manual review.")
        else:
            print(f"❌ LIB_PATH not defined; no log will be saved.")

        print("\n⚠️ The following tracks were NOT found in Plex:")
        for track in missing_tracks:
            print(f" - {track}")
    else:
        print("\n✅ All tracks were matched successfully!")

    # Save updated GUID mapping if new entries were added
    if guid_data_updated:
        save_guid_data(guid_data)

    # Create playlist in Plex using the M3U file's base name (underscores replaced by spaces)
    playlist_name = os.path.splitext(os.path.basename(m3u_path))[0].replace('_', ' ')
    create_playlist(plex, playlist_name, matched_tracks)


# Exposed function for module use

def import_playlists(plex_serv_url, plex_token):
    """
    Connects to the Plex server and prompts the user to import a playlist from an M3U file.
    
    Parameters:
      - plex_serv_url: URL of the Plex server.
      - plex_token: Plex API token.
    """
    try:
        plex = PlexServer(plex_serv_url, plex_token)
    except Exception as e:
        print(f"❌ Error connecting to Plex: {e}")
        return

    # Prompt for M3U file path
    m3u_path = input("Enter the full path to the M3U file to export to Plex: ").strip()
    if not os.path.exists(m3u_path):
        print(f"❌ M3U file not found: {m3u_path}")
        return

    # Determine default music directory from LIB_PATH/Music if available
    default_music_dir = os.path.join(lib_path, "Spawn", "Music") if lib_path else ""
    if default_music_dir and os.path.isdir(default_music_dir):
        music_dir = default_music_dir
        print(f"Using default music directory: {music_dir}")
    else:
        music_dir = input("Enter the base directory for the music files: ").strip()
        if not os.path.isdir(music_dir):
            print(f"❌ Invalid music directory: {music_dir}")
            return

    import_m3u_to_plex(plex, m3u_path, music_dir)

if __name__ == '__main__':
    # If run directly, prompt for Plex credentials interactively.
    plex_serv_url = input("Enter Plex Server URL (e.g., http://192.168.86.67:32400): ").strip()
    plex_token = input("Enter Plex Token: ").strip()
    import_playlists(plex_serv_url, plex_token)
