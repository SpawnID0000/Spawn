# Plex_Playlist_Importer.py

import os
import argparse
import re
import unicodedata
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from plexapi.server import PlexServer
from plexapi.exceptions import NotFound
from fuzzywuzzy import fuzz

# --- Utility Functions ---

lib_path = os.environ.get("LIB_PATH", "").strip()

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
    """Tries multiple ways to find a track in Plex, including fuzzy matching."""
    try:
        print(f"🔍 Searching for track in Plex: '{track_title}' by {artist} from {album}")
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
    """Imports an M3U file into Plex, attempting to match each track."""
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

    for track_path in track_paths:
        artist, album, track_title = extract_metadata_from_file(track_path)
        if not artist or not album or not track_title:
            print(f"⚠️ Could not extract metadata from: {track_path}")
            missing_tracks.append(track_path)
            continue
        track = find_track_in_plex(plex, artist, album, track_title)
        if track:
            matched_tracks.append(track)
        else:
            missing_tracks.append(track_path)

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
        else:
            print(f"❌ LIB_PATH not defined; no log will be saved.")

        with open(log_file, 'w') as f:
            for track in missing_tracks:
                f.write(track + "\n")
        print(f"⚠️ Log of tracks missing from Plex saved to '{log_file}' for manual review.")
        print("\n⚠️ The following tracks were NOT found in Plex:")
        for track in missing_tracks:
            print(f" - {track}")
    else:
        print("\n✅ All tracks were matched successfully!")


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
