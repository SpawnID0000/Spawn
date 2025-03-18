# Plex_track_finder.py

import json
import argparse
import requests
import os
from plexapi.server import PlexServer
from mutagen.mp4 import MP4

# Hardcoded Plex Playlist Name
PLEX_PLAYLIST_NAME = "All Music"

def get_track_metadata(item):
    """
    Extract metadata from a Plex music track item.
    Returns a dictionary with artist, album, track, plex_guid, spawn_id, and media type.
    """
    try:
        artist = getattr(item, "grandparentTitle", "Unknown Artist")
        album = getattr(item, "parentTitle", "Unknown Album")
        track = getattr(item, "title", "Unknown Track")
        plex_guid = getattr(item, "guid", "")
        # Remove "plex://track/" prefix if present
        if plex_guid.startswith("plex://track/"):
            plex_guid = plex_guid.replace("plex://track/", "")
        media_type = "track"
        spawn_id = ""
        
        # Extract spawn_id if possible
        if hasattr(item, "media") and item.media:
            media = item.media[0]
            if hasattr(media, "parts") and media.parts:
                file_path = media.parts[0].file
                try:
                    audio = MP4(file_path)
                    spawn_ids = audio.tags.get("----:com.apple.iTunes:spawn_ID", [])
                    if spawn_ids:
                        spawn_id = spawn_ids[0]
                        if isinstance(spawn_id, bytes):
                            spawn_id = spawn_id.decode("utf-8", errors="ignore")
                except Exception as e:
                    error_message = f"⚠️ Failed to extract spawn_id from {file_path}: {e}"
                    print(error_message)
                    return None, error_message
        
        return {
            "artist": artist,
            "album": album,
            "track": track,
            "plex_guid": plex_guid,
            "spawn_id": spawn_id,
            "media type": media_type,
        }, None
    except Exception as e:
        error_message = f"❌ Error processing track: {e}"
        print(error_message)
        return None, error_message

def export_all_music(plex_serv_url, plex_token, lib_path):
    """
    Connects to Plex, retrieves all tracks in the "All Music" playlist, and exports metadata to a JSON file.
    Logs errors and provides a summary at the end.
    """
    try:
        plex = PlexServer(plex_serv_url, plex_token)
        
        # Retrieve all playlists from Plex and find "All Music"
        playlists = plex.playlists()
        all_music_playlist = None
        for pl in playlists:
            if pl.title.lower() == PLEX_PLAYLIST_NAME.lower():
                all_music_playlist = pl
                break
        
        if not all_music_playlist:
            print(f"❌ No '{PLEX_PLAYLIST_NAME}' playlist found in Plex.")
            return
        
        print(f"🔍 Fetching tracks from '{PLEX_PLAYLIST_NAME}'...")
        tracks = all_music_playlist.items()
        
        metadata_list = []
        error_list = []
        for track in tracks:
            metadata, error = get_track_metadata(track)
            if metadata:
                metadata_list.append(metadata)
            if error:
                error_list.append(error)
        
        # Create output directories if they don't exist
        output_dir = os.path.join(lib_path, "Spawn", "aux", "glob")
        log_dir = os.path.join(lib_path, "Spawn", "aux", "temp")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Save JSON output
        output_file = os.path.join(output_dir, "plex_guid.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2)
        
        # Save error log if there were failures
        if error_list:
            error_file = os.path.join(log_dir, "plex_errors.log")
            with open(error_file, "w", encoding="utf-8") as f:
                f.write("\n".join(error_list))
            print(f"⚠️ {len(error_list)} errors encountered. See {error_file} for details.")
        
        print(f"✅ Exported {len(metadata_list)} tracks successfully to {output_file}")
    except Exception as e:
        print(f"❌ Error retrieving Plex data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export all tracks from the 'All Music' playlist in Plex.")
    parser.add_argument("--PLEX_SERV_URL", required=True, help="Plex server URL")
    parser.add_argument("--PLEX_TOKEN", required=True, help="Plex authentication token")
    parser.add_argument("--LIB_PATH", required=True, help="Output directory for JSON file")
    
    args = parser.parse_args()
    
    export_all_music(args.PLEX_SERV_URL, args.PLEX_TOKEN, args.LIB_PATH)
