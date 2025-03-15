import json
import os
import re
from datetime import datetime

# Optional: if Plex API is available for metadata lookup.
try:
    from plexapi.server import PlexServer
except ImportError:
    PlexServer = None

from mutagen.mp4 import MP4

def get_metadata_from_plex(plex, key):
    """
    Fetch media metadata from Plex using the given key.
    If the key is numeric, it prepends "/library/metadata/".
    Returns a 4-tuple: (title, artist, spawn_id, media_type)
    """
    if not key.startswith("/"):
        if key.isdigit():
            key = f"/library/metadata/{key}"
        else:
            print(f"⚠️ Invalid key format: {key}. Skipping Plex lookup.")
            return None, None, "", ""
    try:
        item = plex.fetchItem(key)
        title = item.title
        artist = getattr(item, "grandparentTitle", "")
        spawn_id = ""
        media_type = getattr(item, "type", "unknown")
        # Attempt to extract spawn_id from the file (for MP4/M4A files)
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
                    print(f"⚠️ Failed to extract spawn_id from file {file_path}: {e}")
        return title, artist, spawn_id, media_type
    except Exception as e:
        print(f"⚠️ Plex lookup failed for key {key}: {e}")
        return None, None, "", ""

def export_recently_played(plex_serv_url, plex_token, output_file=None):
    """
    Connects to Plex, retrieves the playlist named "Recently Played" (case-insensitive),
    and exports its tracks to a JSON file.

    Each record includes:
      - "artist"
      - "album"
      - "track"
      - "last played": Converted from lastViewedAt (as a human-readable datetime)
      - "spawn_id" (from metadata lookup)
      - "duration (s)" (in seconds)
      - "media type"
    
    If output_file is not provided, defaults to "recently_played.json" saved under:
        LIB_PATH/Spawn/aux/temp/recently_played.json
    or in the current directory if LIB_PATH is not defined.

    Instead of overwriting the output file, the function loads existing records (if any)
    and adds new entries that are not already present (based on a unique key).
    All records are sorted in descending order (most recent first) by "Last Played".
    """
    try:
        plex = PlexServer(plex_serv_url, plex_token)
    except Exception as e:
        print(f"Error connecting to Plex: {e}")
        return
    
    # Retrieve all playlists from Plex and find the "Recently Played" one
    playlists = plex.playlists()
    recently_played = None
    for pl in playlists:
        if pl.title.lower() == "recently played":
            recently_played = pl
            break
    if not recently_played:
        print("No 'Recently Played' playlist found in Plex.")
        return

    items = recently_played.items()
    print(f"Found 'Recently Played' playlist with {len(items)} items.")

    new_records = []
    for track in items:

        # Determine last played time (using lastViewedAt)
        last_viewed_at = getattr(track, 'lastViewedAt', None)
        if last_viewed_at:
            try:
                if isinstance(last_viewed_at, (int, float, str)):
                    # If it's a timestamp, convert it.
                    last_viewed = datetime.fromtimestamp(int(last_viewed_at)).strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(last_viewed_at, datetime):
                    # If it's already a datetime, format it directly.
                    last_viewed = last_viewed_at.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    last_viewed = "N/A"
            except Exception as e:
                print(f"Error converting lastViewedAt: {e}")
                last_viewed = "N/A"
        else:
            last_viewed = "N/A"

        track_title = track.title
        artist = getattr(track, 'grandparentTitle', "")
        album = getattr(track, 'parentTitle', "")
        
        duration_ms = getattr(track, 'duration', 0)
        duration_s = duration_ms / 1000.0 if duration_ms else 0
        
        _, _, spawn_id, media_type = get_metadata_from_plex(plex, track.key)
        
        record = {
            "artist": artist,
            "album": album,
            "track": track_title,
            "last played": last_viewed,
            "spawn_id": spawn_id,
            "duration (s)": f"{duration_s:.2f}" if duration_s else "",
            "media type": media_type
        }
        new_records.append(record)
    
    # Set default output file name if not provided
    if not output_file:
        output_file = "recently_played.json"
    
    # If LIB_PATH is defined and valid, save under LIB_PATH/Spawn/aux/temp
    lib_path = os.environ.get("LIB_PATH", "").strip()
    if lib_path and os.path.isdir(lib_path):
        target_dir = os.path.join(lib_path, "Spawn", "aux", "temp")
        os.makedirs(target_dir, exist_ok=True)
        output_file = os.path.join(target_dir, output_file)
    else:
        output_file = os.path.abspath(output_file)

   # Load existing records (if any)
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_records = json.load(f)
        except Exception as e:
            print(f"Error loading existing JSON file: {e}")
            existing_records = []
    else:
        existing_records = []

    # Define a function to generate a unique key for each record
    def record_key(rec):
        # Prefer "spawn_id" if available; otherwise, use a combination of title, artist, and album.
        if rec.get("spawn_id"):
            return rec["spawn_id"]
        return f"{rec.get('track','')}_{rec.get('artist','')}_{rec.get('album','')}"

    existing_keys = {record_key(rec) for rec in existing_records}

    # Filter new_records to include only records not already present
    new_unique_records = [rec for rec in new_records if record_key(rec) not in existing_keys]
    if new_unique_records:
        print(f"Found {len(new_unique_records)} new records to add.")
    else:
        print("No new records found.")

    # Combine new records with existing ones
    combined_records = new_unique_records + existing_records

    # Sort records by "Last Viewed" (most recent first)
    def parse_last_viewed(rec):
        try:
            return datetime.strptime(rec["Last Viewed"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.min

    combined_records.sort(key=parse_last_viewed, reverse=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_records, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(combined_records)} records to {output_file}")

if __name__ == "__main__":
    plex_serv_url = input("Enter Plex Server URL (e.g., http://192.168.86.67:32400): ").strip()
    plex_token = input("Enter Plex Token: ").strip()
    output_file = input("Enter output JSON file (default: recently_played.json): ").strip()
    if not output_file:
        output_file = "recently_played.json"
    export_recently_played_json(plex_serv_url, plex_token, output_file)
