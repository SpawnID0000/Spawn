import json
import os
import re
import requests
import sys
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
      - "media type"
    
    If output_file is not provided, defaults to "plex_play_log.json" saved under:
        LIB_PATH/Spawn/aux/temp/plex_play_log.json
    or in the current directory if LIB_PATH is not defined.

    Instead of overwriting the output file, the function loads existing records (if any)
    and adds new entries that are not already present (based on a unique key).
    All records are sorted in descending order (most recent first) by "last played".
    """
    try:
        # Optionally increase timeout if needed, e.g. timeout=60
        plex = PlexServer(plex_serv_url, plex_token, timeout=60)
    except Exception as e:
        print(f"Error connecting to Plex: {e}")
        return []
    
    # Retrieve all playlists from Plex and find the "Recently Played" one
    playlists = plex.playlists()
    recently_played = None
    for pl in playlists:
        if pl.title.lower() == "recently played":
            recently_played = pl
            break
    if not recently_played:
        print("No 'Recently Played' playlist found in Plex.")
        return []

    items = recently_played.items()
    print(f"\nFound 'Recently Played' playlist with {len(items)} items.")

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
        
        _, _, spawn_id, media_type = get_metadata_from_plex(plex, track.key)
        
        record = {
            "artist": artist,
            "album": album,
            "track": track_title,
            "last played": last_viewed,
            "spawn_id": spawn_id,
            "media type": media_type
        }
        new_records.append(record)
    return new_records

def fetch_remote_recent_plays(plex_user_uuid, plex_token):
    """
    Polls Plex.tv for recent plays using the provided user UUID (or accountID) and Plex token.
    Returns a list of raw items from the endpoint.
    
    Endpoint:
       https://plex.tv/api/v2/user/settings?uuid=<plex_user_uuid>&key=RecentPlays&X-Plex-Token=<plex_token>
    """
    url = f"https://plex.tv/api/v2/user/settings?uuid={plex_user_uuid}&key=RecentPlays&X-Plex-Token={plex_token}"
    print(f"Fetching 'Recent Plays' from URL: {url}", flush=True)
    try:
        response = requests.get(url, headers={"Accept": "application/json"})
        # Print additional debug information:
        #print("Remote response status code:", response.status_code, flush=True)
        #print("Remote response headers:", response.headers, flush=True)
        #print("Raw response text:", flush=True)
        #print(response.text, flush=True)
        #print("Raw response content (decoded):", flush=True)
        #print(response.content.decode('utf-8', errors='replace'), flush=True)

        response.raise_for_status()
        data = response.json()
        remote_items = data.get("data", [])
        print(f"Found 'Recent Plays' with {len(remote_items)} items.", flush=True)
        return remote_items
    except Exception as e:
        print(f"Error fetching remote recent plays: {e}", flush=True)
        return []

def parse_remote_item(item):
    """
    Parses a single item from the remote Recent Plays endpoint into our record schema.
    
    For remote items:
      - For type "track": uses 'title' as track, 'grandparentTitle' as artist, 'parentTitle' as album.
      - For type "album": uses 'title' as album, 'parentTitle' as artist.
      - For type "playlist": uses 'title' as track (or playlist name), leaving artist/album blank.
    Uses "updatedAt" (or if not available, "addedAt") as the timestamp.
    """
    timestamp = item.get("updatedAt") or item.get("addedAt")
    if timestamp:
        try:
            last_viewed = datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Error converting remote timestamp: {e}", flush=True)
            last_viewed = "N/A"
    else:
        last_viewed = "N/A"

    item_type = item.get("type", "")
    if item_type == "track":
        track_val = item.get("title", "")
        artist_val = item.get("grandparentTitle", "")
        album_val = item.get("parentTitle", "")
    elif item_type == "album":
        track_val = ""  # No individual track info
        album_val = item.get("title", "")
        artist_val = item.get("parentTitle", "")
    elif item_type == "playlist":
        # For playlists, we'll put the playlist name in "track"
        track_val = item.get("title", "")
        artist_val = ""
        album_val = ""
    else:
        track_val = item.get("title", "")
        artist_val = item.get("grandparentTitle", "")
        album_val = item.get("parentTitle", "")

    spawn_id = item.get("ratingKey", "")
    return {
        "artist": artist_val,
        "album": album_val,
        "track": track_val,
        "last played": last_viewed,
        "spawn_id": spawn_id,
        "media type": item_type
    }

def record_key(rec):
    """
    Generates a unique key for a record.
    Uses "spawn_id" if available; otherwise, uses a composite of track, artist, and album.
    """
    if rec.get("spawn_id"):
        return rec["spawn_id"]
    return f"{rec.get('track','')}_{rec.get('artist','')}_{rec.get('album','')}"

def export_recent_plays_json(plex_serv_url, plex_token, plex_user_uuid, output_file=None):
    """
    Combines local Recently Played items (from your Plex server) and remote Recent Plays items (from plex.tv)
    into a unified JSON file.
    
    The function:
      - Loads local records from the "Recently Played" playlist.
      - Polls the Plex.tv endpoint for Recent Plays using plex_user_uuid and plex_token.
      - Parses both sources into a common record schema.
      - Loads an existing JSON file (if any) and merges in only new records (based on unique keys).
      - Sorts all records in descending order by "last played" (most recent first).
      - Writes the combined records back to the output file.
      
    If output_file is not provided, defaults to "plex_play_log.json" under LIB_PATH/Spawn/aux/temp (or current directory if LIB_PATH is not defined).
    """
    local_records = export_recently_played(plex_serv_url, plex_token)
    #print(f"Local records count: {len(local_records)}", flush=True)
    remote_raw = fetch_remote_recent_plays(plex_user_uuid, plex_token)
    remote_records = [parse_remote_item(item) for item in remote_raw]
    #print(f"Remote records count: {len(remote_records)}", flush=True)

    all_new_records = local_records + remote_records
    #print(f"Total new records from both sources: {len(all_new_records)}", flush=True)

    # Set default output file name if not provided
    if not output_file:
        output_file = "plex_play_log.json"
    
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
            print(f"Error loading existing JSON file: {e}", flush=True)
            existing_records = []
    else:
        existing_records = []

    def record_key_inner(rec):
        if rec.get("spawn_id"):
            return rec["spawn_id"]
        return f"{rec.get('track','')}_{rec.get('artist','')}_{rec.get('album','')}"

    existing_keys = {record_key_inner(rec) for rec in existing_records}

    # Filter new_records to include only records not already present
    new_unique_records = [rec for rec in all_new_records if record_key_inner(rec) not in existing_keys]
    if new_unique_records:
        print(f"\nFound {len(new_unique_records)} new records to add.", flush=True)
    else:
        print("\nNo new records found.", flush=True)

    # Combine new records with existing ones
    combined_records = new_unique_records + existing_records

    # Sort records by "last played" (most recent first)
    def parse_last_viewed(rec):
        try:
            return datetime.strptime(rec["last played"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.min

    combined_records.sort(key=parse_last_viewed, reverse=True)

    # Debug print full output file path before writing
    print(f"Writing combined records to: {output_file}", flush=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_records, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(combined_records)} records to {output_file}", flush=True)

if __name__ == "__main__":
    plex_serv_url = input("Enter Plex Server URL (e.g., http://192.168.86.67:32400): ").strip()
    plex_token = input("Enter Plex Token: ").strip()
    plex_user_uuid = input("Enter your Plex user UUID (or accountID): ").strip()
    output_file = input("Enter output JSON file (default: plex_play_log.json): ").strip()
    if not output_file:
        output_file = "plex_play_log.json"
    export_recent_plays_json(plex_serv_url, plex_token, plex_user_uuid, output_file)
