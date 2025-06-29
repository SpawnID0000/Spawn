# favs.py

import os
import json
import difflib
import shlex
import sqlite3
import tempfile
import unicodedata
import re
from mutagen import File as AudioFile

FAVS_FOLDER = "Spawn/aux/user/favs"  # subfolder relative to LIB_PATH

def update_favorites_menu(spawn_root):
    """
    Main entry point for the 'Update favorites' feature.
    Asks which list of favorites we are updating (Artists, Albums, or Tracks),
    or alternatively, exports favorite tracks as an M3U playlist,
    imports favorite tracks from Plex, or rates Plex playlists.
    
    :param spawn_root: The root path to the Spawn project (where "aux/user/favs" resides).
    """
    print("\nWhich list of favorites would you like to update?")
    print("    1) Artists")
    print("    2) Albums")
    print("    3) Tracks")
    print("\nOr, alternatively:")
    print("    4) Export M3U of favorite tracks")
    print("    5) Import favorite tracks from Plex")
    print("    6) Mark all tracks in a Plex playlist as favorites")
    valid_choices = ["1", "2", "3", "4", "5", "6"]
    choice = input("\nEnter choice: ").strip()

    if choice not in valid_choices:
        print("[WARN] Invalid choice. Returning to main menu.")
        return

    if choice == "4":
        export_favorite_tracks_m3u(spawn_root)
        return

    elif choice == "5":
        import_from_plex_favorites(spawn_root)
        return

    elif choice == "6":
        # Ensure Plex API parameters are available
        plex_serv_url = os.environ.get("PLEX_SERV_URL", "").strip()
        plex_token = os.environ.get("PLEX_TOKEN", "").strip()
        if not plex_serv_url:
            plex_serv_url = input("Enter Plex Server URL (e.g., http://192.168.86.67:32400): ").strip()
        if not plex_token:
            plex_token = input("Enter Plex Token: ").strip()
        from .plex.Plex_Playlist_Rater import rate_playlists
        rate_playlists(plex_serv_url, plex_token)
        return

    # --- Original logic for choices 1-3 ---
    input_path = input("Enter the path to an M3U, CSV, or JSON file listing your favorites, or directly enter Spawn IDs here: ").strip()
    parts = shlex.split(input_path)
    if not parts:
        print("[ERROR] No input provided.")
        return
    input_path = " ".join(parts)

    extinf_entries = []
    spawn_ids = []

    if os.path.isfile(input_path):
        ext = input_path.lower()
        if ext.endswith(".m3u"):
            print("[INFO] Parsing M3U file...")
            extinf_entries = parse_m3u_custom(input_path)
        elif ext.endswith(".csv"):
            print("[INFO] Parsing CSV file for Spawn IDs...")
            spawn_ids = parse_csv_for_spawn_ids(input_path)
        elif ext.endswith(".json"):
            print("[INFO] Parsing JSON file for Spawn IDs...")
            spawn_ids = parse_json_for_spawn_ids(input_path)
        else:
            print("[ERROR] Unsupported file format. Please provide an M3U, CSV, or JSON file.")
            return
    else:
        spawn_ids = [id.strip() for id in input_path.split(",") if id.strip()]

    if not extinf_entries and not spawn_ids:
        print("[INFO] No valid entries found. Returning to main menu.")
        return

    fav_type_map = {"1": "artists", "2": "albums", "3": "tracks"}
    fav_type = fav_type_map[choice]

    # Prepare path for fav_tracks.json
    favs_folder = os.path.join(spawn_root, FAVS_FOLDER)
    os.makedirs(favs_folder, exist_ok=True)
    fav_file = os.path.join(favs_folder, f"fav_{fav_type}.json")
    existing_tracks = load_favorites_file(fav_file)
    if not isinstance(existing_tracks, list):
        existing_tracks = []

    existing_favs = load_favorites_file(fav_file)
    db_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "spawn_catalog.db")

    # --- Parse existing favorites based on type ---
    if fav_type == "artists":
        existing_artists_map = {}
        for item in existing_favs:
            if isinstance(item, dict):
                art = item.get("artist", "").strip()
                art_mbid = item.get("artist_mbid", None)
                if art:
                    existing_artists_map[art] = {"artist": art, "artist_mbid": art_mbid}
            elif isinstance(item, str):
                art = item.strip()
                if art:
                    existing_artists_map[art] = {"artist": art, "artist_mbid": None}
        matched_artists_data = []
        existing_album_pairs = None
        existing_tracks_map = None
    elif fav_type == "albums":
        existing_album_pairs = set()
        for item in existing_favs:
            if isinstance(item, dict):
                art = item.get("artist", "").strip()
                alb = item.get("album", "").strip()
                existing_album_pairs.add((art, alb))
            elif isinstance(item, str):
                existing_album_pairs.add(("", item.strip()))
        matched_albums_data = []
        matched_artists_data = None
        existing_artists_map = None
        existing_tracks_map = None
    else:  # tracks
        existing_tracks_map = {}
        for item in existing_favs:
            if isinstance(item, dict):
                art = item.get("artist", "").strip()
                alb = item.get("album", "").strip()
                trk = item.get("track", "").strip()
                mbid = item.get("track_mbid", None)
                sp_id = item.get("spawn_id", None)
                key = (art, alb, trk)
                existing_tracks_map[key] = {"artist": art, "album": alb, "track": trk, "track_mbid": mbid, "spawn_id": sp_id}
            elif isinstance(item, str):
                trk = item.strip()
                if trk:
                    key = ("", "", trk)
                    existing_tracks_map[key] = {"artist": "", "album": "", "track": trk, "track_mbid": None, "spawn_id": None}
        matched_tracks_data = []
        matched_artists_data = None
        existing_album_pairs = None

    matched_items = set()
    matched_albums_data = matched_albums_data if fav_type == "albums" else None
    matched_tracks_data = matched_tracks_data if fav_type == "tracks" else None
    nonmatched_items = set()

    # --- Process M3U entries (if any) ---
    for entry in extinf_entries:
        artist_guess = entry["artist"] or ""
        track_guess  = entry["track"] or ""
        album_guess  = entry["album"] or ""

        if fav_type == "artists":
            found_value = artist_guess
        elif fav_type == "albums":
            found_value = (artist_guess, album_guess)
        else:
            found_value = track_guess

        if not any(found_value):
            continue

        matched_ids = []
        if os.path.isfile(db_path):
            matched_ids = lookup_in_spawn_catalog(db_path, artist=artist_guess, track=track_guess, album=album_guess)

        if matched_ids:
            print(f"[INFO] Found {len(matched_ids)} match(es) in spawn_catalog.db for "
                  f"Artist='{artist_guess}', Track='{track_guess}', Album='{album_guess}' => {matched_ids}")
            first_id = matched_ids[0]
            if fav_type == "artists":
                art_mbid = get_artist_mbid(db_path, first_id)
                matched_artists_data.append({"artist": artist_guess, "artist_mbid": art_mbid})
            elif fav_type == "albums":
                rg_mbid = get_release_group_mbid(db_path, first_id)
                matched_albums_data.append({"artist": artist_guess, "album": album_guess, "release_group_mbid": rg_mbid})
            else:
                track_obj = get_track_data(db_path, first_id)
                if not track_obj:
                    track_obj = {"artist": artist_guess, "album": album_guess, "track": track_guess, "track_mbid": None, "spawn_id": first_id}
                matched_tracks_data.append(track_obj)
        else:
            print(f"[WARN] No match found in spawn_catalog.db for "
                  f"Artist='{artist_guess}', Track='{track_guess}', Album='{album_guess}'")
            nonmatched_items.add(found_value)

    # --- Process direct Spawn IDs (if provided) ---
    if spawn_ids:
        for spawn_id in spawn_ids:
            if os.path.isfile(db_path):
                track_obj = get_track_data(db_path, spawn_id)
                if track_obj:
                    if fav_type == "artists":
                        art_mbid = get_artist_mbid(db_path, spawn_id)
                        matched_artists_data.append({"artist": track_obj.get("artist", ""), "artist_mbid": art_mbid})
                    elif fav_type == "albums":
                        rg_mbid = get_release_group_mbid(db_path, spawn_id)
                        matched_albums_data.append({"artist": track_obj.get("artist", ""), "album": track_obj.get("album", ""), "release_group_mbid": rg_mbid})
                    else:
                        matched_tracks_data.append(track_obj)
                else:
                    print(f"[WARN] No match found in spawn_catalog.db for Spawn ID '{spawn_id}'")
                    nonmatched_items.add(spawn_id)

    # --- Merge & Save ---
    if fav_type == "artists":
        if matched_artists_data:
            for artist_obj in matched_artists_data:
                art_name = artist_obj["artist"]
                art_mbid = artist_obj.get("artist_mbid")
                existing_artists_map[art_name] = {"artist": art_name, "artist_mbid": art_mbid}
            final_list = list(existing_artists_map.values())
            final_list.sort(key=lambda x: x["artist"].lower())
            save_favorites_file(fav_file, final_list)
            print(f"[INFO] Updated your 'artists' favorites with {len(matched_artists_data)} matched item(s).")
            print(f"       See => {fav_file}")
        else:
            print("[INFO] No new matched artists found. JSON file not updated.")
    elif fav_type == "albums":
        if matched_albums_data:
            existing_album_dict = {}
            for (art, alb) in existing_album_pairs:
                existing_album_dict[(art, alb)] = {"artist": art, "album": alb, "release_group_mbid": None}
            for album_obj in matched_albums_data:
                key = (album_obj["artist"], album_obj["album"])
                existing_album_dict[key] = album_obj
            final_list = list(existing_album_dict.values())
            final_list.sort(key=lambda x: (x["artist"].lower(), x["album"].lower()))
            save_favorites_file(fav_file, final_list)
            print(f"[INFO] Updated your 'albums' favorites with {len(matched_albums_data)} matched item(s).")
            print(f"       See => {fav_file}")
        else:
            print("[INFO] No new matched albums found. JSON file not updated.")
    else:
        if matched_tracks_data:
            for t_obj in matched_tracks_data:
                art = t_obj.get("artist", "").strip()
                alb = t_obj.get("album", "").strip()
                trk = t_obj.get("track", "").strip()
                key = (art, alb, trk)
                existing_tracks_map[key] = t_obj
            final_list = list(existing_tracks_map.values())
            final_list.sort(key=lambda x: (x["artist"].lower(), x["album"].lower(), x["track"].lower()))
            save_favorites_file(fav_file, final_list)
            print(f"[INFO] Updated your 'tracks' favorites with {len(matched_tracks_data)} matched item(s).")
        else:
            print(f"[INFO] No new matched {fav_type} found. JSON file not updated.")

    # --- Write non-matched items ---
    if nonmatched_items:
        nonmatch_txt_file = os.path.join(favs_folder, f"non-matched_fav_{fav_type}.txt")
        with open(nonmatch_txt_file, "w", encoding="utf-8") as f_txt:
            if fav_type == "albums":
                for val in sorted(nonmatched_items):
                    if isinstance(val, tuple):
                        a, b = val
                        f_txt.write(f"{a} - {b}\n")
            else:
                for val in sorted(nonmatched_items):
                    f_txt.write(f"{val}\n")
        print(f"[INFO] Wrote {len(nonmatched_items)} non-matched {fav_type} to => {nonmatch_txt_file}")
    else:
        print(f"[INFO] No non-matched {fav_type} to report.")


def export_favorite_tracks_m3u(spawn_root):
    """
    Exports the favorite tracks (from fav_tracks.json) to an M3U playlist.
    For each track, the function locates the associated symlink in
    LIB_PATH/Spawn/aux/user/linx using the Spawn ID, then uses its target
    file path for the M3U output. If the symlink or its target is missing,
    the user is prompted to supply the correct file path (which will update the symlink).
    The track duration (in seconds) is determined via Mutagen.
    
    The M3U is saved as:
        LIB_PATH/Spawn/Playlists/Favorites/favorite_tracks.m3u
    """
    favs_folder = os.path.join(spawn_root, FAVS_FOLDER)
    fav_tracks_file = os.path.join(favs_folder, "fav_tracks.json")
    if not os.path.isfile(fav_tracks_file):
        print("[INFO] No favorite tracks file found to export.")
        return

    favorite_tracks = load_favorites_file(fav_tracks_file)
    if not favorite_tracks:
        print("[INFO] Favorite tracks file is empty, nothing to export.")
        return

    export_folder = os.path.join(spawn_root, "Spawn", "Playlists", "Favorites")
    os.makedirs(export_folder, exist_ok=True)
    export_file = os.path.join(export_folder, "favorite_tracks.m3u")
    total_tracks = len(favorite_tracks)
    print(f"[INFO] Processing {total_tracks} track(s). Please wait...")

    with open(export_file, "w", encoding="utf-8") as m3u:
        m3u.write("#EXTM3U\n")
        for idx, track in enumerate(favorite_tracks, 1):
            if isinstance(track, dict):
                artist = track.get("artist", "Unknown Artist")
                title = track.get("track", "Unknown Title")
                spawn_id = track.get("spawn_id", "").strip()
                if not spawn_id:
                    print(f"[WARN] Track '{artist} - {title}' has no Spawn ID; skipping.")
                    continue

                target, symlink_path = get_symlink_target(spawn_root, spawn_id)
                if not target:
                    print(f"\n[WARN] Symlink for Spawn ID '{spawn_id}' not found or its target is missing.")
                    user_input = input(f"Enter the full path for track '{artist} - {title}' (or leave blank to skip): ").strip()
                    if user_input and os.path.isfile(user_input):
                        target = os.path.abspath(user_input)
                        update_symlink(symlink_path, target)
                    else:
                        print(f"[INFO] Skipping track '{artist} - {title}'.")
                        continue

                duration = get_audio_duration(target) if os.path.isfile(target) else -1
                m3u.write(f"#EXTINF:{duration},{artist} - {title}\n")
                m3u.write(f"{target}\n")
            elif isinstance(track, str):
                m3u.write(f"#EXTINF:-1,{track}\n")
                m3u.write(f"{track}\n")
            print(f"[INFO] Processed {idx}/{total_tracks} tracks...", end="\r")
    print(f"\n[INFO] Exported favorite tracks to M3U: {export_file}")


def normalize_text(s):
    """
    Normalizes text for matching by:
      1. Converting to Unicode NFKD form and encoding to ASCII (dropping non-ASCII characters)
      2. Converting to lowercase.
      3. Removing all whitespace and non-alphanumeric characters.
    """
    if not s:
        return ""
    # Normalize and drop non-ASCII characters
    normalized = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    normalized = normalized.lower().strip()
    # Remove all whitespace and punctuation to get a compact key for matching
    normalized = re.sub(r'\s+', '', normalized)      # remove whitespace
    normalized = re.sub(r'[^\w]', '', normalized)      # remove non-alphanumeric characters
    return normalized


def get_embedded_spawn_id(file_path):
    """
    Attempts to read the embedded spawn_ID tag from the given file.
    The tag key is "----:com.apple.iTunes:spawn_ID". Returns the value as a string if found;
    otherwise returns None.
    """
    try:
        audio = AudioFile(file_path)
        if audio and audio.tags:
            tag_val = audio.tags.get("----:com.apple.iTunes:spawn_ID")
            if tag_val:
                if isinstance(tag_val, list):
                    tag_val = tag_val[0]
                # If the tag is in bytes, decode it.
                if isinstance(tag_val, bytes):
                    tag_val = tag_val.decode('utf-8', errors='ignore')
                return tag_val.strip()
    except Exception as e:
        print(f"[WARN] Could not read embedded spawn_ID from file {file_path}: {e}")
    return None


def import_from_plex_favorites(spawn_root):
    """
    Imports favorite tracks from the Plex playlist titled '❤️ Tracks'
    by exporting the playlist to a temporary M3U file and then parsing it.
    For each parsed entry, it first attempts to read an embedded spawn_ID from the file.
    That value is used as the primary search term for the database lookup.
    If not found, it falls back to matching on artist/album/track.
    """
    # Load Plex credentials from environment variables
    plex_serv_url = os.environ.get("PLEX_SERV_URL", "").strip()
    plex_token = os.environ.get("PLEX_TOKEN", "").strip()

    if not plex_serv_url:
        plex_serv_url = input("Enter Plex Server URL (e.g., http://192.168.86.67:32400): ").strip()
    if not plex_token:
        plex_token = input("Enter Plex Token: ").strip()

    # Import Plex exporter functions
    from .plex import Plex_Playlist_Exporter as plex_exporter

    playlists = plex_exporter.get_playlists(plex_serv_url, plex_token)
    if not playlists:
        print("[ERROR] Could not fetch playlists from Plex.")
        return

    # Find the playlist titled '❤️ Tracks'
    target_playlist = None
    for pl in playlists:
        if pl['title'] == "❤️ Tracks":
            target_playlist = pl
            break

    if not target_playlist:
        print("[ERROR] Playlist '❤️ Tracks' not found in Plex playlists.")
        return

    print(f"[INFO] Found Plex playlist: {target_playlist['title']}. Fetching tracks...")
    tracks = plex_exporter.get_playlist_tracks(plex_serv_url, plex_token, target_playlist['key'])
    if not tracks:
        print("[ERROR] No tracks found in '❤️ Tracks'.")
        return

    # Export tracks to a temporary M3U file
    temp_dir = os.path.join(spawn_root, "Spawn", "aux", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_m3u_path = os.path.join(temp_dir, f"plex_favs_import_log.m3u")
    #import uuid
    #temp_m3u_path = os.path.join(temp_dir, f"temp_plex_export_{uuid.uuid4().hex}.m3u")

    with open(temp_m3u_path, "w", encoding="utf-8") as temp_m3u:
        temp_m3u.write("#EXTM3U\n")
        for track in tracks:
            duration = track.get('duration', -1)
            artist = track.get('artist', '').strip()
            title = track.get('title', '').strip()
            # Write EXTINF line and then the file path (assuming Plex provides an absolute path)
            temp_m3u.write(f"#EXTINF:{duration},{artist} - {title}\n")
            file_path = track.get('file', '').strip()
            temp_m3u.write(f"{file_path}\n")
        temp_m3u_path = temp_m3u.name

    print(f"[INFO] Exported temporary M3U file at: {temp_m3u_path}")

    # Parse the temporary M3U file
    m3u_entries = parse_m3u_custom(temp_m3u_path)

    # Deduplicate m3u_entries by normalized (artist, album, track)
    unique_entries = {}
    for entry in m3u_entries:
        key = (normalize_text(entry.get("artist") or ""),
               normalize_text(entry.get("album") or ""),
               normalize_text(entry.get("track") or ""))
        if key not in unique_entries:
            unique_entries[key] = entry
    m3u_entries = list(unique_entries.values())

    # Remove the temp M3U file
    try:
        os.remove(temp_m3u_path)
    except Exception as e:
        print(f"[WARN] Could not remove temporary M3U file: {e}")

    # Prepare path for fav_tracks.json
    favs_folder = os.path.join(spawn_root, FAVS_FOLDER)
    os.makedirs(favs_folder, exist_ok=True)
    fav_tracks_file = os.path.join(favs_folder, "fav_tracks.json")
    existing_tracks = load_favorites_file(fav_tracks_file)
    if not isinstance(existing_tracks, list):
        existing_tracks = []

    # Build a set of normalized keys from existing tracks (artist, album, track)
    existing_keys = set()
    # Also build a set of existing Spawn IDs
    existing_spawn_ids = set()
    for item in existing_tracks:
        if isinstance(item, dict):
            key = (normalize_text(item.get("artist")),
                   normalize_text(item.get("album")),
                   normalize_text(item.get("track")))
            existing_keys.add(key)
            sid = item.get("spawn_id")
            if sid:
                existing_spawn_ids.add(sid.strip())

    new_tracks = []
    # Get the path to the spawn_catalog.db for database lookups.
    db_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "spawn_catalog.db")

    for entry in m3u_entries:
        artist = (entry.get("artist") or "").strip()
        album = (entry.get("album") or "").strip()
        title = (entry.get("track") or "").strip()
        filepath = (entry.get("filepath") or "").strip()
        # Compute the normalized key from text (for fallback)
        norm_key = (normalize_text(artist), normalize_text(album), normalize_text(title))
        # If this normalized key is already in existing favorites, skip
        if norm_key in existing_keys:
            continue

        # Try to read the embedded spawn_ID from the file.
        embedded_spawn_id = get_embedded_spawn_id(filepath)

        # Now determine spawn_id:
        if embedded_spawn_id:
            spawn_ids = [embedded_spawn_id]
        else:
            spawn_ids = lookup_in_spawn_catalog(db_path, artist=artist, track=title, album=album)

        if spawn_ids:
            spawn_id = spawn_ids[0]
            # Check if this spawn_id is already in the JSON file.
            if spawn_id and spawn_id.strip() in existing_spawn_ids:
                continue
            track_data = get_track_data(db_path, spawn_id)
            if track_data:
                new_track = track_data
            else:
                new_track = {
                    "artist": artist,
                    "album": album,
                    "track": title,
                    "track_mbid": None,
                    "spawn_id": spawn_id
                }
        else:
            new_track = {
                "artist": artist,
                "album": album,
                "track": title,
                "track_mbid": None,
                "spawn_id": None
            }
        new_tracks.append(new_track)
        existing_keys.add(norm_key)
        if new_track.get("spawn_id"):
            existing_spawn_ids.add(new_track.get("spawn_id").strip())

    if new_tracks:
        merged_tracks = existing_tracks + new_tracks
        merged_tracks.sort(key=lambda x: (normalize_text(x.get("artist")),
                                          normalize_text(x.get("album")),
                                          normalize_text(x.get("track"))))
        save_favorites_file(fav_tracks_file, merged_tracks)
        print(f"[INFO] Imported {len(new_tracks)} new favorite tracks from Plex playlist '❤️ Tracks'.")
        print("\n[INFO] New tracks added:")
        for track in new_tracks:
            print(f"  - {track.get('artist', 'Unknown Artist')} - {track.get('track', 'Unknown Title')} (Album: {track.get('album', 'Unknown Album')})")
    else:
        print("[INFO] No new favorite tracks found in Plex playlist '❤️ Tracks'.")


def get_symlink_target(spawn_root, spawn_id):
    """
    Given the spawn_root and a Spawn ID, returns a tuple:
      (target_file_path or None, expected_symlink_path)
    The function looks for a symlink in LIB_PATH/Spawn/aux/user/linx named '<spawn_id>.m4a'.
    It uses os.path.realpath() to resolve the target if the symlink is found.
    """
    # Note: Since spawn_root is your LIB_PATH (e.g. /Volumes/Untitled),
    # we need to include "Spawn" in the path.
    linx_dir = os.path.join(spawn_root, "Spawn", "aux", "user", "linx")
    symlink_path = os.path.join(linx_dir, f"{spawn_id}.m4a")
    
    # Debug prints (optional):
    # print(f"[DEBUG] Checking symlink path: {symlink_path}")
    
    if os.path.lexists(symlink_path):
        if os.path.islink(symlink_path):
            target = os.path.realpath(symlink_path)
        else:
            # If it exists but is not flagged as a link, treat it as a file.
            target = symlink_path
        if os.path.exists(target):
            return os.path.abspath(target), symlink_path

    return None, symlink_path


def update_symlink(symlink_path, target):
    """
    Updates (or creates) the symlink at symlink_path to point to the given target.
    If a file or broken symlink already exists at symlink_path, it is removed first.
    """
    try:
        if os.path.lexists(symlink_path):
            os.remove(symlink_path)
        os.symlink(os.path.abspath(target), symlink_path)
        print(f"[INFO] Created/Updated symlink: {symlink_path} -> {os.path.abspath(target)}")
    except Exception as e:
        print(f"[ERROR] Failed to create/update symlink at {symlink_path}: {e}")


def get_audio_duration(file_path):
    """
    Uses Mutagen to open the audio file and return its duration in seconds (rounded to the nearest integer).
    Returns -1 if the duration cannot be determined.
    """
    try:
        audio = AudioFile(file_path)
        if audio is not None and hasattr(audio.info, 'length'):
            return int(round(audio.info.length))
    except Exception as e:
        print(f"[WARN] Failed to retrieve duration for '{file_path}': {e}")
    return -1


def parse_m3u_custom(m3u_path):
    """
    Parses an M3U file specifically for lines of the form:
        #EXTINF:xxx,Artist - Track
        ../Music/Artist/Album/track.m4a
    Returns a list of dicts with keys "artist", "track", "album", and "filepath".
    """
    result = []
    with open(m3u_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    for i in range(len(lines) - 1):
        line = lines[i].strip()
        if line.startswith('#EXTINF:'):
            file_line = lines[i+1].strip() if (i+1 < len(lines)) else None
            if file_line and not file_line.startswith('#'):
                entries.append((line, file_line))

    for extinf_line, file_path in entries:
        maybe_artist, maybe_track, album = parse_extinf_and_filepath(extinf_line, file_path)
        result.append({
            "artist": maybe_artist,
            "track": maybe_track,
            "album": album,
            "filepath": file_path
        })
    return result


def parse_extinf_and_filepath(extinf_line, file_path):
    """
    Given an EXTINF line and its following file path, deduce the artist, track, and album.
    """
    line_data = extinf_line.split(",", 1)
    if len(line_data) < 2:
        return (None, None, None)
    name_part = line_data[1].strip()
    name1, name2 = parse_name_part(name_part)
    full_path_norm = os.path.normpath(file_path)
    path_parts = full_path_norm.split(os.sep)
    filename = os.path.splitext(path_parts[-1])[0]
    parent_dirs = path_parts[:-1]
    dir_1_up = parent_dirs[-1] if len(parent_dirs) >= 1 else ""
    dir_2_up = parent_dirs[-2] if len(parent_dirs) >= 2 else ""
    artist_guess, track_guess = guess_artist_track(name1, name2, filename, dir_1_up, dir_2_up)
    album_guess = None
    if artist_guess and dir_2_up:
        ratio_dir2 = difflib.SequenceMatcher(None, artist_guess.lower(), dir_2_up.lower()).ratio()
        if ratio_dir2 > 0.6:
            album_guess = dir_1_up
    if not artist_guess or not track_guess:
        print("\n[WARNING] Could not confidently determine artist/track from:")
        print(f"   EXTINF => {extinf_line}")
        print(f"   PATH   => {file_path}")
        artist_guess = artist_guess or input("   Enter the correct artist: ").strip()
        track_guess  = track_guess  or input("   Enter the correct track: ").strip()
    return (artist_guess, track_guess, album_guess)


def parse_name_part(name_part):
    """
    Splits a string like 'Artist - Track' into two parts.
    """
    parts = name_part.split(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    else:
        return (name_part.strip(), None)


def guess_artist_track(name1, name2, filename, dir_1_up, dir_2_up):
    """
    Returns (artist_guess, track_guess) by comparing provided names with filename and directory names.
    """
    if name1 and name2 and name1.strip().lower() == name2.strip().lower():
        return (name1, name2)
    def similarity(a, b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    if name1 and not name2:
        score_filename = similarity(name1, filename)
        score_dir1 = similarity(name1, dir_1_up)
        score_dir2 = similarity(name1, dir_2_up)
        if score_filename >= max(score_dir1, score_dir2):
            return (None, name1)
        else:
            return (name1, None)
    elif name2 and not name1:
        score_filename = similarity(name2, filename)
        score_dir1 = similarity(name2, dir_1_up)
        score_dir2 = similarity(name2, dir_2_up)
        if score_filename >= max(score_dir1, score_dir2):
            return (None, name2)
        else:
            return (name2, None)
    score1_file = similarity(name1, filename)
    score2_file = similarity(name2, filename)
    score1_dir1 = similarity(name1, dir_1_up)
    score1_dir2 = similarity(name1, dir_2_up)
    score2_dir1 = similarity(name2, dir_1_up)
    score2_dir2 = similarity(name2, dir_2_up)
    DIR_WEIGHT = 2.0
    FILE_INVERSE_WEIGHT = 1.0
    def artist_score(file_sim, dir1_sim, dir2_sim):
        return (dir1_sim + dir2_sim) * DIR_WEIGHT + (1.0 - file_sim) * FILE_INVERSE_WEIGHT
    artist_score1 = artist_score(score1_file, score1_dir1, score1_dir2)
    artist_score2 = artist_score(score2_file, score2_dir1, score2_dir2)
    if artist_score1 > artist_score2:
        return (name1, name2)
    elif artist_score2 > artist_score1:
        return (name2, name1)
    else:
        return (None, None)


def lookup_in_spawn_catalog(db_path, artist=None, track=None, album=None):
    """
    Searches spawn_catalog.db for rows in 'tracks' whose tag_data matches artist, track, album.
    Returns a list of matching spawn_ids.
    """
    artist_lower = artist.lower() if artist else None
    track_lower  = track.lower()  if track  else None
    album_lower  = album.lower()  if album  else None
    matches = []
    if not os.path.isfile(db_path):
        print(f"[ERROR] spawn_catalog.db not found at: {db_path}")
        return matches
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT spawn_id, tag_data FROM tracks")
        rows = cursor.fetchall()
        for (spawn_id, tag_data_str) in rows:
            try:
                tag_data = json.loads(tag_data_str)
            except json.JSONDecodeError:
                continue
            db_artists = [a.lower() for a in tag_data.get("©ART", []) if isinstance(a, str)]
            db_tracks  = [t.lower() for t in tag_data.get("©nam", []) if isinstance(t, str)]
            db_albums  = [al.lower() for al in tag_data.get("©alb", []) if isinstance(al, str)]
            artist_ok = (not artist_lower) or any(artist_lower == x for x in db_artists)
            track_ok  = (not track_lower)  or any(track_lower  == x for x in db_tracks)
            album_ok  = (not album_lower)  or any(album_lower  == x for x in db_albums)
            if artist_ok and track_ok and album_ok:
                matches.append(spawn_id)
        conn.close()
    except sqlite3.Error as e:
        print(f"[ERROR] SQLite Error: {e}")
    return matches


def get_release_group_mbid(db_path, spawn_id):
    """
    Given a spawn_id, returns the first MusicBrainz Release Group Id from tag_data.
    """
    if not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT tag_data FROM tracks WHERE spawn_id = ?", (spawn_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        tag_data_str = row[0]
        try:
            tag_json = json.loads(tag_data_str)
        except json.JSONDecodeError:
            return None
        rg_list = tag_json.get("----:com.apple.iTunes:MusicBrainz Release Group Id")
        if isinstance(rg_list, list) and rg_list:
            return rg_list[0]
        return None
    except sqlite3.Error as e:
        print(f"[ERROR] get_release_group_mbid: {e}")
        return None


def load_favorites_file(filepath):
    """
    Loads a JSON file from 'filepath'. Returns an empty list if not found or invalid.
    """
    if not os.path.isfile(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except:
        return []


def get_artist_mbid(db_path, spawn_id):
    """
    Given a spawn_id, returns the first MusicBrainz Artist Id from tag_data.
    """
    import os, sqlite3, json
    if not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT tag_data FROM tracks WHERE spawn_id = ?", (spawn_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        tag_data_str = row[0]
        try:
            tag_json = json.loads(tag_data_str)
        except json.JSONDecodeError:
            return None
        mbid_list = tag_json.get("----:com.apple.iTunes:MusicBrainz Artist Id")
        if isinstance(mbid_list, list) and mbid_list:
            return mbid_list[0]
        return None
    except sqlite3.Error as e:
        print(f"[ERROR] get_artist_mbid: {e}")
        return None


def get_track_data(db_path, spawn_id):
    """
    Given a spawn_id, returns a dict with track details from tag_data.
    """
    import os, sqlite3, json
    if not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT tag_data FROM tracks WHERE spawn_id = ?", (spawn_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        tag_data_str = row[0]
        try:
            tag_json = json.loads(tag_data_str)
        except json.JSONDecodeError:
            return None
        db_artists = tag_json.get("©ART", [])
        db_albums  = tag_json.get("©alb", [])
        db_titles  = tag_json.get("©nam", [])
        db_mbid_list = tag_json.get("----:com.apple.iTunes:MusicBrainz Track Id", [])
        artist_str = db_artists[0] if db_artists else ""
        album_str  = db_albums[0]  if db_albums  else ""
        title_str  = db_titles[0]  if db_titles  else ""
        track_mbid = db_mbid_list[0] if db_mbid_list else None
        return {"artist": artist_str, "album": album_str, "track": title_str, "track_mbid": track_mbid, "spawn_id": spawn_id}
    except sqlite3.Error as e:
        print(f"[ERROR] get_track_data: {e}")
        return None


def save_favorites_file(filepath, data):
    """
    Saves 'data' to the specified JSON file with pretty formatting.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_csv_for_spawn_ids(csv_path):
    """
    Parses a CSV file to extract Spawn IDs.
    This placeholder assumes one Spawn ID per line.
    """
    spawn_ids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                spawn_ids.append(line)
    return spawn_ids


def parse_json_for_spawn_ids(json_path):
    """
    Load a JSON file containing either a single object or a list of
    objects with a 'spawn_id' field, and return a list of those IDs.
    """
    spawn_ids = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to open or parse JSON file: {e}")
        return spawn_ids

    # normalize into a list of entries
    entries = data if isinstance(data, list) else [data]

    for entry in entries:
        if isinstance(entry, dict):
            sid = entry.get("spawn_id", "")
            if isinstance(sid, str) and sid.strip():
                spawn_ids.append(sid.strip())
            else:
                print(f"[WARNING] JSON entry missing valid 'spawn_id': {entry}")
        else:
            print(f"[WARNING] Skipping non-object JSON entry: {entry}")

    return spawn_ids


if __name__ == "__main__":
    spawn_root_input = input("Enter the Spawn root path: ").strip()
    update_favorites_menu(spawn_root_input)
