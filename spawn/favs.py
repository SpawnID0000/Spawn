# favs.py

import os
import json
import difflib
import shlex
import sqlite3


FAVS_FOLDER = "Spawn/aux/user/favs"  # subfolder relative to LIB_PATH

def update_favorites_menu(spawn_root):
    """
    Main entry point for the 'Update favorites' feature.
    Asks which type of favorites we are updating (Artists, Albums, or Tracks),
    then prompts user to enter M3U/CSV file path or direct list of Spawn IDs.
    It then parses the input and saves the resulting favorites to separate
    JSON files (fav_artists.json, etc.).
    
    :param spawn_root: The root path to the Spawn project (where "aux/user/favs" resides).
    """
    print("\nWhich list of favorites would you like to update?")
    print("    1) Artists")
    print("    2) Albums")
    print("    3) Tracks")

    while True:
        choice = input("\nEnter choice: ").strip()

        if choice not in ["1", "2", "3"]:
            print("[WARN] Invalid choice. Please select from the options listed.")
        else:
            break

    # Prompt for input: M3U file, CSV file, or direct Spawn IDs
    input_path = input("Enter the path to an M3U or CSV file listing your favorites, or directly enter Spawn IDs here: ").strip()
    parts = shlex.split(input_path)
    if not parts:
        print("[ERROR] No input provided.")
        return
    input_path = " ".join(parts)

    extinf_entries = []
    spawn_ids = []

    if os.path.isfile(input_path):
        if input_path.lower().endswith(".m3u"):
            print("[INFO] Parsing M3U file...")
            extinf_entries = parse_m3u_custom(input_path)
        elif input_path.lower().endswith(".csv"):
            print("[INFO] Parsing CSV file for Spawn IDs...")
            spawn_ids = parse_csv_for_spawn_ids(input_path)
        else:
            print("[ERROR] Unsupported file format. Please provide an M3U or CSV file.")
            return
    else:
        # Assume input is a direct list of Spawn IDs
        spawn_ids = [id.strip() for id in input_path.split(",") if id.strip()]

    if not extinf_entries and not spawn_ids:
        print("[INFO] No valid entries found. Returning to main menu.")
        return

    # Map user choice to favorite type
    fav_type_map = {
        "1": "artists",
        "2": "albums",
        "3": "tracks"
    }
    fav_type = fav_type_map[choice]

    # Build path to the favorites subfolder and the corresponding JSON file
    favs_folder = os.path.join(spawn_root, FAVS_FOLDER)
    os.makedirs(favs_folder, exist_ok=True)

    # e.g. "Spawn/aux/user/favs/fav_artists.json"
    fav_file = os.path.join(favs_folder, f"fav_{fav_type}.json")

    # Load existing favorites or initialize empty
    existing_favs = load_favorites_file(fav_file)

    # Build path to the Spawn catalog .db
    db_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "spawn_catalog.db")

    # -----------------------------------------------------------------
    # 1) Parse existing favorites depending on fav_type
    # -----------------------------------------------------------------

    if fav_type == "artists":
        # Convert existing_favs => dict keyed by artist_name -> {artist, artist_mbid}
        existing_artists_map = {}
        for item in existing_favs:
            if isinstance(item, dict):
                art = item.get("artist", "").strip()
                art_mbid = item.get("artist_mbid", None)
                if art:
                    existing_artists_map[art] = {
                        "artist": art,
                        "artist_mbid": art_mbid
                    }
            elif isinstance(item, str):
                # Legacy strings: store them as {artist=that_string, artist_mbid=None}
                art = item.strip()
                if art:
                    existing_artists_map[art] = {
                        "artist": art,
                        "artist_mbid": None
                    }
        # Store new matches in a list of dicts, then merge them into existing_artists_map
        matched_artists_data = []
        existing_album_pairs = None
        existing_tracks_map = None
    elif fav_type == "albums":
        # Convert existing_favs into a set of (artist, album) for easy "union"
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



    else:  # fav_type == "tracks"
        # We'll store them as {artist, album, track, track_mbid, spawn_id}
        existing_tracks_map = {}
        for item in existing_favs:
            if isinstance(item, dict):
                art   = item.get("artist", "").strip()
                alb   = item.get("album", "").strip()
                trk   = item.get("track", "").strip()
                mbid  = item.get("track_mbid", None)
                sp_id = item.get("spawn_id", None)
                key = (art, alb, trk)
                existing_tracks_map[key] = {
                    "artist": art,
                    "album": alb,
                    "track": trk,
                    "track_mbid": mbid,
                    "spawn_id": sp_id
                }
            elif isinstance(item, str):
                trk = item.strip()
                if trk:
                    key = ("", "", trk)
                    existing_tracks_map[key] = {
                        "artist": "",
                        "album": "",
                        "track": trk,
                        "track_mbid": None,
                        "spawn_id": None
                    }
        matched_artists_data = None
        existing_album_pairs = None
        matched_tracks_data = []

    matched_items = set()
    matched_albums_data = matched_albums_data if fav_type == "albums" else None
    matched_tracks_data = matched_tracks_data if fav_type == "tracks" else None
    nonmatched_items = set()

    # -----------------------------------------------------------------
    # Step 2a: Process M3U entries (if any)
    # -----------------------------------------------------------------
    for entry in extinf_entries:
        artist_guess = entry["artist"] or ""
        track_guess  = entry["track"]  or ""
        album_guess  = entry["album"]  or ""

        # Depending on user choice, "found_value" is the item they'd want to favorites
        if fav_type == "artists":
            found_value = artist_guess
        elif fav_type == "albums":
            # Store a tuple (artist, album) in memory
            found_value = (artist_guess, album_guess)
        else:  # tracks
            found_value = track_guess

        # If there's no found_value at all, skip
        if not any(found_value):
            continue

        # Attempt database lookup if the DB file exists
        matched_ids = []
        if os.path.isfile(db_path):
            matched_ids = lookup_in_spawn_catalog(
                db_path,
                artist=artist_guess,
                track=track_guess,
                album=album_guess
            )

        if matched_ids:
            print(f"[INFO] Found {len(matched_ids)} match(es) in spawn_catalog.db for "
                  f"Artist='{artist_guess}', Track='{track_guess}', Album='{album_guess}' => {matched_ids}")

            # If there's more than one, just pick the first; could also unify them
            first_id = matched_ids[0]

            if fav_type == "artists":
                art_mbid = get_artist_mbid(db_path, first_id)
                matched_artists_data.append({
                    "artist": artist_guess,
                    "artist_mbid": art_mbid
                })

            elif fav_type == "albums":
                rg_mbid = get_release_group_mbid(db_path, first_id)
                matched_albums_data.append({
                    "artist": artist_guess,
                    "album": album_guess,
                    "release_group_mbid": rg_mbid  # might be None if not found
                })
            else:
                # Tracks => gather the full track data from DB
                track_obj = get_track_data(db_path, first_id)
                # If get_track_data returns None, we fallback to just what we gleaned
                if not track_obj:
                    track_obj = {
                        "artist": artist_guess,
                        "album": album_guess,
                        "track": track_guess,
                        "track_mbid": None,
                        "spawn_id": first_id
                    }
                matched_tracks_data.append(track_obj)
        else:
            print(f"[WARN] No match found in spawn_catalog.db for "
                  f"Artist='{artist_guess}', Track='{track_guess}', Album='{album_guess}'")
            nonmatched_items.add(found_value)

    # -----------------------------------------------------------------
    # Step 2b: Process direct Spawn IDs (if provided)
    # -----------------------------------------------------------------
    if spawn_ids:
        for spawn_id in spawn_ids:
            if os.path.isfile(db_path):
                track_obj = get_track_data(db_path, spawn_id)
                if track_obj:
                    if fav_type == "artists":
                        art_mbid = get_artist_mbid(db_path, spawn_id)
                        matched_artists_data.append({
                            "artist": track_obj.get("artist", ""),
                            "artist_mbid": art_mbid
                        })
                    elif fav_type == "albums":
                        rg_mbid = get_release_group_mbid(db_path, spawn_id)
                        matched_albums_data.append({
                            "artist": track_obj.get("artist", ""),
                            "album": track_obj.get("album", ""),
                            "release_group_mbid": rg_mbid
                        })
                    else:  # tracks
                        matched_tracks_data.append(track_obj)
                else:
                    print(f"[WARN] No match found in spawn_catalog.db for Spawn ID '{spawn_id}'")
                    nonmatched_items.add(spawn_id)

    # -----------------------------------------------------------------
    # Step 3: Merge & Save
    # -----------------------------------------------------------------
    if fav_type == "artists":
        if matched_artists_data:
            # Merge these new artists into existing_artists_map
            for artist_obj in matched_artists_data:
                art_name = artist_obj["artist"]
                art_mbid = artist_obj.get("artist_mbid")
                # If already in map, you could override or keep old if you want
                existing_artists_map[art_name] = {
                    "artist": art_name,
                    "artist_mbid": art_mbid
                }

            # Convert to a list, sorted by artist name
            final_list = list(existing_artists_map.values())
            final_list.sort(key=lambda x: x["artist"].lower())
            save_favorites_file(fav_file, final_list)
            print(f"[INFO] Updated your 'artists' favorites with {len(matched_artists_data)} matched item(s).")
            print(f"       See => {fav_file}")
        else:
            print("[INFO] No new matched artists found. JSON file not updated.")

    elif fav_type == "albums":
        if matched_albums_data:
            # Convert existing_album_pairs to a dict keyed by (artist, album)
            existing_album_dict = {}
            for (art, alb) in existing_album_pairs:
                existing_album_dict[(art, alb)] = {
                    "artist": art,
                    "album": alb,
                    "release_group_mbid": None
                }

            # Merge in matched_albums_data
            for album_obj in matched_albums_data:
                key = (album_obj["artist"], album_obj["album"])
                existing_album_dict[key] = album_obj  # overwrite or add new

            # Convert back to a list
            final_list = list(existing_album_dict.values())
            # Sort by (artist, album) for consistency
            final_list.sort(key=lambda x: (x["artist"].lower(), x["album"].lower()))

            save_favorites_file(fav_file, final_list)
            print(f"[INFO] Updated your 'albums' favorites with {len(matched_albums_data)} matched item(s).")
            print(f"       See => {fav_file}")
        else:
            print("[INFO] No new matched albums found. JSON file not updated.")

    else:
        # Tracks => matched_tracks_data
        if matched_tracks_data:
            # Merge into existing_tracks_map
            for t_obj in matched_tracks_data:
                art = t_obj.get("artist", "").strip()
                alb = t_obj.get("album", "").strip()
                trk = t_obj.get("track", "").strip()
                key = (art, alb, trk)
                existing_tracks_map[key] = t_obj

            # Convert map to list, sort if you want by (artist, album, track)
            final_list = list(existing_tracks_map.values())
            final_list.sort(key=lambda x: (x["artist"].lower(),
                                           x["album"].lower(),
                                           x["track"].lower()))
            save_favorites_file(fav_file, final_list)
            print(f"[INFO] Updated your 'tracks' favorites with "
                  f"{len(matched_tracks_data)} matched item(s).")
        else:
            print(f"[INFO] No new matched {fav_type} found. JSON file not updated.")

    # -----------------------------------------------------------------
    # Step 4: Write non-matched items to a plain text file
    # -----------------------------------------------------------------

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
        print(f"[INFO] Wrote {len(nonmatched_items)} non-matched {fav_type} "
              f"to => {nonmatch_txt_file}")
    else:
        print(f"[INFO] No non-matched {fav_type} to report.")


def parse_m3u_custom(m3u_path):
    """
    Parses an M3U file specifically for lines of the form:
        #EXTINF:xxx,[name1] - [name2]
        ../Music/Artist/Album/track.m4a

    Returns a list of dicts:
        [
          { "artist": <artist_name>, "album": <album_name_or_None>, "track": <track_name> },
          ...
        ]

    We’ll parse the #EXTINF line, then check the file path to deduce the artist/track,
    and possibly the album.  If there's a conflict, we prompt the user.

    This version demonstrates one possible logic flow, using simple substring checks
    and directory traversal. You can refine the fuzzy matching as needed.
    """
    result = []
    lines = []
    with open(m3u_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # We'll store pairs of (extinf_line, path_line)
    entries = []
    for i in range(len(lines) - 1):
        line = lines[i].strip()
        if line.startswith('#EXTINF:'):
            # The next line should be the file path
            file_line = lines[i+1].strip() if (i+1 < len(lines)) else None
            if file_line and not file_line.startswith('#'):
                entries.append((line, file_line))

    for extinf_line, file_path in entries:
        # Example extinf_line: "#EXTINF:184,Artist - Track"
        # or "#EXTINF:184,Track - Artist"
        # We want to parse out name1, name2
        maybe_artist, maybe_track, album = parse_extinf_and_filepath(extinf_line, file_path)
        result.append({
            "artist": maybe_artist,
            "track": maybe_track,
            "album": album
        })

    return result


def parse_extinf_and_filepath(extinf_line, file_path):
    """
    Given the #EXTINF line and the subsequent file path, attempt to deduce
    the artist name, track name, and album name using the logic:

    - Parse extinf_line for something like "[name1] - [name2]"
    - Compare those names with:
      1) the file basename (minus extension),
      2) the directory 1 level up,
      3) the directory 2 levels up.
    - If conflict, prompt the user to resolve it.

    Returns (artist, track, album).
    """
    # 1) Extract name1 and name2 from the #EXTINF line
    #    e.g. "#EXTINF:184,Artist - Track"
    #         we want "Artist" and "Track"
    line_data = extinf_line.split(",", 1)
    if len(line_data) < 2:
        return (None, None, None)

    # Something like "Artist - Track"
    name_part = line_data[1].strip()
    name1, name2 = parse_name_part(name_part)

    # 2) Get the file name (without extension) and up to two levels of directories
    full_path_norm = os.path.normpath(file_path)  
    # e.g. "../Music/Artist/Album/track.m4a" -> 
    # splits: ["..", "Music", "Artist", "Album", "track.m4a"]
    path_parts = full_path_norm.split(os.sep)

    filename = os.path.splitext(path_parts[-1])[0]  # "track"
    parent_dirs = path_parts[:-1]                   # ["..", "Music", "Artist", "Album"]
    dir_1_up = parent_dirs[-1] if len(parent_dirs) >= 1 else ""
    dir_2_up = parent_dirs[-2] if len(parent_dirs) >= 2 else ""

    # 3) Attempt to figure out which is artist vs. track by comparing name1, name2 with filename & directory
    #    For a quick approach, we'll do simple "similar substring" checks.  For more robust results, use difflib.
    artist_guess, track_guess = guess_artist_track(name1, name2, filename, dir_1_up, dir_2_up)

    # 4) If the dir_2_up matches the final guess for artist, then dir_1_up is likely the album
    album_guess = None
    if artist_guess and dir_2_up:
        # Compare with dir_2_up for closeness
        ratio_dir2 = difflib.SequenceMatcher(None, artist_guess.lower(), dir_2_up.lower()).ratio()
        if ratio_dir2 > 0.6:
            # If it's a decent match, assume dir_1_up is the album
            album_guess = dir_1_up

    # 5) If there's a conflict or we have incomplete data, you can prompt the user
    #    For example, if we ended up with no confident guess.  
    #    In this example, we do a quick check for obviously missing data:
    if not artist_guess or not track_guess:
        print("\n[WARNING] Could not confidently determine artist/track from:")
        print(f"   EXTINF => {extinf_line}")
        print(f"   PATH   => {file_path}")
        # We'll prompt the user for a resolution:
        artist_guess = artist_guess or input("   Enter the correct artist: ").strip()
        track_guess  = track_guess  or input("   Enter the correct track:  ").strip()

    return (artist_guess, track_guess, album_guess)


def parse_name_part(name_part):
    """
    Attempts to split a string like "Artist - Track" or "Track - Artist"
    into (name1, name2). If no hyphen, return them unmodified, or None if blank.
    """
    # #EXTINF lines often look like "xxx,Artist - Track"
    # So name_part might be "Artist - Track"
    # We'll try to split on ' - '
    parts = name_part.split(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    else:
        # We can't find the " - ", so we just return the entire thing as name1, None as name2
        return (name_part.strip(), None)


def guess_artist_track(name1, name2, filename, dir_1_up, dir_2_up):
    """
    Returns (artist_guess, track_guess) using a weighted approach:
      - High directory similarity => more "artist-like"
      - High filename similarity  => more "track-like"
    """

    # print("\n--- DEBUG: guess_artist_track ---")
    # print(f"name1: {name1}")
    # print(f"name2: {name2}")
    # print(f"filename: {filename}")
    # print(f"dir_1_up: {dir_1_up}")
    # print(f"dir_2_up: {dir_2_up}")

    if name1 and name2 and name1.strip().lower() == name2.strip().lower():
        # The user says if artist & track are the same name, it really doesn't matter
        # which is which. So we'll just take them literally as the same string.
        return (name1, name2)

    # For simplicity, let's define a small helper to compute a similarity ratio:
    def similarity(a, b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # We'll gather a few similarity scores
    # Hypothesis 1: name1 == track, name2 == artist
    # Hypothesis 2: name1 == artist, name2 == track
    # Then compare which has a higher match ratio to filename, dir_1_up, dir_2_up, etc.

    # We only want to test if name1 and name2 are non-empty
    if name1 and not name2:
        # We only have one piece of data => maybe that's the track or the artist
        # We'll do a quick guess:
        score_filename = similarity(name1, filename)
        score_dir1 = similarity(name1, dir_1_up)
        score_dir2 = similarity(name1, dir_2_up)
        # If it's closer to filename than a directory, guess it's the track
        if score_filename >= max(score_dir1, score_dir2):
            return (None, name1)  # no clear artist
        else:
            return (name1, None)  # no clear track
    elif name2 and not name1:
        # Similar logic, but reversed
        score_filename = similarity(name2, filename)
        score_dir1 = similarity(name2, dir_1_up)
        score_dir2 = similarity(name2, dir_2_up)
        if score_filename >= max(score_dir1, score_dir2):
            return (None, name2)
        else:
            return (name2, None)

    # If we have two names:
    score1_file = similarity(name1, filename)
    score2_file = similarity(name2, filename)

    # We'll guess that whichever name is more similar to the filename is the track
    # and the other is the artist. But then we also check directories to see if name1
    # or name2 is similar to dir_1_up or dir_2_up.
    score1_dir1 = similarity(name1, dir_1_up)
    score1_dir2 = similarity(name1, dir_2_up)
    score2_dir1 = similarity(name2, dir_1_up)
    score2_dir2 = similarity(name2, dir_2_up)

    # print("\n--- DEBUG: Similarity Scores ---")
    # print(f" score1_file:  {score1_file:.3f}   (name1 -> filename)")
    # print(f" score2_file:  {score2_file:.3f}   (name2 -> filename)")
    # print(f" score1_dir1:  {score1_dir1:.3f}   (name1 -> dir_1_up)")
    # print(f" score1_dir2:  {score1_dir2:.3f}   (name1 -> dir_2_up)")
    # print(f" score2_dir1:  {score2_dir1:.3f}   (name2 -> dir_1_up)")
    # print(f" score2_dir2:  {score2_dir2:.3f}   (name2 -> dir_2_up)")

    # -- Define the weights --
    DIR_WEIGHT = 2.0
    FILE_INVERSE_WEIGHT = 1.0

    # artist_score: we WANT strong directory match, and we DISLIKE strong file match
    def artist_score(file_sim, dir1_sim, dir2_sim):
        return (dir1_sim + dir2_sim) * DIR_WEIGHT + (1.0 - file_sim) * FILE_INVERSE_WEIGHT

    # Compute an "artist-like" score for each name
    artist_score1 = artist_score(score1_file, score1_dir1, score1_dir2)
    artist_score2 = artist_score(score2_file, score2_dir1, score2_dir2)

    # print(f"\n--- DEBUG: Weighted Scores (Artist-Like) ---")
    # print(f" artist_score1 (name1): {artist_score1:.3f}")
    # print(f" artist_score2 (name2): {artist_score2:.3f}")

    # Whichever name has the higher "artist-like" score => that name is the artist
    if artist_score1 > artist_score2:
        # print("DEBUG: name1 is more 'artist-like' => (artist, track) = (name1, name2)")
        return (name1, name2)
    elif artist_score2 > artist_score1:
        # print("DEBUG: name2 is more 'artist-like' => (artist, track) = (name2, name1)")
        return (name2, name1)
    else:
        # print("DEBUG: Perfect tie => returning (None, None) or you could prompt user here.")
        return (None, None)


def lookup_in_spawn_catalog(db_path, artist=None, track=None, album=None):
    """
    Searches the spawn_catalog.db file for any rows in 'tracks' whose tag_data
    matches artist, track, album (case-insensitive).
    Returns a list of matching spawn_ids (or could return the entire row data).
    """

    # Normalize everything to lowercase for matching
    artist_lower = artist.lower() if artist else None
    track_lower  = track.lower()  if track  else None
    album_lower  = album.lower()  if album  else None

    matches = []

    if not os.path.isfile(db_path):
        print(f"[ERROR] spawn_catalog.db not found at: {db_path}")
        return matches  # empty

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # We'll just select everything and manually filter in Python
        cursor.execute("SELECT spawn_id, tag_data FROM tracks")
        rows = cursor.fetchall()

        for (spawn_id, tag_data_str) in rows:
            try:
                tag_data = json.loads(tag_data_str)
            except json.JSONDecodeError:
                # If the JSON is invalid, skip
                continue

            # Typically:
            #   "©ART" -> list of artist names
            #   "©nam" -> list of track names
            #   "©alb" -> list of album names
            # We'll do case-insensitive membership checks
            db_artists = [a.lower() for a in tag_data.get("©ART", []) if isinstance(a, str)]
            db_tracks  = [t.lower() for t in tag_data.get("©nam", []) if isinstance(t, str)]
            db_albums  = [al.lower() for al in tag_data.get("©alb", []) if isinstance(al, str)]

            # Check if we have a match:
            #   If 'artist' is given, it must match one of the db_artists
            #   If 'track'  is given, it must match one of the db_tracks
            #   If 'album'  is given, it must match one of the db_albums
            # If any field is None, we ignore it.
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
    Given a single spawn_id, return the first
    'MusicBrainz Release Group Id' found in the tag_data JSON.
    If it's missing or the row doesn't exist, return None.
    """
    if not os.path.isfile(db_path):
        return None
    
    import sqlite3, json

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT tag_data FROM tracks WHERE spawn_id = ?", (spawn_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        
        tag_data_str = row[0]  # the tag_data column
        try:
            tag_json = json.loads(tag_data_str)
        except json.JSONDecodeError:
            return None

        # The release group MBID is typically in:
        #   "----:com.apple.iTunes:MusicBrainz Release Group Id": ["some-uuid"]
        rg_list = tag_json.get("----:com.apple.iTunes:MusicBrainz Release Group Id")
        if isinstance(rg_list, list) and rg_list:
            return rg_list[0]  # if there are multiple, we just return the first
        return None

    except sqlite3.Error as e:
        print(f"[ERROR] get_release_group_mbid: {e}")
        return None


def load_favorites_file(filepath):
    """
    Loads a JSON file (list of strings) from 'filepath'.
    Returns an empty list if not found or invalid.
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
    Given a single spawn_id, return the first
    'MusicBrainz Artist Id' found in the tag_data JSON.
    If it's missing or row doesn't exist, return None.
    """
    import os
    import sqlite3
    import json

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

        # "----:com.apple.iTunes:MusicBrainz Artist Id": ["1882fe91-cdd9-49c9-9956-8e06a3810bd4"]
        mbid_list = tag_json.get("----:com.apple.iTunes:MusicBrainz Artist Id")
        if isinstance(mbid_list, list) and mbid_list:
            return mbid_list[0]  # Return the first if multiple
        return None

    except sqlite3.Error as e:
        print(f"[ERROR] get_artist_mbid: {e}")
        return None


def get_track_data(db_path, spawn_id):
    """
    Given a single spawn_id, returns a dict of:
      {
        "artist": ...,
        "album": ...,
        "track": ...,
        "track_mbid": ...,
        "spawn_id": spawn_id
      }
    ...based on the 'tracks' row's tag_data.
    If missing or invalid, return None.
    """
    import os
    import sqlite3
    import json

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

        # For each, we just pick the first string if present.
        artist_str = db_artists[0] if db_artists else ""
        album_str  = db_albums[0]  if db_albums  else ""
        title_str  = db_titles[0]  if db_titles  else ""
        track_mbid = db_mbid_list[0] if db_mbid_list else None

        return {
            "artist": artist_str,
            "album": album_str,
            "track": title_str,
            "track_mbid": track_mbid,
            "spawn_id": spawn_id
        }

    except sqlite3.Error as e:
        print(f"[ERROR] get_track_data: {e}")
        return None


def save_favorites_file(filepath, data):
    """
    Saves 'data' (list of strings) to the specified JSON file, prettified.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
