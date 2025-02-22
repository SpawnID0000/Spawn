# year_checker.py

#!/usr/bin/env python3
import os
import sys
from mutagen.mp4 import MP4
import json
import sqlite3

def extract_year(file_path):
    """
    Extracts the year from the '©day' tag in an m4a file.
    If the tag contains more than 4 characters (e.g. a full date),
    only the first 4 characters (the year) are returned.
    """
    try:
        audio = MP4(file_path)
        # '©day' tag holds the release date; its first element should be a string.
        tag = audio.tags.get('©day')
        if tag and tag[0]:
            return tag[0][:4]
    except Exception:
        pass
    return None

def extract_spawn_id(file_path):
    """
    Extracts the spawn_ID from the '----:com.apple.iTunes:spawn_ID' tag in an m4a file.
    If the value is a bytes object, it is decoded to a UTF-8 string.
    """
    try:
        audio = MP4(file_path)
        tag = audio.tags.get("----:com.apple.iTunes:spawn_ID")
        if tag and tag[0]:
            value = tag[0]
            if isinstance(value, bytes):
                return value.decode('utf-8')
            return str(value)
    except Exception:
        pass
    return None

def process_album(album_path):
    """
    Processes all .m4a files in an album folder and returns a dictionary mapping
    each unique year value to the number of tracks with that year.
    Skips files that start with "._" to avoid hidden or resource files.
    """
    year_counts = {}
    for item in os.listdir(album_path):
        if item.startswith("._"):
            continue
        if item.lower().endswith('.m4a'):
            file_path = os.path.join(album_path, item)
            year = extract_year(file_path)
            if year not in year_counts:
                year_counts[year] = 0
            year_counts[year] += 1
    return year_counts

def update_db_for_file(file_path, db_conn, new_year):
    """
    Updates the database entry (in the tracks table, tag_data column) that matches the file's spawn_ID.
    Loads the JSON stored in tag_data and looks for either a key named "@day" or "©day":
      - If one is found, its value is updated.
      - Otherwise, the JSON is dumped for debugging, the script pauses, and a new "@day" key is added.
    """
    spawn_id = extract_spawn_id(file_path)
    if not spawn_id:
        print(f"No spawn_ID found for {file_path}. Skipping DB update.")
        return

    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT rowid, tag_data FROM tracks WHERE tag_data LIKE ?", ('%' + spawn_id + '%',))
        row = cursor.fetchone()
        if row is None:
            print(f"No database entry found for spawn_ID {spawn_id} from file {file_path}.")
            return

        rowid, tag_data = row
        data = json.loads(tag_data)
        found_key = None
        if "@day" in data:
            found_key = "@day"
        elif "©day" in data:
            found_key = "©day"

        if found_key is None:
            print(f"No '@day' or '©day' key found in DB for spawn_ID {spawn_id}; dumping all tag_data for debugging:")
            print(json.dumps(data, indent=2))
            input("Press Enter to continue...")
            print(f"Adding '@day': {new_year} to DB for spawn_ID {spawn_id}.")
            data["@day"] = new_year
        else:
            print(f"Found existing {found_key} value: {data[found_key]} in DB for spawn_ID {spawn_id}; updating to {new_year}.")
            data[found_key] = new_year

        new_tag_data = json.dumps(data)
        cursor.execute("UPDATE tracks SET tag_data = ? WHERE rowid = ?", (new_tag_data, rowid))
        db_conn.commit()
        print(f"Updated DB for spawn_ID {spawn_id} in file {file_path}.")
    except Exception as e:
        print(f"Error updating DB for {file_path}: {e}")

def update_album_year(album_path, new_year, db_conn=None):
    """
    Updates the '©day' tag of all .m4a files in the album folder.
    If a db_conn is provided, also update the corresponding database entry for each file.
    Skips files that start with "._" to avoid processing hidden files.
    """
    for item in os.listdir(album_path):
        if item.startswith("._"):
            continue
        if item.lower().endswith('.m4a'):
            file_path = os.path.join(album_path, item)
            try:
                audio = MP4(file_path)
                audio.tags['©day'] = [new_year]
                audio.save()
                print(f"Updated file: {file_path}")
            except Exception as e:
                print(f"Error updating {file_path}: {e}")
            if db_conn:
                update_db_for_file(file_path, db_conn, new_year)

def main():
    # Usage: python year_checker.py path/to/Music [path/to/spawn_catalog.db]
    if len(sys.argv) not in [2, 3]:
        print("Usage: python year_checker.py path/to/Music [path/to/spawn_catalog.db]")
        sys.exit(1)

    music_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) == 3 else None

    if not os.path.isdir(music_path):
        print("Invalid music directory:", music_path)
        sys.exit(1)

    db_conn = None
    if db_path:
        if not os.path.isfile(db_path):
            print("Invalid database file:", db_path)
            sys.exit(1)
        try:
            db_conn = sqlite3.connect(db_path)
        except Exception as e:
            print("Error connecting to database:", e)
            sys.exit(1)

    # Walk through the music directory; an album folder is one that contains .m4a files and no subdirectories.
    for root, dirs, files in os.walk(music_path):
        if not dirs and any(f.lower().endswith('.m4a') for f in files):
            year_counts = process_album(root)
            # Flag the album if more than one unique year is found.
            if len(year_counts) > 1:
                print(f"\nAlbum folder: {root}")
                # Format the output as "year (count)"
                tag_summary = ", ".join(f"{year} ({count})" for year, count in year_counts.items())
                print("Found year tags:", tag_summary)
                prompt = (
                    "This album folder has mismatched years. If intentional, press 'y'.\n"
                    "Otherwise, type the correct 4-digit year for all output files: "
                )
                user_input = input(prompt).strip()
                if user_input.lower() == 'y':
                    print("Leaving album unchanged.")
                elif len(user_input) == 4 and user_input.isdigit():
                    update_album_year(root, user_input, db_conn)
                    print(f"Updated album folder with year {user_input}.")
                else:
                    print("Invalid input. Skipping update for this album.")

    if db_conn:
        db_conn.close()

if __name__ == '__main__':
    main()
