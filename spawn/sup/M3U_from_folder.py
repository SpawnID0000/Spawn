import os
import sys
from mutagen import File

def get_track_info(file_path):
    """ Extract track metadata including album, disc number, and track number. """
    try:
        audio = File(file_path, easy=True)
        if audio is None:
            return None

        duration = int(audio.info.length)
        title = audio.get('title', ['Unknown Title'])[0]
        artist = audio.get('artist', ['Unknown Artist'])[0]
        album = audio.get('album', ['Unknown Album'])[0]
        track_number = audio.get('tracknumber', [0])[0]
        disc_number = audio.get('discnumber', [0])[0]

        # Handling different types for track numbers and disc numbers
        track_number = parse_number(track_number)
        disc_number = parse_number(disc_number)

        return duration, title, artist, album, disc_number, track_number
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def parse_number(number):
    """ Parse a number which could be in a string format like '1/10'. """
    if isinstance(number, str):
        try:
            return int(number.split('/')[0])
        except ValueError:
            return 0
    elif isinstance(number, int):
        return number
    else:
        return 0

def normalize_artist_name(artist_name):
    """Normalize the artist name by removing 'The ' prefix if present."""
    if artist_name.lower().startswith('the '):
        return artist_name[4:]
    return artist_name

def generate_m3u(music_dir, m3u_file_path):
    entries = []
    
    for root, _, files in os.walk(music_dir):
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue

            if file.endswith(('.opus', '.m4a', '.mp3', '.flac')):
                file_path = os.path.join(root, file)
                track_info = get_track_info(file_path)
                if track_info:
                    duration, title, artist, album, disc_number, track_number = track_info

                    relative_path = os.path.join('..', os.path.relpath(file_path, os.path.dirname(m3u_file_path)))
                    entries.append((artist, album, disc_number, track_number, title, duration, relative_path))

    # Sorting by normalized artist name, album, disc number, and then track number
    entries.sort(key=lambda x: (normalize_artist_name(x[0]).lower(), x[1].lower(), x[2], x[3]))

    with open(m3u_file_path, 'w') as m3u_file:
        m3u_file.write("#EXTM3U\n")
        for entry in entries:
            m3u_file.write(f"#EXTINF:{entry[5]},{entry[4]} - {entry[0]}\n")
            m3u_file.write(f"{entry[6]}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python M3U_creator_from_Music_folder.py <path_to_music_directory>")
        sys.exit(1)

    music_dir = sys.argv[1]
    m3u_file_path = os.path.join(os.path.dirname(music_dir), "playlist.m3u")
    generate_m3u(music_dir, m3u_file_path)
    print(f"Playlist generated at {m3u_file_path}")
