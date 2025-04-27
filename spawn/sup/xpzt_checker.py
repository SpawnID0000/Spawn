# xpzt_checker.py

# Requirements: requests, fuzzywuzzy, AtomicParsley installed and in PATH

import os
import sys
import requests
import base64
from urllib.parse import quote
from fuzzywuzzy import fuzz
import subprocess
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FUZZY_MATCH_THRESHOLD = 80

def create_session_with_retries():
    session = requests.Session()
    retries = Retry(
        total=5,                  # Retry up to 5 times
        backoff_factor=1,         # Exponential backoff: 1s, 2s, 4s, etc.
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
        allowed_methods=["GET", "POST"],  # Apply to both GET and POST
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def process_file(session, file_path, token, artist_name, song_name, client_id, client_secret, copy_mode=False, output_base=None, input_root=None, clean_mode=False):
    try:
        tracks, token = search_track(session, token, artist_name, song_name, client_id, client_secret)
    except Exception as e:
        print(f"❌ Error while searching Spotify for '{song_name}' by '{artist_name}': {e}\n")
        return token

    if not tracks:
        print(f"ℹ️ No tracks returned from Spotify for '{song_name}' by '{artist_name}'. Skipping.\n")
        return token

    explicit_found = False
    clean_available = False

    for track in tracks:
        track_name = track['name']
        track_artist = track['artists'][0]['name']
        score_name = fuzz.token_set_ratio(song_name.lower(), track_name.lower())
        score_artist = fuzz.token_set_ratio(artist_name.lower(), track_artist.lower())

        if score_name >= FUZZY_MATCH_THRESHOLD and score_artist >= FUZZY_MATCH_THRESHOLD:
            if track['explicit']:
                print(f"✅ Explicit version found: \"{track_name}\" by {track_artist} (Spotify URL: {track['external_urls']['spotify']})")
                explicit_found = True
            else:
                clean_available = True  # Found a clean version match

    if not explicit_found:
        print("ℹ️ No explicit versions found for this track on Spotify.\n")
        return token

    if clean_mode:
        if clean_available:
            print("🧼 A clean version is ALSO available on Spotify!")
        else:
            print("❌ No clean version found on Spotify.")
            if copy_mode:
                print("⚠️ Skipping copy/tag — Requires BOTH explicit and clean versions.\n")
                return token

    is_explicit = check_file_explicit(file_path)

    if is_explicit is None:
        print("⚠️ Could not determine if the file is marked explicit.\n")
        return token
    elif is_explicit:
        print("✅ The file is correctly marked as explicit.\n")
        return token
    else:
        print("\n🚨 WARNING: Spotify reports an explicit version is available, but THIS FILE IS NOT MARKED AS EXPLICIT!\n")

        target_file = file_path
        if copy_mode and output_base:
            rel_path = os.path.relpath(file_path, input_root)
            target_file = os.path.join(output_base, rel_path)
            out_dir = os.path.dirname(target_file)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(file_path, target_file)
            print(f"📄 Copied to: {target_file}")

        print("➡️ Applying explicit tag...")
        add_explicit_tag(target_file)

    return token

def extract_tags(file_path):
    try:
        result = subprocess.run(['AtomicParsley', file_path, '-t'], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        print(f"⏰ AtomicParsley timed out while reading tags from {file_path}")
        return None, None
    
    if result.returncode != 0:
        print(f"❌ AtomicParsley failed to read tags from {file_path}")
        return None, None

    output = result.stdout
    artist = None
    song = None

    for line in output.splitlines():
        if 'Atom "©ART"' in line:
            artist = line.split('contains:')[1].strip()
        if 'Atom "©nam"' in line:
            song = line.split('contains:')[1].strip()

    return artist, song

def get_spotify_token(session, client_id, client_secret):
    auth_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_header}"}
    data = {"grant_type": "client_credentials"}

    response = session.post(auth_url, headers=headers, data=data, timeout=10)
    response.raise_for_status()
    return response.json()["access_token"]

def search_track(session, token, artist_name, song_name, client_id, client_secret):
    query = f"track:{quote(song_name)} artist:{quote(artist_name)}"
    url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=20"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"🔎 Querying Spotify for: '{song_name}' by '{artist_name}'...")

    try:
        response = session.get(url, headers=headers, timeout=10)
    except requests.exceptions.Timeout:
        print(f"⏰ Timeout while searching for '{song_name}' by '{artist_name}'. Skipping.\n")
        return [], token
    except requests.exceptions.ConnectionError:
        print(f"🔌 Connection error while searching for '{song_name}' by '{artist_name}'. Skipping.\n")
        return [], token

    if response.status_code == 401:
        print("🔄 Token expired. Refreshing Spotify token...")
        token = get_spotify_token(session, client_id, client_secret)
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = session.get(url, headers=headers, timeout=10)
        except requests.exceptions.RequestException:
            print(f"❌ Failed after token refresh for '{song_name}' by '{artist_name}'. Skipping.\n")
            return [], token

        print("✅ Retried request after token refresh.")

        if response.status_code == 401:
            print("❌ Token refresh failed or invalid credentials.")
            raise Exception("Unauthorized after token refresh. Check your Client ID and Secret.")

    response.raise_for_status()
    return response.json()["tracks"]["items"], token

def check_file_explicit(file_path):
    try:
        result = subprocess.run(['AtomicParsley', file_path, '-t'], capture_output=True, text=True, timeout=15)
        output = result.stdout

        # print("\n🔧 AtomicParsley Raw Output:")
        # print(output)

        for line in output.splitlines():
            if 'Atom "rtng"' in line:
                print(f"\n🔎 Found: {line.strip()}")
                if 'Explicit Content' in line:
                    return True
                elif 'Clean' in line:
                    return False

        print("\nℹ️ No rtng advisory tag found in file.")
        return False

    except Exception as e:
        print(f"❌ Failed to check advisory tag: {e}")
        return None

def add_explicit_tag(file_path):
    try:
        print(f"\n⚡ Applying explicit tag using AtomicParsley for {file_path} ...")
        subprocess.run(['AtomicParsley', file_path, '--advisory', 'explicit', '--overWrite'], check=True)
        print(f"✅ AtomicParsley successfully marked the file as Explicit.\n")
    except Exception as e:
        print(f"❌ Failed to apply explicit tag with AtomicParsley: {e}")


def main():
    if not shutil.which('AtomicParsley'):
        print("❌ AtomicParsley is not installed. Cannot proceed.")
        sys.exit(1)

    copy_mode = False
    clean_mode = False
    output_base = None

    args = sys.argv[1:]

    if '-copy' in args:
        copy_index = args.index('-copy')
        try:
            output_base = args[copy_index + 1]
            copy_mode = True
            # Remove from args for easier parsing
            args = args[:copy_index] + args[copy_index+2:]
        except IndexError:
            print("❌ Error: You must specify a path after '-copy'")
            sys.exit(1)

    if '-clean' in args:
        clean_mode = True
        args.remove('-clean')

    if len(args) == 3:
        client_id, client_secret, file_path = args

    elif len(args) in [4, 5]:
        client_id = args[0]
        client_secret = args[1]
        artist_name = args[2]
        song_name = args[3]
        file_path = args[4] if len(args) == 5 else None

    else:
        print("Usage:")
        print("  python xpzt_checker.py CLIENT_ID CLIENT_SECRET [-copy /output/path] [-clean] \"Artist\" \"Song\" [file.m4a]")
        print("  OR")
        print("  python xpzt_checker.py CLIENT_ID CLIENT_SECRET [-copy /output/path] [-clean] path/to/file_or_directory")
        print("  Note: When using BOTH -copy and -clean, files are only copied if BOTH explicit AND clean versions exist on Spotify.")
        sys.exit(1)

    session = create_session_with_retries()

    try:
        token = get_spotify_token(session, client_id, client_secret)

        if not os.path.exists(file_path):
            print(f"❌ Error: The path '{file_path}' does not exist.")
            sys.exit(1)
        if os.path.isdir(file_path):
            print(f"\n📂 Scanning directory: {file_path}")

            for root, dirs, files in os.walk(file_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    if file.endswith('.m4a') and not file.startswith('.'):
                        full_path = os.path.join(root, file)

                        # Extract tags
                        artist, song = extract_tags(full_path)
                        if not artist or not song:
                            print(f"\n⚠️ Skipping {full_path} (missing Artist or Song tags)")
                            continue

                        print(f"\n------------------------")
                        print(f"🎵 Processing: {song} by {artist}")
                        token = process_file(session, full_path, token, artist, song, client_id, client_secret, copy_mode, output_base, file_path, clean_mode)
        else:
            if len(args) == 3:
                artist_name, song_name = extract_tags(file_path)
                if not artist_name or not song_name:
                    print("❌ Error: Could not extract Artist or Song Name from file tags.")
                    sys.exit(1)
                print(f"\n🎵 Processing: {song_name} by {artist_name}")

            print(f"\nChecking if '{song_name}' by '{artist_name}' is available in explicit form on Spotify...")
            token = process_file(session, file_path, token, artist_name, song_name, client_id, client_secret, copy_mode, output_base, file_path, clean_mode)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user.")