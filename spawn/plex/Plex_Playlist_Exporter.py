# Plex_Playlist_Exporter.py

import requests
import argparse
import xml.etree.ElementTree as ET
import os
import re

def get_playlists(server, token):
    headers = {
        'X-Plex-Token': token
    }
    try:
        response = requests.get(f'{server}/playlists', headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses
        print('Successfully connected to Plex server!')

        try:
            # Parse the XML response
            root = ET.fromstring(response.content)
            playlists = []
            for playlist in root.findall('Playlist'):
                playlists.append({
                    'title': playlist.get('title'),
                    'key': playlist.get('key')
                })
            return sorted(playlists, key=lambda x: x['title'])

        except ET.ParseError:
            print('Failed to parse XML response.')
            print('Response content:', response.text)
            return None

    except requests.exceptions.RequestException as e:
        print(f'Error connecting to Plex server: {e}')
        return None

def get_playlist_tracks(server, token, playlist_key):
    headers = {
        'X-Plex-Token': token
    }
    try:
        response = requests.get(f'{server}{playlist_key}', headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses

        try:
            # Parse the XML response
            root = ET.fromstring(response.content)
            tracks = []
            for track in root.findall('.//Track'):
                duration_ms = track.find('.//Media').get('duration')
                duration_s = int(duration_ms) // 1000 if duration_ms else -1
                tracks.append({
                    'title': track.get('title'),
                    'artist': track.get('grandparentTitle'),
                    'album': track.get('parentTitle'),
                    'file': track.find('.//Media/Part').get('file'),
                    'duration': duration_s
                })
            return tracks

        except ET.ParseError:
            print('Failed to parse XML response.')
            print('Response content:', response.text)
            return None

    except requests.exceptions.RequestException as e:
        print(f'Error connecting to Plex server: {e}')
        return None

def save_tracks_to_m3u(tracks, playlist_title):
    # Clean up the playlist title to create a valid filename
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', playlist_title)
    filename = f"{safe_title}.m3u"

    # Determine the target directory based on LIB_PATH
    lib_path = os.environ.get("LIB_PATH", "").strip()
    if lib_path and os.path.isdir(lib_path):
        target_dir = os.path.join(lib_path, "Spawn", "Playlists")
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = "."
    
    filepath = os.path.join(target_dir, filename)

    # Determine the base music directory dynamically based on the first track's file path
    if not tracks:
        print("No tracks to save.")
        return
    
    first_track_path = tracks[0]['file']
    music_collection_dir = os.path.dirname(os.path.dirname(os.path.dirname(first_track_path)))
    with open(filepath, 'w') as f:
        f.write("#EXTM3U\n")
        for track in tracks:
            f.write(f"#EXTINF:{track['duration']},{track['artist']} - {track['title']}\n")
            relative_path = os.path.relpath(track['file'], os.path.join(music_collection_dir, '..'))
            f.write(f"../{relative_path}\n")
    print(f"Track information saved to {filepath}")


# Exposed function for module use

def export_playlists(plex_serv_url, plex_token):
    """
    Connects to the Plex server and exports a selected playlist to an M3U file.
    
    Parameters:
      - plex_serv_url: URL of the Plex server.
      - plex_token: Plex API token.
    """
    playlists = get_playlists(plex_serv_url, plex_token)
    if playlists:
        print("Playlists:")
        for idx, playlist in enumerate(playlists, start=1):
            print(f"{idx}. {playlist['title']}")
        try:
            playlist_number = int(input("Enter the number of the playlist you want to export: "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return
        if 1 <= playlist_number <= len(playlists):
            selected_playlist = playlists[playlist_number - 1]
            print(f"Fetching tracks for playlist: {selected_playlist['title']}")
            print(f"Playlist key: {selected_playlist['key']}")
            tracks = get_playlist_tracks(plex_serv_url, plex_token, selected_playlist['key'])
            if tracks:
                save_tracks_to_m3u(tracks, selected_playlist['title'])
            else:
                print("No tracks found or failed to fetch tracks.")
        else:
            print("Invalid playlist number.")
    else:
        print("No playlists found or failed to fetch playlists.")

if __name__ == '__main__':
    # If run directly, prompt for Plex credentials interactively.
    plex_serv_url = input("Enter Plex Server URL (e.g., http://<your-plex-server-ip>:32400): ").strip()
    plex_token = input("Enter Plex Token: ").strip()
    export_playlists(plex_serv_url, plex_token)
