# Plex_Playlist_Rater.py

import requests
import argparse
import xml.etree.ElementTree as ET

def get_playlists(server, token):
    headers = {'X-Plex-Token': token}
    try:
        response = requests.get(f'{server}/playlists', headers=headers)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        playlists = [
            {'title': playlist.get('title'), 'key': playlist.get('key')}
            for playlist in root.findall('Playlist')
        ]
        return sorted(playlists, key=lambda x: x['title'])

    except requests.exceptions.RequestException as e:
        print(f'Error fetching playlists: {e}')
        return []

def get_playlist_tracks(server, token, playlist_key):
    headers = {'X-Plex-Token': token}
    try:
        response = requests.get(f'{server}{playlist_key}', headers=headers)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        tracks = [
            {'title': track.get('title'), 'artist': track.get('grandparentTitle'), 
             'album': track.get('parentTitle'), 'ratingKey': track.get('ratingKey')}
            for track in root.findall('.//Track')
        ]
        return tracks

    except requests.exceptions.RequestException as e:
        print(f'Error fetching playlist tracks: {e}')
        return []

def rate_track(server, token, rating_key, rating=10):
    url = f"{server}/:/rate"
    params = {
        'key': rating_key,
        'identifier': 'com.plexapp.plugins.library',
        'rating': rating
    }
    headers = {'X-Plex-Token': token}

    print(f"Attempting to rate track - ratingKey: {rating_key}")

    try:
        response = requests.put(url, params=params, headers=headers)
        if response.status_code == 200:
            print(f"✅ Successfully rated track {rating_key}")
            return True
        else:
            print(f"❌ Failed to rate track {rating_key} - HTTP {response.status_code} - {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Error rating track {rating_key}: {e}")
        return False


# Exposed function for module use

def rate_playlists(plex_serv_url, plex_token):
    """
    Connects to the Plex server and sets a 5-star rating (10) for all tracks in a selected playlist.
    
    Parameters:
      - plex_serv_url: URL of the Plex server.
      - plex_token: Plex API token.
    """
    playlists = get_playlists(plex_serv_url, plex_token)
    if not playlists:
        print("No playlists found.")
        return

    print("\nAvailable Playlists:")
    for idx, playlist in enumerate(playlists, start=1):
        print(f"{idx}. {playlist['title']}")

    # Select a playlist
    try:
        playlist_number = int(input("\nEnter the number of the playlist to rate: "))
        if not (1 <= playlist_number <= len(playlists)):
            raise ValueError
    except ValueError:
        print("Invalid selection.")
        return

    selected_playlist = playlists[playlist_number - 1]
    print(f"\nFetching tracks for: {selected_playlist['title']}")

    # Get tracks in the selected playlist
    tracks = get_playlist_tracks(plex_serv_url, plex_token, selected_playlist['key'])
    if not tracks:
        print("No tracks found in the selected playlist.")
        return

    # Track failed ratings
    failed_tracks = []

    # Rate all tracks
    for track in tracks:
        print(f"Rating: {track['title']} by {track['artist']} ({track['album']})")
        success = rate_track(plex_serv_url, plex_token, track['ratingKey'])
        
        if not success:
            failed_tracks.append(f"{track['title']} by {track['artist']} ({track['album']})")

    # Summary Report
    print("\n==== Rating Summary ====")
    print(f"Total tracks in playlist: {len(tracks)}")
    print(f"Successfully rated: {len(tracks) - len(failed_tracks)}")
    print(f"Failed to rate: {len(failed_tracks)}")

    if failed_tracks:
        print("\n⚠️ The following tracks were NOT rated:")
        for failed_track in failed_tracks:
            print(f" - {failed_track}")
    else:
        print("\n✅ All tracks were rated successfully!")

if __name__ == '__main__':
    # If run directly, prompt for Plex credentials interactively.
    plex_serv_url = input("Enter Plex Server URL (e.g., http://<your-plex-server-ip>:32400): ").strip()
    plex_token = input("Enter Plex Token: ").strip()
    rate_playlists(plex_serv_url, plex_token)
