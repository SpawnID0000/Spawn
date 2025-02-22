# Spawn

**Spawn** is a Python-based audio manager that imports music tracks, standardizes and enriches metadata (tags, MBIDs, ReplayGain, etc.), provides a flexible favorites system, creates curated playlists, and can play M3U files (via [mpv](https://mpv.io/)).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [System Requirements](#system-requirements)
  - [Python Dependencies](#python-dependencies)
- [Usage](#usage)
  - [1. Import Tracks](#1-import-tracks)
  - [2. Update Favorites](#2-update-favorites)
  - [3. Create Curated Playlist](#3-create-curated-playlist)
  - [4. Play M3U Playlist](#4-play-m3u-playlist)
- [Modules Overview](#modules-overview)
- [Additional Configuration](#additional-configuration)
- [License](#license)
- [Contributing](#contributing)

---

## Features

- **Audio Import & Standardization:**  
  Cleans and repackages AAC/ALAC/FLAC to a uniform M4A format, embedding [MusicBrainz](https://musicbrainz.org/) IDs, [AcoustID](https://acoustid.org/) fingerprints, and more.

- **Favorites System:**  
  Track your favorite artists, albums, or tracks. Filter curated playlists by those favorites.

- **Curated Playlists:**  
  - Group tracks by into canonical genre clusters
  - Leverage a detailed genre dictionary ([`dic_spawnre.py`](./spawn/dic_spawnre.py)) that supports subgenres, *multi-parent* relationships, and synonyms.
  - Reorder clusters by “related” subgenres, optionally filter by favorites, and output an `.m3u` playlist.
  - Optionally use advanced embeddings or numeric audio features to achieve *smooth transitions*.

- **M3U Playback (via mpv):**  
  Select an `.m3u` file and easily play it with mpv from the command line.

- **Optional API Integrations:**  
  - [Spotify](https://spotipy.readthedocs.io/) for searching and matching track IDs or audio features.  
  - [Last.fm](https://www.last.fm/api) for genre/fingerprint lookups.  
  - [MusicBrainzNGS](https://python-musicbrainzngs.readthedocs.io/) for MBIDs.

---

## Installation

### System Requirements

- **Python 3.6 or higher** (tested up to Python 3.10).  
- **ReplayGain**
  - On macOS, install via 'brew install rsgain'
  - On Ubuntu/Debian, 'sudo apt install rsgain'
  - On Windows, see https://github.com/complexlogic/rsgain
- **mpv** media player (for playlist playback):  
  - On macOS, install via `brew install mpv`  
  - On Ubuntu/Debian, `sudo apt-get install mpv`  
  - On Windows, see [mpv.io installation guide](https://mpv.io/installation/)  
- For advanced features (optional):
  - [ffmpeg](https://ffmpeg.org/) (for ALAC/FLAC conversions).
  - [MP4Box](https://gpac.wp.imt.fr/downloads/) (for AAC cleaning).
  - [chromaprint](https://acoustid.org/chromaprint) / `fpcalc` (for AcoustID fingerprinting).

### Python Dependencies

Install via **pip** from the repository or after cloning:

```bash
git clone https://github.com/SpawnID0000/Spawn.git
cd Spawn
pip install -r requirements.txt
```

Or if using the setup.py:

```bash
git clone https://github.com/SpawnID0000/Spawn.git
cd Spawn
python setup.py install
```

Dependencies (summarized):
- mutagen (https://mutagen.readthedocs.io/)
- spotipy (https://spotipy.readthedocs.io/)
- musicbrainzngs (https://python-musicbrainzngs.readthedocs.io/)
- requests, Pillow, python-dotenv
- python-mpv (wrapping mpv; mpv must still be installed system-wide) (https://pypi.org/project/python-mpv/)
- …and others in requirements.txt.

## Usage

If you installed via setup.py (which includes the console script entry), you can run:
```bash
spawn
```

Or, if you’re running from source, do:
```bash
python -m spawn.main
```

You’ll see a text menu like:
```
Select one of the following options:
   (or press any other key to exit)

    1) Import tracks
    2) Update favorites
    3) Create curated playlist
    4) Play M3U playlist

Enter choice:
```

### 1. Import Tracks
- **Purpose:** Repackage, standardize metadata, optionally fetch MBIDs, compute ReplayGain, etc.
- **Implementation:** track_importer.py
- **Usage:**
  - Enter “1” from the main menu.
  - Provide input folder (where your raw audio is).
  - Provide (or confirm) the LIB_PATH for your local Spawn library.

Tracks will be processed, cleaned, assigned a spawn_ID and stored in your Spawn/Music folder with standardized tags and naming convention.

### 2. Update Favorites
- **Purpose:** Add or edit your favorite artists, albums, or tracks.
- **Implementation:** favs.py
- **Usage:**
  - Enter “2” from the main menu.
  - Choose which favorites you want to update (artists, albums, or tracks).
  - Provide an .m3u that references them.
  - Favorites are saved in JSON under Spawn/aux/user/favs/fav_artists.json, fav_albums.json, fav_tracks.json.

### 3. Create Curated Playlist
- **Purpose:** Build an .m3u from your library, grouping by “spawnre” or fallback genre, optionally filtering by favorites.
- **Implementation:** curator.py
- **Usage:**
  - Enter “3” from the main menu.
  - Optionally filter by favorite artists, albums, or tracks.
  - The script clusters your tracks by genre, allows reordering, and writes a curated .m3u to Spawn/Playlists/Curated/.
    - **Basic Mode:** Randomly shuffle tracks within each genre cluster.
    - **Feature Mode:** Use numeric audio features (tempo, loudness, etc.) to do a nearest-neighbor chain within each cluster.
    - **Advanced Mode:** Use embeddings plus a “chain-based” approach to refine cluster membership (with an “outliers” bucket if needed).
                          Includes code & trained model from Deej-AI: https://github.com/teticio/Deej-AI).

### 4. Play M3U Playlist
- **Purpose:** Launch mpv with a chosen .m3u.
- **Implementation:** player.py
- **Usage:**
  - Enter “4” from the main menu.
  - Select from discovered .m3u files in Spawn/Playlists/ or manually type a path.
  - mpv then plays the tracks in that playlist.

## Modules Overview
1. **track_importer.py**
   Handles audio standardization, tag rewriting, MBIDs, replaygain, etc.
2. **favs.py**
   Manages your JSON favorites lists (fav_artists.json, fav_albums.json, fav_tracks.json).
3. **curator.py**
   Creates curated playlists from your library, optionally filtering by favorites, grouping by genre, reordering clusters via “Related” relationships, or advanced embedding approaches.
4. **player.py**
   Finds .m3u files and plays them in mpv.
5. **main.py**
   Provides the CLI menu. Imports the other modules and orchestrates user input.

## Additional Configuration
- **APId.env or settings.env**  
  You can store environment variables (like Spotify Client ID, Last.fm API key, etc.) in .env files. The code uses python-dotenv to load them.
- **Spawn Catalog DB**  
  We store track metadata in spawn_catalog.db (SQLite) under Spawn/aux/glob/.
- **File Paths**  
  - Library path: Typically LIB_PATH/Spawn/Music. Set LIB_PATH in settings.env.
  - Logs & Aux: In Spawn/aux/temp, etc.
- **ReplayGain (optional)**  
  For album-level loudness balancing, install bs1770gain, and track_importer.py can call it.
- **Audio Feature-based Curation (optional future step)**  
  For advanced track ordering, librosa is used to extract numeric features (tempo, loudness, key, etc.) and store them in spawn_catalog.db. Then curator.py can read these features for advanced ordering logic. Future improvements or ML-based heuristics can further refine features like valence, danceability, speechiness, etc.

## License

Distributed under the GNU General Public License v3 (GPLv3). See the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to:
- Open an issue or pull request on GitHub: https://github.com/SpawnID0000/Spawn
- Submit bug reports, feature requests, or code improvements.

Please ensure that your contributions are consistent with the project license. For major changes, open an issue first to discuss.

Thanks for using Spawn! If you have any questions or issues, please contact us at spawn.id.0000@gmail.com.
