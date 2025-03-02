# Spawn

**Spawn** is a Python‑based audio manager that imports music tracks, standardizes and enriches metadata (tags, MBIDs, ReplayGain, etc.), and provides a flexible system for favorites, curation, and playback. It also automatically creates symlinks for all imported tracks to enable universal playlist functionality.

## Table of Contents

- [Spawn](#spawn)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [System Requirements](#system-requirements)
    - [Python Dependencies](#python-dependencies)
    - [Before Running Spawn](#before-running-spawn)
      - [If you have Python \>= 3.13](#if-you-have-python--313)
  - [Library Directory Structure](#library-directory-structure)
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
  Import audio files (AAC, ALAC, FLAC, MP4) and convert them to a standardized M4A format. During import, Spawn:
  - Cleans and repackages audio using ffmpeg, MP4Box, and bs1770gain (for ReplayGain).
  - Automatically extracts, canonicalizes, and enriches metadata including MusicBrainz IDs, AcoustID fingerprints, and Spotify track IDs.
  - Computes album‑level ReplayGain and assigns a unique 8‑digit hexadecimal Spawn ID to each track.
  - Generates advanced track embeddings (using a Deej‑AI‑inspired encoder) for improved playlist curation.

- **Favorites System:**  
  Manage and update your favorite artists, albums, or tracks. Favorites are stored in JSON files and can be used to filter curated playlists.

- **Curated Playlists:**  
  Spawn supports several playlist curation methods:
  - **Basic Curation:** Groups tracks by genre (using a comprehensive genre dictionary with subgenre and synonym support), then shuffles tracks within each cluster.
  - **Feature‑Based Curation:** Uses numeric audio features (e.g. tempo, loudness) to reorder tracks in each genre cluster.
  - **Advanced Curation:** Leverages track embeddings and a trained ML model to implement a “chain‑based” nearest‑neighbor approach for smooth transitions between tracks, based on the Deej-AI project (https://github.com/teticio/Deej-AI)
  - **Personalized Recommendations:** After filtering by favorites, users can opt to generate a playlist of tracks with a high `like_likelihood` score, selecting tracks similar to their favorites.

- **M3U Playback (via mpv):**  
  Easily play curated playlists with mpv. Spawn scans for `.m3u` files and launches mpv with the selected playlist.

- **Symlink Generation:**  
  During import, Spawn automatically creates symbolic links for every track with a unique Spawn ID. These symlinks are stored directly in the `Spawn/aux/user/linx/` folder (named `<SpawnID>.m4a`) to enable consistent file paths for universal playlists.

- **Embeddings & Metadata Storage:**  
  - Track embeddings are stored in `mp4tovec.p`, a pickle file located in `Spawn/aux/glob/`, and updated whenever new tracks are added.
  - Track metadata is stored in `spawn_catalog.db` (for all cataloged tracks) and `spawn_library.db` (for user-specific tracks).

- **API Integrations:**  
  Spawn integrates with external services such as Spotify, Last.fm, MusicBrainz, and AcoustID for enhanced metadata and audio fingerprinting.

- **Admin vs. User Mode:**  
  - **Admin Mode:** New or updated tracks are written to `spawn_catalog.db`, embeddings are generated, and all metadata is updated.
  - **User Mode:** Users access a subset of the catalog via `spawn_library.db`, where `cat_tracks` holds catalog tracks and `lib_tracks` holds unique user tracks.

---

## Installation

### System Requirements

- **Python 3.6 or higher** (tested with Python 3.10).
- **ReplayGain:**  
  - macOS: `brew install rsgain`
  - Ubuntu/Debian: `sudo apt install rsgain`
  - Windows: see [rsgain on GitHub](https://github.com/complexlogic/rsgain)
- **mpv** (for playlist playback):  
  - macOS: `brew install mpv`
  - Ubuntu/Debian: `sudo apt-get install mpv`
  - Windows: see [mpv installation guide](https://mpv.io/installation/)
- **Optional (for audio conversion and fingerprinting):**  
  - ffmpeg  
  - MP4Box  
  - chromaprint/fpcalc

### Python Dependencies

Install via **pip** from the repository or after cloning:

```bash
git clone https://github.com/SpawnID0000/Spawn.git
cd Spawn
pip install -r requirements.txt
```

Or, if using setup.py:

```bash
git clone https://github.com/SpawnID0000/Spawn.git
cd Spawn
python setup.py install
```

Dependencies include:
- mutagen
- spotipy
- musicbrainzngs
- requests, Pillow, python-dotenv, python-mpv
- numpy, torch
- …and others listed in requirements.txt.

---

### Before Running Spawn

- Make sure to have `git lfs` in your system. [Installation Steps](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
- Run `git lfs install` if you have not already.
- Run `git lfs pull`.
- Now the `diffusion_pytorch_model.safetensors` should be the actual file instead of the Git LFS pointer file.

#### If you have Python >= 3.13

- In Python 3.13, some modules like `aifc` and `sunau` have been removed. These modules are used in generating the embeddings. As a workaround, please run this:

```bash
pip install standard-aifc standard-sunau
```

## Library Directory Structure

When you import tracks, Spawn organizes your library under your LIB_PATH/Spawn directory. An example structure is as follows:

```
LIB_PATH/Spawn
├── Music
│   └── Artist
│       └── Album
│           ├── D-TT [spawn_id] - Title.m4a
│           └── cover.jpg
├── Playlists
│   ├── Curated
│   │   └── curate_YYYY-MM-DD_HHMMSS.m3u
│   └── Imported
│       └── import_YYYY-MM-DD_HHMM.m3u
└── aux
    ├── glob
    │   ├── mp4tovec.p
    │   └── spawn_catalog.db
    ├── temp
    │   └── import_log.txt
    └── user
        ├── favs
        │   ├── fav_albums.json
        │   ├── fav_artists.json
        │   └── fav_tracks.json
        ├── hart
        │   └── Artist
        │       └── Album
        │           └── cover_hi-res.jpg
        ├── linx
        │   └── spawn_id.m4a
        └── spawn_library.db
```

---

## Usage

After installation, you can run Spawn from the command line:

```bash
spawn
```

Or from source:

```bash
python -m spawn.main
```

You will see a text menu:

```
Select one of the following options:
   (or type 'quit' to exit)

    1) Import tracks
    2) Update favorites
    3) Create curated playlist
    4) Play M3U playlist
```

### 1. Import Tracks
- **Purpose:**  
  Repackage raw audio, standardize metadata (including MBIDs, ReplayGain, and embeddings), and assign a unique Spawn ID to each track.
- **Details:**  
  The import process supports both Admin and User modes. In Admin Mode, new tracks are stored in the main catalog database; in User Mode, they are stored in the user library database.
- **Usage:**  
  Select “1” from the main menu and follow the prompts.

### 2. Update Favorites
- **Purpose:**  
  Manage your favorite artists, albums, and tracks.
- **Usage:**  
  Select “2” and follow the prompts to update your JSON‑formatted favorites lists, which are stored under `Spawn/aux/user/favs/`.

### 3. Create Curated Playlist
- **Purpose:**  
  Generate an `.m3u` playlist from your library. You can choose between basic, feature‑based, or advanced (Deej‑AI) curation.
- **Usage:**  
  Select “3” from the main menu. You may filter by favorites and then choose between advanced or feature‑based ordering.
- **Note:**  
  In User Mode, only tracks stored in the user library database (those with Spawn IDs in the `cat_tracks` table) are included.

### 4. Play M3U Playlist
- **Purpose:**  
  Launch mpv to play a selected `.m3u` playlist.
- **Usage:**  
  Select “4” and choose a playlist from the displayed list or enter a custom path.


## Modules Overview

- **main.py**  
  The central CLI menu that integrates all functionalities.
- **track_importer.py**  
  Handles audio file processing, metadata standardization, MBID and ReplayGain lookups, embedding generation, file renaming, and symlink creation.
- **favs.py**  
  Manages your JSON‑formatted favorites lists (fav_artists.json, fav_albums.json, fav_tracks.json).
- **curator.py**  
  Provides multiple curation methods (basic, feature‑based, advanced) for grouping tracks by genre, ordering clusters via relationships or embeddings, and writing curated M3U playlists.
- **audiodiffusion/**  
  Contains Deej-AI–inspired modules for embedding generation.
- **likey.py**  
  Computes `like_likelihood` scores based on embeddings to recommend new tracks similar to the user’s favorites.
- **symlinker.py**  
  Ensures every track with a unique Spawn ID has a symlink for universal playlist support.
- **player.py**  
  Finds `.m3u` files and plays them with mpv.

## Additional Configuration

- **Environment Files:**  
  Use `APId.env` or `settings.env` to set API keys (Spotify, Last.fm, etc.) and other configuration values (such as LIB_PATH).
- **Database:**  
  Track metadata is stored in SQLite databases:
  - Admin mode uses `Spawn/aux/glob/spawn_catalog.db` (table “tracks”).
  - User mode uses `Spawn/aux/user/spawn_library.db` (table “cat_tracks”).
- **Auxiliary Directories:**  
  - Library path: Typically LIB_PATH/Spawn/Music.
  - Logs and temporary files are saved under `Spawn/aux/temp/`.
  - Custom cluster orders for curation are stored in `Spawn/aux/user/cur8/`.
- **ReplayGain (optional):**  
  For album-level loudness balancing, install bs1770gain.
- **Audio Feature-based Curation (optional):**  
  For advanced track ordering, librosa and/or a Deej‑AI model are used.

## License

Distributed under the GNU General Public License v3 (GPLv3). See the LICENSE file for details.

## Contributing

Contributions are welcome! To contribute:
- Open an issue or pull request on GitHub: [https://github.com/SpawnID0000/Spawn](https://github.com/SpawnID0000/Spawn)
- Submit bug reports, feature requests, or code improvements.
- For major changes, please open an issue first to discuss your ideas.

Thanks for using Spawn! If you have any questions or issues, please contact us at spawn.id.0000@gmail.com.