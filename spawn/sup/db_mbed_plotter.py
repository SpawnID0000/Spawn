#!/usr/bin/env python3

"""
db_mbed_plotter.py

Purpose:
    1. Load mp4tovec embeddings from 'Spawn/aux/glob/mp4tovec.p'
    2. Load track metadata from 'Spawn/aux/glob/spawn_catalog.db'
    3. Match each track's spawn_id to the embedding
    4. Run a dimensionality reduction (t-SNE) to project from 100D -> 2D (or 3D)
    5. Plot the resulting scatter using matplotlib with an input box to highlight specific points

Usage:
    python db_mbed_plotter.py --spawn-root /path/to/your/library
    python db_mbed_plotter.py --spawn-root /path/to/your/library --components 3

Requirements:
    pip install scikit-learn matplotlib fuzzywuzzy python-Levenshtein
"""

import os
import sys
import sqlite3
import pickle
import argparse
import numpy as np
import logging
import torch
import mplcursors
import random
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.widgets import TextBox
from matplotlib.collections import PathCollection
from sklearn.manifold import TSNE
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

current_m3u_indices = None

def load_embeddings(glob_dir: str):
    """
    Loads embeddings from mp4tovec.p found in the given 'glob_dir'
    Returns a dict { spawn_id: np.array([...]) } or {} if not found or on error.
    """
    print(f"Checking for embeddings file at {os.path.join(glob_dir, 'mp4tovec.p')}")
    emb_path = os.path.join(glob_dir, "mp4tovec.p")
    if not os.path.isfile(emb_path):
        logger.error(f"No mp4tovec.p found at '{emb_path}'. Cannot plot.")
        return {}

    try:
        with open(emb_path, "rb") as f:
            emb_dict = pickle.load(f)
        return emb_dict
    except Exception as e:
        logger.error(f"Failed to load embeddings from {emb_path}: {e}")
        return {}


def load_catalog_tracks(glob_dir: str):
    """
    Connects to spawn_catalog.db, returns a list of dicts, each containing:
        {
          'spawn_id': 'ABCD1234',
          '©ART': [...],
          '----:com.apple.iTunes:spawnre': [...],
          ...
        }
    or an empty list on error.
    """
    print(f"Checking for database file at {os.path.join(glob_dir, 'spawn_catalog.db')}")
    db_path = os.path.join(glob_dir, "spawn_catalog.db")
    if not os.path.isfile(db_path):
        logger.error(f"No spawn_catalog.db found at '{db_path}'. Cannot plot.")
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT spawn_id, tag_data FROM tracks")
        rows = cursor.fetchall()
        conn.close()

        results = []
        import json
        for (spawn_id, tag_json_str) in rows:
            try:
                tag_dict = json.loads(tag_json_str)
                tag_dict["spawn_id"] = spawn_id
                results.append(tag_dict)
            except Exception as e:
                logger.warning(f"Failed to parse JSON for spawn_id={spawn_id}: {e}")
        return results

    except sqlite3.Error as e:
        logger.error(f"SQLite error reading spawn_catalog.db: {e}")
        return []


def do_dimensionality_reduction(embedding_array, method='tsne', n_components=2):
    """
    Takes a NxD embedding_array, returns Nx2 or Nx3 matrix of projected points.
    """
    if method.lower() == 'tsne':
        n_samples = embedding_array.shape[0]
        
        # Ensure perplexity < n_samples. If n<3, perplexity=1 is the fallback.
        # Otherwise default to 30 or something else
        if n_samples <= 2:
            # If you only have 1 or 2 points, t-SNE is not going to do much.
            perplexity = 1
        else:
            # pick min(30, n_samples-1)
            perplexity = min(30, n_samples - 1)

        print(f"Using perplexity={perplexity} for t-SNE with {n_samples} samples.")

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components,
                    perplexity=perplexity,
                    random_state=42)
        transformed = tsne.fit_transform(embedding_array)
        return transformed
    else:
        raise NotImplementedError(f"Method '{method}' not implemented.")


def build_color_map(tracks):
    """
    Creates a color array for each track, keyed by the main spawnre genre (©gen).
    Returns a list of colors aligned with 'tracks'.
    """
    import random

    # Gather the main spawnre (©gen) for each track, defaulting to 'unknown'
    main_genres = []
    for t in tracks:
        genre_val = t.get("©gen", "unknown")
        if isinstance(genre_val, list):
            genre_val = genre_val[0] if genre_val else "unknown"
        if isinstance(genre_val, bytes):
            genre_val = genre_val.decode("utf-8", errors="replace")
        genre_val = str(genre_val).strip().lower()
        main_genres.append(genre_val)

    # Assign random colors to each unique main genre
    unique_genres = list(set(main_genres))
    color_map = {g: (random.random(), random.random(), random.random()) for g in unique_genres}

    # Map colors to tracks based on the main genre
    color_list = [color_map[g] for g in main_genres]
    return color_list


def normalize_name(name: str) -> str:
    """
    Normalizes artist names and search terms by stripping leading 'The' and converting to lowercase.
    """
    name = name.strip().lower()
    if name.startswith("the "):
        name = name[4:]  # Remove "The " from the start of the name
    return name


def find_spawn_id_for_file(file_path: str, valid_tracks):
    """
    Given a file path (from an M3U file), attempts to find the matching spawn_id from the valid_tracks.
    If the filename contains a pattern "[XXXXXXXX]" where X is an alphanumeric hex digit,
    that value is used as the spawn_id.
    Otherwise, it first checks for an exact match using the file basename (without extension).
    If no exact match is found, it falls back to fuzzy matching on the spawn_id field.
    Returns the matched spawn_id (string) or None if no good match is found.
    """
    base = os.path.basename(file_path)

    # Try to extract a spawn id from pattern "[XXXXXXXX]"
    hex_match = re.search(r'\[([0-9A-Fa-f]{8})\]', base)
    if hex_match:
        return hex_match.group(1)
    
    # No explicit spawn id in filename; use the basename (without extension)
    name, ext = os.path.splitext(base)
    # Try direct exact match
    for track in valid_tracks:
        if track.get("spawn_id", "").lower() == name.lower():
            return track.get("spawn_id", "")
    # Fuzzy matching fallback
    best_ratio = 0
    best_spawn_id = None
    for track in valid_tracks:
        spawn_id = track.get("spawn_id", "")
        ratio = fuzz.ratio(name.lower(), spawn_id.lower())
        if ratio > best_ratio:
            best_ratio = ratio
            best_spawn_id = spawn_id
    # Return match if ratio is high enough
    if best_ratio > 70:
        return best_spawn_id
    return None


def process_m3u_command(text, coords, valid_tracks, scatter, fig, ax, original_colors):
    """
    Processes the "@m3u" command from the search box.
    Expected syntax:
         @m3u [/path/to/playlist.m3u]
         or with an index: @m3u-<index> [/path/to/playlist.m3u]
    It loads the M3U file, takes 20 tracks starting at the given index (default is first track),
    maps the file names to spawn_ids (using find_spawn_id_for_file), highlights those points,
    and draws lines connecting them in playlist order. The connecting line segments transition
    through a custom color gradient.
    """

    # Pattern to extract an optional index and the file path (preserving original case for the file path)
    pattern = r'@m3u(?:-(\d+))?\s+(.+)'
    m = re.search(pattern, text)
    if not m:
        print("Invalid @m3u command format. Expected '@m3u [playlist_path]' or '@m3u-<index> [playlist_path]'.")
        return

    start_index_str = m.group(1)
    m3u_path = m.group(2).strip()
    start_index = int(start_index_str) - 1 if start_index_str else 0  # Convert to 0-based index

    if not os.path.isfile(m3u_path):
        print(f"M3U file not found at {m3u_path}")
        return

    with open(m3u_path, 'r') as f:
        lines = f.readlines()

    # Filter out comments (lines starting with "#") and empty lines
    playlist_files = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    if start_index < 0 or start_index >= len(playlist_files):
        print("Start index out of bounds in the M3U file.")
        return

    # Take up to 20 tracks from the playlist starting at start_index
    selected_files = playlist_files[start_index: start_index + 20]
    m3u_spawn_indices = []
    m3u_dir = os.path.dirname(m3u_path)

    for file_path in selected_files:
        # If the path is relative, convert it to an absolute path based on the M3U file's directory.
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(os.path.join(m3u_dir, file_path))
        spawn_id = find_spawn_id_for_file(file_path, valid_tracks)
        if spawn_id is None:
            print(f"Could not match spawn_id for file: {file_path}")
            continue
        # Find the index of the track with the matching spawn_id in valid_tracks
        idx = next((i for i, t in enumerate(valid_tracks) if t.get("spawn_id", "").lower() == spawn_id.lower()), None)
        if idx is not None:
            m3u_spawn_indices.append(idx)
        else:
            print(f"Spawn ID {spawn_id} not found in valid tracks for file: {file_path}")

    if not m3u_spawn_indices:
        print("No matching tracks found for the M3U playlist.")
        return

    global current_m3u_indices
    current_m3u_indices = m3u_spawn_indices

    # Update the scatter plot:
    # Grey out all data points first.
    new_colors = np.full((len(coords), 4), [0.9, 0.9, 0.9, 0.5])
    new_sizes = np.full(len(coords), 8)
    # Then, for the playlist points, restore their original colors and increase marker size.
    for idx in m3u_spawn_indices:
        new_colors[idx] = original_colors[idx]  # Retain the original color from @gen tag.
        new_sizes[idx] = 30                     # Increase marker size.
    scatter.set_facecolors(new_colors)
    scatter.set_sizes(new_sizes)

    # Remove any existing overlay scatter for M3U points (if any)
    for child in list(ax.collections):
        if hasattr(child, "m3u_overlay") and child.m3u_overlay:
            child.remove()
    
    # Create an overlay scatter for the M3U-connected points to bring them to the front.
    m3u_coords = coords[m3u_spawn_indices]
    if coords.shape[1] == 2:
        m3u_overlay = ax.scatter(m3u_coords[:, 0], m3u_coords[:, 1],
                                 c=[original_colors[i] for i in m3u_spawn_indices],
                                 s=20, zorder=3)
    elif coords.shape[1] == 3:
        m3u_overlay = ax.scatter(m3u_coords[:, 0], m3u_coords[:, 1], m3u_coords[:, 2],
                                 c=[original_colors[i] for i in m3u_spawn_indices],
                                 s=20, zorder=3)
    # Tag this overlay for future removal
    m3u_overlay.m3u_overlay = True

    # Remove existing playlist lines (if any)
    for line in list(ax.lines):
        if hasattr(line, "get_color") and line.get_color() is not None:
            line.remove()


    # Draw gradient lines connecting the points in playlist order.
    num_segments = len(m3u_coords) - 1
    if num_segments > 0:
        # Create a custom colormap transitioning through multiple colors.
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_gradient",
            ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
        )
        for i in range(num_segments):
            # Compute fraction so that the first segment is red and the last is violet.
            frac = i / (num_segments - 1) if num_segments > 1 else 0.5
            color = cmap(frac)
            x_vals = [m3u_coords[i, 0], m3u_coords[i+1, 0]]
            y_vals = [m3u_coords[i, 1], m3u_coords[i+1, 1]]
            if coords.shape[1] == 2:
                ax.plot(x_vals, y_vals, color=color, lw=2)
            elif coords.shape[1] == 3:
                z_vals = [m3u_coords[i, 2], m3u_coords[i+1, 2]]
                ax.plot(x_vals, y_vals, z_vals, color=color, lw=2)

    plt.draw()
    plt.pause(0.001)

def on_search_submit(text, coords, valid_tracks, scatter, fig, ax, original_colors):
    """
    Highlights points that match the entered search term using case-insensitive and fuzzy matching.
    Supports OR (",") and AND ("+") search logic.
    Allows filtering by specific metadata fields using "@gen" syntax.
    Also supports an "@m3u" command to load a playlist and draw connecting lines.
    """
    raw_text = text.strip()  # Preserve original case for file paths
    text_lower = raw_text.lower()  # Use this for command detection and search matching

    text = text.strip().lower()

    # Check for the special "@m3u" command first.
    if '@m3u' in text_lower:
        process_m3u_command(raw_text, coords, valid_tracks, scatter, fig, ax, original_colors)
        return

    if text_lower.startswith("the "):
        text_lower = text_lower[4:]  # Remove leading "the "

    # If search is blank, reset colors and sizes
    if not text_lower:
        scatter.set_facecolors(original_colors)  # Reset colors
        scatter.set_sizes([8] * len(coords))  # Reset point sizes
        scatter.set_zorder(1)  # Reset z-order to default

        # Remove any additional scatter plots (highlighted points)
        for child in ax.get_children():
            if isinstance(child, PathCollection) and child != scatter:
                child.remove()

        plt.draw()
        plt.pause(0.001)
        return

    # Determine if search is filtered by @gen or a general search
    filter_gen_only = "@gen" in text
    if filter_gen_only:
        text = text.replace("@gen", "").strip()

    # Determine the search mode: OR (",") or AND ("+")
    if "+" in text:
        search_terms = [term.strip() for term in text.split("+")]
        search_mode = "AND"
    else:
        search_terms = [term.strip() for term in text.split(",")]
        search_mode = "OR"

    matching_indices = []
    for idx, track_info in enumerate(valid_tracks):
        # Gather searchable metadata based on filter mode
        searchable_fields = []

        if filter_gen_only:
            # Only include the main genre (©gen)
            main_genre = track_info.get("©gen", "unknown")
            if isinstance(main_genre, list):
                main_genre = main_genre[0] if main_genre else "unknown"
            if isinstance(main_genre, bytes):
                main_genre = main_genre.decode("utf-8", errors="replace")
            searchable_fields.append(str(main_genre).lower())

        else:
            # Include all searchable fields for general searches
            searchable_fields.append(track_info.get("spawn_id", "").lower())

            # Main genre (©gen)
            main_genre = track_info.get("©gen", "unknown")
            if isinstance(main_genre, list):
                main_genre = main_genre[0] if main_genre else "unknown"
            if isinstance(main_genre, bytes):
                main_genre = main_genre.decode("utf-8", errors="replace")
            searchable_fields.append(str(main_genre).lower())

            # Spawnre tags (----:com.apple.iTunes:spawnre)
            spawnre_val = track_info.get("----:com.apple.iTunes:spawnre", "unknown")
            if isinstance(spawnre_val, list):
                spawnre_val = " ".join([str(item).lower() for item in spawnre_val])
            elif isinstance(spawnre_val, bytes):
                spawnre_val = spawnre_val.decode("utf-8", errors="replace").lower()
            else:
                spawnre_val = str(spawnre_val).lower()
            searchable_fields.append(spawnre_val)

            # Spawnre_hex value
            spawnre_hex = track_info.get("----:com.apple.iTunes:spawnre_hex", "unknown")
            if isinstance(spawnre_hex, list):
                spawnre_hex = " ".join([str(item).lower() for item in spawnre_hex])
            elif isinstance(spawnre_hex, bytes):
                spawnre_hex = spawnre_hex.decode("utf-8", errors="replace").lower()
            else:
                spawnre_hex = str(spawnre_hex).lower()
            searchable_fields.append(spawnre_hex)

            # Artist name (©ART)
            artist_data = track_info.get("©ART", ["unknown"])
            if isinstance(artist_data, list):
                artist_name = artist_data[0] if artist_data else "unknown"
            elif isinstance(artist_data, bytes):
                artist_name = artist_data.decode("utf-8", errors="replace").lower()
            else:
                artist_name = str(artist_data).lower()
            searchable_fields.append(normalize_name(artist_name))

        # Check for matches based on the search mode
        match_found = False
        if search_mode == "OR":
            # OR logic: match if any search term is in any searchable field
            for term in search_terms:
                if any(term in field for field in searchable_fields):
                    match_found = True
                    break
        elif search_mode == "AND":
            # AND logic: match only if all search terms are present in the fields
            match_found = all(
                any(term in field for field in searchable_fields) for term in search_terms
            )

        if match_found:
            matching_indices.append(idx)

    # Update colors and sizes for the main scatter plot (greyed out non-matching points)
    new_colors = np.full((len(coords), 4), [0.9, 0.9, 0.9, 0.5])  # Grey for all
    new_sizes = np.full(len(coords), 8)  # Small size for all

    # Apply the changes to the main scatter plot
    scatter.set_facecolors(new_colors)
    scatter.set_sizes(new_sizes)
    scatter.set_zorder(1)  # Keep non-matching points at a lower z-order

    # Now handle the highlighted points separately
    highlighted_points = np.array([coords[i] for i in matching_indices])
    highlight_colors = np.array([[0, 0, 0, 1]] * len(matching_indices))  # Black color
    highlight_sizes = np.full(len(matching_indices), 20)  # Larger size

    # Remove any existing highlighted scatter plot
    for child in ax.get_children():
        if isinstance(child, PathCollection) and child != scatter:
            child.remove()

    # Add a new scatter plot for highlighted points with higher z-order
    if len(highlighted_points) > 0:
        ax.scatter(highlighted_points[:, 0], highlighted_points[:, 1],
                   c=highlight_colors, s=highlight_sizes, alpha=1, zorder=3)

    plt.draw()
    plt.pause(0.001)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a 2D/3D visualization of track embeddings from spawn_catalog.db + mp4tovec.p"
    )
    parser.add_argument("--spawn-root",
                        help="Path to the Spawn project root (containing aux/glob).",
                        required=True)
    parser.add_argument("--method",
                        help="Dimensionality reduction method (tsne, umap, etc.)",
                        default="tsne")
    parser.add_argument("--components",
                        help="Number of output dimensions (2 or 3).",
                        type=int, default=2)
    parser.add_argument("-m3u", "--m3u", action="store_true",
                        help="Prompt for M3U playlist input")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Build the path to "aux/glob"
    spawn_root = os.path.abspath(args.spawn_root)
    glob_dir = os.path.join(spawn_root, "aux", "glob")

    # 1) Load embeddings
    print("Step 1: Loading embeddings...")
    emb_dict = load_embeddings(glob_dir)
    if not emb_dict:
        logger.error("No embeddings loaded. Exiting.")
        sys.exit(1)

    # 2) Load catalog tracks
    print("Step 2: Loading catalog tracks...")
    catalog_tracks = load_catalog_tracks(glob_dir)
    if not catalog_tracks:
        logger.error("No tracks found in spawn_catalog.db. Exiting.")
        sys.exit(1)

    # 3) Match each track's spawn_id with an embedding in emb_dict
    print("Step 3: Matching embeddings to catalog tracks...")
    valid_tracks = []
    embed_array = []

    for t in catalog_tracks:
        sid = t.get("spawn_id")
        embedding = emb_dict.get(sid, None)
        if embedding is None:
            continue    # No embedding or mismatch

        # 1) If it's a torch.Tensor, convert to NumPy
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()  # now a NumPy array

        # 2) If it's a list, convert to NumPy
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)

        # 3) Possibly squeeze from (1, D) -> (D,) if needed
        if isinstance(embedding, np.ndarray):
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

            # Accept any 1D array with length > 0
            if embedding.ndim == 1 and embedding.size > 0:
                valid_tracks.append(t)
                embed_array.append(embedding)
            else:
                logger.warning(f"Skipping spawn_id={sid} => shape={embedding.shape} not 1D or empty.")
        else:
            logger.warning(f"Skipping spawn_id={sid} => not a NumPy array.")

    if not valid_tracks:
        logger.error("No valid embeddings match the DB tracks. Exiting.")
        sys.exit(1)

    # 4) Convert to NumPy for dimensionality reduction
    print(f"Step 4: Found {len(valid_tracks)} valid tracks.")
    embed_array = np.array(embed_array)  # shape (N, 100) typically

    logger.info(f"Running dimensionality reduction on {len(valid_tracks)} tracks...")
    coords = do_dimensionality_reduction(
        embedding_array=embed_array,
        method=args.method,
        n_components=args.components
    )

    # coords => shape (N, 2) or (N, 3)

    # 5) Assign colors
    color_list = build_color_map(valid_tracks)

    # 6) Plot
    if args.components == 2:
        fig, ax = plt.subplots(figsize=(8,6))

        # Set the cursor to a crosshair
        plt.get_current_fig_manager().canvas.set_cursor(3)

        # Maximize the window
        try:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        except AttributeError:
            try:
                figManager.window.state('zoomed')
            except Exception:
                pass

        # Create the 2D scatter plot
        sc = ax.scatter(coords[:,0], coords[:,1], c=color_list, s=8, alpha=0.8)
        original_colors = sc.get_facecolors()
        
        ax.set_title("Track Embeddings (2D)")
        #ax.set_xlabel("Dimension 1")
        #ax.set_ylabel("Dimension 2")

        # TextBox widget for search functionality
        text_box_ax = plt.axes([0.2, 0.01, 0.6, 0.05])
        text_box = TextBox(text_box_ax, 'Search: ', initial="")
        text_box.on_submit(lambda text: on_search_submit(text, coords, valid_tracks, sc, fig, ax, original_colors))

        # Ensure the text box is focused when clicked, enabling keyboard shortcuts
        def focus_text_box(event):
            if text_box_ax == event.inaxes:
                text_box.set_active(True)

        fig.canvas.mpl_connect('button_press_event', focus_text_box)

        # Ensure the text box is focused initially so keyboard shortcuts work
        text_box_ax.set_navigate(False)  # Prevents the plot from interpreting keyboard events
        text_box_ax.figure.canvas.mpl_connect('button_press_event', lambda event: text_box.begin_typing())

        # If the CLI flag is used, repeatedly prompt for M3U input
        if args.m3u:
            while True:
                user_input = input("Please enter path to M3U playlist (and optionally also starting track location): ")
                if user_input.strip() == "":
                    # If input is empty, remove M3U overlays/lines and reset the plot.
                    sc.set_facecolors(original_colors)
                    sc.set_sizes([8] * len(coords))
                    # Remove overlay scatter objects marked as m3u_overlay
                    for child in list(ax.collections):
                        if hasattr(child, "m3u_overlay") and child.m3u_overlay:
                            child.remove()
                    # Remove any lines (playlist connecting lines)
                    for line in list(ax.lines):
                        line.remove()
                    plt.draw()
                    plt.pause(0.001)
                    print("M3U overlay removed.")
                    break
                else:
                    # Parse the input using shlex to support paths with spaces.
                    import shlex
                    tokens = shlex.split(user_input)
                    if not tokens:
                        print("No input provided. Please try again.")
                        continue
                    playlist_path = tokens[0]
                    start_track = tokens[1] if len(tokens) > 1 else "1"
                    # Build command string expected by process_m3u_command:
                    if start_track != "1":
                        cmd_str = f"@m3u-{start_track} {playlist_path}"
                    else:
                        cmd_str = f"@m3u {playlist_path}"
                    process_m3u_command(cmd_str, coords, valid_tracks, sc, fig, ax, original_colors)

        # Enable mplcursors for hover tooltips in 2D
        cursor2 = mplcursors.cursor(sc, hover=True)
        active_annotations = []

        @cursor2.connect("add")
        def on_add(sel):
            global current_m3u_indices
            # If in M3U mode and the hovered index is not in the M3U indices, skip showing a tooltip
            if current_m3u_indices is not None and sel.index not in current_m3u_indices:
                sel.annotation.set_visible(False)
                return
            track_info = valid_tracks[sel.index]
            
            # Get artist name
            artist_data = track_info.get("©ART", ["Unknown"])
            artist_name = artist_data[0] if isinstance(artist_data, list) and artist_data else str(artist_data)
            
            # Get spawn ID
            spawn_id_str = track_info.get("spawn_id", "No ID")
            
            # Get main genre (©gen)
            main_genre = track_info.get("©gen", "unknown")
            if isinstance(main_genre, list):
                main_genre = main_genre[0] if main_genre else "unknown"
            if isinstance(main_genre, bytes):
                main_genre = main_genre.decode("utf-8", errors="replace")
            main_genre = str(main_genre).strip().lower()

            # Get all associated spawnre tags
            spawnre_val = track_info.get("----:com.apple.iTunes:spawnre", "unknown")
            if isinstance(spawnre_val, list):
                spawnre_val = " ".join([str(item).lower() for item in spawnre_val])
            elif isinstance(spawnre_val, bytes):
                spawnre_val = spawnre_val.decode("utf-8", errors="replace").lower()
            else:
                spawnre_val = str(spawnre_val).lower()

            # Get spawnre_hex value
            spawnre_hex = track_info.get("----:com.apple.iTunes:spawnre_hex", "No spawnre_hex")
            if isinstance(spawnre_hex, list):
                spawnre_hex = spawnre_hex[0] if spawnre_hex else "No spawnre_hex"
            elif isinstance(spawnre_hex, bytes):
                spawnre_hex = spawnre_hex.decode("utf-8", errors="replace")
            spawnre_hex = str(spawnre_hex).strip().lower()

            # Update the tooltip text
            sel.annotation.set_text(f"{artist_name}\nSpawn ID: {spawn_id_str}\nspawnre_hex: {spawnre_hex}\ngenre: {main_genre}")
            #sel.annotation.set_text(f"{artist_name}\nSpawn ID: {spawn_id_str}\nspawnre_hex: {spawnre_hex}\nmain genre: {main_genre}\nall genres: {spawnre_val}")
            active_annotations.append(sel.annotation)


        # Function to clear all annotations on right-click or ESC
        def clear_annotations(event):
            if (hasattr(event, "button") and event.button == 3) or (getattr(event, "key", None) == "escape"):
                for ann in active_annotations:
                    ann.set_visible(False)
                active_annotations.clear()
                plt.draw()
                plt.pause(0.001)

        # Attach event listeners to the figure
        fig.canvas.mpl_connect("button_press_event", clear_annotations)
        fig.canvas.mpl_connect("key_press_event", clear_annotations)

    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        # Create the 3D scatter plot with picking enabled
        sc3 = ax.scatter(
            coords[:,0], coords[:,1], coords[:,2],
            c=color_list, s=8, alpha=0.8, picker=True
        )
        ax.set_title("Track Embeddings (3D)")
        #ax.set_xlabel("Dim 1")
        #ax.set_ylabel("Dim 2")
        #ax.set_zlabel("Dim 3")

        # Set fixed limits so the view doesn't change during panning
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # Hide the default 3D panes and grid
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.grid(False)
        # Remove the pane edges explicitly
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        # Also disable the default 3D box (the outside border)
        if hasattr(ax, '_axinfo') and "box" in ax._axinfo:
            ax._axinfo["box"]["visible"] = False
            ax._axinfo["box"]["edgecolor"] = (1,1,1,0)  # transparent edge
            ax._axinfo["box"]["linewidth"] = 0

        # Additionally, disable the 3D axes entirely
        ax._axis3don = False

        # Remove tick marks and labels (the numerical values)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Compute the center of the data for fixed axes lines
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        z_center = (z_min + z_max) / 2.0

        # Draw custom fixed axes lines (black lines intersecting at the center)
        ax.plot([x_min, x_max], [y_center, y_center], [z_center, z_center], color='black', lw=1)
        ax.plot([x_center, x_center], [y_min, y_max], [z_center, z_center], color='black', lw=1)
        ax.plot([x_center, x_center], [y_center, y_center], [z_min, z_max], color='black', lw=1)

        # Enable mplcursors for interactive tooltips in 3D
        cursor3 = mplcursors.cursor(sc3, hover=False)    # Use click to show tooltips

        # List to keep track of active annotations
        active_annotations = []

        @cursor3.connect("add")
        def on_add_3d(sel):
            global current_m3u_indices
            # If in M3U mode and the hovered index is not in the M3U indices, skip showing a tooltip
            if current_m3u_indices is not None and sel.index not in current_m3u_indices:
                sel.annotation.set_visible(False)
                return
            track_info = valid_tracks[sel.index]

            # Get artist name
            artist_data = track_info.get("©ART", ["Unknown"])
            artist_name = artist_data[0] if isinstance(artist_data, list) and artist_data else str(artist_data)
            
            # Get spawn ID
            spawn_id_str = track_info.get("spawn_id", "No ID")
            
            # Get main genre (©gen)
            main_genre = track_info.get("©gen", "unknown")
            if isinstance(main_genre, list):
                main_genre = main_genre[0] if main_genre else "unknown"
            if isinstance(main_genre, bytes):
                main_genre = main_genre.decode("utf-8", errors="replace")
            main_genre = str(main_genre).strip().lower()

            # Get all associated spawnre tags
            spawnre_val = track_info.get("----:com.apple.iTunes:spawnre", "unknown")
            if isinstance(spawnre_val, list):
                spawnre_val = " ".join([str(item).lower() for item in spawnre_val])
            elif isinstance(spawnre_val, bytes):
                spawnre_val = spawnre_val.decode("utf-8", errors="replace").lower()
            else:
                spawnre_val = str(spawnre_val).lower()

            # Get spawnre_hex value
            spawnre_hex = track_info.get("----:com.apple.iTunes:spawnre_hex", "No spawnre_hex")
            if isinstance(spawnre_hex, list):
                spawnre_hex = spawnre_hex[0] if spawnre_hex else "No spawnre_hex"
            elif isinstance(spawnre_hex, bytes):
                spawnre_hex = spawnre_hex.decode("utf-8", errors="replace")
            spawnre_hex = str(spawnre_hex).strip().lower()

            # Update the tooltip text
            annotation = sel.annotation
            annotation.set_text(f"{artist_name}\nSpawn ID: {spawn_id_str}\nspawnre_hex: {spawnre_hex}\ngenre: {main_genre}")
            #annotation.set_text(f"{artist_name}\nSpawn ID: {spawn_id_str}\nspawnre_hex: {spawnre_hex}\nmain genre: {main_genre}\nall genres: {spawnre_val}")
            active_annotations.append(annotation)

        # Function to clear all annotations on right-click or ESC
        def clear_annotations(event):
            if (hasattr(event, "button") and event.button == 3) or (getattr(event, "key", None) == "escape"):
                for ann in active_annotations:
                    ann.set_visible(False)
                active_annotations.clear()
                plt.draw()
                plt.pause(0.001)

        # Attach event listeners to the figure
        fig.canvas.mpl_connect("button_press_event", clear_annotations)
        fig.canvas.mpl_connect("key_press_event", clear_annotations)

    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

if __name__ == "__main__":
    main()
