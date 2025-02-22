#!/usr/bin/env python3

"""
db_mbed_plotter.py

Purpose:
    1. Load mp4tovec embeddings from 'Spawn/aux/glob/mp4tovec.p'
    2. Load track metadata from 'Spawn/aux/glob/spawn_catalog.db'
    3. Match each track's spawn_id to the embedding
    4. Run a dimensionality reduction (t-SNE) to project from 100D -> 2D (or 3D)
    5. Plot the resulting scatter using matplotlib

Usage:
    python db_mbed_plotter.py --spawn-root /path/to/your/library
    python db_mbed_plotter.py --spawn-root /path/to/your/library --components 3

Requirements:
    pip install scikit-learn matplotlib
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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def load_embeddings(glob_dir: str):
    """
    Loads embeddings from mp4tovec.p found in the given 'glob_dir'
    Returns a dict { spawn_id: np.array([...]) } or {} if not found or on error.
    """
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
    Creates a color array for each track, keyed by "spawnre" if present.

    Returns:
      list_of_colors aligned with 'tracks'.
    """
    import random

    # Gather each track's spawnre, if available, else 'unknown'
    spawnres = []
    for t in tracks:
        spawnre_val = t.get("----:com.apple.iTunes:spawnre")
        if spawnre_val:
            # might be list or bytes
            if isinstance(spawnre_val, list):
                spawnre_val = spawnre_val[0] if spawnre_val else "unknown"
            if isinstance(spawnre_val, bytes):
                spawnre_val = spawnre_val.decode("utf-8", errors="replace")
            spawnre_val = str(spawnre_val).strip().lower()
        else:
            spawnre_val = "unknown"
        spawnres.append(spawnre_val)

    # Assign random colors to each unique spawnre
    unique_spawnres = list(set(spawnres))
    color_map = {}
    for g in unique_spawnres:
        color_map[g] = (random.random(), random.random(), random.random())

    color_list = [color_map[g] for g in spawnres]
    return color_list


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
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Build the path to "aux/glob"
    spawn_root = os.path.abspath(args.spawn_root)
    glob_dir = os.path.join(spawn_root, "aux", "glob")

    # 1) Load embeddings
    emb_dict = load_embeddings(glob_dir)
    if not emb_dict:
        logger.error("No embeddings loaded. Exiting.")
        sys.exit(1)

    # 2) Load catalog tracks
    catalog_tracks = load_catalog_tracks(glob_dir)
    if not catalog_tracks:
        logger.error("No tracks found in spawn_catalog.db. Exiting.")
        sys.exit(1)

    # 3) Match each track's spawn_id with an embedding in emb_dict
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
        
        ax.set_title("Track Embeddings (2D)")
        #ax.set_xlabel("Dimension 1")
        #ax.set_ylabel("Dimension 2")

        # Enable mplcursors for hover tooltips in 2D
        cursor2 = mplcursors.cursor(sc, hover=True)
        active_annotations = []

        @cursor2.connect("add")
        def on_add(sel):
            idx = sel.index
            track_info = valid_tracks[idx]
            # Get artist name
            artist_data = track_info.get("©ART", ["Unknown"])
            artist_name = artist_data[0] if isinstance(artist_data, list) and artist_data else str(artist_data)
            # Get spawn_id
            spawn_id_str = track_info.get("spawn_id", "No ID")
            # Get spawnre value
            spawnre_val = track_info.get("----:com.apple.iTunes:spawnre")
            if spawnre_val:
                if isinstance(spawnre_val, list):
                    spawnre_val = spawnre_val[0] if spawnre_val else "unknown"
                if isinstance(spawnre_val, bytes):
                    spawnre_val = spawnre_val.decode("utf-8", errors="replace")
                spawnre_val = str(spawnre_val).strip().lower()
            else:
                spawnre_val = "unknown"
            sel.annotation.set_text(f"{artist_name}\nspawn_id={spawn_id_str}\nspawnre={spawnre_val}")
            active_annotations.append(sel.annotation)

        # Function to clear all annotations on right-click or ESC
        def clear_annotations(event):
            if (hasattr(event, "button") and event.button == 3) or (getattr(event, "key", None) == "escape"):
                for ann in active_annotations:
                    ann.set_visible(False)
                active_annotations.clear()
                plt.draw()

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
            idx = sel.index
            track_info = valid_tracks[idx]

            artist_data = track_info.get("©ART", ["Unknown"])
            artist_name = artist_data[0] if isinstance(artist_data, list) and artist_data else str(artist_data)
            spawn_id_str = track_info.get("spawn_id", "No ID")

            spawnre_val = track_info.get("----:com.apple.iTunes:spawnre")
            if spawnre_val:
                if isinstance(spawnre_val, list):
                    spawnre_val = spawnre_val[0] if spawnre_val else "unknown"
                if isinstance(spawnre_val, bytes):
                    spawnre_val = spawnre_val.decode("utf-8", errors="replace")
                spawnre_val = str(spawnre_val).strip().lower()
            else:
                spawnre_val = "unknown"

            annotation = sel.annotation
            annotation.set_text(f"{artist_name}\nspawn_id={spawn_id_str}\nspawnre={spawnre_val}")
            
            # Store annotation reference for later removal
            active_annotations.append(annotation)

        # Function to clear all annotations on right-click or ESC
        def clear_annotations(event):
            if (hasattr(event, "button") and event.button == 3) or (getattr(event, "key", None) == "escape"):
                for ann in active_annotations:
                    ann.set_visible(False)
                active_annotations.clear()
                plt.draw()

        # Attach event listeners to the figure
        fig.canvas.mpl_connect("button_press_event", clear_annotations)
        fig.canvas.mpl_connect("key_press_event", clear_annotations)

    plt.tight_layout()
    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()


if __name__ == "__main__":
    main()