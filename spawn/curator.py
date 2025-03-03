# curator.py

import os
import sys
import sqlite3
import json
import logging
import random
import re
import math
import time
#import librosa
import pickle
import numpy as np
import torch
#import tensorflow as tf

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional
#from audiodiffusion.audio_encoder import AudioEncoder
from .audiodiffusion.audio_encoder import AudioEncoder
#from tqdm import tqdm
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import TFSMLayer
#from .MP4ToVec import load_mp4tovec_model_tf
#from .MP4ToVec import load_mp4tovec_model_torch
from .MP4ToVec import load_mp4tovec_model_diffusion, generate_embedding

from .dic_spawnre import genre_mapping, subgenre_to_parent, genre_synonyms

logger = logging.getLogger(__name__)


def decode_spawnre_hex(spawnre_hex: str) -> str:
    """
    Given a spawnre_hex value like 'x011500', parse it into pairs ('01','15','00'),
    look each up in genre_mapping to get the Genre name(s), and return a comma-
    separated string like 'classic rock, blues rock, rock'.
    """
    # Remove leading "x" if present
    if spawnre_hex.startswith("x"):
        spawnre_hex = spawnre_hex[1:]  # e.g. "011500"

    # Split into pairs of 2 hex chars
    pairs = [spawnre_hex[i:i+2] for i in range(0, len(spawnre_hex), 2)]
    # Example: "011500" => ["01", "15", "00"]

    decoded_genres = []
    for pair in pairs:
        # Turn "01" into "0x01"
        hex_val = f"0x{pair.lower()}"
        
        # Search genre_mapping for an entry where info["Hex"] == hex_val
        found_genre = None
        for code_key, info in genre_mapping.items():
            if info.get("Hex", "").lower() == hex_val:
                found_genre = info.get("Genre", "Unknown")
                break
        
        if found_genre:
            decoded_genres.append(found_genre)
        else:
            # If nothing matches, optionally store a placeholder
            decoded_genres.append(f"Unknown({hex_val})")
    
    # Join them with commas
    return ", ".join(decoded_genres)

###############################################################################
# BASIC CURATION
###############################################################################

def run_curator_basic(spawn_root: str, is_admin: bool = True):
    """
    BASIC curation approach:
      - Possibly filter by favorites
      - Group tracks by spawnre or ©gen
      - Optionally merge similar clusters (using embeddings)
      - Shuffle each cluster
      - Write M3U
    """

    # 1) Determine the database path and table name based on mode.
    if is_admin:
        db_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "spawn_catalog.db")
        table_name = "tracks"
    else:
        db_path = os.path.join(spawn_root, "Spawn", "aux", "user", "spawn_library.db")
        table_name = "cat_tracks"

    if not os.path.isfile(db_path):
        print(f"[ERROR] Database not found at: {db_path}")
        return

    # 2) Load all track rows from the appropriate table.
    all_tracks = load_tracks_from_db(db_path, table_name=table_name)
    if not all_tracks:
        print("[INFO] No tracks found in the database. Nothing to curate.")
        return

    # Filter tracks to only include those with valid audio files
    filter_existing_files = True  # Set to False to disable filtering by file existence (i.e. to test curation of all catalog tracks)
    if filter_existing_files:
        original_count = len(all_tracks)
        all_tracks = [t for t in all_tracks if track_file_exists(t, spawn_root)]
        print(f"[INFO] Filtered tracks by file existence: {original_count} -> {len(all_tracks)}")

    # In user mode, only keep tracks that have a Spawn ID.
    if not is_admin:
        all_tracks = [t for t in all_tracks if t.get("spawn_id")]

    # Ask if user wants to filter by favorites
    favorites_filter_desc = None
    suffix = ""
    only_favorites_ans = input("\nWould you like to only use favorites for the curation? (y/[n]): ").strip().lower()
    if only_favorites_ans == "y":
        # Prompt user: which favorite set?
        print("\nWhich would you like to include in the curated playlist?")
        print("    1. favorite artists")
        print("    2. favorite albums")
        print("    3. favorite tracks")
        print("    4. favorite artists & favorite tracks")
        fav_choice = input("Enter your selection: ").strip()

        if fav_choice in ["1", "2", "3", "4"]:
            all_tracks = filter_tracks_by_favorites(all_tracks, spawn_root, fav_choice)
            if not all_tracks:
                print("[INFO] No tracks remain after applying favorites filter. Aborting curation.")
                return
            if fav_choice == "1":
                favorites_filter_desc = "favorite artists"
            elif fav_choice == "2":
                favorites_filter_desc = "favorite albums"
            elif fav_choice == "3":
                favorites_filter_desc = "favorite tracks"
            elif fav_choice == "4":
                favorites_filter_desc = "favorite artists & favorite tracks"
            suffix = "_favs"  # Append _favs to the M3U filename
        else:
            print("[WARN] Invalid selection. Proceeding without favorites filtering.")


    # 3) Group into clusters by 'spawnre' or fallback '©gen'
    clusters = group_tracks_by_genre(all_tracks)

    # 4) Optional: Merge similar clusters based on embeddings
    merge_choice = input("\nMerge similar clusters based on embeddings? (y/[n]): ").strip().lower()
    if merge_choice == "y":
        embeddings_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "mp4tovec.p")
        if not os.path.isfile(embeddings_path):
            print(f"[ERROR] Embeddings file not found at: {embeddings_path}. Skipping merge.")
        else:
            try:
                with open(embeddings_path, "rb") as f:
                    embeddings = pickle.load(f)
                if not isinstance(embeddings, dict):
                    raise ValueError("Embeddings file does not contain a dictionary.")
            except Exception as e:
                print(f"[ERROR] Failed to load embeddings: {e}. Skipping merge.")
            else:
                # Attach embedding to each track (if available)
                for tracks in clusters.values():
                    for track in tracks:
                        sid = track.get("spawn_id")
                        if sid in embeddings:
                            track['embedding'] = embeddings[sid]
                clusters = merge_similar_clusters(clusters, threshold=0.97)
                print("[INFO] Similar clusters have been merged.")

    # 5) Order the clusters by genre relationships (initial)
    ordered_genres = order_clusters_by_relationships(clusters)
    if not ordered_genres:
        print("[ERROR] Could not order clusters by relationship. Exiting.")
        return

    # 6) Allow user to optionally modify cluster order
    ordered_genres = maybe_modify_cluster_order(ordered_genres, clusters, spawn_root)

    # 7) Basic approach => just shuffle each cluster
    curated_clusters = []
    for genre in ordered_genres:
        track_list = clusters[genre]
        random.shuffle(track_list)
        curated_clusters.append((genre, track_list))

    # 8) Write M3U
    final_m3u_path = write_curated_m3u(spawn_root, curated_clusters, favorites_filter_desc, suffix)
    if final_m3u_path:
        print(f"[INFO] Curated M3U created at: {final_m3u_path}")
    else:
        print("[ERROR] Could not write the curated M3U playlist.")

    # 9) Summaries
    print("\n[Summary of curated clusters]:")
    for g, tracks in curated_clusters:
        print(f"  * {g} => {len(tracks)} tracks")


###############################################################################
# 2) FEATURE-BASED CURATION
###############################################################################
def run_curator_feature(spawn_root: str, is_admin: bool = True):
    """
    FEATURE-BASED curation approach:
      - Possibly filter by favorites
      - Group tracks by spawnre or ©gen
      - Optionally merge similar clusters (using embeddings)
      - Reorder each cluster with feature_based_curate
      - Write M3U
    """

    # 1) Determine the database path and table name based on mode.
    if is_admin:
        db_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "spawn_catalog.db")
        table_name = "tracks"
    else:
        db_path = os.path.join(spawn_root, "Spawn", "aux", "user", "spawn_library.db")
        table_name = "cat_tracks"

    if not os.path.isfile(db_path):
        print(f"[ERROR] Database not found at: {db_path}")
        return

    # 2) Load all track rows from the appropriate table.
    all_tracks = load_tracks_from_db(db_path, table_name=table_name)
    if not all_tracks:
        print("[INFO] No tracks found in the database. Nothing to curate.")
        return

    # Filter tracks to only include those with valid audio files
    filter_existing_files = True  # Set to False to disable filtering by file existence (i.e. to test curation of all catalog tracks)
    if filter_existing_files:
        original_count = len(all_tracks)
        all_tracks = [t for t in all_tracks if track_file_exists(t, spawn_root)]
        print(f"[INFO] Filtered tracks by file existence: {original_count} -> {len(all_tracks)}")

    # In user mode, only include tracks with a valid Spawn ID.
    if not is_admin:
        all_tracks = [t for t in all_tracks if t.get("spawn_id")]

    # 3) Ask if user wants to filter by favorites
    favorites_filter_desc = None
    suffix = ""
    only_favorites_ans = input("\nWould you like to only use favorites for the curation? (y/[n]): ").strip().lower()
    if only_favorites_ans == "y":
        print("\nWhich would you like to include in the curated playlist?")
        print("    1. favorite artists")
        print("    2. favorite albums")
        print("    3. favorite tracks")
        print("    4. favorite artists & favorite tracks")
        fav_choice = input("Enter your selection: ").strip()

        if fav_choice in ["1", "2", "3", "4"]:
            all_tracks = filter_tracks_by_favorites(all_tracks, spawn_root, fav_choice)
            if not all_tracks:
                print("[INFO] No tracks remain after applying favorites filter. Aborting curation.")
                return
            if fav_choice == "1":
                favorites_filter_desc = "favorite artists"
            elif fav_choice == "2":
                favorites_filter_desc = "favorite albums"
            elif fav_choice == "3":
                favorites_filter_desc = "favorite tracks"
            elif fav_choice == "4":
                favorites_filter_desc = "favorite artists & favorite tracks"
            suffix = "_favs"  # Append _favs to the M3U filename
        else:
            print("[WARN] Invalid selection. Proceeding without favorites filtering.")

    # 4) Group by spawnre / ©gen
    clusters = group_tracks_by_genre(all_tracks)

    # 5) Optional: Merge similar clusters based on embeddings
    merge_choice = input("\nMerge similar clusters based on embeddings? (y/[n]): ").strip().lower()
    if merge_choice == "y":
        embeddings_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "mp4tovec.p")
        if not os.path.isfile(embeddings_path):
            print(f"[ERROR] Embeddings file not found at: {embeddings_path}. Skipping merge.")
        else:
            try:
                with open(embeddings_path, "rb") as f:
                    embeddings = pickle.load(f)
                if not isinstance(embeddings, dict):
                    raise ValueError("Embeddings file does not contain a dictionary.")
            except Exception as e:
                print(f"[ERROR] Failed to load embeddings: {e}. Skipping merge.")
            else:
                for tracks in clusters.values():
                    for track in tracks:
                        sid = track.get("spawn_id")
                        if sid in embeddings:
                            track['embedding'] = embeddings[sid]
                clusters = merge_similar_clusters(clusters, threshold=0.97)
                print("[INFO] Similar clusters have been merged.")

    # 6) Order by relationships
    ordered_genres = order_clusters_by_relationships(clusters)
    if not ordered_genres:
        print("[ERROR] Could not order clusters by relationship. Exiting.")
        return

    # 7) Optionally modify cluster order
    ordered_genres = maybe_modify_cluster_order(ordered_genres, clusters, spawn_root)

    # 8) Feature-based reorder each cluster
    curated_clusters = []
    for genre in ordered_genres:
        track_list = clusters[genre]
        # Reorder with numeric feature distance
        reordered_list = feature_based_curate(track_list)
        curated_clusters.append((genre, reordered_list))

    # 9) Write M3U
    final_m3u_path = write_curated_m3u(spawn_root, curated_clusters, favorites_filter_desc, suffix)
    if final_m3u_path:
        print(f"[INFO] Feature-based M3U created at: {final_m3u_path}")
    else:
        print("[ERROR] Could not write the feature-based M3U playlist.")

    # 10) Summary
    print("\n[Summary of curated clusters]:")
    for g, tracks in curated_clusters:
        print(f"  * {g} => {len(tracks)} tracks")


###############################################################################
# 3) ADVANCED (DEEJ-AI) CURATION
###############################################################################

def run_curator_advanced(spawn_root: str, is_admin: bool = True):
    """
    Advanced curation approach leveraging embeddings and a trained model, with:
      - refined cluster membership (cosine centroid approach),
      - chain-based ordering within each cluster,
      - optional favorites filtering,
      - optional merging of similar clusters based on embeddings.
    """

    # 1) Determine the database path and table name based on mode.
    if is_admin:
        db_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "spawn_catalog.db")
        table_name = "tracks"
    else:
        db_path = os.path.join(spawn_root, "Spawn", "aux", "user", "spawn_library.db")
        table_name = "cat_tracks"

    if not os.path.isfile(db_path):
        print(f"[ERROR] Database not found at: {db_path}")
        return

    # 2) Validate embeddings file path
    embeddings_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "mp4tovec.p")
    if not os.path.isfile(embeddings_path):
        print(f"[ERROR] Embeddings file not found at: {embeddings_path}")
        return

    # 3) Load embeddings
    try:
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        if not isinstance(embeddings, dict):
            raise ValueError("Embeddings file does not contain a dictionary.")
    except Exception as e:
        print(f"[ERROR] Failed to load embeddings: {e}")
        return

    # 4) Load all track rows from the appropriate table.
    all_tracks = load_tracks_from_db(db_path, table_name=table_name)
    if not all_tracks:
        print("[INFO] No tracks found in the database. Nothing to curate.")
        return

    # Filter tracks to only include those with valid audio files
    filter_existing_files = True  # Set to False to disable filtering by file existence (i.e. to test curation of all catalog tracks)
    if filter_existing_files:
        original_count = len(all_tracks)
        all_tracks = [t for t in all_tracks if track_file_exists(t, spawn_root)]
        print(f"[INFO] Filtered tracks by file existence: {original_count} -> {len(all_tracks)}")

    # In user mode, only include tracks with a valid Spawn ID.
    if not is_admin:
        all_tracks = [t for t in all_tracks if t.get("spawn_id")]

    # # 5) Load Trained Model
    # model_path = os.path.join(os.path.dirname(__file__), "audio-encoder")
    # #model_path = os.path.join(os.path.dirname(__file__), "diffusion_pytorch_model.bin")
    # #model_path = os.path.join(os.path.dirname(__file__), "speccy_model.h5")
    # try:
    #     #model = AudioEncoder.from_pretrained("teticio/audio-encoder")
    #     model = AudioEncoder.from_pretrained(model_path)
    #     #model = load_mp4tovec_model_diffusion(model_path)
    #     #model = load_mp4tovec_model_torch(model_path)
    #     #model = load_mp4tovec_model_tf(model_path)
    # except Exception as e:
    #     print(f"[ERROR] Could not load model: {e}")
    #     return

    # 6) Optionally filter tracks by favorites
    favorites_filter_desc = None
    suffix = ""
    only_favorites_ans = input("\nWould you like to only use favorites for advanced curation? (y/[n]): ").strip().lower()
    if only_favorites_ans == "y":
        print("\nWhich would you like to include in the curated playlist?")
        print("    1. favorite artists")
        print("    2. favorite albums")
        print("    3. favorite tracks")
        print("    4. favorite artists & favorite tracks")
        fav_choice = input("Enter your selection: ").strip()

        if fav_choice in ["1", "2", "3", "4"]:
            all_tracks = filter_tracks_by_favorites(all_tracks, spawn_root, fav_choice)
            if not all_tracks:
                print("[INFO] No tracks remain after applying favorites filter. Aborting.")
                return
            if fav_choice == "1":
                favorites_filter_desc = "favorite artists"
            elif fav_choice == "2":
                favorites_filter_desc = "favorite albums"
            elif fav_choice == "3":
                favorites_filter_desc = "favorite tracks"
            elif fav_choice == "4":
                favorites_filter_desc = "favorite artists & favorite tracks"
            suffix = "_favs"  # Append _favs to the M3U filename
        else:
            print("[WARN] Invalid selection. Proceeding without favorites filtering.")

    # 7) Build base clusters (from spawnre/©gen)
    base_clusters = group_tracks_by_genre(all_tracks)

    # 8) Optional: Merge similar clusters based on embeddings
    merge_choice = input("\nMerge similar clusters based on embeddings? (y/[n]): ").strip().lower()
    if merge_choice == "y":
        # Attach embedding to each track (if available)
        for tracks in base_clusters.values():
            for track in tracks:
                sid = track.get("spawn_id")
                if sid in embeddings:
                    track['embedding'] = embeddings[sid]
        base_clusters = merge_similar_clusters(base_clusters, threshold=0.97)
        print("[INFO] Similar clusters have been merged.")

    # 9) Refine cluster membership based on cosine distance to each centroid
    refined_clusters = refine_clusters_by_embeddings(all_tracks, embeddings, base_clusters, distance_threshold=0.15)

    # 10) Sort clusters, with outliers last
    all_genre_keys = sorted(g for g in refined_clusters.keys() if g != "outliers")
    if "outliers" in refined_clusters:
        all_genre_keys.append("outliers")

    curated_clusters = []
    last_track_embedding = None

    # 11) Chain-based ordering within each cluster, bridging from previous cluster
    for genre in all_genre_keys:
        track_list = refined_clusters[genre]
        chain_ordered = chain_based_curation(track_list, embeddings, last_track_embedding)
        curated_clusters.append((genre, chain_ordered))

        # Update last track embedding
        if chain_ordered:
            last_track = chain_ordered[-1]
            sid = last_track.get("spawn_id")
            if sid in embeddings:
                last_track_embedding = embeddings[sid]
            else:
                last_track_embedding = None

    # 12) Write M3U
    final_m3u_path = write_curated_m3u(spawn_root, curated_clusters, favorites_filter_desc, suffix)
    if final_m3u_path:
        print(f"[INFO] Advanced curated M3U created at: {final_m3u_path}")
    else:
        print("[ERROR] Could not write the advanced curated M3U playlist.")

    # 13) Summary
    print("\n[Summary of refined clusters]:")
    for g, tracks in curated_clusters:
        print(f"\n  * {g} => {len(tracks)} tracks")

        # If this g looks like 'x...' => decode it into genres
        if g.startswith("x"):
            decoded = decode_spawnre_hex(g)
            # Print them with some indentation
            print(f"    {decoded}")

    # 14) Recommendations playlist
    recs_ans = input("\nWould you also like to generate a curated playlist of other tracks you might like? ([y]/n): ").strip().lower()
    if recs_ans in ["", "y"]:
        from .likey import generate_recommended_playlist
        generate_recommended_playlist(spawn_root, embeddings, threshold=0.98)


def chain_based_curation(
    track_list: List[dict],
    embeddings: Dict[str, np.ndarray],
    last_track_embedding: Optional[np.ndarray]
) -> List[dict]:
    """
    Chain-based approach for smooth track-to-track flow in a cluster.
    If last_track_embedding is given, pick the cluster track closest to it as start.
    Otherwise, pick a random start track.
    Then do a nearest-neighbor chain to order the rest.
    """

    # 1) Gather tracks with embeddings
    tracks_with_emb = [t for t in track_list if t.get("spawn_id") in embeddings]
    if not tracks_with_emb:
        logger.warning("No embedded tracks in this cluster. Returning as-is.")
        return track_list

    # 2) Build embedding array
    emb_list = [embeddings[t["spawn_id"]] for t in tracks_with_emb]
    emb_array = np.array(emb_list)
    n = len(tracks_with_emb)

    # 3) Build pairwise distance matrix
    dist_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = 1 - np.dot(emb_array[i], emb_array[j]) / (
                np.linalg.norm(emb_array[i]) * np.linalg.norm(emb_array[j])
            )
            dist_mat[i,j] = d
            dist_mat[j,i] = d

    # 4) Pick start index
    if last_track_embedding is not None:
        best_i = None
        best_d = float("inf")
        for i in range(n):
            d = 1 - np.dot(emb_array[i], last_track_embedding) / (
                np.linalg.norm(emb_array[i]) * np.linalg.norm(last_track_embedding)
            )
            if d < best_d:
                best_d = d
                best_i = i
        start_index = best_i
        logger.info(f"Starting chain in this cluster with track index={start_index}, distance={best_d:.4f}")
    else:
        import random
        start_index = random.randrange(n)
        logger.info(f"No previous cluster track. Starting chain with random index={start_index}")

    # 5) Nearest-neighbor chain
    unvisited = list(range(n))
    ordered_idx = []
    current = start_index
    ordered_idx.append(current)
    unvisited.remove(current)

    while unvisited:
        best_next = None
        best_dist = float("inf")
        for cand in unvisited:
            d = dist_mat[current, cand]
            if d < best_dist:
                best_dist = d
                best_next = cand
        ordered_idx.append(best_next)
        unvisited.remove(best_next)
        current = best_next

    # 6) Build final chain
    chain_tracks = [tracks_with_emb[i] for i in ordered_idx]

    # (Optional) Debug prints
    logger.info("Chain-based ordering within cluster:")
    for idx in range(1, len(ordered_idx)):
        prev_i = ordered_idx[idx-1]
        curr_i = ordered_idx[idx]
        d = dist_mat[prev_i, curr_i]
        sim = 1 - d
        logger.debug(f"  {tracks_with_emb[prev_i].get('©nam', ['Unknown'])[0]} -> "
                     f"{tracks_with_emb[curr_i].get('©nam', ['Unknown'])[0]}: dist={d:.4f}, sim={sim:.4f}")

    return chain_tracks


def refine_clusters_by_embeddings(
    all_tracks: List[dict],
    embeddings: Dict[str, np.ndarray],
    base_clusters: Dict[str, List[dict]],
    distance_threshold: float = 0.4,
) -> Dict[str, List[dict]]:
    """
    1) Compute centroid for each base cluster
    2) For each track (across entire library), measure cosine distance to each centroid
    3) If distance <= distance_threshold => add track to that cluster
       If a track doesn't belong to any => put in 'outliers'
    4) TBD: A track can be in multiple clusters (multi-membership)
        or each track joins at most one cluster (single-best-cluster approach)
    """

    print("[DEBUG] ========== refine_clusters_by_embeddings() using cosine distance ==========")
    centroids = compute_cluster_centroids(base_clusters, embeddings)

    refined = defaultdict(list)
    refined["outliers"] = []

    print(f"[DEBUG] distance_threshold = {distance_threshold:.2f}")

    for track_info in all_tracks:
        sid = track_info.get("spawn_id")
        # If no embedding, goes straight to outliers
        if sid not in embeddings:
            print(f"[DEBUG] Track {sid} => no embedding => outliers")
            refined["outliers"].append(track_info)
            continue

        track_emb = embeddings[sid]
        track_name = safe_extract_first(track_info, "©nam") or "<no title>"
        added_any = False

        # Pre-compute norm for speed
        track_norm = np.linalg.norm(track_emb)

        # Single-best cluster approach:
        best_cluster = None
        best_dist = float("inf")

        for genre, c_emb in centroids.items():
            if c_emb is None:
                continue

            # Cosine distance
            dot_val = np.dot(track_emb, c_emb)
            denom = track_norm * np.linalg.norm(c_emb)
            cos_dist = 1.0 - (dot_val / denom)

            # Save the lowest dist
            if cos_dist < best_dist:
                best_dist = cos_dist
                best_cluster = genre

        # Now place the track in either best_cluster or outliers
        if best_cluster and best_dist <= distance_threshold:
            print(f"[DEBUG] Track {sid} \"{track_name}\" => best cluster='{best_cluster}', dist={best_dist:.4f}")
            refined[best_cluster].append(track_info)
        else:
            print(f"[DEBUG] Track {sid} \"{track_name}\" => outliers (best_dist={best_dist:.4f} > {distance_threshold:.2f})")
            refined["outliers"].append(track_info)

        # for genre, c_emb in centroids.items():
        #     # If centroid is None => skip (no embeddings in that cluster)
        #     if c_emb is None:
        #         continue

        #     # --- COSINE DISTANCE ---
        #     dot_val = np.dot(track_emb, c_emb)
        #     denom = track_norm * np.linalg.norm(c_emb)
        #     cos_dist = 1.0 - (dot_val / denom)  # in [0,2] if embeddings not normalized

        #     print(f"[DEBUG] Track {sid} \"{track_name}\" -> centroid[{genre}] cos_dist={cos_dist:.4f}")

        #     if cos_dist <= distance_threshold:
        #         print(f"  -> ADDED to '{genre}' (cos_dist={cos_dist:.4f} ≤ {distance_threshold:.2f})")
        #         refined[genre].append(track_info)
        #         added_any = True

        # if not added_any:
        #     print(f"[DEBUG] Track {sid} => outliers (all cos_dist > {distance_threshold:.2f})")
        #     refined["outliers"].append(track_info)

    return dict(refined)



def compute_cluster_centroids(
    clusters: Dict[str, List[dict]],
    embeddings: Dict[str, np.ndarray]
) -> Dict[str, Optional[np.ndarray]]:
    """
    For each cluster in 'clusters', average the embeddings of its tracks
    to get a centroid. Return {genre -> centroid_vector} or None if no embeddings.
    """
    print("[DEBUG] ====== compute_cluster_centroids() ======")
    centroids = {}
    for genre, track_list in clusters.items():
        emb_list = []
        for t in track_list:
            sid = t.get("spawn_id")
            if sid in embeddings:
                emb_list.append(embeddings[sid])

        if emb_list:
            arr = np.vstack(emb_list)
            centroid = arr.mean(axis=0)
            centroids[genre] = centroid
            # Print debug: how many embeddings and partial centroid
            print(f"[DEBUG] genre='{genre}': {len(emb_list)} embeddings => centroid first 5 dims: {centroid[:5]}")
        else:
            centroids[genre] = None
            print(f"[DEBUG] genre='{genre}': 0 embeddings => centroid=None")

    return centroids


# def advanced_curation(track_list: List[dict], embeddings: Dict[str, np.ndarray], model) -> List[dict]:
#     """
#     Reorder tracks using embeddings in a sum-of-distances approach.
#     Exactly as before, but we can keep the name "advanced_curation".
#     """
#     tracks_with_embeddings = []
#     for track in track_list:
#         spawn_id = track.get("spawn_id")
#         if spawn_id in embeddings:
#             embedding = embeddings[spawn_id]
#             tracks_with_embeddings.append((track, embedding))
#         else:
#             logger.warning(f"No embedding found for Spawn ID: {spawn_id}")

#     if not tracks_with_embeddings:
#         logger.warning("No tracks with embeddings found. Returning original order.")
#         return track_list

#     # Extract embeddings and track metadata
#     embeddings_only = np.array([e for _, e in tracks_with_embeddings])
#     tracks_only = [t for t, _ in tracks_with_embeddings]

#     # Compute cosine similarity distances
#     num_embeddings = len(embeddings_only)
#     cosine_distances = np.zeros((num_embeddings, num_embeddings), dtype=np.float32)

#     logger.info("Calculating pairwise cosine distances...")
#     for i in range(num_embeddings):
#         for j in range(i + 1, num_embeddings):
#             distance = 1 - np.dot(embeddings_only[i], embeddings_only[j]) / (
#                 np.linalg.norm(embeddings_only[i]) * np.linalg.norm(embeddings_only[j])
#             )
#             cosine_distances[i, j] = distance
#             cosine_distances[j, i] = distance

#     logger.info("Reordering tracks based on sum-of-distances...")
#     sorted_indices = np.argsort(cosine_distances.sum(axis=1))
#     curated_list = [tracks_only[i] for i in sorted_indices]

#     # Debug prints for consecutive pairs
#     for idx in range(1, len(sorted_indices)):
#         prev_idx = sorted_indices[idx - 1]
#         curr_idx = sorted_indices[idx]
#         dist = cosine_distances[prev_idx, curr_idx]
#         sim = 1.0 - dist
#         print(f"[DEBUG] {tracks_only[prev_idx].get('©nam', ['Unknown'])[0]}"
#               f" -> {tracks_only[curr_idx].get('©nam', ['Unknown'])[0]}"
#               f": distance={dist:.4f}, similarity={sim:.4f}")

#     return curated_list


###############################################################################
# Load tracks from database
###############################################################################

#def load_tracks_from_db(db_path: str) -> List[dict]:
def load_tracks_from_db(db_path: str, table_name: str = "tracks") -> List[dict]:
    """
    Returns a list of dicts, each representing a track's tags as stored in the given table.
    For spawn_catalog.db, table_name should be "tracks".
    For spawn_library.db in user mode, table_name should be "cat_tracks".
    """
    results = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        #cursor.execute("SELECT spawn_id, tag_data FROM tracks")
        query = f"SELECT spawn_id, tag_data FROM {table_name}"
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        for (spawn_id, tag_json_str) in rows:
            try:
                tag_dict = json.loads(tag_json_str)
            except json.JSONDecodeError:
                continue # skip malformed JSON
            tag_dict["spawn_id"] = spawn_id
            results.append(tag_dict)

    except sqlite3.Error as e:
        logger.error(f"SQLite error while reading DB: {e}")
    return results


###############################################################################
# Optional favorites filtering
###############################################################################

def filter_tracks_by_favorites(all_tracks: List[dict], spawn_root: str, fav_choice: str) -> List[dict]:
    """
    Returns a subset of 'all_tracks' that match user's selected favorites filter.
    fav_choice is one of: "1" (fav artists), "2" (fav albums), 
                          "3" (fav tracks), or "4" (fav artists + fav tracks).
    Load the JSON from: 
       spawn_root/Spawn/aux/user/favs/fav_artists.json
       spawn_root/Spawn/aux/user/favs/fav_albums.json
       spawn_root/Spawn/aux/user/favs/fav_tracks.json
    and filter accordingly.
    """

    favs_folder = os.path.join(spawn_root, "Spawn", "aux", "user", "favs")
    fav_artists = load_favs_artists(os.path.join(favs_folder, "fav_artists.json"))
    fav_albums  = load_favs_albums(os.path.join(favs_folder, "fav_albums.json"))
    fav_tracks  = load_favs_tracks(os.path.join(favs_folder, "fav_tracks.json"))

    # For quick lookups, build sets or dictionaries
    # 1) Artists => set of (artist_lower) plus optional set of MBIDs
    artist_names_set = set()
    artist_mbids_set = set()
    for item in fav_artists:
        if isinstance(item, dict):
            art = item.get("artist", "")
            art_lower = art.strip().lower()
            if art_lower:
                artist_names_set.add(art_lower)

            mbid = item.get("artist_mbid", "")
            if mbid:
                artist_mbids_set.add(mbid.strip().lower())
        else:
            # if it's just a string
            art_lower = str(item).strip().lower()
            if art_lower:
                artist_names_set.add(art_lower)

    # 2) Albums => set of (artist_lower, album_lower) or optional MBIDs
    album_pairs_set = set()
    album_mbids_set = set()
    for item in fav_albums:
        if isinstance(item, dict):
            art = item.get("artist", "").strip().lower()
            alb = item.get("album", "").strip().lower()
            if art or alb:
                album_pairs_set.add((art, alb))

            rg_mbid = item.get("release_group_mbid", "")
            if rg_mbid:
                album_mbids_set.add(rg_mbid.strip().lower())
        else:
            # If it's just a string, interpret it as album name alone
            alb = str(item).strip().lower()
            album_pairs_set.add(("", alb))

    # 3) Tracks => set of spawn_ids, or (artist, album, track)
    track_spawn_ids = set()
    track_triples = set()
    for item in fav_tracks:
        if isinstance(item, dict):
            spid = item.get("spawn_id")
            if spid:
                track_spawn_ids.add(spid.strip().lower())
            art = item.get("artist", "").strip().lower()
            alb = item.get("album", "").strip().lower()
            trk = item.get("track", "").strip().lower()
            track_triples.add((art, alb, trk))
        else:
            # If it's just a string, interpret it as a track name alone
            track_triples.add(("", "", str(item).strip().lower()))

    # Now define the "keep" logic based on fav_choice
    filtered = []

    for track_info in all_tracks:
        # Extract relevant fields
        spawn_id = track_info.get("spawn_id", "").strip().lower()

        artist_name = safe_extract_first(track_info, "©ART") or ""
        album_name  = safe_extract_first(track_info, "©alb") or ""
        track_name  = safe_extract_first(track_info, "©nam") or ""

        art_lower = artist_name.strip().lower()
        alb_lower = album_name.strip().lower()
        trk_lower = track_name.strip().lower()

        # Also read MBIDs if you want
        artist_mbid_val = safe_extract_first(track_info, "----:com.apple.iTunes:MusicBrainz Artist Id") or ""
        artist_mbid_val = artist_mbid_val.strip().lower()
        # For "release_group_mbid", you might store it in "----:com.apple.iTunes:MusicBrainz Release Group Id" or similar.

        # Helper booleans
        is_fav_artist = (art_lower in artist_names_set) or (artist_mbid_val in artist_mbids_set if artist_mbid_val else False)
        is_fav_album  = ((art_lower, alb_lower) in album_pairs_set)  # ignoring MBID for simplicity
        is_fav_track  = (spawn_id in track_spawn_ids) or ((art_lower, alb_lower, trk_lower) in track_triples)

        # Evaluate conditions
        if fav_choice == "1":  # favorite artists
            if is_fav_artist:
                filtered.append(track_info)

        elif fav_choice == "2":  # favorite albums
            if is_fav_album:
                filtered.append(track_info)

        elif fav_choice == "3":  # favorite tracks
            if is_fav_track:
                filtered.append(track_info)

        elif fav_choice == "4":  # favorite artists OR favorite tracks
            if is_fav_artist or is_fav_track:
                filtered.append(track_info)

    return filtered


def load_favs_artists(filepath: str) -> List[dict]:
    if not os.path.isfile(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # data could be a list of dicts or strings
            if isinstance(data, list):
                return data
    except:
        pass
    return []


def load_favs_albums(filepath: str) -> List[dict]:
    if not os.path.isfile(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except:
        pass
    return []


def load_favs_tracks(filepath: str) -> List[dict]:
    if not os.path.isfile(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except:
        pass
    return []



###############################################################################
# Group by genre/spawnre
###############################################################################

def group_tracks_by_genre(all_tracks: List[dict]) -> Dict[str, List[dict]]:
    """
    Return a dict of {genre_name -> list_of_track_dicts}.
    We do NOT have an explicit file_path stored, so we must 
    store the entire track data dict for use in final reconstruction.
    """
    clusters = defaultdict(list)
    for track_info in all_tracks:
        # Prefer spawnre_hex for normalized, standardized genre codes
        spawnre_hex_val = safe_extract_first(track_info, "----:com.apple.iTunes:spawnre_hex")
        if spawnre_hex_val:
            g = spawnre_hex_val.lower()
        else:
            # fallback to the genre tag
            fallback_gen = safe_extract_first(track_info, "genre")
            g = fallback_gen.lower() if fallback_gen else "unknown"
        clusters[g].append(track_info)
    return dict(clusters)


def merge_similar_clusters(clusters, threshold=0.97):
    """
    clusters: dict mapping cluster_id to list of track dictionaries.
              Each track is expected to have an 'embedding' key.
    threshold: cosine similarity threshold above which clusters are merged.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Compute centroids for each cluster
    centroids = {
        cid: np.mean([track['embedding'] for track in tracks if 'embedding' in track], axis=0)
        for cid, tracks in clusters.items()
    }
    
    labels = list(centroids.keys())
    merged_clusters = {}
    merged = set()
    
    for i, label_i in enumerate(labels):
        if label_i in merged:
            continue
        # Start with current cluster
        current_cluster = clusters[label_i][:]
        centroid_i = centroids[label_i]
        
        for j in range(i + 1, len(labels)):
            label_j = labels[j]
            if label_j in merged:
                continue
            centroid_j = centroids[label_j]
            sim = cosine_similarity(centroid_i.reshape(1, -1), centroid_j.reshape(1, -1))[0][0]
            if sim > threshold:
                current_cluster.extend(clusters[label_j])
                merged.add(label_j)
        merged_clusters[label_i] = current_cluster
    
    return merged_clusters


def safe_extract_first(tag_dict: dict, key: str) -> Optional[str]:
    """
    Helps decode iTunes-style tags that can be list-of-bytes or list-of-strings, etc.
    """
    raw = tag_dict.get(key)
    if not raw:
        return None

    if isinstance(raw, list):
        if not raw:
            return None
        first_item = raw[0]
        if isinstance(first_item, bytes):
            return first_item.decode("utf-8", errors="replace").strip()
        return str(first_item).strip()
    elif isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    else:
        return str(raw).strip()


###############################################################################
# Order clusters by related genre logic
###############################################################################

def order_clusters_by_relationships(clusters: Dict[str, List[dict]]) -> List[str]:
    """
    1) Sort by descending size
    2) Link related genres
    3) Return a list of genre keys in an "intuitive" order
    """

    # 1) Sort by size
    cluster_counts = {g: len(tracks) for g, tracks in clusters.items()}
    sorted_by_size = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_by_size:
        return []

    # final order
    ordered_genres = []
    used = set()

    # 2) Start with biggest
    start_genre = sorted_by_size[0][0]
    ordered_genres.append(start_genre)
    used.add(start_genre)

    current_genre = start_genre
    while len(used) < len(clusters):
        # find next related
        next_g = find_next_related_genre(current_genre, clusters, used)
        if next_g:
            ordered_genres.append(next_g)
            used.add(next_g)
            current_genre = next_g
        else:
            # fallback to next largest unused
            leftover = [(g, c) for (g, c) in sorted_by_size if g not in used]
            if leftover:
                next_g = leftover[0][0]
                ordered_genres.append(next_g)
                used.add(next_g)
                current_genre = next_g
            else:
                break

    return ordered_genres


def find_next_related_genre(current: str, clusters: Dict[str, List[dict]], used: set) -> Optional[str]:
    related_list = get_related_genres(current)
    for r in related_list:
        if r in clusters and r not in used:
            return r
    return None


def get_related_genres(genre_name: str) -> List[str]:
    """
    Finds 'Related' codes from genre_mapping if genre_name 
    matches the code's .lower() 'Genre' field.
    """
    from .dic_spawnre import genre_mapping
    glower = genre_name.lower()
    matched_codes = []
    for code, details in genre_mapping.items():
        if details["Genre"].lower() == glower:
            matched_codes.append(code)

    related_genres = []
    for c in matched_codes:
        rel_codes = genre_mapping[c].get("Related", [])
        for rc in rel_codes:
            rname = genre_mapping.get(rc, {}).get("Genre", "").lower()
            if rname:
                related_genres.append(rname)
    return related_genres


###############################################################################
# Final reorder: optionally modify cluster order
###############################################################################

def maybe_modify_cluster_order(
    ordered_genres: List[str], 
    clusters: Dict[str, List[dict]],
    spawn_root: str
) -> List[str]:
    """
    Allows the user to:
     - see current cluster order
     - optionally load a previously saved order
     - or manually enter a new comma-separated list of genres
     - or skip reordering
    Then returns the final list of genres in the chosen order.

    If user chooses to save the new order, we store it in 
      spawn_root/Spawn/aux/user/cur8/custom_cluster_order.json
    """

    # 1) Present summary
    print("\nCurrent clusters in M3U order:")
    for genre in ordered_genres:
        print(f" - {genre}: {len(clusters[genre])} tracks")

    # 2) Prompt user whether they want to reorder
    ans = input("\nWould you like to modify the genre order? (y/[n]): ").strip().lower()
    if ans == "":
        ans = "n"
    if ans not in ["y", "n"]:
        print("[WARN] Invalid input, skipping reordering.")
        return ordered_genres
    if ans == "n":
        return ordered_genres

    # If user says yes, give them a few options
    while True:
        print("\nHow would you like to specify the cluster order?")
        print("   1) Load previously saved custom cluster order")
        print("   2) Enter a new comma-separated list of genres")
        print("   3) Use the existing order (do nothing)")
        choice = input("Enter choice: ").strip()

        if choice not in ["1", "2", "3"]:
            print("[WARN] Invalid choice, try again.")
            continue

        if choice == "3":
            # user wants to keep the existing order
            return ordered_genres

        if choice == "1":
            # load from JSON
            new_order = load_custom_cluster_order(spawn_root)
            if not new_order:
                print("[INFO] No custom cluster order found or invalid. Returning to menu.")
                continue

            # Validate & combine
            validated_order = [g for g in new_order if g in clusters]
            leftover = [g for g in ordered_genres if g not in validated_order]
            final_order = validated_order + leftover

            # present summary
            print("\nUpdated order from custom file:")
            for genre in final_order:
                print(f" - {genre}: {len(clusters[genre])} tracks")

            # Done
            return final_order

        if choice == "2":
            # user manually enters a new order
            # First show the initial order in a single line
            initial_order_str = ", ".join(ordered_genres)
            print(f"\nInitial genre order: {initial_order_str}")
            manual_input = input("Enter the new genre order, comma-separated: ").strip()
            if not manual_input:
                print("[INFO] Nothing entered, returning to menu.")
                continue

            new_order_list = [x.strip().lower() for x in manual_input.split(",") if x.strip()]
            if not new_order_list:
                print("[INFO] No valid genres recognized, returning to menu.")
                continue

            # Validate
            validated_order = [g for g in new_order_list if g in clusters]
            # If user typed some invalid genres, we skip them
            invalid_genres = [g for g in new_order_list if g not in clusters]
            if invalid_genres:
                print(f"[WARN] The following genres aren't present in current clusters, ignoring: {', '.join(invalid_genres)}")

            # add leftover
            leftover = [g for g in ordered_genres if g not in validated_order]
            final_order = validated_order + leftover

            print("\nUpdated cluster order:")
            for genre in final_order:
                print(f" - {genre}: {len(clusters[genre])} tracks")

            # Let user optionally save it
            ans_save = input("\nWould you like to save this new order for future use? (y/[n]): ").strip().lower()
            if ans_save == "":
                ans_save = "n"
            if ans_save == "y":
                save_custom_cluster_order(spawn_root, final_order)
            return final_order


###############################################################################
# Save / Load custom cluster order
###############################################################################

def save_custom_cluster_order(spawn_root: str, cluster_order: List[str]):
    """
    Save the user-defined cluster order to 
       spawn_root/Spawn/aux/user/cur8/custom_cluster_order.json
    """
    cur8_dir = os.path.join(spawn_root, "Spawn", "aux", "user", "cur8")
    os.makedirs(cur8_dir, exist_ok=True)

    out_file = os.path.join(cur8_dir, "custom_cluster_order.json")
    try:
        data = {"cluster_order": cluster_order}
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Saved custom cluster order to: {out_file}")
    except Exception as e:
        logger.error(f"Could not save cluster order: {e}")
        print("[ERROR] Failed to save custom cluster order.")


def load_custom_cluster_order(spawn_root: str) -> Optional[List[str]]:
    """
    Load a previously saved cluster order from 
      spawn_root/Spawn/aux/user/cur8/custom_cluster_order.json
    Returns the list of genres or None if not found/invalid.
    """
    cur8_dir = os.path.join(spawn_root, "Spawn", "aux", "user", "cur8")
    in_file = os.path.join(cur8_dir, "custom_cluster_order.json")
    if not os.path.isfile(in_file):
        print("[INFO] No custom cluster order file found.")
        return None

    try:
        with open(in_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        possible_order = data.get("cluster_order")
        if not (possible_order and isinstance(possible_order, list)):
            print("[WARN] 'cluster_order' missing or not a list in custom file.")
            return None
        return [g.strip().lower() for g in possible_order if g.strip()]
    except Exception as e:
        logger.error(f"Could not load custom cluster order: {e}")
        print("[ERROR] Failed to load custom cluster order.")
        return None


###############################################################################
# Feature-based ordering
###############################################################################

# These are the keys in track_importer.py that store numeric values in the database
FEATURE_KEYS = [
    "----:com.apple.iTunes:feature_valence",
    "----:com.apple.iTunes:feature_time_signature",
    "----:com.apple.iTunes:feature_tempo",
    "----:com.apple.iTunes:feature_speechiness",
    "----:com.apple.iTunes:feature_mode",
    "----:com.apple.iTunes:feature_loudness",
    "----:com.apple.iTunes:feature_liveness",
    "----:com.apple.iTunes:feature_key",
    "----:com.apple.iTunes:feature_instrumentalness",
    "----:com.apple.iTunes:feature_energy",
    "----:com.apple.iTunes:feature_danceability",
    "----:com.apple.iTunes:feature_acousticness",
]

def feature_based_curate(cluster: List[dict]) -> List[dict]:
    """
    1) Separate cluster into tracks that have full feature data vs. those that do not.
    2) For the ones with features, do a distance-based chain.
    3) Append the feature-missing tracks at the end (shuffled).
    """
    # separate cluster
    with_feats = []
    missing_feats = []
    for t in cluster:
        vec = get_numeric_features(t)
        if vec is not None:
            with_feats.append({"track": t, "feat_vec": vec})
        else:
            missing_feats.append(t)

    # distance-based ordering for with_feats
    if len(with_feats) <= 1:
        # no need for ordering
        ordered_with_feats = [x["track"] for x in with_feats]
    else:
        # chain them
        ordered_with_feats = feature_based_order(with_feats)

    random.shuffle(missing_feats)
    return ordered_with_feats + missing_feats

def feature_based_order(track_dicts: List[Dict]) -> List[dict]:
    """
    track_dicts is a list of {"track": <track_info>, "feat_vec": [floats]}
    We'll pick track_dicts[0] as the start, then chain to the next closest track by Eucl. distance in feature space.
    """
    # Make a copy so we don't mutate original
    todo = track_dicts[:]
    # pick the first track as the start
    ordered = [todo.pop(0)]

    while todo:
        last_vec = ordered[-1]["feat_vec"]
        next_idx = None
        min_dist = float("inf")
        for i, candidate in enumerate(todo):
            dist = euclidean_distance(last_vec, candidate["feat_vec"])
            if dist < min_dist:
                min_dist = dist
                next_idx = i

        if next_idx is not None:
            # chain it
            ordered.append(todo.pop(next_idx))
        else:
            # fallback if something weird
            ordered.append(todo.pop(0))

    return [d["track"] for d in ordered]

def euclidean_distance(vecA: List[float], vecB: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vecA, vecB)))

def get_numeric_features(track_info: dict) -> Optional[List[float]]:
    """
    Extract float values from the track's DB tags for each item in FEATURE_KEYS.
    If any are missing or fail to parse, return None to indicate incomplete data.
    """
    feat_vec = []
    for k in FEATURE_KEYS:
        val_str = safe_extract_first(track_info, k)
        if not val_str:
            return None  # missing
        try:
            fval = float(val_str)
        except ValueError:
            return None  # invalid parse
        feat_vec.append(fval)
    return feat_vec


###############################################################################
# Build final file paths & write .m3u
###############################################################################

def write_curated_m3u(
    spawn_root: str, 
    curated_clusters: List[Tuple[str, List[dict]]],
    favorites_filter_desc: Optional[str] = None,
    suffix: str = ""
) -> Optional[str]:
    """
    Save a curated M3U to: spawn_root/Spawn/Playlists/Curated/curate_YYYY-MM-DD_HHMMSS{suffix}.m3u
    Each track line => "../../Music/Artist/Album/D-TT - Title.m4a"
    Also insert a line just below #EXTM3U if a favorites filter was used.
    The optional suffix parameter (default empty) is appended to the filename.
    """
    # 1) Build the subfolder
    curated_dir = os.path.join(spawn_root, "Spawn", "Playlists", "Curated")
    os.makedirs(curated_dir, exist_ok=True)

    # 2) Build the .m3u filename (include seconds for uniqueness)
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    m3u_name = f"curate_{ts}{suffix}.m3u"
    m3u_path = os.path.join(curated_dir, m3u_name)

    # 3) Write the file
    prefix = "../../Music"
    try:
        with open(m3u_path, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            
            if favorites_filter_desc:
                f.write(f"# Filtered by {favorites_filter_desc}\n")

            f.write("\n")

            for (genre_key, track_list) in curated_clusters:
                # If the key starts with 'x', it's a spawnre_hex, so decode it
                if genre_key.startswith("x"):
                    # Write raw hex, then the decoded genres
                    f.write(f"# SPAWNRE_HEX: {genre_key}\n")
                    decoded = decode_spawnre_hex(genre_key)
                    f.write(f"# GENRES: {decoded}\n")
                else:
                    # Otherwise just label it as "GENRE:"
                    f.write(f"# GENRE: {genre_key}\n")

                # Then list the tracks
                for track_info in track_list:
                    rel_path = build_relative_path(track_info, prefix)
                    f.write(f"{rel_path}\n")
                f.write("\n")

        return m3u_path
    except Exception as e:
        logger.error(f"Error writing curated M3U => {e}")
        return None


def track_file_exists(track_info: dict, spawn_root: str) -> bool:
    """
    Returns True if the audio file for the track exists in the Music directory.
    Uses similar logic to build_relative_path(), but constructs an absolute path.
    """
    import os
    # Define the absolute Music directory (assumes Music folder is under spawn_root)
    music_dir = os.path.join(spawn_root, "Music")
    
    artist = safe_extract_first(track_info, "©ART") or "Unknown"
    album  = safe_extract_first(track_info, "©alb") or "Unknown"
    title  = safe_extract_first(track_info, "©nam") or "Untitled"
    spawn_id = track_info.get("spawn_id", "unknown")
    
    # Extract disc and track numbers (with safe fallbacks)
    disc_tag = track_info.get("disk")
    track_tag = track_info.get("trkn")
    disc_main = 1
    if disc_tag and isinstance(disc_tag, list) and disc_tag:
        try:
            disc_main = int(disc_tag[0][0])
        except Exception:
            pass
    track_main = 0
    if track_tag and isinstance(track_tag, list) and track_tag:
        try:
            track_main = int(track_tag[0][0])
        except Exception:
            pass

    # Build the file name using the same helper as before
    file_name = build_d_tt_spawn_id_title_filename(disc_main, track_main, spawn_id, title)
    
    # Sanitize artist and album as done in build_relative_path
    artist_dir = sanitize_for_directory(artist)
    album_dir = sanitize_for_directory(album)
    
    # Build the absolute path to the audio file
    abs_path = os.path.join(music_dir, artist_dir, album_dir, file_name)
    return os.path.isfile(abs_path)


def build_relative_path(track_tags: dict, prefix: str) -> str:
    """
    Reconstructs file path in the format:
    ../../Music/Artist/Album/D-TT [spawn_id] - Title.m4a
    """

    artist_raw = safe_extract_first(track_tags, "©ART") or "Unknown"
    album_raw  = safe_extract_first(track_tags, "©alb") or "Unknown"
    disc_tag   = track_tags.get("disk")  # typically [(disc_main, total)]
    track_tag  = track_tags.get("trkn")  # typically [(track_main, total)]
    title_raw  = safe_extract_first(track_tags, "©nam") or "Untitled"
    spawn_id   = track_tags.get("spawn_id", "unknown")

    disc_main = 1
    if disc_tag and isinstance(disc_tag, list) and disc_tag:
        try:
            disc_main = int(disc_tag[0][0])
        except:
            pass

    track_main = 0
    if track_tag and isinstance(track_tag, list) and track_tag:
        try:
            track_main = int(track_tag[0][0])
        except:
            pass

    # Sanitize fields for filenames
    artist_dir = sanitize_for_directory(artist_raw)
    album_dir  = sanitize_for_directory(album_raw)
    file_name  = build_d_tt_spawn_id_title_filename(disc_main, track_main, spawn_id, title_raw)

    return f"{prefix}/{artist_dir}/{album_dir}/{file_name}"


def sanitize_for_directory(name: str, max_len: int = 50) -> str:
    name = re.sub(r'[\\/:*?"<>|]', '_', name.strip())
    if len(name) > max_len:
        name = name[:max_len].rstrip("_- ")
    return name or "Unknown"


def build_d_tt_spawn_id_title_filename(disc_num: int, track_num: int, spawn_id: str, track_title: str) -> str:
    """
    Builds filename in the format: D-TT [spawn_id] - Title.m4a
    """

    if not disc_num or disc_num < 1:
        disc_num = 1
    if not track_num or track_num < 1:
        track_num = 0
    track_str = f"{track_num:02d}"
    safe_title = sanitize_title_for_filename(track_title)

    return f"{disc_num}-{track_str} [{spawn_id}] - {safe_title}.m4a"


def sanitize_title_for_filename(title: str, max_len: int = 60) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', title.strip())
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].rstrip("_- ")
    return sanitized
