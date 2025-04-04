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
import unicodedata
#import librosa
import pickle
import numpy as np
import torch
#import tensorflow as tf

from collections import defaultdict, Counter
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
from sklearn.metrics.pairwise import cosine_similarity
from .likey import compute_like_likelihoods
from .track_loader import load_and_merge

from .dic_spawnre import genre_mapping, subgenre_to_parent, genre_synonyms

logger = logging.getLogger(__name__)


###############################################################################
# HELPER: Build embeddings dictionary from merged track data.
###############################################################################
def build_embeddings_dict(tracks: List[dict]) -> Dict[str, np.ndarray]:
    """
    Builds a dictionary mapping an identifier (spawn_id if available, else local_ID)
    to its embedding (if present) for all tracks.
    """
    emb = {}
    for t in tracks:
        key = t.get("spawn_id") or t.get("local_id")
        if key and t.get("embedding") is not None:
            emb[key] = t["embedding"]
    return emb


###############################################################################
# GENRE DECODING
###############################################################################

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


def get_album_genre_metadata(album_id: Tuple[str, str], all_tracks: List[dict]) -> Dict[str, set]:
    """
    Given an album identifier as (artist_lower, album_lower) and a list of all tracks,
    finds all tracks that belong to that album and consolidates the genre metadata.
    
    It returns a dictionary with keys:
       - "genre": from the 'genre' tag (if present)
       - "spawnre": from a tag named, for example, "spawnre" (if used)
       - "spawnre_hex": from the '----:com.apple.iTunes:spawnre_hex' tag
    
    The values are sets of the unique values found.
    """
    genres = set()
    spawnres = set()
    spawnre_hexes = set()

    for track in all_tracks:
        # Compare using normalized values
        artist = (safe_extract_first(track, "©ART") or "").strip().lower()
        album  = (safe_extract_first(track, "©alb") or "").strip().lower()

        if artist == album_id[0] and album == album_id[1]:
            g = safe_extract_first(track, "©gen")
            if g:
                genres.add(g.strip())
            sr = safe_extract_first(track, "----:com.apple.iTunes:spawnre")
            if sr:
                spawnres.add(sr.strip())
            sr_hex = safe_extract_first(track, "----:com.apple.iTunes:spawnre_hex")
            if sr_hex:
                spawnre_hexes.add(sr_hex.strip())

    #print(f"[DEBUG] For album {album_id}, found genres: {genres}, spawnre: {spawnres}, spawnre_hex: {spawnre_hexes}")
    return {
        "genre": genres,
        "spawnre": spawnres,
        "spawnre_hex": spawnre_hexes,
    }


def build_album_comments(excluded_album_m3us: List[str],
                         all_tracks: List[dict],
                         spawn_root: str) -> Dict[Tuple[str, str], str]:
    """
    Builds a dictionary mapping album_id (artist_lower, album_lower) to the album comment string.
    The comment is constructed using the album's relative M3U path, main genre, and spawnre_hex tokens.
    """
    album_comments = {}
    for album_m3u in excluded_album_m3us:
        base_name = os.path.basename(album_m3u)
        try:
            artist_part, album_part = os.path.splitext(base_name)[0].split(" - ", 1)
            album_id = (artist_part.strip().lower(), album_part.strip().lower())
        except Exception as e:
            print(f"[WARN] Failed to parse album id from '{base_name}': {e}")
            continue

        # Filter full library for tracks matching this album.
        album_tracks = [t for t in all_tracks if
                        (safe_extract_first(t, "©ART", normalize=False) or "").strip().lower() == album_id[0] and
                        (safe_extract_first(t, "©alb", normalize=False) or "").strip().lower() == album_id[1]]
        # Build album metadata.
        album_meta = get_album_genre_metadata(album_id, all_tracks) if all_tracks else {}
        meta_parts = []
        if album_meta.get("genre"):
            meta_parts.append("main genre: " + ", ".join(sorted(album_meta["genre"])))
        if album_meta.get("spawnre_hex"):
            token_counter = Counter()
            for hex_val in album_meta["spawnre_hex"]:
                decoded_str = decode_spawnre_hex(hex_val)
                tokens = [token.strip() for token in decoded_str.split(",") if token.strip()]
                token_counter.update(tokens)
            top_tokens = [token for token, count in token_counter.most_common(5)]
            meta_parts.append("spawnre_hex: " + ", ".join(top_tokens))
        rel_album_path = os.path.relpath(album_m3u, start=os.path.join(spawn_root, "Spawn", "Playlists"))
        comment = f"# FAVORITE ALBUM: {rel_album_path}"
        if meta_parts:
            comment += " | " + " | ".join(meta_parts)
        album_comments[album_id] = comment
    return album_comments


def get_album_candidate_blocks_for_cluster(cluster_key: str, 
                                           excluded_albums: set, 
                                           all_tracks_full: List[dict], 
                                           embeddings: Dict[str, np.ndarray],
                                           album_comments: Dict[Tuple[str, str], str],
                                           album_to_cluster: Dict[Tuple[str, str], str]
                                           ) -> List[dict]:
    """
    For a given cluster (identified by its key), scan the full track library to find all favorite album tracks
    (from excluded_albums, a set of (artist_lower, album_lower) tuples) that are assigned to this cluster based on the 
    best matching cluster assignment. Returns a list of album candidate blocks, each with:
         "is_album": True,
         "album_tracks": [track1, track2, ...] (sorted in sequential order),
         "album_comment": the precomputed album comment.
    A block is only created if both the first and last track have valid embeddings.
    """
    def get_track_order(t: dict):
        disk = 1
        track_num = 0
        try:
            disk_tag = t.get("disk")
            if disk_tag and isinstance(disk_tag, list) and disk_tag:
                disk = int(disk_tag[0][0])
        except Exception:
            disk = 1
        try:
            track_tag = t.get("trkn")
            if track_tag and isinstance(track_tag, list) and track_tag:
                track_num = int(track_tag[0][0])
        except Exception:
            track_num = 0
        return (disk, track_num)

    album_blocks = []
    for album in excluded_albums:  # album is a tuple: (artist_lower, album_lower)
        # Only include the album if it is assigned to the current cluster.
        if album_to_cluster.get(album) != cluster_key:
            continue

        album_tracks = [t for t in all_tracks_full if
                        (safe_extract_first(t, "©ART", normalize=True) == album[0] and
                         safe_extract_first(t, "©alb", normalize=True) == album[1])]
        if album_tracks:
            # Sort album tracks in sequential order.
            album_tracks = sorted(album_tracks, key=get_track_order)
            first_spawn = album_tracks[0].get("spawn_id")
            last_spawn  = album_tracks[-1].get("spawn_id")
            if first_spawn in embeddings and last_spawn in embeddings:
                # Look up the album comment from album_comments.
                album_comment = album_comments.get(album, "")
                album_block = {
                    "is_album": True,
                    "album_tracks": album_tracks,
                    "album_comment": album_comment
                }
                album_blocks.append(album_block)
    return album_blocks


def assign_albums_to_clusters(excluded_album_m3us: List[str],
                              all_tracks: List[dict],
                              embeddings: Dict[str, np.ndarray],
                              cluster_centroids: Dict[str, np.ndarray],
                              spawn_root: str,
                              similarity_threshold: float = 0.0
                              ) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], str]]:
    """
    For each favorite album (provided as an album M3U path), compute its centroid
    based on all tracks in the full library that belong to that album. Also extract
    its spawnre_hex metadata (decoded into tokens). Then, for each candidate cluster
    (using its centroid from embeddings), compute both the cosine similarity and the
    token overlap with the album’s tokens. The album is assigned to the candidate
    cluster that has the highest token overlap (using cosine similarity as a tiebreaker)
    provided that the similarity meets the threshold and there is at least one overlapping token.
    
    If no candidate meets the criteria, the album is assigned to the "outliers" group.
    
    Returns a tuple:
      - A dictionary mapping cluster keys to lists of album comment lines.
      - A dictionary mapping album IDs (tuple of artist_lower, album_lower) to their assigned cluster.
    """
    album_assignments = {key: [] for key in cluster_centroids.keys()}
    album_assignments.setdefault("outliers", [])
    album_to_cluster = {}
    
    for album_m3u in excluded_album_m3us:
        base_name = os.path.basename(album_m3u)
        try:
            # Expected filename format: "Artist - Album.m3u"
            artist_part, album_part = os.path.splitext(base_name)[0].split(" - ", 1)
            album_id = (artist_part.strip().lower(), album_part.strip().lower())
        except Exception as e:
            print(f"[WARN] Failed to parse album id from '{base_name}': {e}")
            continue
        
        # Filter full library for tracks matching this album using raw (non-normalized) comparisons.
        album_tracks = [t for t in all_tracks if
                        (safe_extract_first(t, "©ART", normalize=False) or "").strip().lower() == album_id[0] and
                        (safe_extract_first(t, "©alb", normalize=False) or "").strip().lower() == album_id[1]]
        album_embs = [embeddings[t.get("spawn_id")] for t in album_tracks if t.get("spawn_id") in embeddings]
        if not album_embs:
            print(f"[DEBUG] No embeddings for album {album_id}; skipping.")
            continue
        arr = np.vstack(album_embs)
        album_centroid = arr.mean(axis=0)

        # Retrieve album metadata and compute its token set from spawnre_hex.
        album_meta = get_album_genre_metadata(album_id, all_tracks) if all_tracks else {}
        album_tokens = set()
        if album_meta.get("spawnre_hex"):
            for hex_val in album_meta["spawnre_hex"]:
                decoded_str = decode_spawnre_hex(hex_val)
                tokens = {token.strip() for token in decoded_str.split(",") if token.strip()}
                album_tokens.update(tokens)
        
        # Iterate over candidate clusters to find the best match.
        best_cluster = None
        best_overlap = 0
        best_sim = 0.0
        for cluster, centroid in cluster_centroids.items():
            if centroid is None:
                continue
            sim = cosine_similarity(album_centroid.reshape(1, -1), centroid.reshape(1, -1))[0][0]
            # Get candidate tokens from cluster.
            candidate_tokens = set()
            if cluster.startswith("x"):
                decoded_cluster = decode_spawnre_hex(cluster)
                candidate_tokens = {token.strip() for token in decoded_cluster.split(",") if token.strip()}
            else:
                # For non-"x" clusters, treat the cluster key itself as the token.
                candidate_tokens = {cluster}
            
            overlap = len(album_tokens.intersection(candidate_tokens))
            # Select candidate with maximum overlap; if tied, choose one with higher similarity.
            if overlap > best_overlap or (overlap == best_overlap and sim > best_sim):
                best_overlap = overlap
                best_sim = sim
                best_cluster = cluster
        
        rel_album_path = os.path.relpath(album_m3u, start=os.path.join(spawn_root, "Spawn", "Playlists"))
        # Build metadata comment.
        meta_str = []
        if album_meta.get("genre"):
            meta_str.append("main genre: " + ", ".join(sorted(album_meta["genre"])))
        if album_meta.get("spawnre_hex"):
            token_counter = Counter()
            for hex_val in album_meta["spawnre_hex"]:
                decoded_str = decode_spawnre_hex(hex_val)
                tokens = [token.strip() for token in decoded_str.split(",") if token.strip()]
                token_counter.update(tokens)
            top_tokens = [token for token, count in token_counter.most_common(5)]
            meta_str.append("spawnre_hex: " + ", ".join(top_tokens))
        comment = f"# FAVORITE ALBUM: {rel_album_path}"
        if meta_str:
            comment += " | " + " | ".join(meta_str)
        
        if best_cluster is not None and best_sim >= similarity_threshold and best_overlap > 0:
            album_assignments.setdefault(best_cluster, []).append(comment)
            album_to_cluster[album_id] = best_cluster
        else:
            print(f"[WARN] Album {album_id} did not meet genre overlap criteria; assigning to outliers.")
            comment = f"# FAVORITE ALBUM: {rel_album_path} | WARNING: low genre overlap"
            if meta_str:
                comment += " | " + " | ".join(meta_str)
            album_assignments["outliers"].append(comment)
            album_to_cluster[album_id] = "outliers"

    return album_assignments, album_to_cluster



###############################################################################
# CURATION
###############################################################################

def run_curator(spawn_root: str, is_admin: bool = True):
    """
    Entry point for curation that loads tracks using track_loader,
    builds an embeddings dictionary from the merged tracks, and then proceeds
    with filtering, clustering, and ordering.

    Steps performed:
      1) Determine DB & load tracks
      2) Filter tracks by valid file existence & Spawn ID
      3) Ask if user wants to filter by favorites
      4) Group tracks by genre
      5) Optional merge of similar clusters
      6) If advanced => refine cluster membership + chain-based ordering
         Else => keep clusters as-is + random shuffle
      7) Write M3U
      8) Summarize results
      9) If advanced => optionally generate recommended playlist
    """
    # ------------------------------------------------------------------------
    # 0) Ask if user wants advanced curation
    # ------------------------------------------------------------------------
    choice = input("\nWould you like to run advanced curation? ([y]/n): ").strip().lower()
    do_advanced = (choice != "n")  # "y" or Enter => True, "n" => False

    # ------------------------------------------------------------------------
    # 1) Load tracks (catalog tracks only for admin mode; catalog & local tracks for user mode)
    # ------------------------------------------------------------------------
    user_mode = not is_admin
    tracks_df = load_and_merge(user_mode=user_mode, spawn_root=spawn_root)
    all_tracks_full = tracks_df.to_dict(orient="records")
    all_tracks = list(all_tracks_full)
    
    if not all_tracks:
        print("[INFO] No tracks found. Nothing to curate.")
        return

    # ------------------------------------------------------------------------
    # 2) Optionally filter by ownership
    # ------------------------------------------------------------------------
    own_filter = None
    own_input = input(
        "\nIf you'd like to filter by file ownership, enter one of the following options:\n"
        "   cat = (catalog) global catalog only, not in library\n"
        "   lib = (library) all library tracks\n"
        "   mat = (matched) library tracks matched to catalog\n"
        "   nom = (not matched) library tracks not matched to catalog\n"
        "   noo = (not owned) global catalog tracks not in library\n"
        "\nOtherwise, to include all tracks in catalog & library, just press Enter: "
    ).strip()
    if own_input.lower() in ["cat", "lib", "mat", "nom", "noo"]:
        own_filter = own_input.lower()
    elif own_input != "":
        print("[WARN] Invalid value provided; no ownership filtering will be applied.")

    if own_filter:
        before_origin_filter = len(all_tracks)
        if own_filter == "lib":
            # Library tracks include both matched ("mat") and non-matched ("nom")
            all_tracks = [t for t in all_tracks if t.get("origin") in ["mat", "nom"]]
        elif own_filter in ["cat", "mat", "nom"]:
            all_tracks = [t for t in all_tracks if t.get("origin") == own_filter]
        elif own_filter == "noo":
            # Admin mode only: from the global catalog (origin=="cat"),
            # exclude those that appear in the user library's cat_tracks.
            if is_admin:
                lib_db_path = os.path.join(spawn_root, "Spawn", "aux", "user", "spawn_library.db")
                try:
                    with sqlite3.connect(lib_db_path) as conn:
                        df_cat = pd.read_sql_query("SELECT spawn_id FROM cat_tracks", conn)
                    cat_ids = set(df_cat["spawn_id"].astype(str).str.strip())
                    all_tracks = [t for t in all_tracks if t.get("spawn_id", "").strip() not in cat_ids]
                except Exception as e:
                    print(f"[WARN] Could not apply @own:noo filter: {e}")
            else:
                # In user mode, fallback to filtering as if @own:cat was specified.
                all_tracks = [t for t in all_tracks if t.get("origin") == "cat"]
        print(f"[INFO] Applied @own filter '{own_filter}': {before_origin_filter} -> {len(all_tracks)} tracks")

    # ------------------------------------------------------------------------
    # 3) Filter to valid audio files
    # ------------------------------------------------------------------------
    filter_existing_files = True

    if filter_existing_files:
        original_count = len(all_tracks)
        valid_tracks = []
        missing_tracks = []
        for t in all_tracks:
            actual_path = track_file_exists(t, spawn_root)
            if actual_path:
                t["file_path"] = actual_path
                valid_tracks.append(t)
            else:
                missing_tracks.append(t)
        all_tracks = valid_tracks
        if original_count != len(all_tracks):
            print(f"[INFO] Filtered tracks by file existence: {original_count} -> {len(all_tracks)}")
            print(f"[INFO] Missing tracks: {len(missing_tracks)}")
            for track in missing_tracks:
                title = safe_extract_first(track, "©nam") or "Unknown Title"
                artist = safe_extract_first(track, "©ART") or "Unknown Artist"
                spawn_id = track.get("spawn_id", "N/A")
                print(f"  - Spawn ID: {spawn_id}, Title: {title}, Artist: {artist}")

    # In user mode, keep only tracks that have a Spawn ID
    if not is_admin:
        before_user_filter = len(all_tracks)
        all_tracks = [t for t in all_tracks if (t.get("spawn_id") or t.get("local_id"))]
        if before_user_filter != len(all_tracks):
            print(f"[INFO] Retained {len(all_tracks)} user tracks with a valid ID (catalog or local).")

    # ------------------------------------------------------------------------
    # 4) Optionally filter by favorites, recommendations, or none
    # ------------------------------------------------------------------------
    favorites_filter_desc = None
    suffix = ""
    excluded_album_m3us = []
    excluded_albums = set()

    # Save a copy of the full track set (after file and Spawn ID filtering)
    all_tracks_for_recs = list(all_tracks)

    # Determine if any favorites JSON files exist
    favs_folder = os.path.join(spawn_root, "Spawn", "aux", "user", "favs")
    has_favs = (
        os.path.isfile(os.path.join(favs_folder, "fav_artists.json")) or
        os.path.isfile(os.path.join(favs_folder, "fav_albums.json")) or
        os.path.isfile(os.path.join(favs_folder, "fav_tracks.json"))
    )

    print("\nFiltering options:")
    if has_favs:
        print("  1. Filter by favorites")
        print("  2. Filter by recommendations")
        print("  3. No filtering (use full catalog)")
        filter_choice = input("\nEnter selection (1, 2, and/or [3]): ").strip()
    else:
        print("[INFO] No favorites metadata found; using full catalog (no filtering).")
        filter_choice = "3"

    if filter_choice == "1" and has_favs:
        # --- Favorites filtering branch ---
        print("\nWhich favorites would you like to include?")
        print("    1. favorite artists")
        print("    2. favorite albums")
        print("    3. favorite tracks")

        # Prompt user for input, ignoring commas/spaces/other text
        fav_input = input("Enter your selection; any combination of 1, 2, and/or [3]: ").strip()
        if not fav_input:
            # Default to "3" (favorite tracks only) if user just hits Enter
            fav_options = {"3"}
        else:
            # Extract only the digits '1','2','3' from the string
            # so "2,3", "2 3", "23", or "2 blah 3" all become {"2", "3"}
            matches = re.findall(r'[1-3]', fav_input)
            fav_options = set(matches)
            if not fav_options:
                # If user typed something but no digits 1-3, skip filtering
                print("[WARN] No valid favorites options recognized. Proceeding without favorites filtering.")
                fav_options = set()  # empty => skip filtering logic

        if fav_options:
            # This function updates the track set by removing tracks that match favorite albums (for example)
            all_tracks, excluded_albums, excluded_tracks = filter_tracks_by_favorites(all_tracks, spawn_root, fav_options)

            # Convert (artist_lower, album_lower) -> album_m3u_path
            excluded_album_m3us = []
            for (a_lower, alb_lower) in excluded_albums:
                artist_display = a_lower.title()
                album_display = alb_lower.title()
                album_m3u_path = os.path.join(
                    spawn_root, "Spawn", "aux", "user", "albm",
                    f"{artist_display} - {album_display}.m3u"
                )
                #print(f"[DEBUG] Generated Album M3U path: {album_m3u_path}")
                excluded_album_m3us.append(album_m3u_path)

            # Special branch: if only favorite albums were selected, build album blocks from the full library.
            if fav_options == {"2"}:

                embeddings_temp = build_embeddings_dict(all_tracks_full)

                # Build album comments (this uses the full list of album paths).
                album_comments = build_album_comments(excluded_album_m3us, all_tracks_full, spawn_root)

                # Helper: Define a function to preserve the album track order.
                def get_track_order(t: dict):
                    disk = 1
                    track_num = 0
                    try:
                        disk_tag = t.get("disk")
                        if disk_tag and isinstance(disk_tag, list) and disk_tag:
                            disk = int(disk_tag[0][0])
                    except Exception:
                        pass
                    try:
                        track_tag = t.get("trkn")
                        if track_tag and isinstance(track_tag, list) and track_tag:
                            track_num = int(track_tag[0][0])
                    except Exception:
                        pass
                    return (disk, track_num)

                # Build album blocks: For each favorite album, retrieve its full track list (preserving order)
                album_blocks = []
                for album in excluded_albums:
                    album_tracks = [
                        t for t in all_tracks_full
                        if (safe_extract_first(t, "©ART", normalize=True) == album[0] and
                            safe_extract_first(t, "©alb", normalize=True) == album[1])
                    ]
                    if album_tracks:
                        # Sort tracks in album order (using disk and track number)
                        album_tracks = sorted(album_tracks, key=get_track_order)
                        album_comment = album_comments.get(album, "")
                        album_block = {
                            "is_album": True,
                            "album_tracks": album_tracks,
                            "album_comment": album_comment
                        }
                        album_blocks.append(album_block)

                # Chain-based ordering on album blocks (so adjacent albums have similar embeddings)
                chain_ordered = chain_based_curation(album_blocks, embeddings_temp, None)
                #chain_ordered = chain_based_curation(album_blocks, embeddings, last_track_embedding)
                curated_clusters = [("favorite albums", chain_ordered)]

                final_m3u_path = write_curated_m3u(
                    spawn_root,
                    curated_clusters,
                    favorites_filter_desc="favorite albums",
                    suffix="_fav_alb",
                    excluded_album_m3us=excluded_album_m3us,
                    all_tracks=all_tracks_full,
                    album_assignments=None
                )
                print(f"[INFO] Curated M3U created at: {final_m3u_path}")
                return

            # For non-"only albums" favorites (e.g. favorite tracks or favorite artists/tracks):
            if not all_tracks:
                print("[INFO] No tracks remain after filtering and no excluded albums to reference. Aborting.")
                return

            # Build a user-friendly description, e.g. "favorite artists & favorite albums"
            desc_parts = []
            if "1" in fav_options:
                desc_parts.append("favorite artists")
            if "2" in fav_options:
                desc_parts.append("favorite albums")
            if "3" in fav_options:
                desc_parts.append("favorite tracks")
            favorites_filter_desc = " & ".join(desc_parts)

            # Build suffix for M3U filename
            option_to_suffix = {"1": "art", "2": "alb", "3": "trk"}
            custom_order = ["3", "2", "1"]
            suffix_parts = [option_to_suffix[opt] for opt in custom_order if opt in fav_options]
            # Build the suffix using "fav" as the base.
            suffix = "_fav" + "".join(f"_{part}" for part in suffix_parts)

        else:
            favorites_filter_desc = None
            suffix = ""

    elif filter_choice == "2" and has_favs:
        # --- Recommendation filtering branch (based on favorites ratings) ---
        # Load favorites JSON to build explicit ratings.
        fav_artists = load_favs_artists(os.path.join(favs_folder, "fav_artists.json"))
        fav_albums  = load_favs_albums(os.path.join(favs_folder, "fav_albums.json"))
        fav_tracks  = load_favs_tracks(os.path.join(favs_folder, "fav_tracks.json"))
        explicit_ratings = {}
        # Build explicit ratings from favorite tracks
        for item in fav_tracks:
            if isinstance(item, dict):
                spid = item.get("spawn_id")
            else:
                spid = str(item)
            if spid:
                explicit_ratings[spid.strip()] = 1

        # Load embeddings
        embeddings_temp = build_embeddings_dict(all_tracks_full)
        threshold = 0.98
        #threshold = 0.982

        # Use compute_like_likelihoods to get scores
        like_scores = compute_like_likelihoods(embeddings_temp, explicit_ratings, sigma=0.15)

        recommended_tracks = []
        for track in all_tracks_for_recs:
            sid = (track.get("spawn_id") or track.get("local_id") or "").strip()
            if sid in explicit_ratings:
                continue

            if sid in like_scores:
                score = like_scores[sid]

                # DEBUG: Print the like_likelihood score for each candidate
                title = safe_extract_first(track, "©nam", normalize=False) or "Untitled"
                artist = safe_extract_first(track, "©ART", normalize=False) or "Unknown Artist"
                #print(f"[DEBUG] {artist} - {title} [{sid}] => like_likelihood = {score:.4f}")
                
                if score < 1.0 and score >= threshold:
                    recommended_tracks.append(track)
        all_tracks = recommended_tracks
        favorites_filter_desc = f"recommended (like_likelihood >= {threshold:.3f})"
        suffix = "_recs"

        #print(f"[DEBUG] Total recommended tracks (before extra filtering): {len(all_tracks)}")
        extra_filter = input(
            "\nIf you'd like to restrict recommendations to specific genres and/or artists,\n"
            "enter '@gen' or '@art' followed by a comma-separated list of values.\n"
            "Otherwise, just press Enter to continue: "
        ).strip()
        if extra_filter:
            extra_filter_str = extra_filter
            extra_filter_lower = extra_filter.lower()
            filtered_tracks = []
            if extra_filter_lower.startswith("@gen"):
                genres_str = extra_filter[4:].strip()
                genres_list = [g.strip().lower() for g in genres_str.split(",") if g.strip()]
                for track in all_tracks:
                    gen_tag = safe_extract_first(track, "©gen", normalize=True) or ""
                    spawnre_tag = safe_extract_first(track, "----:com.apple.iTunes:spawnre", normalize=True) or ""
                    combined_genres = (gen_tag + " " + spawnre_tag).lower()
                    if combined_genres and any(g in combined_genres for g in genres_list):
                        filtered_tracks.append(track)
                all_tracks = filtered_tracks
            elif extra_filter_lower.startswith("@art"):
                artists_str = extra_filter[4:].strip()
                artists_list = [a.strip().lower() for a in artists_str.split(",") if a.strip()]
                for track in all_tracks:
                    artist = safe_extract_first(track, "©ART", normalize=True) or ""
                    if artist.lower() and any(a in artist.lower() for a in artists_list):
                        filtered_tracks.append(track)
                all_tracks = filtered_tracks
            else:
                print("[WARN] Unrecognized extra filter format; skipping extra filtering.")
            favorites_filter_desc += f" | Extra filter: {extra_filter_str}"

    elif filter_choice == "3" or not filter_choice:
        # --- No filtering branch ---
        favorites_filter_desc = None
        suffix = ""
    else:
        # Fallback if invalid selection or favorites metadata is missing.
        favorites_filter_desc = None
        suffix = ""

    # ------------------------------------------------------------------------
    # 5) Group into clusters by spawnre/©gen
    # ------------------------------------------------------------------------
    base_clusters = group_tracks_by_genre(all_tracks)
    #print(f"[DEBUG] Final track count passed to M3U writer: {len(all_tracks)}")
    #print(f"\n[DEBUG] Initial genre cluster breakdown (before merging):")
    for genre_key, cluster_tracks in base_clusters.items():
        print(f"  - Cluster '{genre_key}': {len(cluster_tracks)} tracks")

    # ------------------------------------------------------------------------
    # 6) Optional merge of similar clusters via embeddings
    # ------------------------------------------------------------------------
    merge_choice = input("\nMerge similar clusters? ([y]/n): ").strip().lower()

    # Load both catalog and user embeddings
    catalog_embed_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "mp4tovec.p")
    user_embed_path    = os.path.join(spawn_root, "Spawn", "aux", "user", "mp4tovec_local.p")

    catalog_embeds = {}
    user_embeds = {}

    if os.path.isfile(catalog_embed_path):
        with open(catalog_embed_path, "rb") as f:
            catalog_embeds = pickle.load(f)

    if os.path.isfile(user_embed_path):
        with open(user_embed_path, "rb") as f:
            user_embeds = pickle.load(f)
            #print(f"[DEBUG] Sample keys in user_embeds: {list(user_embeds.keys())[:5]}")

    all_embed_dict = {**catalog_embeds, **user_embeds}

    embeddings = all_embed_dict
    #embeddings = build_embeddings_dict(all_tracks_full)  # Build embeddings dict from merged tracks.

    local_ids = [t.get("local_id") for t in all_tracks_full if t.get("origin") == "nom"]
    #print(f"[DEBUG] Sample local_ids from all_tracks_full: {local_ids[:5]}")
    normalized_embed_keys = {k.strip().lower(): v for k, v in all_embed_dict.items()}
    embeddings = normalized_embed_keys
    present = [lid for lid in (lid.strip().lower() for lid in local_ids if lid) if lid in embeddings]
    print(f"[INFO] Local ID embeddings present: {len(present)} of {len(local_ids)}\n")

    if merge_choice in ["", "y"]:
        print("    Merging similar clusters...")
        # Since embeddings are already attached, we can pass our dictionary directly.
        base_clusters = merge_similar_clusters_with_breadth(base_clusters, embeddings,
                                                             centroid_threshold=0.97,
                                                             breadth_tolerance=0.5)
        #print("Similar clusters have been merged.")

    # ------------------------------------------------------------------------
    # 7) (Advanced-Only) Load embeddings & refine cluster membership
    # ------------------------------------------------------------------------
    curated_clusters = []
    last_track_embedding = None

    if do_advanced:

        # DEBUG
        #print("[DEBUG] Cluster sizes before refinement:")
        for g, tracks in base_clusters.items():
            print(f"  - Cluster '{g}': {len(tracks)} tracks")

        # 7a) Refine membership
        refined_clusters = refine_clusters_by_embeddings(
            all_tracks, embeddings, base_clusters, distance_threshold=0.15
        )

        # DEBUG
        #print("[DEBUG] Cluster sizes after refinement:")
        for g, tracks in refined_clusters.items():
            print(f"  - Cluster '{g}': {len(tracks)} tracks")

        # 7b) Sort clusters, placing "outliers" last
        all_genre_keys = sorted(g for g in refined_clusters.keys() if g != "outliers")
        if "outliers" in refined_clusters:
            all_genre_keys.append("outliers")

        # Precompute album comments and assign albums to clusters
        album_comments = build_album_comments(excluded_album_m3us, all_tracks_full, spawn_root)
        cluster_centroids = compute_cluster_centroids(refined_clusters, embeddings)
        album_assignments, album_to_cluster = assign_albums_to_clusters(
            excluded_album_m3us,
            all_tracks_full,
            embeddings,
            cluster_centroids,
            spawn_root,
            similarity_threshold=0.0  # or choose a threshold if desired
        )

        # 7c) Chain-based ordering with album blocks
        for genre in all_genre_keys:
            # Get the refined tracks for this genre.
            track_list = refined_clusters[genre]
            # Get album blocks for this genre using the precomputed album_comments.
            album_blocks = get_album_candidate_blocks_for_cluster(
                genre, 
                excluded_albums, 
                all_tracks_full, 
                embeddings, 
                album_comments, 
                album_to_cluster
            )
            # Combine normal tracks with album blocks.
            combined_candidates = track_list + album_blocks
            # Order candidates with chain-based ordering
            #print(f"[DEBUG] Refining genre cluster '{genre}': {len(track_list)} tracks + {len(album_blocks)} album blocks")
            chain_ordered = chain_based_curation(combined_candidates, embeddings, last_track_embedding)
            #print(f"[DEBUG] => Result: {len(chain_ordered)} tracks kept after chain-based ordering")

            curated_clusters.append((genre, chain_ordered))

            # Update last_track_embedding based on the final track in the chain.
            if chain_ordered:
                last_track = chain_ordered[-1]
                sid = last_track.get("spawn_id") or last_track.get("local_id")
                last_track_embedding = embeddings.get(sid)

    else:
        # BASIC PATH
        # 7a) Order clusters by relationships
        ordered_genres = order_clusters_by_relationships(base_clusters)
        if not ordered_genres:
            print("[ERROR] Could not order clusters by relationship. Exiting.")
            return

        # 7b) Allow user to optionally modify cluster order
        ordered_genres = maybe_modify_cluster_order(ordered_genres, base_clusters, spawn_root)

        # 7c) Shuffle each cluster
        for genre in ordered_genres:
            track_list = base_clusters[genre]
            random.shuffle(track_list)
            curated_clusters.append((genre, track_list))

    # After track ordering, decide which clusters to use for album assignments:
    if do_advanced:
        clusters_for_albums = refined_clusters  # defined in advanced mode
    else:
        clusters_for_albums = base_clusters      # basic mode

    # ------------------------------------------------------------------------
    # 8) Write M3U
    # ------------------------------------------------------------------------

    #print(f"[DEBUG] Final track count passed to M3U writer: {len(all_tracks)}")
    final_m3u_path = write_curated_m3u(
        spawn_root,
        curated_clusters,
        favorites_filter_desc=favorites_filter_desc,
        suffix=suffix,
        excluded_album_m3us=excluded_album_m3us,
        all_tracks=all_tracks_full,
        album_assignments=album_assignments if do_advanced else None
    )

    if final_m3u_path:
        msg = "Advanced" if do_advanced else "Basic"
        print(f"[INFO] {msg} curated M3U created at: {final_m3u_path}")
    else:
        print("[ERROR] Could not write the curated M3U playlist.")
        return

    # ------------------------------------------------------------------------
    # 9) Summaries
    # ------------------------------------------------------------------------
    if do_advanced:
        print("\n[Summary of refined clusters]:")
    else:
        print("\n[Summary of curated clusters]:")

    for g, tracks in curated_clusters:
        print(f"\n  * {g} => {len(tracks)} tracks")
        # If spawnre_hex
        if g.startswith("x"):
            decoded = decode_spawnre_hex(g)
            print(f"    {decoded}")


###############################################################################
# CHAIN-BASED ORDERING
###############################################################################

def get_embedding_key(track_dict):
    sid = track_dict.get("spawn_id") or track_dict.get("local_id") or ""
    return sid.strip().lower()

def chain_based_curation(
    candidates: List[dict],
    embeddings: Dict[str, np.ndarray],
    last_track_embedding: Optional[np.ndarray] = None
) -> List[dict]:
    """
    Orders candidate items (individual tracks and album blocks) into a smooth chain based on embeddings.
    
    For a normal track, its own embedding is used. For an album block (with "is_album": True),
    the embedding of its first track is used to measure proximity.
    
    This updated implementation uses a single unified candidate pool so that album blocks are
    inserted at appropriate positions amongst non-album tracks. When an album block is chosen,
    its entire block is inserted, with the header and footer markers, and the current embedding
    is updated to the embedding of its last track.
    
    Returns a list of items (mix of track dicts and marker dicts).
    Marker dicts have "is_album_marker": True.
    """
    import random
    remaining = candidates.copy()
    ordered = []

    #print(f"[DEBUG] Starting chain curation with {len(remaining)} candidates.")

    # Initialize current_embedding: if none, choose a random candidate.
    if last_track_embedding is None and remaining:
        chosen_index = random.randrange(len(remaining))
        candidate = remaining.pop(chosen_index)
        if candidate.get("is_album"):
            header = {
                "is_album_marker": True,
                "marker_type": "header",
                "album_comment": candidate["album_comment"]
            }
            ordered.append(header)
            ordered.extend(candidate["album_tracks"])
            footer = {
                "is_album_marker": True,
                "marker_type": "footer"
            }
            ordered.append(footer)
            album_last = candidate["album_tracks"][-1]
            last_track_embedding = embeddings.get(get_embedding_key(album_last))
        else:
            ordered.append(candidate)
            last_track_embedding = embeddings.get(get_embedding_key(candidate))

    # Process all remaining candidates in one unified loop.
    while remaining and last_track_embedding is not None:
        best_candidate = None
        best_index = None
        best_distance = float("inf")

        for i, candidate in enumerate(remaining):
            # Determine candidate embedding based on candidate type.
            if candidate.get("is_album"):
                first_track = candidate["album_tracks"][0]
                cand_emb = embeddings.get(get_embedding_key(first_track))
            else:
                cand_emb = embeddings.get(get_embedding_key(candidate))
            if cand_emb is None:
                track_name = first_track.get("©nam") if candidate.get("is_album") else candidate.get("©nam")
                print(f"[DEBUG] Skipping candidate '{name}' — no embedding found (key: {get_embedding_key(first_track or candidate)})")
                continue

            # Compute cosine similarity (or distance)
            dot_val = np.dot(last_track_embedding, cand_emb)
            denom = np.linalg.norm(last_track_embedding) * np.linalg.norm(cand_emb)
            if denom == 0:
                track_name = first_track.get("©nam") if candidate.get("is_album") else candidate.get("©nam")
                print(f"[DEBUG] Skipping candidate '{track_name}' — zero norm in embedding")
                continue
            similarity = dot_val / denom
            distance = 1 - similarity

            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate
                best_index = i

        if best_candidate is None:
            print("[DEBUG] No more valid candidates could be chained; ending curation.")
            break  # No valid candidate found.
        # Remove the chosen candidate from remaining.
        remaining.pop(best_index)

        # Insert candidate and update current embedding.
        if best_candidate.get("is_album"):
            header = {
                "is_album_marker": True,
                "marker_type": "header",
                "album_comment": best_candidate["album_comment"]
            }
            ordered.append(header)
            ordered.extend(best_candidate["album_tracks"])
            footer = {
                "is_album_marker": True,
                "marker_type": "footer"
            }
            ordered.append(footer)
            album_last = best_candidate["album_tracks"][-1]
            last_track_embedding = embeddings.get(get_embedding_key(album_last))

            album_name = best_candidate["album_comment"]
            print(f"[DEBUG] Added album block: '{album_name}' with {len(best_candidate['album_tracks'])} tracks")
        else:
            ordered.append(best_candidate)
            last_track_embedding = embeddings.get(get_embedding_key(best_candidate))

            track_title = best_candidate.get("©nam") or "Untitled"
            track_artist = best_candidate.get("©ART") or "Unknown Artist"
            #print(f"[DEBUG] Added track: {track_artist} - {track_title} [{get_embedding_key(best_candidate)}], distance={best_distance:.4f}")

    #print(f"[DEBUG] [chain_based_curation] Finished with {len(ordered)} items in final chain.")

    return ordered


###############################################################################
# REFINING CLUSTERS WITH EMBEDDINGS
###############################################################################

def refine_clusters_by_embeddings(
    all_tracks: List[dict],
    embeddings: Dict[str, np.ndarray],
    base_clusters: Dict[str, List[dict]],
    distance_threshold: float = 0.4,
) -> Dict[str, List[dict]]:
    """
    1) Compute centroid for each base cluster
    2) For each track, measure cosine distance to each centroid
    3) Assign track to the cluster of its best (lowest) distance if within threshold
       Otherwise put it in 'outliers'
    """

    print("\nRefining clusters by embeddings...")
    #print("[DEBUG] ========== refine_clusters_by_embeddings() using cosine distance ==========")
    centroids = compute_cluster_centroids(base_clusters, embeddings)
    refined = defaultdict(list)
    refined["outliers"] = []

    #print(f"[DEBUG] distance_threshold = {distance_threshold:.2f}")

    for track_info in all_tracks:
        sid_raw = track_info.get("spawn_id") or track_info.get("local_id")
        sid = sid_raw.strip().lower() if sid_raw else None

        # If no embedding, it goes to outliers
        if not sid or sid not in embeddings:
            print(f"[DEBUG] Track {sid_raw} => no embedding => outliers")
            refined["outliers"].append(track_info)
            continue

        track_emb = embeddings[sid]
        track_name = safe_extract_first(track_info, "©nam") or "<no title>"
        best_cluster = None
        best_dist = float("inf")
        track_norm = np.linalg.norm(track_emb)

        # Compare to each centroid
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
            #print(f"[DEBUG] Track {sid} \"{track_name}\" => best cluster='{best_cluster}', dist={best_dist:.4f}")
            refined[best_cluster].append(track_info)
        else:
            #print(f"[DEBUG] Track {sid} \"{track_name}\" => outliers (best_dist={best_dist:.4f} > {distance_threshold:.2f})")
            refined["outliers"].append(track_info)

    return dict(refined)


def compute_cluster_centroids(
    clusters: Dict[str, List[dict]],
    embeddings: Dict[str, np.ndarray]
) -> Dict[str, Optional[np.ndarray]]:
    """
    For each cluster, average the embeddings of its tracks to form a centroid.
    Returns a dict {genre -> centroid_vector} or None if no embeddings in cluster.
    """
    #print("[DEBUG] ====== compute_cluster_centroids() ======")
    centroids = {}
    for genre, track_list in clusters.items():
        emb_list = []
        for t in track_list:
            sid = t.get("spawn_id") or t.get("local_id")
            if sid in embeddings:
                emb_list.append(embeddings[sid])

        if emb_list:
            arr = np.vstack(emb_list)
            centroid = arr.mean(axis=0)
            centroids[genre] = centroid
            #print(f"[DEBUG] genre='{genre}': {len(emb_list)} embeddings => centroid first 5 dims: {centroid[:5]}")
        else:
            centroids[genre] = None
            #print(f"[DEBUG] genre='{genre}': 0 embeddings => centroid=None")

    return centroids


###############################################################################
# Optional favorites filtering
###############################################################################

def filter_tracks_by_favorites(all_tracks: List[dict], spawn_root: str, fav_options: set) -> Tuple[List[dict], set, List[dict]]:
    """
    Applies multi-category favorites filtering to 'all_tracks'.

    Interpretation:
      - "1" => Keep tracks if they're by a favorite artist.
      - "2" => EXCLUDE tracks if they're on a favorite album (unless selected alone).
      - "3" => Keep tracks if they're a favorite track (by ID or A-A-T triple).

    If the user chose multiple options (e.g. {"1","2"}), then we keep a track if it's a favorite artist (1),
    but if it's on a favorite album (2), we remove it (even if it also qualifies as a favorite artist).
    
    Returns a tuple:
      (filtered_tracks, excluded_albums, excluded_tracks)
        - filtered_tracks: list of tracks that pass the filter.
        - excluded_albums: set of (artist_lower, album_lower) tuples for which tracks were excluded.
        - excluded_tracks: list of track dictionaries that were excluded due to favorite album match.
    """

    # 1) Load favorites data from JSON
    favs_folder = os.path.join(spawn_root, "Spawn", "aux", "user", "favs")
    fav_artists = load_favs_artists(os.path.join(favs_folder, "fav_artists.json"))
    fav_albums  = load_favs_albums(os.path.join(favs_folder, "fav_albums.json"))
    fav_tracks  = load_favs_tracks(os.path.join(favs_folder, "fav_tracks.json"))

    # 2) Build sets for quick lookups
    artist_names_set = set()
    artist_mbids_set = set()
    for item in fav_artists:
        if isinstance(item, dict):
            art_lower = item.get("artist", "").strip().lower()
            if art_lower:
                artist_names_set.add(art_lower)
            mbid = item.get("artist_mbid", "")
            if mbid:
                artist_mbids_set.add(mbid.strip().lower())
        else:
            art_lower = str(item).strip().lower()
            artist_names_set.add(art_lower)

    album_pairs_set = set()
    for item in fav_albums:
        if isinstance(item, dict):
            art = item.get("artist", "").strip().lower()
            alb = item.get("album", "").strip().lower()
            album_pairs_set.add((art, alb))
        else:
            alb = str(item).strip().lower()
            album_pairs_set.add(("", alb))

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
            track_triples.add(("", "", str(item).strip().lower()))

    # 3) Determine which user-chosen favorites apply
    do_artists = "1" in fav_options  # keep tracks by favorite artists
    do_albums  = "2" in fav_options  # normally exclude tracks on favorite albums
    do_tracks  = "3" in fav_options  # keep tracks that are favorite tracks

    excluded_albums = set()   # to collect album IDs (artist, album) of excluded tracks
    excluded_tracks = []      # to collect the track info that are excluded

    # 4) Loop over tracks and check if they match any chosen category
    filtered = []
    for track_info in all_tracks:
        spawn_id = track_info.get("spawn_id", "").strip().lower()
        artist_name = safe_extract_first(track_info, "©ART") or ""
        album_name  = safe_extract_first(track_info, "©alb") or ""
        track_name  = safe_extract_first(track_info, "©nam") or ""

        art_lower = artist_name.strip().lower()
        alb_lower = album_name.strip().lower()
        trk_lower = track_name.strip().lower()

        artist_mbid_val = safe_extract_first(track_info, "----:com.apple.iTunes:MusicBrainz Artist Id") or ""
        artist_mbid_val = artist_mbid_val.strip().lower()

        is_fav_artist = (art_lower in artist_names_set) or (artist_mbid_val in artist_mbids_set)
        is_fav_album  = (art_lower, alb_lower) in album_pairs_set
        is_fav_track  = (spawn_id in track_spawn_ids) or ((art_lower, alb_lower, trk_lower) in track_triples)

        # Start with keep=False
        keep = False

        # Special handling if ONLY favorite albums option is selected.
        if fav_options == {"2"}:
            if is_fav_album:
                keep = True
                excluded_albums.add((art_lower, alb_lower))
            else:
                keep = False
        else:
            # Combined options logic:
            if do_artists and is_fav_artist:
                keep = True
            if do_tracks and is_fav_track:
                keep = True
            if do_albums and is_fav_album:
                # When album option is combined with others, we exclude favorite album tracks.
                keep = False
                excluded_albums.add((art_lower, alb_lower))
                excluded_tracks.append(track_info)
        if keep:
            filtered.append(track_info)

    return filtered, excluded_albums, excluded_tracks


def load_favs_artists(filepath: str) -> List[dict]:
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
            #fallback_gen = safe_extract_first(track_info, "©gen")
            g = fallback_gen.lower() if fallback_gen else "unknown"
        clusters[g].append(track_info)
    return dict(clusters)


def merge_similar_clusters_with_breadth(clusters: Dict[str, List[dict]], embeddings: Dict[str, np.ndarray],
                                          centroid_threshold: float = 0.97, breadth_tolerance: float = 0.5) -> Dict[str, List[dict]]:
    cluster_stats = {}
    def extract_genre_tokens(track_list):
        genre_tokens = set()
        for track in track_list:
            if "spawnre" in track and track["spawnre"]:
                genre_tokens.update(track["spawnre"].split(","))
            if "spawnre_hex" in track and track["spawnre_hex"]:
                hex_decoded = decode_spawnre_hex(track["spawnre_hex"])
                genre_tokens.update(hex_decoded.split(","))
        if not genre_tokens:
            genre_tokens.add("unknown_genre")
        return genre_tokens

    for cid, tracks in clusters.items():
        emb_list = [
            embeddings[t.get("spawn_id") or t.get("local_id")]
            for t in tracks
            if (t.get("spawn_id") or t.get("local_id")) in embeddings
        ]
        if emb_list:
            arr = np.vstack(emb_list)
            centroid = arr.mean(axis=0)
            breadth = sum(np.linalg.norm(emb - centroid) for emb in emb_list) / len(emb_list)
            genre_tokens = extract_genre_tokens(tracks)
            cluster_stats[cid] = (centroid, breadth, genre_tokens)
        else:
            cluster_stats[cid] = (None, None, {"unknown_genre"})
    labels = list(cluster_stats.keys())
    merged_clusters = {}
    merged = set()
    for i, label_i in enumerate(labels):
        if label_i in merged:
            continue
        current_cluster = clusters[label_i][:]
        centroid_i, breadth_i, genres_i = cluster_stats[label_i]
        for j in range(i + 1, len(labels)):
            label_j = labels[j]
            if label_j in merged:
                continue
            centroid_j, breadth_j, genres_j = cluster_stats[label_j]
            if centroid_i is None or centroid_j is None:
                continue
            sim = cosine_similarity(centroid_i.reshape(1, -1), centroid_j.reshape(1, -1))[0][0]
            if sim > centroid_threshold:
                if breadth_i is not None and breadth_j is not None:
                    avg_breadth = (breadth_i + breadth_j) / 2
                    rel_diff = abs(breadth_i - breadth_j) / avg_breadth if avg_breadth > 0 else 0
                    if rel_diff > breadth_tolerance:
                        continue
                common_genres = genres_i.intersection(genres_j)
                if len(common_genres) >= 1:
                    current_cluster.extend(clusters[label_j])
                    merged.add(label_j)
        merged_clusters[label_i] = current_cluster
    return merged_clusters


def normalize_tag_value(val: str) -> str:
    """
    Normalizes a tag value for comparison purposes:
      - Strips leading/trailing whitespace,
      - Converts to lowercase,
      - Replaces problematic punctuation characters with nothing.
      
    Adjust the regex as needed for your use case.
    """
    normalized = val.strip().lower()
    # Remove common problematic punctuation (you can also replace them with underscores if preferred)
    normalized = re.sub(r'[\\/:*?"<>|]', '', normalized)
    return normalized


def safe_extract_first(tag_dict: dict, key: str, normalize: bool = True) -> Optional[str]:
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
            first_item = first_item.decode("utf-8", errors="replace")
        result = str(first_item).strip()
    elif isinstance(raw, bytes):
        result = raw.decode("utf-8", errors="replace").strip()
    else:
        result = str(raw).strip()

    if normalize:
        return normalize_tag_value(result)
    else:
        return result


###############################################################################
# Order clusters by related genre logic
###############################################################################


def order_clusters_by_relationships(clusters: Dict[str, List[dict]]) -> List[str]:
    """
    1) Sort by descending cluster size
    2) Then chain them by "related" genre logic
    3) Return a list of genre keys in an "intuitive" order
    """

    # Sort by size
    cluster_counts = {g: len(tracks) for g, tracks in clusters.items()}
    sorted_by_size = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_by_size:
        return []

    ordered_genres = []
    used = set()

    # Start with the largest cluster
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
            # fallback to next-largest unused
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
    Looks up "Related" codes in genre_mapping if genre_name matches one of the known entries.
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
    Allows user to see/optionally reorder clusters.
    - Can load a previously saved custom order
    - Can manually specify a new comma-separated list
    - Or skip reordering
    """

    print("\nCurrent clusters in M3U order:")
    for genre in ordered_genres:
        print(f" - {genre}: {len(clusters[genre])} tracks")

    ans = input("\nWould you like to modify the genre order? (y/[n]): ").strip().lower()
    if ans == "":
        ans = "n"
    if ans not in ["y", "n"]:
        print("[WARN] Invalid input, skipping reordering.")
        return ordered_genres
    if ans == "n":
        return ordered_genres

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
            return ordered_genres

        if choice == "1":
            new_order = load_custom_cluster_order(spawn_root)
            if not new_order:
                print("[INFO] No custom cluster order found or invalid. Returning to menu.")
                continue
            validated_order = [g for g in new_order if g in clusters]
            leftover = [g for g in ordered_genres if g not in validated_order]
            final_order = validated_order + leftover
            print("\nUpdated order from custom file:")
            for genre in final_order:
                print(f" - {genre}: {len(clusters[genre])} tracks")
            return final_order

        if choice == "2":
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

            validated_order = [g for g in new_order_list if g in clusters]
            invalid_genres = [g for g in new_order_list if g not in clusters]
            if invalid_genres:
                print(f"[WARN] The following genres aren't present, ignoring: {', '.join(invalid_genres)}")

            leftover = [g for g in ordered_genres if g not in validated_order]
            final_order = validated_order + leftover

            print("\nUpdated cluster order:")
            for genre in final_order:
                print(f" - {genre}: {len(clusters[genre])} tracks")

            ans_save = input("\nWould you like to save this new order for future use? (y/[n]): ").strip().lower()
            if ans_save == "":
                ans_save = "n"
            if ans_save == "y":
                save_custom_cluster_order(spawn_root, final_order)
            return final_order


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
# Build final file paths & write .m3u
###############################################################################

def write_curated_m3u(
    spawn_root: str, 
    curated_clusters: List[Tuple[str, List[dict]]],
    favorites_filter_desc: Optional[str] = None,
    suffix: str = "",
    excluded_album_m3us: Optional[List[str]] = None,
    all_tracks: Optional[List[dict]] = None,
    album_assignments: Optional[Dict[str, List[str]]] = None
) -> Optional[str]:
    """
    Save a curated M3U to: 
      spawn_root/Spawn/Playlists/Curated/curate_YYYY-MM-DD_HHMMSS{suffix}.m3u

    Inserts #EXTM3U at the top, plus an optional comment about favorites.
    If excluded_album_m3us is provided (a list of album M3U paths) and all_tracks
    is provided (the full track library), then for each album the function attempts 
    to gather consolidated genre metadata and prints a comment below the album reference.
    """
    curated_dir = os.path.join(spawn_root, "Spawn", "Playlists", "Curated")
    os.makedirs(curated_dir, exist_ok=True)

    ts = time.strftime("%Y-%m-%d_%H%M%S")
    m3u_name = f"cur8{suffix}_{ts}.m3u"
    m3u_path = os.path.join(curated_dir, m3u_name)

    try:
        with open(m3u_path, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            if favorites_filter_desc:
                f.write(f"# Filtered by {favorites_filter_desc}\n")
            f.write("\n")

            for (genre_key, track_list) in curated_clusters:
                # Write a header for the cluster
                if genre_key.startswith("x"):
                    f.write(f"# SPAWNRE_HEX: {genre_key}\n")
                    decoded = decode_spawnre_hex(genre_key)
                    f.write(f"# GENRES: {decoded}\n")
                else:
                    f.write(f"# GENRE: {genre_key}\n")

                # # Insert album M3U comments (if any) for this cluster.
                # if album_assignments and genre_key in album_assignments:
                #     for album_comment in album_assignments[genre_key]:
                #         f.write(f"{album_comment}\n")

                # Write each track (or album block markers and tracks)
                for item in track_list:
                    if isinstance(item, dict) and item.get("is_album_marker"):
                        if item.get("marker_type") == "header":
                            # Use "album_comment" if available.
                            album_comment = item.get("album_comment", "")
                            f.write(f"{album_comment}\n")
                            f.write("# fav_albm_srt\n")
                        elif item.get("marker_type") == "footer":
                            f.write("# fav_albm_end\n")
                    else:
                        # Recalculate the file path on the fly
                        actual_path = track_file_exists(item, spawn_root)
                        if actual_path:
                            rel_path = os.path.relpath(
                                actual_path, 
                                start=os.path.join(spawn_root, "Spawn", "Playlists")
                            )
                            f.write(f"{rel_path}\n")
                        else:
                            print(f"[WARNING] Track not found for M3U: {item.get('©nam', 'Unknown')}")

                f.write("\n")

        return m3u_path
    except Exception as e:
        logger.error(f"Error writing curated M3U => {e}")
        return None


###############################################################################
# FILE PATH CHECK
###############################################################################

def track_file_exists(track_info: dict, spawn_root: str) -> Optional[str]:
    """
    Checks if the audio file for the track exists. Returns the actual path if found,
    otherwise None.
    """

    music_dir = os.path.join(spawn_root, "Spawn", "Music")

    # Get raw values for file construction
    artist = safe_extract_first(track_info, "©ART", normalize=False) or "Unknown"
    album  = safe_extract_first(track_info, "©alb", normalize=False) or "Unknown"
    title  = safe_extract_first(track_info, "©nam", normalize=False) or "Untitled"
    spawn_id = track_info.get("spawn_id") or track_info.get("local_id") or "unknown"

    disc_tag = track_info.get("disk")
    track_tag = track_info.get("trkn")
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

    # Build filename
    file_name = build_d_tt_spawn_id_title_filename(disc_main, track_main, spawn_id, title)
    
    artist_dir = sanitize_for_directory(artist)
    album_dir = sanitize_for_directory(album)
    
    abs_path = os.path.join(music_dir, artist_dir, album_dir, file_name)
    if os.path.isfile(abs_path):
        return abs_path

    # Try a sanitized fallback (ASCII only)
    sanitized_artist = sanitize_for_directory_ascii(artist)
    sanitized_album  = sanitize_for_directory_ascii(album)
    sanitized_title  = sanitize_for_directory_ascii(title)
    sanitized_file_name = build_d_tt_spawn_id_title_filename(disc_main, track_main, spawn_id, sanitized_title)
    
    sanitized_path = os.path.join(music_dir, sanitized_artist, sanitized_album, sanitized_file_name)
    if os.path.isfile(sanitized_path):
        return sanitized_path

    raw_artist = track_info.get("©ART")
    raw_album = track_info.get("©alb")
    raw_title = track_info.get("©nam")
    #print(f"\n[DEBUG] Raw values: Artist: {raw_artist}, Album: {raw_album}, Title: {raw_title}")
    #print(f"[DEBUG] Attempting file path: {abs_path}")
    #print(f"[DEBUG] Attempting sanitized path: {sanitized_path}")

    return None


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


def sanitize_for_directory(name: str, max_len: int = 50) -> str:
    name = re.sub(r'[\\/:*?"<>|]', '_', name.strip())
    if len(name) > max_len:
        name = name[:max_len].rstrip("_- ")
    return name or "Unknown"


def sanitize_for_directory_ascii(name: str, max_len: int = 50) -> str:
    """
    Removes or replaces problematic characters, ASCII-only approach.
    """
    normalized = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', normalized.strip())
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].rstrip("_- ")
    return sanitized or "Unknown"


def sanitize_title_for_filename(title: str, max_len: int = 60) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', title.strip())
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].rstrip("_- ")
    return sanitized

def normalize_tag(value: str) -> str:
    """
    Normalize a metadata tag for comparison:
      - Strip leading/trailing whitespace
      - Lowercase everything
      - Remove problematic punctuation (same characters used in directory sanitization)
      - Collapse multiple spaces into one
    """
    normalized = value.strip().lower()
    normalized = re.sub(r'[\\/:*?"<>|]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

