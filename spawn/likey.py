# likey.py

"""

This module implements the like_likelihood feature for Spawn and integrates it
into the curation workflow. Tracks explicitly marked by the user as "favorite" 
are assigned a rating of 1, while all other tracks get a continuous like_likelihood
score between -1 and 1 based on their proximity in the embedding space to the 
explicitly rated tracks.

It also provides an integration function, generate_recommended_playlist(),
which, when triggered, will compute like_likelihood scores for all tracks and then
select those with a score below 1 (i.e. not favorites) but above a given threshold 
(e.g., 0.85). These recommended tracks are then grouped and passed to the curator 
to generate a curated M3U file (with filename appended with "_recs").

Note: This integration is triggered (for now) only after the user selects advanced
curation with a favorites filter. In run_curator_advanced() (in curator.py), after 
favorites filtering, add a prompt such as:

    "Would you also like to generate a curated playlist of other tracks you might like? ([y]/n): "

If the user answers "y", then call generate_recommended_playlist() below.
"""

import os
import numpy as np
import math
import pickle
import time
import shutil

# -------------------------
# Core like_likelihood code
# -------------------------

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize a given embedding vector.
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two normalized vectors.
    (Assumes vec1 and vec2 are already normalized.)
    """
    return np.dot(vec1, vec2)

def gaussian_weight(distance: float, sigma: float) -> float:
    """
    Compute a Gaussian weight given a distance and sigma.
    
    The weight is computed as:
         exp( - (distance^2) / (2 * sigma^2) )
    """
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

def compute_like_likelihoods(embeddings: dict, explicit_ratings: dict, sigma: float = 0.15, min_weight_threshold: float = 1e-3) -> dict:
    """
    Computes a like_likelihood for each track (by spawn_id) based on the distances
    to explicitly rated tracks. For explicitly rated tracks, the rating is returned directly.
    For non-explicit tracks, the weighted average cosine distance is computed, and the
    like_likelihood is defined as 1.0 minus this average distance.
    
    Parameters:
      embeddings: dict mapping spawn_id (str) to its numpy.ndarray embedding.
      explicit_ratings: dict mapping spawn_id (str) to an explicit rating (1 for favorite, -1 for disliked).
      sigma: Gaussian kernel sigma parameter (default 0.15).
      min_weight_threshold: if the total weight is below this, the score is set to 0.
      
    Returns:
      A dict mapping spawn_id to a like_likelihood score (float, typically between 0 and 1).
    """
    # Pre-normalize all embeddings for cosine similarity.
    norm_embeddings = {}
    for sid, emb in embeddings.items():
        norm_embeddings[sid] = normalize_embedding(emb)
    
    # Build lists of explicitly rated tracks with their normalized embeddings and ratings.
    explicit_ids = []
    explicit_norms = []
    explicit_vals = []
    for sid, rating in explicit_ratings.items():
        if sid in norm_embeddings:
            explicit_ids.append(sid)
            explicit_norms.append(norm_embeddings[sid])
            explicit_vals.append(rating)
    if explicit_norms:
        explicit_norms = np.array(explicit_norms)  # shape (n, d)
        explicit_vals = np.array(explicit_vals)      # shape (n,)
    else:
        # If there are no explicit ratings, return 0 for all tracks.
        return {sid: 0.0 for sid in norm_embeddings}

    results = {}
    # For each track, either use the explicit rating or compute a weighted score.
    for sid, emb in norm_embeddings.items():
        if sid in explicit_ratings:
            # Use the user-provided rating (1 or -1).
            results[sid] = float(explicit_ratings[sid])
        else:
            # Compute cosine similarities with all explicit tracks.
            sims = np.dot(explicit_norms, emb)  # shape (n,)
            distances = 1.0 - sims            # cosine distance in [0, 1] (since sims in [0,1])
            weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))
            weights[sims < 0] = 0.0
            total_weight = np.sum(weights)
            if total_weight < min_weight_threshold:
                results[sid] = 0.0
            else:
                weighted_distance = np.sum(weights * distances) / total_weight
                # Define like_likelihood: 1 if distance=0, 0 if distance=1.
                score = 1.0 - weighted_distance
                results[sid] = score
    return results

# -------------------------
# Integration: Generate Recommended Playlist
# -------------------------

def generate_recommended_playlist(spawn_root: str, embeddings: dict, threshold: float = 0.85):
    """
    Generate a curated playlist of recommended tracks (those not explicitly marked as favorites)
    based on their like_likelihood scores. Tracks with a like_likelihood less than 1 (i.e. not favorites)
    but greater than or equal to 'threshold' (e.g., 0.85) are selected.
    
    The selected tracks are grouped by genre and then ordered within each genre cluster using
    chain-based curation. The final output is written as a curated M3U file with '_recs' appended.
    
    Parameters:
      spawn_root: Root path to the Spawn project.
      embeddings: Dictionary of track embeddings (spawn_id -> numpy.ndarray).
      threshold: Float threshold for like_likelihood to include a track (default 0.85).
    """
    # Import required functions using relative imports.
    from .curator import load_tracks_from_db, group_tracks_by_genre, write_curated_m3u, filter_tracks_by_favorites, safe_extract_first, chain_based_curation

    # 1. Load all tracks from the catalog database.
    db_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "spawn_catalog.db")
    full_tracks = load_tracks_from_db(db_path)
    if not full_tracks:
        print("[INFO] No tracks found in the database for recommendations.")
        return

    # 2. Build explicit ratings from the favorites.
    favorites, _, _ = filter_tracks_by_favorites(full_tracks, spawn_root, "3")
    explicit_ratings = {}
    for track in favorites:
        sid = track.get("spawn_id")
        if sid:
            explicit_ratings[sid] = 1  # rating for favorites

    if not explicit_ratings:
        print("[INFO] No explicit favorite ratings found; cannot generate recommendations.")
        return

    # 3. Compute like_likelihood scores for all tracks.
    like_scores = compute_like_likelihoods(embeddings, explicit_ratings, sigma=0.15)

    # 4. Select recommended tracks: those with like_score >= threshold and < 1.
    recommended_tracks = []
    for track in full_tracks:
        sid = track.get("spawn_id")
        if sid in like_scores:
            score = like_scores[sid]
            if score < 1.0 and score >= threshold:
                recommended_tracks.append(track)

    if not recommended_tracks:
        print(f"[INFO] No recommended tracks found with like_likelihood >= {threshold}.")
        return

    # Output debug information for each recommended track.
    for track in recommended_tracks:
        sid = track.get("spawn_id", "unknown")
        title = safe_extract_first(track, "©nam") or "Untitled"
        score = like_scores.get(sid, 0.0)
        print(f"[DEBUG] Track {sid} \"{title}\" => like_likelihood={score:.4f}")

    print(f"[INFO] Found {len(recommended_tracks)} recommended tracks with like_likelihood >= {threshold}.")

    # 5. Group recommended tracks by genre.
    clusters = group_tracks_by_genre(recommended_tracks)

    # 6. Apply chain-based ordering within each genre cluster.
    for genre, tracks in clusters.items():
        clusters[genre] = chain_based_curation(tracks, embeddings)

    # 7. Write the recommended curated M3U file with a '_recs' suffix.
    m3u_path = write_curated_m3u(spawn_root, list(clusters.items()), favorites_filter_desc="recommended", suffix="_recs")

    if m3u_path:
        print(f"[INFO] Recommended curated playlist created at: {m3u_path}")
    else:
        print("[ERROR] Could not write the recommended curated M3U playlist.")
