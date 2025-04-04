# likey.py

"""

This module implements the like_likelihood feature for Spawn and integrates it
into the curation workflow. Tracks explicitly marked by the user as "favorite" 
are assigned a rating of 1, while all other tracks get a continuous like_likelihood
score between -1 and 1 based on their proximity in the embedding space to the 
explicitly rated tracks.

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
