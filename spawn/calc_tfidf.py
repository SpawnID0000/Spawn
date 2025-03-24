#!/usr/bin/env python3
"""
calc_tfidf.py

This module computes TF‑IDF–based embeddings from a collection of raw snippet vectors 
(i.e. “mp4tovecs”) for a group of tracks. Unlike the simple averaging method used in mp4tovec.py, 
this version computes a TF‑IDF weighted vector for each track and saves the resulting dictionary 
of Spawn IDs to their TF‑IDF embeddings as a pickle file.

The new TF‑IDF pickle file is saved (by default) as “aux/glob/mp4tovecTFIDF.p” relative to the given spawn root.
If the raw snippet pickle file (mp4tovec_raw.p) is not found, this module will generate it from the audio files in the Music directory.
"""

import argparse
import concurrent.futures
import os
import pickle
import random
from time import sleep
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import mutagen.mp4
import librosa
import logging

logger = logging.getLogger(__name__)

def init_worker():
    import logging
    # Remove any existing handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Set up basic configuration for the worker.
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
    # Also force flush output on stdout (optional)
    import sys
    sys.stdout.flush()

def generate_raw_snippet_vectors(music_dir: str, snippet_duration: float = 5.0, n_mels: int = 40, sr: int = 22050) -> Dict[str, List[np.ndarray]]:
    """
    Walks through the Music directory and for each .m4a file that has a Spawn ID tag,
    loads the audio, splits it into snippets, computes a mel-spectrogram for each snippet,
    averages the spectrogram over time to obtain a fixed-length vector, and returns a mapping
    from Spawn ID to a list of snippet vectors.
    """
    raw_snippets = {}
    for root, dirs, files in os.walk(music_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file.lower().endswith(".m4a") and not file.startswith("."):
                file_path = os.path.join(root, file)
                logger.debug("Starting processing file: %s", file_path)
                try:
                    y, _ = librosa.load(file_path, sr=sr, mono=True)
                    audio = mutagen.mp4.MP4(file_path)
                    tags = audio.tags
                    if not tags or "----:com.apple.iTunes:spawn_ID" not in tags:
                        continue  # Skip files without spawn_ID
                    raw_val = tags["----:com.apple.iTunes:spawn_ID"][0]
                    spawn_id = raw_val.decode("utf-8", errors="replace") if isinstance(raw_val, bytes) else str(raw_val)
                    spawn_id = spawn_id.strip()
                    if not spawn_id:
                        continue

                    snippet_length = int(sr * snippet_duration)
                    snippets = []
                    for start in range(0, len(y), snippet_length):
                        snippet = y[start:start + snippet_length]
                        # if len(snippet) < snippet_length:
                        #     logger.warning(f"Snippet starting at {start} in file {file_path} is incomplete (length {len(snippet)} < expected {snippet_length}). Skipping snippet.")
                        #     continue
                        S = np.array(librosa.feature.melspectrogram(y=snippet, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels))
                        if S.ndim < 2:
                            logger.warning(f"Mel-spectrogram has unexpected shape {S.shape} for snippet starting at {start} in file {file_path}. Skipping snippet.")
                            continue
                        logger.debug(f"File: {file_path}, snippet starting at {start}: S shape = {S.shape}")
                        if S.shape[1] == 0:
                            logger.warning(f"Empty mel spectrogram for snippet starting at {start} in file {file_path}. Skipping snippet.")
                            continue
                        vec = np.atleast_1d(np.mean(S, axis=1)).flatten().astype(float)
                        # Expected dimension is n_mels (default 40)
                        expected_dim = n_mels
                        if vec.ndim != 1 or vec.shape[0] != expected_dim:
                            logger.warning(f"Snippet vector shape {vec.shape} does not match expected shape ({expected_dim},) for snippet starting at {start} in file {file_path}. Skipping snippet.")
                            continue
                        logger.debug(f"Computed snippet vector shape: {vec.shape} for snippet starting at {start} in file {file_path}")
                        snippets.append(vec)
                    if snippets:
                        # Consistency check: ensure all snippet vectors have the same shape.
                        expected_shape = None
                        valid_snippets = []
                        for idx, snip in enumerate(snippets):
                            snip = np.array(snip, dtype=float).flatten()
                            if expected_shape is None:
                                expected_shape = snip.shape
                            elif snip.shape != expected_shape:
                                logger.warning(f"Inconsistent snippet vector shape {snip.shape} vs expected {expected_shape} for snippet {idx} in file {file_path}. Skipping snippet.")
                                continue
                            valid_snippets.append(snip)
                        if valid_snippets:
                            raw_snippets[spawn_id] = valid_snippets
                            logger.debug("Generated %d snippet vectors for Spawn ID %s", len(valid_snippets), spawn_id)
                        else:
                            logger.error("No valid snippet vectors for Spawn ID %s in file %s", spawn_id, file_path)
                except Exception as e:
                    logger.error("Failed to process %s: %s", file_path, e)
    return raw_snippets


def calc_idf(
    mp3s: List[str], mp3_indices: Dict[str, List[int]], close: np.ndarray
) -> np.ndarray:
    """
    Calculates the inverse document frequency (IDF) for each vector index.
    For each vector dimension, count the number of MP3s in which at least one snippet vector appears (close==True).
    Then compute idf[i] = -log((# MP3s in which i appears) / total_MP3s).

    Args:
        mp3s: List of MP3 (track) IDs.
        mp3_indices: Mapping from MP3 ID to list of indices (rows) in the raw snippet vectors.
        close: Boolean matrix (n_vectors x n_total_snippets) where close[i, j] is True if the cosine distance between 
               snippet vector i and snippet vector j is less than a given epsilon.

    Returns:
        idfs: A numpy array of shape (n_vectors,) containing the IDF for each vector.
    """
    logger.debug("Calculating IDF for %d dimensions...", close.shape[0])
    vec_in_mp3 = np.zeros((close.shape[0], len(mp3s)))
    for i, mp3 in enumerate(mp3s):
        vec_in_mp3[:, i] = close[:, mp3_indices[mp3]].any(axis=1)
    idfs = -np.log((vec_in_mp3.sum(axis=1) + 1e-10) / len(mp3s))
    logger.debug("Completed IDF calculation.")
    return idfs


def calc_tf_and_mp3tovec(
    mp3s: List[str],
    mp3_indices: Dict[str, List[int]],
    close: np.ndarray,
    idfs: np.ndarray,
    mp3_vecs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Calculates TF (term frequency) for each track and returns a single TF‑IDF vector per track.

    Args:
        mp3s: List of MP3 (track) IDs.
        mp3_indices: Mapping from MP3 ID to list of indices in mp3_vecs.
        close: Boolean matrix (n_vectors x n_total_snippets) indicating snippet similarity.
        idfs: Inverse document frequencies for each snippet vector (as computed by calc_idf).
        mp3_vecs: Numpy array of raw snippet vectors.

    Returns:
        A dictionary mapping each MP3 ID to its TF‑IDF vector.
    """
    logger.debug("Calculating TF and combining to form TF-IDF vectors for each track...")
    mp3tovec = {}
    for mp3 in mp3s:
        # Calculate term frequency for this MP3: count occurrences among its snippet vectors.
        num_snippets = len(mp3_indices[mp3])
        logger.debug("Processing Spawn ID: %s with %d snippet vectors", mp3, num_snippets)
        tf = np.sum(close[mp3_indices[mp3], :][:, mp3_indices[mp3]], axis=1)
        logger.debug("Spawn ID %s: term frequency (TF) values: %s", mp3, tf)
        tfidf_vector = np.sum(
            mp3_vecs[mp3_indices[mp3]]
            * tf[:, np.newaxis]
            * idfs[np.newaxis, :],
            axis=0,
        )
        mp3tovec[mp3] = tfidf_vector
        logger.debug("Completed TF-IDF vector for Spawn ID: %s", mp3)
    logger.debug("Completed TF-IDF vector calculation for current batch.")
    return mp3tovec

    logger.debug("Calculating TF and combining to form TF-IDF vectors for each track...")
    mp3tovec = {}
    for mp3 in mp3s:
        # Calculate term frequency for this MP3: count occurrences among its snippet vectors.
        logger.debug("Processing Spawn ID: %s", mp3)
        tf = np.sum(close[mp3_indices[mp3], :][:, mp3_indices[mp3]], axis=1)
        mp3tovec[mp3] = np.sum(
            mp3_vecs[mp3_indices[mp3]]
            * tf[:, np.newaxis]
            * idfs[mp3_indices[mp3]][:, np.newaxis],
            axis=0,
        )
        logger.debug("Completed Spawn ID: %s", mp3)
    logger.debug("Completed TF-IDF vector calculation for current batch.")
    return mp3tovec


def compute_idf(mp3tovecs: Dict[str, List[np.ndarray]], dims: int, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute the IDF vector (per mel band feature) over all tracks.
    For each mel band, count the number of tracks in which at least one snippet has a value above 'threshold'.
    Then, idf[i] = -log((df[i] + 1e-10) / total_tracks)
    
    Args:
        mp3tovecs: Dictionary mapping track ID to a list of snippet vectors.
        dims: Expected dimension of each snippet vector (e.g. 40).
        threshold: Minimal value to consider that a feature is "present".
    
    Returns:
        idf: A 1D numpy array of length dims.
    """
    n_tracks = len(mp3tovecs)
    df = np.zeros(dims)
    for track, snippets in mp3tovecs.items():
        if not snippets:
            continue
        try:
            stacked = np.stack(snippets)  # shape: (n_snippets, dims)
        except Exception as e:
            continue
        # For each mel band, mark as 1 if any snippet has a value above threshold.
        present = (stacked > threshold).any(axis=0)  # shape: (dims,)
        df += present.astype(float)
    idf = -np.log((df + 1e-10) / n_tracks)
    return idf


def calc_mp3tovec(mp3s: List[str],
                  mp3tovecs: Dict[str, List[np.ndarray]],
                  epsilon: float,  # epsilon is accepted for compatibility but not used
                  dims: int) -> Dict[str, np.ndarray]:
    """
    Compute a TF‑IDF embedding for each track.
    
    For each track:
      - The TF is computed as the average of its snippet vectors.
      - The IDF vector is computed across all tracks (using compute_idf).
      - The TF‑IDF vector is then: tfidf = TF * IDF (element-wise).
    
    Args:
        mp3s: List of track (Spawn) IDs.
        mp3tovecs: Dictionary mapping track ID to a list of snippet vectors.
        epsilon: (Unused) Cosine proximity threshold.
        dims: Expected dimension of each snippet vector (e.g. 40).
    
    Returns:
        A dictionary mapping each track ID to its TF‑IDF embedding.
    """
    # Compute the IDF vector across all tracks.
    idf = compute_idf(mp3tovecs, dims)
    mp3tovec = {}
    for mp3 in mp3s:
        snippets = mp3tovecs.get(mp3, [])
        if not snippets:
            continue
        try:
            stacked = np.stack(snippets)  # shape: (n_snippets, dims)
        except Exception as e:
            logger.error(f"Failed to stack snippets for track {mp3}: {e}")
            continue
        # Compute the track's term frequency (TF) as the average of its snippet vectors.
        tf = np.mean(stacked, axis=0)  # shape: (dims,)
        # Compute TF-IDF by element-wise multiplying TF and IDF.
        tfidf_vector = tf * idf
        mp3tovec[mp3] = tfidf_vector
        logger.debug(f"Computed TF-IDF vector for track {mp3} with shape {tfidf_vector.shape}")
    return mp3tovec

def generate_tfidf_embedding(file_path: str, epsilon: float = 0.001) -> np.ndarray:
    """
    Generates a TF‑IDF embedding for a single audio file.
    Returns a 1D numpy array for the track’s TF‑IDF embedding, or None on failure.
    """
    logger.debug("Generating TF-IDF embedding for file: %s", file_path)
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        snippet_length = int(sr * 5)  # 5-second snippets
        snippets = []
        for start in range(0, len(y), snippet_length):
            snippet = y[start:start+snippet_length]
            # # Skip snippets that are incomplete regardless of position.
            # if len(snippet) < snippet_length:
            #     logger.warning(f"Snippet starting at {start} in file {file_path} is incomplete (length {len(snippet)} < expected {snippet_length}). Skipping snippet.")
            #     continue

            # Compute mel-spectrogram (using 40 mel bands)
            S = np.array(librosa.feature.melspectrogram(y=snippet, sr=sr, n_fft=2048, hop_length=512, n_mels=40))
            if S.ndim < 2:
                logger.warning(f"Mel-spectrogram has unexpected shape {S.shape} for snippet starting at {start} in file {file_path}. Skipping snippet.")
                continue
            logger.debug(f"File: {file_path}, snippet starting at {start}: S shape = {S.shape}")
            if S.shape[1] == 0:
                logger.warning(f"Empty mel spectrogram for snippet starting at {start} in file {file_path}. Skipping snippet.")
                continue

            # Compute mean over time to get a fixed-length vector.
            vec = np.atleast_1d(np.mean(S, axis=1)).flatten().astype(float)
            # Expected dimension is 40 (since n_mels=40)
            expected_dim = 40
            if vec.ndim != 1 or vec.shape[0] != expected_dim:
                logger.warning(f"Snippet vector shape {vec.shape} does not match expected shape ({expected_dim},) for snippet starting at {start} in file {file_path}. Skipping snippet.")
                continue
            logger.debug(f"Computed snippet vector shape: {vec.shape} for snippet starting at {start} in file {file_path}")
            snippets.append(vec)

        if not snippets:
            logger.error("No valid snippets generated from %s", file_path)
            return None

        # Consistency check: ensure all snippet vectors have the same shape.
        expected_shape = None
        valid_snippets = []
        for idx, snip in enumerate(snippets):
            snip = np.array(snip, dtype=float).flatten()
            if expected_shape is None:
                expected_shape = snip.shape
            elif snip.shape != expected_shape:
                logger.warning(f"Inconsistent snippet vector shape {snip.shape} vs expected {expected_shape} for snippet {idx} in file {file_path}. Skipping snippet.")
                continue
            valid_snippets.append(snip)
        if not valid_snippets:
            logger.error("No valid snippets after consistency check for file %s", file_path)
            return None
        snippets = valid_snippets

        # Attempt to stack the snippets.
        try:
            stacked = np.stack(snippets)
        except Exception as e:
            logger.error(f"Failed to stack snippets for file {file_path}: {e}")
            for idx, snip in enumerate(snippets):
                logger.error(f"  snippet {idx}: type={type(snip)}, shape={snip.shape}, value={snip}")
            return None

        # Extra check: ensure stacked array has at least 2 dimensions.
        if stacked.ndim < 2:
            logger.error(f"Stacked snippet array for file {file_path} has unexpected shape {stacked.shape}. Cannot proceed.")
            return None

        dims = stacked.shape[1]
        mp3s = ["dummy"]
        mp3tovecs = {"dummy": snippets}
        # Compute the TF-IDF embedding using the calc_mp3tovec function.
        tfidf_vec = calc_mp3tovec(mp3s, mp3tovecs, epsilon, dims)["dummy"]
        logger.debug("Generated TF-IDF embedding for %s", file_path)
        return tfidf_vec
    except Exception as e:
        logger.error(f"Failed to generate TF-IDF embedding for {file_path}: {e}")
        return None


def main() -> None:
    """
    Main function for the calc_tfidf script.

    This script computes TF‑IDF–based embeddings from a raw snippet vectors pickle file
    (e.g. generated by an alternative method) and saves the per‑track TF‑IDF embeddings
    to a new pickle file.

    Optional arguments:
      --batch_size: Batch size for processing (default: 1000)
      --epsilon: Cosine proximity threshold (default: 0.001)
      --max_workers: Maximum number of worker processes (default: number of CPU cores)
      --mp3tovecs_file: Input pickle file with raw snippet vectors (default: aux/glob/mp4tovec_raw.p)
      --mp3tovec_file: Output pickle file for TF‑IDF embeddings (default: aux/glob/mp4tovecTFIDF.p)
      --spawn_root: The root directory for the Spawn library (e.g. /path/to/Spawn). If provided,
                    the input and output pickle file paths are resolved relative to this directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for TF-IDF calculation",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="Minimum cosine proximity for two vectors to be considered equal",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count() if os.cpu_count() is not None else 1,
        help="Maximum number of cores to use",
    )
    parser.add_argument(
        "--mp3tovecs_file",
        type=str,
        default="aux/glob/mp4tovec_raw.p",
        help="Path to the input pickle file containing raw snippet vectors (per track)",
    )
    parser.add_argument(
        "--mp3tovec_file",
        type=str,
        default="aux/glob/mp4tovecTFIDF.p",
        help="Path to the output pickle file for TF-IDF–based embeddings",
    )
    parser.add_argument(
        "--spawn_root",
        type=str,
        default="",
        help="(Optional) Root directory for the Spawn library. If provided, input/output pickle paths are resolved relative to this directory.",
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="",
        help="(Optional) Directory for music files. Defaults to Spawn/Music relative to spawn_root if not provided.",
    )
    args = parser.parse_args()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers, initializer=init_worker) as executor:

        for start in range(0, len(y), snippet_length):
            snippet = y[start:start+snippet_length]
            if start != 0 and len(snippet) < snippet_length:
                print(f"DEBUG: In file {file_path}, breaking at start {start} due to incomplete snippet.")
                break
            # if len(snippet) == 0:
            #     print(f"DEBUG: In file {file_path}, snippet at start {start} is empty.")
            #     break
            S = np.array(librosa.feature.melspectrogram(y=snippet, sr=sr, n_fft=2048, hop_length=512, n_mels=40))
            print(f"DEBUG: For file {file_path}, snippet starting at {start}: S shape = {S.shape}")
            if S.shape[1] == 0:
                print(f"WARNING: Empty mel spectrogram for snippet starting at {start} in file {file_path}")
                continue
            vec = np.atleast_1d(np.mean(S, axis=1)).flatten().astype(float)
            #print(f"DEBUG: For file {file_path}, snippet starting at {start}: vec type = {type(vec)}, shape = {getattr(vec, 'shape', 'no shape')}")
            #print(f"DEBUG: For file {file_path}, snippet starting at {start}: vec type = {type(vec)}, shape = {vec.shape}, first 5 = {vec[:5]}")
            logger.debug(f"Computed snippet vector shape: {vec.shape} for snippet starting at {start} in file {file_path}")
            if not (isinstance(vec, np.ndarray) and vec.ndim == 1 and vec.shape[0] > 0):
                print(f"WARNING: Invalid snippet vector for snippet starting at {start} in file {file_path}")
                continue
            snippets.append(vec)

        # If spawn_root is provided, adjust the input/output file paths.
        if args.spawn_root:
            spawn_root = os.path.abspath(args.spawn_root)
            args.mp3tovecs_file = os.path.join(spawn_root, args.mp3tovecs_file)
            args.mp3tovec_file = os.path.join(spawn_root, args.mp3tovec_file)
            logger.debug("Spawn root provided. Using mp3tovecs_file: %s and mp3tovec_file: %s", args.mp3tovecs_file, args.mp3tovec_file)
            if not args.music_dir:
                args.music_dir = os.path.join(spawn_root, "Music")
            else:
                args.music_dir = os.path.abspath(args.music_dir)

        # If the raw snippet pickle file doesn't exist, generate it.
        if not os.path.isfile(args.mp3tovecs_file):
            print(f"[INFO] {os.path.basename(args.mp3tovecs_file)} not found. Generating raw snippet vectors from music directory {args.music_dir}...")
            raw_snippets = generate_raw_snippet_vectors(args.music_dir)
            try:
                with open(args.mp3tovecs_file, "wb") as f:
                    pickle.dump(raw_snippets, f)
                print(f"[INFO] Raw snippet vectors saved to {args.mp3tovecs_file}")
            except Exception as e:
                print(f"[ERROR] Failed to save raw snippet vectors to {args.mp3tovecs_file}: {e}")
                return

        # Load the raw snippet vectors from the specified input file.
        try:
            with open(args.mp3tovecs_file, "rb") as f:
                mp3tovecs = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load raw snippet vectors from {args.mp3tovecs_file}: {e}")
            return

        # Get the dimensionality from the first snippet vector.
        first_key = list(mp3tovecs.keys())[0]
        dims = np.array(mp3tovecs[first_key][0], dtype=float).shape[0]
        #print(f"DEBUG: Determined snippet dimensionality: {dims}")

        # Process in batches using a process pool to compute TF-IDF–based embeddings.
        mp3tovec = {}
        keys = list(mp3tovecs.keys())
        random.shuffle(keys)
        logger.info("Processing %d tracks (random order) in batches of %d", len(keys), args.batch_size)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            futures = {}
            for i in tqdm(range(0, len(keys), args.batch_size), desc="Setting up jobs"):
                batch_keys = keys[i : i + args.batch_size]
                logger.debug("Submitting batch starting at index %d with Spawn IDs: %s", i, batch_keys)
                futures[executor.submit(calc_mp3tovec, batch_keys, mp3tovecs, args.epsilon, dims)] = i

            for i, future in enumerate(
                tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating TF-IDF")
            ):
                result = future.result()
                for k, v in result.items():
                    mp3tovec[k] = v
                logger.debug("Completed batch %d", i+1)
                if (i + 1) % 10 == 0:
                    with open(args.mp3tovec_file, "wb") as f:
                        pickle.dump(mp3tovec, f)
                    logger.info("Saved interim TF-IDF embeddings to %s after %d batches", args.mp3tovec_file, i+1)

        with open(args.mp3tovec_file, "wb") as f:
            pickle.dump(mp3tovec, f)
        logger.info("TF-IDF–based embeddings saved to %s", args.mp3tovec_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
