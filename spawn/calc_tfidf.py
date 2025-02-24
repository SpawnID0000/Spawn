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

def generate_raw_snippet_vectors(music_dir: str, snippet_duration: float = 5.0, n_mels: int = 40, sr: int = 22050) -> Dict[str, List[np.ndarray]]:
    """
    Walks through the Music directory and for each .m4a file that has a Spawn ID tag,
    loads the audio (mono, at sr), splits it into snippets of duration snippet_duration (in seconds),
    computes a mel-spectrogram for each snippet (with n_mels bands), and averages the spectrogram over time
    to obtain a fixed-length vector. Returns a dictionary mapping each Spawn ID to the list of snippet vectors.
    
    Args:
        music_dir: Path to the Music directory.
        snippet_duration: Duration (in seconds) for each snippet (default 5 sec).
        n_mels: Number of mel bands (default 40).
        sr: Sampling rate for audio loading (default 22050).
        
    Returns:
        Dictionary mapping spawn_id (str) to list of numpy arrays (snippet vectors).
    """
    raw_snippets = {}
    # Walk through music_dir recursively.
    for root, dirs, files in os.walk(music_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file.lower().endswith(".m4a") and not file.startswith("."):
                file_path = os.path.join(root, file)
                logger.debug("Starting processing file: %s", file_path)
                try:
                    # Load audio using librosa
                    y, _ = librosa.load(file_path, sr=sr, mono=True)
                    # Read spawn_id from file metadata using mutagen
                    audio = mutagen.mp4.MP4(file_path)
                    tags = audio.tags
                    if not tags or "----:com.apple.iTunes:spawn_ID" not in tags:
                        continue  # skip files without spawn_ID
                    raw_val = tags["----:com.apple.iTunes:spawn_ID"][0]
                    spawn_id = raw_val.decode("utf-8", errors="replace") if isinstance(raw_val, bytes) else str(raw_val)
                    spawn_id = spawn_id.strip()
                    if not spawn_id:
                        continue

                    # Split audio into snippets of snippet_duration seconds.
                    snippet_length = int(sr * snippet_duration)
                    snippets = []
                    for start in range(0, len(y), snippet_length):
                        snippet = y[start:start + snippet_length]
                        # For chunks after the first one, if the snippet is incomplete, break.
                        if start != 0 and len(snippet) < snippet_length:
                            break
                        # If there is no audio data, break.
                        if len(snippet) == 0:
                            break
                        # Compute mel-spectrogram
                        S = librosa.feature.melspectrogram(y=snippet, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
                        # Average over time to obtain a fixed-length vector
                        vec = np.mean(S, axis=1)
                        snippets.append(vec)
                    if snippets:
                        raw_snippets[spawn_id] = snippets
                        logger.debug("Generated %d snippet vectors for Spawn ID %s", len(snippets), spawn_id)
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
            * idfs[mp3_indices[mp3]][:, np.newaxis],
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





def calc_mp3tovec(
    mp3s: List[str], mp3tovecs: Dict[str, List[np.ndarray]], epsilon: float, dims: int
) -> Dict[str, np.ndarray]:
    """
    Processes raw snippet vectors from a group of MP3s (tracks) in batches and computes a single
    TF‑IDF–based vector for each track.

    Args:
        mp3s: List of MP3 IDs.
        mp3tovecs: Dictionary mapping MP3 IDs to a list of raw snippet vectors.
        epsilon: Cosine proximity threshold for similarity.
        dims: Dimensionality of each snippet vector.

    Returns:
        A dictionary mapping each MP3 ID to its computed TF‑IDF vector.
    """
    logger.debug("Processing raw snippet vectors for %d tracks...", len(mp3s))
    mp3tovec = {}
    mp3_indices = {}
    total_snippets = sum(len(mp3tovecs[mp3]) for mp3 in mp3s)
    mp3_vecs = np.empty((total_snippets, dims))
    start = 0
    for mp3 in mp3s:
        end = start + len(mp3tovecs[mp3])
        mp3_vecs[start:end] = np.array(mp3tovecs[mp3])
        # Normalize each snippet vector.
        norms = np.linalg.norm(mp3_vecs[start:end], axis=1)
        norms[norms == 0] = 1   # Avoid division by zero by replacing zero norms with 1
        mp3_vecs[start:end] /= norms[:, np.newaxis]
        mp3_indices[mp3] = list(range(start, end))
        start = end
    assert start == len(mp3_vecs)
    logger.debug("Normalized %d snippet vectors.", total_snippets)

    # Compute cosine proximity: close[i,j] = True if (1 - cosine similarity) < epsilon.
    close = (
        1 - np.einsum(
            "ij,kj->ik",
            mp3_vecs.astype(np.float16),
            mp3_vecs.astype(np.float16),
            dtype=np.float16,
        )
        < epsilon
    ).astype(bool)

    idfs = calc_idf(mp3s, mp3_indices, close)
    mp3tovec = calc_tf_and_mp3tovec(mp3s, mp3_indices, close, idfs, mp3_vecs)
    return mp3tovec

def generate_tfidf_embedding(file_path: str, epsilon: float = 0.001) -> np.ndarray:
    """
    Generates a TF‑IDF embedding for a single audio file.
    This is a basic placeholder implementation:
      - Loads audio via librosa.
      - Splits the audio into 5‑second snippets.
      - Computes a mel‐spectrogram for each snippet and averages it over time to form a snippet vector.
      - Uses these snippet vectors to compute a TF‑IDF weighted vector.
    
    Returns a 1D numpy array for the track’s TF‑IDF embedding, or None on failure.
    """
    logger.debug("Generating TF-IDF embedding for file: %s", file_path)
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        snippet_length = int(sr * 5)  # 5-second snippets
        snippets = []
        for start in range(0, len(y), snippet_length):
            snippet = y[start:start+snippet_length]
            # If this is not the first snippet and it's incomplete, skip it.
            if start != 0 and len(snippet) < snippet_length:
                break
            # Otherwise, if there's any audio data, process the snippet.
            if len(snippet) == 0:
                break
            # Compute a mel-spectrogram (using 40 mel bands)
            S = librosa.feature.melspectrogram(y=snippet, sr=sr, n_fft=2048, hop_length=512, n_mels=40)
            # Average across time to obtain a fixed-length vector
            vec = np.mean(S, axis=1)
            snippets.append(vec)
        if not snippets:
            logger.error("No valid snippets generated from %s", file_path)
            return None
        # Treat this single file as a “track” with ID "dummy"
        mp3s = ["dummy"]
        mp3tovecs = {"dummy": snippets}
        dims = snippets[0].shape[0]
        tfidf_vec = calc_mp3tovec(mp3s, mp3tovecs, epsilon, dims)["dummy"]
        logger.debug("Generated TF-IDF embedding for %s", file_path)
        return tfidf_vec
    except Exception as e:
        print(f"[ERROR] Failed to generate TF-IDF embedding for {file_path}: {e}")
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
    dims = mp3tovecs[first_key][0].shape[0]
    logger.debug("Snippet vector dimensionality: %d", dims)

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
