#!/usr/bin/env python3

"""
embed_inspector.py

CLI Usage:
  python embed_inspector.py --spawn-root /path/to/Spawn/root [-data D] [-only SPAWN_ID]

Examples:
  1) Inspect entire dict, but do not print full arrays:
     python embed_inspector.py --spawn-root /Users/toddmarco/Music/Spawn

  2) Inspect entire dict, printing first 10 dimensions:
     python embed_inspector.py --spawn-root /Users/toddmarco/Music/Spawn -data 10

  3) Inspect only spawn_id=F79722D4, printing first 5 dims:
     python embed_inspector.py --spawn-root /Users/toddmarco/Music/Spawn -data 5 -only F79722D4
"""

import os
import sys
import pickle
import numpy as np
import argparse
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Inspect mp4tovec embeddings. Optionally print partial vector data."
    )
    parser.add_argument("--spawn-root", required=True,
                        help="Path to the Spawn project root containing aux/glob/mp4tovec.p")
    parser.add_argument("-data", type=int, default=0,
                        help="Number of dimensions to print from each embedding (0 => print none, -1 => print all).")
    parser.add_argument("-only", default=None,
                        help="If provided, print only the embedding for this spawn_id.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Locate the pickle file
    glob_dir = os.path.join(os.path.abspath(args.spawn_root), "aux", "glob")
    pickle_path = os.path.join(glob_dir, "mp4tovec.p")

    if not os.path.isfile(pickle_path):
        logger.error(f"mp4tovec.p not found at {pickle_path}")
        sys.exit(1)

    # Load the dictionary from pickle
    try:
        with open(pickle_path, "rb") as f:
            emb_dict = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load embeddings from {pickle_path}: {e}")
        sys.exit(1)

    # Check type
    logger.info(f"Type of data: {type(emb_dict)} (expecting dict with spawn_id -> embedding).")
    logger.info(f"Number of entries: {len(emb_dict)}")

    # If -only is specified, we'll focus on that spawn_id
    if args.only:
        spawn_id_val = args.only
        if spawn_id_val not in emb_dict:
            logger.error(f"No embedding found for spawn_id={spawn_id_val}")
            sys.exit(1)

        e = emb_dict[spawn_id_val]
        e = convert_to_numpy(e)
        shape_str = str(e.shape) if isinstance(e, np.ndarray) else "N/A"
        logger.info(f"spawn_id={spawn_id_val}, embedding shape={shape_str}")

        # If -data != 0 => print partial or full vector
        if args.data != 0:
            print_embedding_data(e, args.data, spawn_id_val)
        sys.exit(0)

    # Otherwise, iterate over entire dictionary
    for sid, embedding in emb_dict.items():
        embedding = convert_to_numpy(embedding)
        shape_str = str(embedding.shape) if isinstance(embedding, np.ndarray) else "N/A"
        logger.info(f"spawn_id={sid}, embedding shape={shape_str}")

        # If user wants to print partial data
        if args.data != 0:
            print_embedding_data(embedding, args.data, sid)

    # Prompt user to view full raw vector data
    while True:
        user_input = input("\nIf you'd like to view the raw vector data for a track, enter its Spawn ID (or press Enter to exit): ").strip()
        if not user_input:
            break
        if user_input not in emb_dict:
            logger.info(f"No embedding found for spawn_id '{user_input}'.")
            continue
        e = convert_to_numpy(emb_dict[user_input])
        logger.info(f"\nRaw vector data for spawn_id '{user_input}':")
        # Print the full vector (using -1 to indicate full print)
        print_embedding_data(e, -1, user_input)


def convert_to_numpy(embedding):
    """
    Convert a torch.Tensor or list => NumPy array, else return as is if already a NumPy array.
    """
    import torch

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    elif isinstance(embedding, list):
        embedding = np.array(embedding, dtype=np.float32)

    return embedding


def print_embedding_data(embedding, data_dim, sid):
    """
    Print up to 'data_dim' elements from the embedding if data_dim > 0.
    If data_dim == -1, print entire vector.
    """
    if not isinstance(embedding, np.ndarray):
        logger.warning(f"Cannot print data for spawn_id={sid}: not a NumPy array.")
        return

    # Flatten if shape is (1, N) or similar
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)

    if data_dim == -1:
        # print entire embedding
        logger.info(f"spawn_id={sid} embedding data => {embedding}")
    else:
        # print first data_dim elements
        subset = embedding[:data_dim]
        logger.info(f"spawn_id={sid} first {data_dim} dims => {subset}")


if __name__ == "__main__":
    main()