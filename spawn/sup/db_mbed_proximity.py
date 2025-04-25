#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def load_embeddings(path):
    """Load a dict of {id: embedding_array} from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    ids, embeds = zip(*data.items())
    X = np.vstack([np.asarray(v) for v in embeds])
    return list(ids), X

def compute_nearest(X):
    """
    Fit a 2-NN model and return:
      - distances[:,1]: cosine distance to nearest distinct neighbor
      - indices[:,1]:   index of that neighbor
    """
    nbrs = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute").fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances[:,1], indices[:,1]

def print_statistics(similarities, eps=1e-5):
    """Print overall stats, plus exact-1 and near-1 counts within eps."""
    exact_ones = np.sum(similarities == 1.0)
    near_ones  = np.sum(similarities >= 1.0 - eps)
    print("\n=== Embedding Proximity Statistics ===")
    print(f"  Count            : {similarities.size}")
    print(f"    Min            : {similarities.min():.4f}")
    print(f"    Max            : {similarities.max():.4f}")
    print(f"    Mean           : {similarities.mean():.4f}")
    print(f"    Median         : {np.median(similarities):.4f}")
    print(f"    Std            : {similarities.std():.4f}")
    print(f"    Exact == 1.0   : {exact_ones}")
    print(f"    ≥ {1.0 - eps:.6f}   : {near_ones}")
    print("======================================\n")

def report_close_pairs(ids, similarities, neighbor_idxs, eps=1e-5):
    """List every pair with similarity ≥ 1–eps, excluding self-matches."""
    cutoff = 1.0 - eps
    print(f"Tracks with similarity ≥ {cutoff:.6f}:")
    for i, sim in enumerate(similarities):
        j = neighbor_idxs[i]
        if sim >= cutoff and j != i:
            print(f"  {ids[i]}  ←→  {ids[j]}  (sim={sim:.7f})")
    print()

def report_top_pairs(ids, similarities, neighbor_idxs, top_n=100):
    """
    List the top_n tracks whose nearest-neighbor sim is highest,
    excluding self-matches and mirrored duplicates.
    """
    print(f"Top {top_n} unique pairs by nearest-neighbor similarity:")
    seen = set()
    count = 0

    # sort indices by descending similarity
    for i in sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True):
        j = neighbor_idxs[i]
        if i == j:
            continue  # skip self-match

        pair = tuple(sorted((i, j)))
        if pair in seen:
            continue  # skip mirrored duplicate

        seen.add(pair)
        print(f"  {ids[i]}  ←→  {ids[j]}  (sim={similarities[i]:.7f})")
        count += 1
        if count >= top_n:
            break

    print()

def plot_histogram(similarities, bins=200):
    """Plot histogram from min(sim) to 1.0 on the x-axis."""
    xmin = similarities.min()
    plt.figure(figsize=(8,4))
    plt.hist(similarities, bins=bins, edgecolor='black')
    plt.title("Cosine Similarity to Nearest Neighbor")
    plt.xlabel("Similarity")
    plt.ylabel("Count of Tracks")
    plt.xlim(xmin, 1.0)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot nearest-neighbor cosine-similarities in an embedding set."
    )
    parser.add_argument(
        "embeddings_path",
        help="Path to mp4tovec pickle file (dict of id→vector)."
    )
    parser.add_argument(
        "-top", "--top",
        type=int,
        default=0,
        help="Also list the top N closest pairs (default: off)"
    )
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings_path} …")
    ids, X = load_embeddings(args.embeddings_path)
    print(f"Loaded {len(ids)} embeddings of dimension {X.shape[1]}.")

    print("Computing nearest-neighbor similarities …")
    distances, neighbor_idxs = compute_nearest(X)
    sims = 1.0 - distances

    # Stats & thresholded pairs
    print_statistics(sims, eps=1e-5)
    report_close_pairs(ids, sims, neighbor_idxs, eps=1e-5)

    # Optional top-N listing
    if args.top > 0:
        report_top_pairs(ids, sims, neighbor_idxs, top_n=args.top)

    # Final histogram
    print("Plotting histogram …")
    plot_histogram(sims, bins=10000)

if __name__ == "__main__":
    main()