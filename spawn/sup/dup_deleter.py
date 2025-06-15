#!/usr/bin/env python3
"""
dup_deleter.py

Pre-filter a batch of new audio tracks by comparing their audio embeddings
against an existing library's embeddings, deleting confirmed duplicates.

Prerequisites (install these before running):
    pip install torch torchvision torchaudio
    pip install librosa scipy safetensors tqdm

Usage:
    python dup_deleter.py [--prompt] /path/to/Spawn_LIB_PATH /path/to/new_tracks

Arguments:
    Spawn_LIB_PATH  Path to the root of your Spawn library (where aux/glob/mp4tovec.p lives).
    new_tracks      Path to the directory containing new audio files to check.

Options:
    -h, --help      Show this help message and exit.
    --prompt        Prompt the user for each detected duplicate.
                    If omitted, duplicates are deleted automatically.

What it does:
  1. Loads existing embeddings from lib_path/mp4tovec_local.p.
  2. Loads the MP4ToVec model (from Spawn.MP4ToVec).
  3. Recursively scans new_tracks (ignoring hidden files/directories) for audio files
     with extensions: .m4a, .mp4, .mp3, .flac, .wav, .aac.
  4. Generates embeddings for each new audio file.
  5. For each new embedding, computes cosine similarity vs all existing embeddings.
     If similarity ≥ 0.9999, prompts the user to confirm deletion.
  6. Deletes any files the user confirms as exact duplicates.
  7. Prints a summary of how many were checked and how many were deleted.
"""

import os
import sys

# ─── DEBUG‐ENABLED Shim to ensure “spawn” is importable ───────────────────────
_script_file = os.path.abspath(__file__)
_script_dir = os.path.abspath(os.path.dirname(_script_file))
# Two levels up: if dup_deleter.py lives at …/Spawn/spawn/sup/dup_deleter.py,
# then …/Spawn is os.path.join(_script_dir, "..", "..")
_spawn_pkg_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))

if _spawn_pkg_root not in sys.path:
    sys.path.insert(0, _spawn_pkg_root)

# ─── Debug Prints (comment out if unnecessary) ─────────────────────────────────
#print("\n[DEBUG] __file__       =", _script_file)
#print("[DEBUG] script_dir      =", _script_dir)
#print("[DEBUG] spawn_pkg_root  =", _spawn_pkg_root)
#print("[DEBUG] sys.path[0:5]   =", sys.path[:5], "\n")
# ──────────────────────────────────────────────────────────────────────────────

# ─── Verify that all runtime dependencies are installed ──────────────────────
missing = []

try:
    import torch
except ImportError:
    missing.append("torch")

try:
    import librosa
except ImportError:
    missing.append("librosa")

try:
    import scipy
except ImportError:
    missing.append("scipy")

try:
    import safetensors
except ImportError:
    missing.append("safetensors")

try:
    import tqdm
except ImportError:
    missing.append("tqdm")

if missing:
    print("[Error] The following Python packages are required but missing:")
    for pkg in missing:
        print(f"  • {pkg}")
    print("\nPlease install them first, for example:")
    print("    pip install torch torchvision torchaudio")
    print("    pip install librosa scipy safetensors tqdm\n")
    sys.exit(1)
# ──────────────────────────────────────────────────────────────────────────────

# Now attempt to import from spawn.MP4ToVec (which itself requires the above packages)
try:
    from spawn.MP4ToVec import load_mp4tovec_model_diffusion, generate_embedding
    #print("[DEBUG] Successfully imported spawn.MP4ToVec\n")
except ImportError as e:
    print("[Error] Could not import MP4ToVec from spawn. Traceback:")
    print(e)
    print("\nMake sure that:")
    print("  • The ‘spawn’ folder is in your PYTHONPATH (see the shim at the top of this script).")
    print("  • You’ve installed all prerequisites as listed above.")
    sys.exit(1)

import argparse
import pickle

from scipy.spatial.distance import cosine

# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_PICKLE = os.path.join(os.getcwd(), 'dup_deleter.p')
try:
    with open(OUTPUT_PICKLE, 'rb') as f:
        new_embeddings = pickle.load(f)
    print(f"Loaded {len(new_embeddings)} previously-computed embeddings from {OUTPUT_PICKLE}")
except FileNotFoundError:
    new_embeddings = {}
    print(f"No existing dup_deleter.p found; starting fresh")

# ──────────────────────────────────────────────────────────────────────────────

EMBEDDING_DUPLICATE_THRESHOLD = 0.99992
POSSIBLE_DUPLICATE_THRESHOLD  = 0.99990
POSSIBLE_EXTS = {'.m4a', '.mp4', '.mp3', '.flac', '.wav', '.aac'}

# ──────────────────────────────────────────────────────────────────────────────

def load_existing_embeddings(pickle_path):
    """
    Load a dict of existing embeddings from the given pickle path.
    Returns {} if the file is missing or cannot be parsed as a dict.
    """
    if os.path.isfile(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    return data
                else:
                    print(f"[Warning] {pickle_path} did not contain a dict; ignoring.")
        except Exception as e:
            print(f"[Warning] Failed to load embeddings from {pickle_path}: {e}")
    return {}

def find_audio_files(root_dir):
    """
    Recursively collect all non-hidden files under root_dir whose extension
    matches the known audio extensions in POSSIBLE_EXTS.
    """
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fname in filenames:
            if fname.startswith('.'):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in POSSIBLE_EXTS:
                audio_files.append(os.path.join(dirpath, fname))
    return audio_files

def main():
    parser = argparse.ArgumentParser(
        description="Pre-filter new tracks by embedding-based duplicates.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  python dup_deleter.py /home/user/spawn/lib /home/user/new_batch\n\n"
            "By default, duplicates are deleted automatically. Use --prompt if confirmation is desired.\n"
            "This will check every .m4a/.mp4/.mp3/.flac/.wav/.aac file under\n"
            "/home/user/new_batch (recursively), compare against embeddings in\n"
            "/home/user/spawn/lib/aux/glob/mp4tovec.p, and prompt to delete any that\n"
            "reach cosine similarity ≥ 0.9999. Hidden files and directories are ignored."
        )
    )
    parser.add_argument(
        '--prompt',
        action='store_true',
        help="If set, prompt the user for each detected duplicate. Defaults to auto-delete."
    )
    parser.add_argument(
        'lib_path',
        help="Path to Spawn library directory (must contain aux/glob/mp4tovec.p)."
    )
    parser.add_argument(
        'new_tracks',
        help="Path to directory of new audio files to check (recursively)."
    )
    args = parser.parse_args()

    lib_path = os.path.abspath(args.lib_path)
    new_tracks_path = os.path.abspath(args.new_tracks)
    prompt_mode = args.prompt

    # Validate inputs
    if not os.path.isdir(lib_path):
        print(f"[Error] lib_path '{lib_path}' is not a directory.")
        sys.exit(1)
    if not os.path.isdir(new_tracks_path):
        print(f"[Error] new_tracks '{new_tracks_path}' is not a directory.")
        sys.exit(1)

    # Load required imports
    try:
        from spawn.MP4ToVec import load_mp4tovec_model_diffusion, generate_embedding
    except ImportError:
        print("[Error] Could not import MP4ToVec from spawn. Ensure the Spawn package is on PYTHONPATH.")
        sys.exit(1)

    # Load existing embeddings from library
    pickle_path = os.path.join(lib_path, "aux", "glob", "mp4tovec.p")
    print(f"Loading existing embeddings from: {pickle_path}")
    existing_embeds = load_existing_embeddings(pickle_path)
    print(f"Loaded {len(existing_embeds)} existing embeddings.")

    # Prepare persistent new-embeddings store
    OUTPUT_PICKLE = os.path.join(os.getcwd(), 'dup_deleter.p')
    try:
        with open(OUTPUT_PICKLE, 'rb') as f:
            new_embeddings = pickle.load(f)
        print(f"Loaded {len(new_embeddings)} previously-computed embeddings from {OUTPUT_PICKLE}")
    except FileNotFoundError:
        new_embeddings = {}
        print(f"No existing {os.path.basename(OUTPUT_PICKLE)} found; starting fresh")

    # Load the MP4ToVec model
    print("Loading MP4ToVec model...")
    model = load_mp4tovec_model_diffusion()
    if model is None:
        print("[Error] MP4ToVec model could not be loaded. Exiting.")
        sys.exit(1)
    print("Model loaded successfully.")

    # Gather all candidate audio files
    print(f"Scanning for audio files under: {new_tracks_path}")
    all_new_files = find_audio_files(new_tracks_path)
    if not all_new_files:
        print("[Info] No audio files found in new_tracks.")
        sys.exit(0)
    print(f"Found {len(all_new_files)} candidate files.\n")

    # Determine which files still need embeddings
    files_to_embed = [p for p in all_new_files if p not in new_embeddings]
    print(f"{len(files_to_embed)} files to embed in batches of 50…")

    # Batch embedding generation
    batch_size = 50
    for offset in range(0, len(files_to_embed), batch_size):
        batch = files_to_embed[offset:offset + batch_size]
        print(f"\nProcessing batch {offset//batch_size + 1} "
              f"(items {offset + 1}–{offset + len(batch)})…")
        for path in batch:
            try:
                emb = generate_embedding(path, model)
                if emb is None:
                    print(f"  [Warning] No embedding for {os.path.basename(path)}; skipping.")
                    continue
                new_embeddings[path] = emb
                print(f"  ✓ Embedded: {os.path.basename(path)}")
            except Exception as e:
                print(f"  [Error] Failed embedding {os.path.basename(path)}: {e}")

        # Persist after each batch
        with open(OUTPUT_PICKLE, 'wb') as f:
            pickle.dump(new_embeddings, f)
        print(f"Saved {len(new_embeddings)} total embeddings to {OUTPUT_PICKLE}")

    # Compare each new embedding against all existing embeddings
    removed = []
    print(f"\nChecking for duplicates with thresholds: "
          f"auto-delete ≥{EMBEDDING_DUPLICATE_THRESHOLD}, "
          f"prompt ≥{POSSIBLE_DUPLICATE_THRESHOLD}")
    for new_path, new_vec in new_embeddings.items():
        best_sim, best_id = -1.0, None
        for exist_id, exist_vec in existing_embeds.items():
            try:
                sim = 1.0 - cosine(new_vec, exist_vec)
            except Exception:
                continue
            if sim > best_sim:
                best_sim, best_id = sim, exist_id

        rel_path = os.path.relpath(new_path, new_tracks_path)
        # High-confidence duplicates
        if best_sim >= EMBEDDING_DUPLICATE_THRESHOLD:
            delete = True
            if prompt_mode:
                resp = input(
                    f"\n⚠️  High-confidence duplicate:\n"
                    f"    {rel_path} → {best_id} (sim={best_sim:.6f})\n"
                    "Delete? ([y]/n): "
                ).strip().lower() or "y"
                delete = (resp == 'y')
            if delete:
                try:
                    os.remove(new_path)
                    print(f"[Deleted] {rel_path}")
                    removed.append(new_path)
                except Exception as e:
                    print(f"[Warning] Could not delete {rel_path}: {e}")
        # Possible duplicates
        elif best_sim >= POSSIBLE_DUPLICATE_THRESHOLD:
            resp = input(
                f"\n⚠️  Potential duplicate:\n"
                f"    {rel_path} → {best_id} (sim={best_sim:.6f})\n"
                "Delete? ([y]/n): "
            ).strip().lower() or "y"
            if resp == 'y':
                try:
                    os.remove(new_path)
                    print(f"[Deleted] {rel_path}")
                    removed.append(new_path)
                except Exception as e:
                    print(f"[Warning] Could not delete {rel_path}: {e}")

    # Final summary
    print("\nSummary:")
    print(f"  Total embeddings checked: {len(new_embeddings)}")
    print(f"  Files deleted:           {len(removed)}")
    print("Done.")

if __name__ == "__main__":
    main()