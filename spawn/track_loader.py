import os
import sqlite3
import pandas as pd
import pickle
import json

def load_tracks(user_mode=True, spawn_root=None):
    """
    Loads track data from the database.
    
    In user mode, it loads both the 'cat_tracks' and 'lib_tracks' tables from
    the spawn_library.db located at <spawn_root>/Spawn/aux/user/spawn_library.db.
    
    - Records from cat_tracks are tagged as "mat" (matched).
    - Records from lib_tracks are tagged as "nom" (non-matched).
    
    In non-user (admin) mode, it loads tracks from the global catalog at
    <spawn_root>/Spawn/aux/glob/spawn_catalog.db (table: tracks) and tags them as "cat".
    
    The function expects each row to contain at least 'spawn_id' and 'tag_data'
    (where tag_data is a JSON string). It parses tag_data and returns a DataFrame
    where each row is a dictionary with the parsed tag data, a 'spawn_id' key,
    and an 'origin' key.
    """
    import os
    if spawn_root is None:
        spawn_root = os.getcwd()
    
    if user_mode:
        library_db_path = os.path.join(spawn_root, "Spawn", "aux", "user", "spawn_library.db")
        with sqlite3.connect(library_db_path) as conn:
            # Read spawn_id and tag_data from both tables
            cat_tracks_df = pd.read_sql_query("SELECT spawn_id, tag_data FROM cat_tracks", conn)
            lib_tracks_df = pd.read_sql_query("SELECT spawn_id, tag_data FROM lib_tracks", conn)
        # Tag matched library tracks and non-matched library tracks distinctly.
        cat_tracks_df["origin"] = "mat"
        lib_tracks_df["origin"] = "nom"
        df = pd.concat([cat_tracks_df, lib_tracks_df], ignore_index=True)
    else:
        catalog_db_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "spawn_catalog.db")
        with sqlite3.connect(catalog_db_path) as conn:
            df = pd.read_sql_query("SELECT spawn_id, tag_data FROM tracks", conn)
        df["origin"] = "cat"
    
    # Parse the JSON stored in the tag_data field.
    def parse_row(row):
        try:
            tag_dict = json.loads(row['tag_data'])
        except Exception:
            tag_dict = {}

        spawn_id = row["spawn_id"]
        tag_dict["spawn_id"] = spawn_id
        # If this is a user-mode track with a local ID, assign it explicitly
        tag_dict["local_id"] = spawn_id if spawn_id.startswith("xxx") else None
        tag_dict["origin"] = row["origin"]
        return tag_dict

    # Apply the JSON parsing to every row and convert to a DataFrame.
    tracks = df.apply(parse_row, axis=1).tolist()
    # Convert list of dicts back into a DataFrame.
    return pd.DataFrame(tracks)

def load_embeddings(user_mode=True, spawn_root=None):
    """
    Loads embedding dictionaries.
    - Catalog embeddings are loaded from <spawn_root>/Spawn/aux/glob/mp4tovec.p.
    - Local embeddings are loaded from <spawn_root>/Spawn/aux/user/mp4tovec_local.p (if available and in user mode).
    """
    if spawn_root is None:
        spawn_root = os.getcwd()
    
    catalog_emb_path = os.path.join(spawn_root, "Spawn", "aux", "glob", "mp4tovec.p")
    with open(catalog_emb_path, "rb") as f:
        catalog_embeddings = pickle.load(f)
    
    local_embeddings = {}
    if user_mode:
        local_emb_path = os.path.join(spawn_root, "Spawn", "aux", "user", "mp4tovec_local.p")
        if os.path.exists(local_emb_path):
            with open(local_emb_path, "rb") as f:
                local_embeddings = pickle.load(f)
    
    return catalog_embeddings, local_embeddings

def merge_tracks_with_embeddings(df, catalog_embeddings, local_embeddings):
    """
    Merges embeddings into the tracks DataFrame and annotates each track with its source:
      - "cat"  if only catalog (spawn_id exists and local_id is missing)
      - "lib"  if only local (local_id exists and spawn_id is missing)
      - "both" if both identifiers exist
      - "unknown" otherwise.
    Adds an 'embedding' column to the DataFrame.
    """
    def get_embedding(row):
        if pd.notna(row.get("spawn_id")):
            emb = catalog_embeddings.get(row["spawn_id"])
            if emb is not None:
                return emb
        if pd.notna(row.get("local_id")):
            return local_embeddings.get(row["local_id"])
        return None

    df["embedding"] = df.apply(get_embedding, axis=1)

    def get_track_source(row):
        spawn_id = row.get("spawn_id")
        local_id = row.get("local_id")
        if pd.notna(spawn_id) and pd.isna(local_id):
            return "cat"
        elif pd.notna(local_id) and pd.isna(spawn_id):
            return "lib"
        elif pd.notna(spawn_id) and pd.notna(local_id):
            return "both"
        else:
            return "unknown"

    df["track_source"] = df.apply(get_track_source, axis=1)
    return df

def load_and_merge(user_mode=True, spawn_root=None):
    """
    Complete loader function:
      - Loads track data from the appropriate database.
      - Loads catalog and (if in user mode) local embeddings.
      - Merges embeddings into the track data and annotates track sources.
    Returns a DataFrame with all track data.
    """
    df = load_tracks(user_mode=user_mode, spawn_root=spawn_root)
    catalog_embeddings, local_embeddings = load_embeddings(user_mode=user_mode, spawn_root=spawn_root)
    df = merge_tracks_with_embeddings(df, catalog_embeddings, local_embeddings)
    return df