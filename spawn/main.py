# main.py

import os
import shutil
import shlex
from dotenv import load_dotenv
from .track_importer import run_import
from .favs import update_favorites_menu
from .curator import run_curator
from .player import play_m3u_menu
from .dumb import convert_m3u_to_file_sequence

def store_key_in_env_file(env_path, key, value):
    """
    Appends or updates a key=value in the .env file at env_path.
    If the key already exists, it's overwritten in the file.
    """
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    found = False
    for i, line in enumerate(lines):
        line_strip = line.strip()
        if line_strip.startswith(key + "="):
            lines[i] = f"{key}={value}\n"
            found = True
            break

    if not found:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Saved {key} to {env_path}.")

def copy_catalog_db_to_lib_path(script_dir, lib_path):
    """
    If 'spawn_catalog.db' & 'mp4tovec.p' exist in the same directory as this script (main.py),
    and LIB_PATH is valid, copy them to LIB_PATH/Spawn/aux/glob if they don't already exist there.
    """
    if not lib_path or not os.path.isdir(lib_path):
        # If LIB_PATH is not set or is invalid, skip copying
        print("[SETUP] LIB_PATH is not set or invalid; skipping copy of spawn_catalog.db/mp4tovec.p.")
        return

    # 1) Copy spawn_catalog.db
    source_db_path = os.path.join(script_dir, "spawn_catalog.db")
    if not os.path.exists(source_db_path):
        # No local spawn_catalog.db => nothing to copy
        print("[SETUP] No spawn_catalog.db found next to main.py; skipping database copy.")
    else:
        # Destination under LIB_PATH
        target_dir = os.path.join(lib_path, "Spawn", "aux", "glob")
        os.makedirs(target_dir, exist_ok=True)
        target_db_path = os.path.join(target_dir, "spawn_catalog.db")

        if os.path.exists(target_db_path):
            print("[SETUP] spawn_catalog.db already exists under LIB_PATH; skipping database copy.")
        else:
            print(f"[SETUP] Found spawn_catalog.db next to main.py. Copying to {target_db_path}...")
            try:
                shutil.copy2(source_db_path, target_db_path)
                print("[SETUP] Catalog database copied successfully.")
            except Exception as e:
                print(f"[SETUP] Error copying spawn_catalog.db: {e}")

    # 2) Copy mp4tovec.p
    source_emb_path = os.path.join(script_dir, "mp4tovec.p")
    if not os.path.exists(source_emb_path):
        # No local mp4tovec.p => nothing to copy
        print("[SETUP] No mp4tovec.p found next to main.py; skipping embeddings copy.")
    else:
        target_dir = os.path.join(lib_path, "Spawn", "aux", "glob")
        # target_dir already ensured above, but if you want to be safe, re-check:
        os.makedirs(target_dir, exist_ok=True)
        target_emb_path = os.path.join(target_dir, "mp4tovec.p")

        if os.path.exists(target_emb_path):
            print("[SETUP] mp4tovec.p already exists under LIB_PATH; skipping embeddings copy.")
        else:
            print(f"[SETUP] Found mp4tovec.p next to main.py. Copying to {target_emb_path}...")
            try:
                shutil.copy2(source_emb_path, target_emb_path)
                print("[SETUP] Embeddings file copied successfully.")
            except Exception as e:
                print(f"[SETUP] Error copying mp4tovec.p: {e}")

def main():
    """
    Main loop for the package. Presents a text menu so the user can select which action to run.
    Also reads SKIP_PROMPTS from settings.env, so user can change from default "False" to "True".
    """

    current_dir = os.path.dirname(__file__)

    # Load APId.env
    apid_env = os.path.join(current_dir, "APId.env")
    load_dotenv(apid_env)

    # Load settings.env
    settings_env = os.path.join(current_dir, "settings.env")
    load_dotenv(settings_env)

    # Retrieve LIB_PATH from settings.env
    lib_path = os.environ.get("LIB_PATH", "").strip()

    # Copy packaged database & pickle file if LIB_PATH is already set and valid
    if lib_path and os.path.isdir(lib_path):
        copy_catalog_db_to_lib_path(current_dir, lib_path)

    # Attempt to load ASCII art from logo.txt
    logo_file = os.path.join(current_dir, "logo.txt")
    if os.path.isfile(logo_file):
        # Resize terminal on some platforms
        print(f"\x1b[8;54;210t", end="")
        with open(logo_file, "r", encoding="utf-8") as f:
            logo_art = f.read()
        print(logo_art)

    skip_prompts_str = os.environ.get("SKIP_PROMPTS", "False").strip()
    skip_prompts_bool = (skip_prompts_str.lower() == "true")

    keep_matched_str = os.environ.get("KEEP_MATCHED", "False").strip()
    keep_matched_bool = (keep_matched_str.lower() == "true")

    # Check admin setting in environment
    admin_env_str = os.environ.get("ADMIN", "False").strip().lower()
    env_is_admin = (admin_env_str == "true")

    while True:
        print("\n\n\nSelect one of the following options:")
        print("   (or type 'quit' to exit)\n")
        print("    1) Import tracks")
        print("    2) Update favorites")
        print("    3) Create curated playlist")
        print("    4) Play M3U playlist")
        print("    5) Convert M3U to file folder (for dumb players)")
        print("    6) Playlist import/export (Plex)")

        while True:
            choice = input("\nEnter choice: ").strip()
            #if choice.lower() == "quit":
            if choice.lower() in ("quit", "exit"):
                print("Exiting...")
                return
            if not choice:
                print("No option entered. Please enter a valid option.")
            else:
                break

        if choice == "1":
            while True:
                in_path  = input("Enter the input path (folder containing music to import): ").strip()
                #in_path = in_path.replace("\\ ", " ").replace("\\&", "&")
                if in_path.lower() in ("quit", "exit"):
                    print("Exiting...")
                    return  # or break out of the outer loop if you prefer
                if in_path.lower() == "back":
                    print("Returning to main menu...")
                    break  # Exit this inner loop to return to the main menu
                if not in_path:
                    print("No path entered. Please enter a valid input path.")
                else:
                    break
            # If the user typed "back", skip further processing and return to the main menu.
            if in_path.lower() == "back":
                continue

            # Remove escape characters properly
            in_path = shlex.split(in_path)[0]  # This correctly processes any escaped characters

            # Expand user (~) and normalize path
            in_path = os.path.expanduser(in_path)
            in_path = os.path.abspath(in_path)

            # Debug
            #print(f"Final processed input path: {repr(in_path)}")  # repr() will show if \ still exists

            # Debug
            #print(f"Final processed input path: {in_path}")

            # Verify if the directory exists
            if not os.path.isdir(in_path):
                print(f"[ERROR] {in_path} is not a valid directory.\n")
            else:
                print(f"Validated input path: {in_path}\n")

            # If LIB_PATH wasn't found in settings.env, prompt user
            if not lib_path:
                print("LIB_PATH not found in settings.env. Please specify it now.")
                lib_path = input("Enter the path to where to save your Spawn library: ").strip()
                if lib_path:
                    store_key_in_env_file(settings_env, "LIB_PATH", lib_path)
                # Now that LIB_PATH has been entered, attempt to copy the database and pickle file
                copy_catalog_db_to_lib_path(current_dir, lib_path)

            # Decide if admin mode is active:
            #  - If env_is_admin is True => always admin
            #  - Or if user typed 'admin'
            #  - Otherwise user mode
            if env_is_admin:
                # We'll assume in_path is a real folder, no need to type "admin"
                print("[ADMIN] Admin mode is enabled via settings.env => ADMIN=True.")
                real_path = in_path
                run_import(
                    output_path=lib_path,
                    music_path=real_path,
                    skip_prompts=skip_prompts_bool,
                    keep_matched=keep_matched_bool,
                    acoustid_key="",
                    spotify_client_id="",
                    spotify_client_secret="",
                    lastfm_key="",
                    is_admin=True
                )
            else:
                # If environment isn't admin, check user-typed command
                if in_path.lower() == "admin":
                    print("[ADMIN] Admin mode unlocked by user input.")
                    real_path = input("Enter the actual folder path: ").strip()
                    real_path = real_path.replace("\\ ", " ")
                    run_import(
                        output_path=lib_path,
                        music_path=real_path,
                        skip_prompts=skip_prompts_bool,
                        keep_matched=keep_matched_bool,
                        acoustid_key="",
                        spotify_client_id="",
                        spotify_client_secret="",
                        lastfm_key="",
                        is_admin=True
                    )
                else:
                    # Normal user mode
                    run_import(
                        output_path=lib_path,
                        music_path=in_path,
                        skip_prompts=skip_prompts_bool,
                        keep_matched=keep_matched_bool,
                        acoustid_key="",
                        spotify_client_id="",
                        spotify_client_secret="",
                        lastfm_key="",
                        is_admin=False
                    )

        elif choice == "2":
            if not lib_path or not os.path.isdir(lib_path):
                lib_path = input("Enter the path to your Spawn project root: ").strip()
                if not os.path.isdir(lib_path):
                    print("[ERROR] Invalid path. Cannot update favorites.")
                    return
                store_key_in_env_file(settings_env, "LIB_PATH", lib_path)
            update_favorites_menu(lib_path)

        elif choice == "3":
            if not lib_path or not os.path.isdir(lib_path):
                lib_path = input("Enter the path to your Spawn project root: ").strip()
                if not os.path.isdir(lib_path):
                    print("[ERROR] Invalid path. Cannot create curated playlist.")
                    continue
            run_curator(lib_path, is_admin=env_is_admin)

        elif choice == "4":
            if not lib_path or not os.path.isdir(lib_path):
                lib_path = input("Enter the path to your Spawn project root: ").strip()
                if not os.path.isdir(lib_path):
                    print("[ERROR] Invalid path. Cannot play M3U.")
                    return
            play_m3u_menu(lib_path)

        elif choice == "5":
            if not lib_path or not os.path.isdir(lib_path):
                lib_path = input("Enter the path to your Spawn project root: ").strip()
                if not os.path.isdir(lib_path):
                    print("[ERROR] Invalid path. Cannot play M3U.")
                    return
            convert_m3u_to_file_sequence(lib_path)

        elif choice == "6":
            if not lib_path or not os.path.isdir(lib_path):
                lib_path = input("Enter the path to your Spawn project root: ").strip()
                if not os.path.isdir(lib_path):
                    print("[ERROR] Invalid path. Cannot play M3U.")
                    return
            # Ensure Plex API parameters are available
            plex_serv_url = os.environ.get("PLEX_SERV_URL", "").strip()
            plex_token = os.environ.get("PLEX_TOKEN", "").strip()
            plex_user_uuid = os.environ.get("PLEX_USER_UUID", "").strip()

            if not plex_serv_url:
                plex_serv_url = input("Enter Plex Server URL (e.g., http://192.168.86.67:32400): ").strip()
                store_key_in_env_file(apid_env, "PLEX_SERV_URL", plex_serv_url)
            if not plex_token:
                plex_token = input("Enter Plex Token: ").strip()
                store_key_in_env_file(apid_env, "PLEX_TOKEN", plex_token)
            if not plex_user_uuid:
                plex_user_uuid = input("Enter your Plex User UUID (or accountID): ").strip()
                store_key_in_env_file(apid_env, "PLEX_USER_UUID", plex_user_uuid)

            # Plex Playlist Operations sub-menu
            print("\nPlex Playlist Operations:")
            print("    1) Export playlist to Plex")
            print("    2) Import playlist from Plex")
            print("    3) Import Plex play log")
            while True:
                plex_choice = input("\nEnter choice: ").strip()
                if not plex_choice:
                    print("No option entered. Please enter a valid option.")
                    continue
                if plex_choice.lower() in ("back"):
                    break
                elif plex_choice.lower() in ("quit", "exit"):
                    print("Exiting...")
                    exit(0)
                elif plex_choice == "1":
                    from .plex.Plex_Playlist_Importer import import_playlists
                    import_playlists(plex_serv_url, plex_token)
                    break
                elif plex_choice == "2":
                    from .plex.Plex_Playlist_Exporter import export_playlists
                    export_playlists(plex_serv_url, plex_token)
                    break
                elif plex_choice == "3":
                    from .plex.Plex_Play_Log_Exporter import export_recent_plays_json
                    export_recent_plays_json(plex_serv_url, plex_token, plex_user_uuid)
                    break
                else:
                    print("Please enter a valid option.")
                    continue

        else:
            print("Please enter a valid option.")
            continue

if __name__ == "__main__":
    main()
