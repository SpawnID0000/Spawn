import sys
import os
from difflib import get_close_matches
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_closest_match(track, track_list, cutoff=0.6):
    matches = get_close_matches(track, track_list, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def compare_m3u_files(m3u1_path, m3u2_path, output_path):
    with open(m3u1_path, 'r', encoding='utf-8') as file1, open(m3u2_path, 'r', encoding='utf-8') as file2:
        tracks1 = [line.strip() for line in file1.readlines() if line.strip() and not line.startswith('#')]
        tracks2 = [line.strip() for line in file2.readlines() if line.strip() and not line.startswith('#')]
    
    df = pd.DataFrame(columns=["m3u1", "m3u2"])
    
    print("Starting comparison...")

    # Use ThreadPoolExecutor to speed up the matching process
    def process_track(track):
        return track, find_closest_match(track, tracks1, cutoff=0.6)
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_track, track): track for track in tracks2}
        for index, future in enumerate(as_completed(futures)):
            track, closest_match = future.result()
            df = pd.concat([df, pd.DataFrame({"m3u2": [track], "m3u1": [closest_match]})], ignore_index=True)
            print(f"Processed track {index + 1}/{len(tracks2)} from m3u2: {track}")

    # Find tracks in m3u1 that aren't in m3u2
    print("Finding unmatched tracks in m3u1...")
    unmatched_tracks1 = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(find_closest_match, track, tracks2, cutoff=0.6): track for track in tracks1}
        for index, future in enumerate(as_completed(futures)):
            closest_match = future.result()
            track = futures[future]
            if closest_match is None:
                unmatched_tracks1.append(track)
            print(f"Checked track {index + 1}/{len(tracks1)} in m3u1")

    # Add the unmatched tracks to the DataFrame
    for track in unmatched_tracks1:
        df = pd.concat([df, pd.DataFrame({"m3u1": [track]})], ignore_index=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)

    print(f"Comparison complete. Results saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 compare_m3u.py path/to/m3u1 path/to/m3u2")
        sys.exit(1)
    
    m3u1_path = sys.argv[1]
    m3u2_path = sys.argv[2]
    
    if not os.path.isfile(m3u1_path) or not os.path.isfile(m3u2_path):
        print("Both arguments must be valid file paths.")
        sys.exit(1)
    
    output_path = os.path.join(os.getcwd(), "compare_m3u.csv")
    compare_m3u_files(m3u1_path, m3u2_path, output_path)
