# db_to_csv.py

import sqlite3
import csv
import json
import sys
import os

def convert_db_to_csv(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]

    # Check if "tracks" table exists
    if "tracks" not in tables:
        print("Error: Table 'tracks' not found in the database.")
        return

    table_name = "tracks"

    # Retrieve column names
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [col[1] for col in cursor.fetchall()]

    if len(columns) != 2:
        print("Error: Expected exactly two columns in the 'tracks' table.")
        return

    primary_column, kv_column = columns

    # Retrieve all records
    cursor.execute(f"SELECT {primary_column}, {kv_column} FROM {table_name};")
    rows = cursor.fetchall()

    # Parse key-value pairs and collect all unique keys
    all_keys = set()
    parsed_data = []
    for row in rows:
        primary_value, kv_data = row
        try:
            kv_dict = json.loads(kv_data) if isinstance(kv_data, str) else {}
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON entry: {kv_data}")
            kv_dict = {}

        all_keys.update(kv_dict.keys())
        parsed_data.append({primary_column: primary_value, **kv_dict})

    # Convert set to sorted list of column names
    sorted_keys = sorted(all_keys)
    fieldnames = [primary_column] + sorted_keys

    # Determine output CSV file name
    csv_filename = os.path.splitext(os.path.basename(db_path))[0] + "_tracks.csv"

    # Write to CSV
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in parsed_data:
            writer.writerow(row)

    print(f"CSV file saved as: {csv_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python db_to_csv.py path/to/db")
    else:
        convert_db_to_csv(sys.argv[1])
