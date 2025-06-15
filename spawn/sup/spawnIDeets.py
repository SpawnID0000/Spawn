#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI tool to retrieve ©ART, ©alb, and ©nam tag values for a given spawn_ID
Usage:
    python spawnIDeets.py path/to/spawn_catalog.db spawn_ID
"""
import argparse
import sqlite3
import json
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve tag data fields ©ART, ©alb, and ©nam for a specified spawn_ID"
    )
    parser.add_argument(
        "db_path",
        help="Path to the spawn_catalog.db SQLite database file"
    )
    parser.add_argument(
        "spawn_id",
        help="The spawn_ID to look up in the tracks table"
    )
    args = parser.parse_args()

    try:
        conn = sqlite3.connect(args.db_path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        print(f"Error opening database: {e}", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    cur.execute("SELECT tag_data FROM tracks WHERE spawn_id = ?", (args.spawn_id,))
    row = cur.fetchone()
    if not row:
        print(f"No entry found for spawn_ID '{args.spawn_id}'", file=sys.stderr)
        sys.exit(1)

    tag_data_blob = row["tag_data"]
    try:
        tag_data = json.loads(tag_data_blob)
    except json.JSONDecodeError as e:
        print(f"Error parsing tag_data JSON: {e}", file=sys.stderr)
        sys.exit(1)

    keys = ["©ART", "©alb", "©nam"]
    for key in keys:
        value = tag_data.get(key)
        if value is None:
            print(f"{key}: <not found>")
        else:
            print(f"{key}: {value}")

    conn.close()

if __name__ == "__main__":
    main()