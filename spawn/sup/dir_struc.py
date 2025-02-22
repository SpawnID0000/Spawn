#!/usr/bin/env python3

import os
import sys

def print_dir_structure(path, prefix=""):
    """
    Recursively prints the directory structure of `path`,
    displaying subfolders and files in a tree-like hierarchy.
    """
    if not os.path.isdir(path):
        print(f"{path} is not a directory or doesn't exist.")
        return

    # Get a sorted list of items in the directory
    items = sorted(os.listdir(path))
    # (Optional) Skip hidden files/folders that start with '.'
    items = [item for item in items if not item.startswith('.')]

    # Track how many items we have to display
    total_items = len(items)

    for index, item in enumerate(items):
        # Determine if this is the last item in the directory
        is_last = (index == total_items - 1)

        # Use a "tree branch" character style
        branch = "└── " if is_last else "├── "

        # Print the current item
        print(prefix + branch + item)
        full_item_path = os.path.join(path, item)

        # If it's a directory, recurse into it
        if os.path.isdir(full_item_path):
            # For each child directory, adjust prefix
            extension = "    " if is_last else "│   "
            print_dir_structure(full_item_path, prefix + extension)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 dir_struc.py /path/to/directory")
        sys.exit(1)

    start_path = sys.argv[1]
    print(f"Directory structure for: {start_path}\n")
    print_dir_structure(start_path)

if __name__ == "__main__":
    main()