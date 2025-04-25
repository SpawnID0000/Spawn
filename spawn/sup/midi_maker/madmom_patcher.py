import os
import re
import sys

def patch_madmom(madmom_path):
    if not os.path.isdir(madmom_path):
        print(f"❌ Error: The path '{madmom_path}' is not a valid directory.")
        return

    patterns = {
        r'\bnp\.float\b': 'float',
        r'\bnp\.int\b': 'int',
        r'\bnp\.bool\b': 'bool'
    }

    patched_files = 0

    for root, dirs, files in os.walk(madmom_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    content = f.read()
                new_content = content
                for pattern, replacement in patterns.items():
                    new_content = re.sub(pattern, replacement, new_content)
                if new_content != content:
                    with open(full_path, 'w') as f:
                        f.write(new_content)
                    print(f"✅ Patched: {full_path}")
                    patched_files += 1

    if patched_files == 0:
        print("No changes needed. Madmom is already patched.")
    else:
        print(f"\n🎉 Finished patching {patched_files} files in Madmom.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python patch_madmom.py path/to/madmom")
        print("   path/to/madmom expected to be /Users/username/.pyenv/versions/3.11.11/lib/python3.11/site-packages/madmom")
        sys.exit(1)

    madmom_dir = sys.argv[1]
    patch_madmom(madmom_dir)