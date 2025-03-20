# setup.py

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README.md (if you have a README.md in the same directory)
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="Spawn",
    version="0.0.12",
    packages=find_packages(include=["spawn", "spawn.*"]),
    install_requires=[
        "mutagen>=1.45.1",
        "spotipy>=2.19.0",
        "musicbrainzngs>=0.7",
        "requests>=2.25.1",
        "pyacoustid>=1.2.0",
        "Pillow>=8.0.0",
        "python-dotenv>=0.19.0",
        "python-mpv>=0.5.2",
        "librosa>=0.10.0",
        "numpy<2",
        "tqdm>=4.64.0",
        #"tensorflow",
        #"tensorflow-macos",
        #"tensorflow-metal",
        "torch>=2.0.0",
        #"audiodiffusion>=0.1.0",
        "safetensors>=0.3.1"
    ],
    entry_points={
        "console_scripts": [
            "spawn=spawn.main:main",
        ],
    },
    author="Todd Marco",
    author_email="spawn.id.0000@gmail.com",
    description="Audio track importer that repackages & standardizes tags, MBIDs, and more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpawnID0000/Spawn",  # or your correct GitHub link
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="GNU General Public License v3 (GPLv3)",
    python_requires=">=3.6, <3.13",
    include_package_data=True,  # If you need non-Python files included, set up MANIFEST.in
    package_data={
        "spawn.audio-encoder": ["*.json", "*.safetensors", "*.txt"],  # include needed files
        "spawn.audiodiffusion": ["*.py"],
    },
)