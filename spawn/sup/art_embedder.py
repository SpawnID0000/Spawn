#art_embedder.py

import os
import sys
import io
import logging
from PIL import Image
import mutagen
from mutagen.mp4 import MP4, MP4Cover

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_DIMENSION = 1000
MAX_FILESIZE = 350 * 1024  # 350 KB
QUALITY_STEPS = [90, 80, 70, 60, 50, 40, 30, 20, 10]

def resize_and_optimize_image(image_path, save_path):
    """Resize and optimize image to be under MAX_DIMENSION and MAX_FILESIZE."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            logger.info(f"Original image size: {width}x{height}")

            # Determine if resizing is needed
            needs_resizing = max(width, height) > MAX_DIMENSION
            if needs_resizing:
                scale_factor = MAX_DIMENSION / max(width, height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                logger.info(f"Resized to: {new_width}x{new_height}")

            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Try saving at different quality levels to fit within MAX_FILESIZE
            for quality in QUALITY_STEPS:
                temp_buffer = io.BytesIO()
                img.save(temp_buffer, format="JPEG", quality=quality, optimize=True)
                file_size = temp_buffer.getbuffer().nbytes

                if file_size <= MAX_FILESIZE:
                    with open(save_path, "wb") as f:
                        f.write(temp_buffer.getvalue())
                    logger.info(f"Saved optimized image: {save_path} ({file_size} bytes, quality={quality})")
                    return save_path
                logger.info(f"Quality {quality} still too large ({file_size} bytes). Retrying...")

            logger.warning("Could not optimize below max file size. Saving at lowest quality.")
            with open(save_path, "wb") as f:
                f.write(temp_buffer.getvalue())

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def save_high_res_copy(image_path, album_dir):
    """Save a high-resolution copy in Spawn/aux/user/hart/Artist/Album/cover_hi-res.jpg."""
    
    # Compute the Spawn directory by going up three levels.
    # For example, if album_dir is /Volumes/Untitled/Spawn/Music/311/Archive,
    # spawn_dir becomes /Volumes/Untitled/Spawn.
    spawn_dir = os.path.dirname(os.path.dirname(os.path.dirname(album_dir)))
    
    hart_root = os.path.join(spawn_dir, "aux", "user", "hart")
    music_base = os.path.join(spawn_dir, "Music")  # Expected base Music directory

    # Ensure album_dir is inside Music using a robust check.
    try:
        if os.path.commonpath([album_dir, music_base]) != music_base:
            logger.error(f"Album directory '{album_dir}' is not under expected 'Music' directory '{music_base}'.")
            return None
    except ValueError:
        logger.error(f"Path comparison failed for '{album_dir}' and '{music_base}'.")
        return None

    # Extract the relative path from Music (e.g. "311/Archive")
    relative_album_path = os.path.relpath(album_dir, music_base)

    # Construct the destination directory for the hi-res image
    high_res_dir = os.path.join(hart_root, relative_album_path)
    os.makedirs(high_res_dir, exist_ok=True)

    # Define the final path for the high-resolution image
    high_res_path = os.path.join(high_res_dir, "cover_hi-res.jpg")

    try:
        with open(image_path, "rb") as src, open(high_res_path, "wb") as dest:
            dest.write(src.read())
        logger.info(f"Saved high-resolution album art: {high_res_path}")
        return high_res_path
    except Exception as e:
        logger.error(f"Failed to save high-res image: {e}")
        return None

def embed_cover_art(audio_file, cover_jpeg_path):
    """Embeds album art into an MP4 (AAC/M4A) audio file."""
    try:

        # Skip hidden macOS files (._ prefixed files)
        if os.path.basename(audio_file).startswith("._"):
            logger.debug(f"Skipping hidden macOS file: {audio_file}")
            return

        audio = MP4(audio_file)

        # Read the JPEG file
        with open(cover_jpeg_path, "rb") as f:
            jpg_data = f.read()

        # Embed cover art
        mp4_cover = MP4Cover(jpg_data, imageformat=MP4Cover.FORMAT_JPEG)
        audio["covr"] = [mp4_cover]
        audio.save()
        logger.info(f"Embedded cover art into: {audio_file}")

    except mutagen.MutagenError as e:
        logger.error(f"Failed to embed cover art into {audio_file}: {e}")

def process_album(cover_path, album_dir):
    """Main function to process album: resize/optimize, save hi-res, and embed art."""
    # Ensure the album directory exists
    if not os.path.isdir(album_dir):
        logger.error(f"Album directory does not exist: {album_dir}")
        return

    # Define output paths
    optimized_cover_path = os.path.join(album_dir, "cover.jpg")

    # Process images
    resize_and_optimize_image(cover_path, optimized_cover_path)
    save_high_res_copy(cover_path, album_dir)

    # Embed album art into all audio files
    for filename in os.listdir(album_dir):
        if filename.endswith(".m4a"):
            audio_path = os.path.join(album_dir, filename)
            embed_cover_art(audio_path, optimized_cover_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python art_embedder.py path/to/cover.jpg path/to/Spawn/Music/Artist/Album")
        sys.exit(1)

    cover_image = sys.argv[1]
    album_directory = sys.argv[2]

    process_album(cover_image, album_directory)