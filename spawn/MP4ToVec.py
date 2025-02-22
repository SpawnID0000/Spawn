# MP4ToVec.py

# Trained model, techniques, and snippets of code adopted
#  from Deej-AI by Robert Dargavel Smith (https://github.com/teticio/Deej-AI)

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message=".*does not appear to have a file named diffusion_pytorch_model.safetensors.*")

import os
import logging
import numpy as np
import pickle
import librosa
import torch

#from audiodiffusion.audio_encoder import AudioEncoder
from .audiodiffusion.audio_encoder import AudioEncoder
#import tensorflow as tf
#from tensorflow.keras.models import load_model
from tqdm import tqdm

logger = logging.getLogger(__name__)


def cosine_proximity(y_true, y_pred):
    # "cosine_proximity" is simply the negative of "cosine_similarity"
    return -tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)

try:
    # If model needed custom objects, e.g. 'cosine_proximity':
    #custom_objs = {'cosine_proximity': tf.compat.v1.keras.losses.cosine_proximity}
    custom_objs = {
    # Provide our custom replacement for "cosine_proximity"
    'cosine_proximity': cosine_proximity
    }
    #custom_objs = {}
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not installed. Will fall back to placeholder embeddings.")
    TENSORFLOW_AVAILABLE = False



def load_mp4tovec_model_diffusion(model_path=None):
    """
    Load the diffusion-based AudioEncoder from a local directory.
    If model_path is None, defaults to the 'audio-encoder' subdirectory.
    """
    if model_path is None:
        # Use the audio-encoder subdirectory relative to this file
        model_path = os.path.join(os.path.dirname(__file__), "audio-encoder")
    try:
        # This will load the model using the local files in model_path
        encoder = AudioEncoder.from_pretrained(model_path)
        logger.info(f"Loaded AudioEncoder from local path: {model_path}")
        return encoder
    except Exception as e:
        logger.error(f"Failed to load AudioEncoder from '{model_path}': {e}")
        return None



# def load_mp4tovec_model_diffusion(model_id_or_path="teticio/audio-encoder"):
#     """
#     Load the Hugging Face audio-encoder from the 'audiodiffusion' library.
#     By default, use the Hub ID 'teticio/audio-encoder'.
#     If you have a local folder with all necessary files, pass that path here.
#     """
#     try:
#         encoder = AudioEncoder.from_pretrained(model_id_or_path)
#         return encoder
#     except Exception as e:
#         logger.error(f"Failed to load AudioEncoder from '{model_id_or_path}': {e}")
#         return None


# def load_mp4tovec_model_torch(model_path=None):
#     """
#     Loads a PyTorch model from a .bin (or .pt) file.
#     If model_path is None, default to 'diffusion_pytorch_model.bin'.
#     """
#     if model_path is None:
#         model_path = os.path.join(os.path.dirname(__file__), "diffusion_pytorch_model.bin")
#     if not os.path.isfile(model_path):
#         logger.error(f"PyTorch model file '{model_path}' not found. Cannot generate embeddings.")
#         return None
    
#     # Example: load a model architecture, then state_dict
#     # For example, if you had your own class MyModel(nn.Module) in the code:
#     #     model = MyModel()
#     #     model.load_state_dict(torch.load(model_path, map_location='cpu'))
#     #     model.eval()
#     #
#     # But if the entire model is pickled (less common these days):
#     #     model = torch.load(model_path, map_location='cpu')
#     #     model.eval()
#     #
#     # For demonstration, let's assume a pickled model that returns 512-dim embeddings:
    
#     try:
#         model = torch.load(model_path, map_location='cpu')
#         model.eval()
#         logger.info(f"Successfully loaded PyTorch model from {model_path}")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load PyTorch model: {e}")
#         logger.error("No valid PyTorch model will be used; returning None.")
#         return None


# def load_mp4tovec_model_tf(model_path=None):
#     """
#     Attempt to load a real MP4ToVec model (e.g. 'speccy_model.h5') via Keras.
#     If no path is given, default to checking 'speccy_model.h5' in this file's dir.
#     If that doesn't exist or if TF isn't installed, we return None (no embeddings).
#     """

#     # 1) If user didn't provide a path, look for speccy_model.h5 in same directory:
#     if not model_path:
#         default_path = os.path.join(os.path.dirname(__file__), "speccy_model.h5")
#         if os.path.isfile(default_path):
#             model_path = default_path
#         else:
#             logger.error("No model path and no speccy_model.h5 found. Cannot generate embeddings.")
#             return None

#     # 2) If the file doesn't exist at model_path, use placeholder
#     if not os.path.isfile(model_path):
#         logger.error(f"TF model file '{model_path}' not found. Cannot generate embeddings.")
#         return None

#     # 3) Try to load a real Keras model (only if TENSORFLOW_AVAILABLE)
#     if not TENSORFLOW_AVAILABLE:
#         logger.error(f"TensorFlow not installed, cannot load '{model_path}'. Returning None.")
#         return None

#     try:
#         # If your model needed custom_objects, pass them in:
#         #model = load_model(model_path, custom_objects=custom_objs, compile=False)
#         model = load_model(model_path, custom_objects=custom_objs, compile=False)
#         #model = load_model(model_path, compile=False)
#         #model = load_model(model_path)
#         if model is not None:
#             logger.info(f"Successfully loaded MP4ToVec model from {model_path}")
#             return model
#         else:
#             return None

#     except Exception as e:
#         logger.error(f"Failed to load model from {model_path}: {e}")
#         logger.error("No valid model will be used; returning None.")
#         return None


def generate_embedding(file_path, model):
    """
    Generate a 100-dim embedding via the loaded AudioEncoder.
    `model` is assumed to be an AudioEncoder instance.
    Returns a 1D numpy array of shape (100,) if successful, or None on failure.
    """
    if model is None:
        logger.warning("No AudioEncoder model loaded; skipping embedding.")
        return None

    try:
        # audio_encoder.encode expects a list of file paths
        #print("Running HF AudioEncoder on:", file_path)
        embeddings = model.encode([file_path])
        if embeddings is not None and len(embeddings) > 0:
            return embeddings[0]  # first (and only) track's embedding
    except Exception as e:
        logger.warning(f"Error generating embedding for {file_path}: {e}")
    return None

# ------------------------------------------------------------------------------
# [REFERENCE] Alternative 1: Manual Librosa Spectrogram Approach
# ------------------------------------------------------------------------------
#
# This demonstrates how you'd manually load audio with librosa, produce
# a mel-spectrogram, optionally slice/normalize, then feed it into your
# TensorFlow or PyTorch model. Useful if your embedding model doesn't
# handle raw audio directly and requires a specific input shape.
#
# def generate_embedding(file_path, model):
#     # 1) Load mono audio at 22050 Hz
#     y, sr = librosa.load(file_path, sr=22050, mono=True)
#     
#     # 2) Create a mel-spectrogram with n_mels=96
#     S = librosa.feature.melspectrogram(
#         y=y,
#         sr=sr,
#         n_fft=2048,
#         hop_length=512,
#         n_mels=96,
#         fmax=sr/2
#     )
#     # S.shape is (96, time_frames)
#
#     # 3) Ensure you have enough frames for your model
#     if S.shape[1] < 216:
#         raise ValueError(
#             f"Spectrogram has only {S.shape[1]} frames; need >= 216."
#         )
#
#     # 4) Slice to 216 frames (or chunk in a loop for multiple slices)
#     log_S = librosa.power_to_db(S[:, :216], ref=np.max)
#
#     # 5) Normalize 0..1 if needed
#     log_S -= log_S.min()
#     if log_S.max() > 0:
#         log_S /= log_S.max()
#
#     # 6) Reshape => shape (1, 96, 216, 1) for typical Keras CNN input
#     x_input = log_S.reshape((1, 96, 216, 1))
#
#     # 7) Run inference
#     embedding = model.predict(x_input, verbose=0)
#     # => typically shape (1, 100) or (1, 512), etc.
#
#     return embedding.flatten()


# ------------------------------------------------------------------------------
# [REFERENCE] Alternative 2: Generic TF/PyTorch Flow
# ------------------------------------------------------------------------------
#
# This snippet shows how you might handle different model APIs. If 'model'
# is a Keras model (has a .predict() method), we pass the input accordingly;
# if it's PyTorch (has .eval()), we do a forward pass with torch.no_grad().
# 
# NOTE: We no longer provide random placeholder embeddings, so if 'model'
# is invalid or unrecognized, you'd return None or raise an error.
#
# def generate_embedding(file_path, model):
#     if not os.path.isfile(file_path):
#         raise FileNotFoundError(f"File not found: {file_path}")
#
#     # Example: you'd replace this with real feature extraction or spectrogram
#     # so you can feed your model the correct shape. For demonstration:
#     # "fake" input => shape (100, 40).
#     features = np.zeros((100, 40), dtype=np.float32)
#
#     # Distinguish different model types:
#     if hasattr(model, 'predict'):  # Keras
#         batch_input = features.reshape((1, 100, 40, 1))
#         embedding = model.predict(batch_input, verbose=0)
#         return embedding.flatten()
#
#     else:
#         # Assume PyTorch if there's an .eval() method or parameters
#         import torch
#         with torch.no_grad():
#             input_tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).float()
#             # => shape: (batch=1, channels=1, time=100, freq=40) or as needed
#             embedding_torch = model(input_tensor)
#             embedding_np = embedding_torch.cpu().numpy().flatten()
#         return embedding_np


def batch_generate_embeddings(file_paths, model, output_pickle):
    """
    Generate embeddings for a batch of audio files and save them in a pickle file.
    """
    embeddings = {}

    for file_path in tqdm(file_paths, desc="Generating embeddings"):
        try:
            embedding = generate_embedding(file_path, model)
            embeddings[file_path] = embedding
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")

    # Save all embeddings
    with open(output_pickle, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"Embeddings saved to {output_pickle}")


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to model (defaults to diffusion_pytorch_model.bin in same folder).", default=None)
    parser.add_argument("--files", nargs="*", help="Audio files to embed.")
    parser.add_argument("--output", help="Pickle file to store embeddings.", default="mp4tovec.p")
    args = parser.parse_args()

    #model = load_mp4tovec_model_tf(args.model)
    #model = load_mp4tovec_model_torch(args.model)
    model = load_mp4tovec_model_diffusion(args.model)

    if not args.files:
        logger.info("No files given. Exiting.")
        sys.exit(0)

    batch_generate_embeddings(args.files, model, args.output)
