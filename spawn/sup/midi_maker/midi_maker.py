#!/usr/bin/env python3
"""
Dependencies:
  pip install librosa pyloudnorm mido music21 Cython numpy wheel demucs torchaudio
  brew install ffmpeg libsndfile
  pip install --no-use-pep517 madmom
  pip install soundfile
  pip install torch torchaudio torchlibrosa

PANNs Setup:
git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git

cp audioset_tagging_cnn/pytorch/models.py
cp audioset_tagging_cnn/pytorch/pytorch_utils.py
cp audioset_tagging_cnn/utils/utilities.py
cp audioset_tagging_cnn/metadata/class_labels_indices.csv

curl -L "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1" \
     -o pretrained_models/Cnn14_16k_mAP=0.438.pth

"""

import sys
import types
import os
import argparse
import warnings
import numpy as np
import librosa
import librosa.display
from librosa.core.audio import __audioread_load
import pyloudnorm as pyln
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo, tempo2bpm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import subprocess
import shutil
import re
import torch
import torchaudio
import csv

try:
    import soundfile
except ImportError:
    pass

warnings.filterwarnings(
    "ignore",
    message=".*librosa\\.core\\.audio\\.__audioread_load.*",
    category=FutureWarning
)

# Compatibility for numpy.complex
if not hasattr(np, 'complex'):
    np.complex = complex

# External analysis tools
try:
    from music21 import roman, key as m21key, chord as m21chord
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

try:
    from madmom.features.chords import CNNChordFeatureProcessor
    from madmom.features.chords import CRFChordRecognitionProcessor
    MADMOM_AVAILABLE = True

except (ImportError, AttributeError) as e:
    error_msg = str(e)
    if ("np.float" in error_msg) or ("np.int" in error_msg):
        print("⚠️ Madmom appears to be broken due to numpy deprecations.")
        print("Attempting to auto-patch Madmom...")

        patcher_path = os.path.join(os.path.dirname(__file__), 'madmom_patcher.py')

        if not os.path.isfile(patcher_path):
            print(f"❌ Madmom patcher script not found at {patcher_path}. Skipping auto-patch.")
            MADMOM_AVAILABLE = False
        else:
            try:
                import madmom
                madmom_path = os.path.dirname(madmom.__file__)
                subprocess.run([sys.executable, patcher_path, madmom_path], check=True)

                # Retry import
                from madmom.features.chords import CNNChordFeatureProcessor
                from madmom.features.chords import CRFChordRecognitionProcessor
                MADMOM_AVAILABLE = True
                print("✅ Madmom successfully patched and loaded!")

            except subprocess.CalledProcessError as sub_err:
                print(f"❌ Madmom patcher failed to execute: {sub_err}")
                MADMOM_AVAILABLE = False
            except Exception as patch_err:
                print(f"❌ Auto-patching failed: {patch_err}")
                MADMOM_AVAILABLE = False
    else:
        print(f"❌ Madmom import failed due to unexpected error: {e}")
        MADMOM_AVAILABLE = False


# PLP constants
PLP_AGG_WINDOW_S = 0.01
PLP_QUANT_BINS   = 64

# For speech vs singing detection
HERE = os.path.dirname(__file__)
from models      import Cnn14


# --- Audio analysis utilities ---

def analyze_audio(file_path, sr=44100):
    # bypass SoundFile entirely, decode via audioread
    y, sr_native = __audioread_load(file_path, offset=0.0, duration=None, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    if sr is not None and sr != sr_native:
        y = librosa.resample(y, orig_sr=sr_native, target_sr=sr)
        sr_native = sr
    return y, sr_native


def compute_loudness_curve(y, sr, window_s=1.0, hop_s=0.2):
    meter = pyln.Meter(sr)
    win = int(sr * window_s)
    hop = int(sr * hop_s)
    curve = []
    for i in range(0, len(y) - win, hop):
        seg = y[i:i+win]
        try:
            lufs = meter.integrated_loudness(seg)
            if not np.isfinite(lufs): lufs = -127.0
        except:
            lufs = -127.0
        cc = int(min(max(abs(lufs), 0), 127))  # clamp to 0–127
        curve.append((i/sr, cc))
    return curve


def detect_beats(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(frames, sr=sr)
    strengths = onset_env[frames]
    tempo = float(tempo) if not hasattr(tempo, 'item') else tempo.item()
    return tempo, beat_times, strengths


def detect_key(file_path, use_stems=False):
    """
    Estimate key using Librosa. If use_stems=True, combines bass + other stems for cleaner detection.
    """
    try:
        target_audio = file_path

        if use_stems:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            audio_dir = os.path.dirname(os.path.abspath(file_path))
            stem_dir  = os.path.join(audio_dir, f"stems_{base_name}")

            bass_stem  = os.path.join(stem_dir, f"{base_name} - bass.m4a")
            other_stem = os.path.join(stem_dir, f"{base_name} - other.m4a")

            if os.path.exists(bass_stem) and os.path.exists(other_stem):
                temp_wav = os.path.join(audio_dir, "temp_key_detect.wav")

                # Mix stems using ffmpeg's amix filter
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", bass_stem, "-i", other_stem,
                    "-filter_complex", "amix=inputs=2:duration=longest",
                    "-c:a", "pcm_s16le",
                    temp_wav
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                target_audio = temp_wav
            else:
                print("⚠️ Bass or Other stem not found. Falling back to full mix.")

        # --- Librosa Key Detection ---
        y, sr = analyze_audio(target_audio)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_profile /= major_profile.sum()
        minor_profile /= minor_profile.sum()

        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        best_score = -np.inf
        best_key, best_mode = None, None

        for i in range(12):
            score_maj = np.corrcoef(np.roll(major_profile, i), chroma_mean)[0,1]
            score_min = np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0,1]
            if score_maj > best_score:
                best_score = score_maj
                best_key = keys[i]
                best_mode = 'major'
            if score_min > best_score:
                best_score = score_min
                best_key = keys[i]
                best_mode = 'minor'

        result = f"{best_key} {best_mode}"

        # Clean up temp file if created
        if use_stems and os.path.exists(target_audio) and target_audio.endswith("temp_key_detect.wav"):
            os.remove(target_audio)

        return result

    except Exception as e:
        print(f"Key detection failed: {e}")
        return "Unknown"


def detect_chords(file_path, use_stems=False):
    """
    Detect chords using Madmom. If use_stems=True, sum bass + other stems before detection.
    """
    if not MADMOM_AVAILABLE:
        print("Madmom not available. Skipping chord detection.")
        return []

    try:
        target_audio = file_path

        if use_stems:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            audio_dir = os.path.dirname(os.path.abspath(file_path))
            stem_dir  = os.path.join(audio_dir, f"stems_{base_name}")

            bass_stem  = os.path.join(stem_dir, f"{base_name} - bass.m4a")
            other_stem = os.path.join(stem_dir, f"{base_name} - other.m4a")

            if os.path.exists(bass_stem) and os.path.exists(other_stem):
                temp_wav = os.path.join(audio_dir, "temp_chord_detect.wav")

                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", bass_stem, "-i", other_stem,
                    "-filter_complex", "amix=inputs=2:duration=longest",
                    "-c:a", "pcm_s16le",
                    temp_wav
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                target_audio = temp_wav
            else:
                print("⚠️ Bass or Other stem not found for chords. Falling back to full mix.")

        # --- Madmom Chord Detection ---
        feats = CNNChordFeatureProcessor()(target_audio)
        chords = CRFChordRecognitionProcessor()([feats])

        # Clean up temp file if created
        if use_stems and os.path.exists(target_audio) and target_audio.endswith("temp_chord_detect.wav"):
            os.remove(target_audio)

        return chords

    except Exception as e:
        print(f"Chord detection failed: {e}")
        return []


def get_music21_key(key_str):
    if not MUSIC21_AVAILABLE or key_str == "Unknown":
        return None
    try:
        tonic, mode = key_str.split()
        return m21key.Key(tonic) if mode.lower() == 'major' else m21key.Key(tonic).relative
    except:
        return None


def chord_to_roman(chord_label, tonal_key):
    if not MUSIC21_AVAILABLE or chord_label == 'N':
        return 'N.C.'
    m = re.match(r'([A-G][#b]?)(.*)', chord_label)
    if not m:
        return chord_label
    root, _ = m.groups()
    rn = roman.romanNumeralFromChord(m21chord.Chord(root), tonal_key)
    return str(rn.figure)


# --- MIDI creation ---

def create_midi(
    audio_path,
    loudness_curve,
    beat_times,
    tempo_bpm,
    key_str,
    chord_segments,
    output_path,
    beat_strengths,
    plp_times,
    plp_curve,
    pyloudnorm_lufs=None,
    rsgain_lufs=None,
    speech_regions=None,
    applause_regions=None,
    char_dist=None,
    temp_dist=None
):
    mid = MidiFile(ticks_per_beat=480)
    tracks = [MidiTrack() for _ in range(4)]
    for t in tracks:
        mid.tracks.append(t)

    # ─── Track Assignments ─────────────────────────────────────────────
    beat_track   = mid.tracks[0]  # Tempo, beat markers, beat strength, and beat salience (PLP)
    loud_track   = mid.tracks[1]  # Dynamic loudness curve & static LUFS values from pyloudnorm & rsgain
    cord_track  = mid.tracks[2]  # Chord labels, Roman numeral notation, key metadata
    lyrx_track  = mid.tracks[3]   # Lyrics track

    beat_track.name = "Beats"
    loud_track.name = "Loudness"
    cord_track.name = "Chords"
    lyrx_track.name = "Lyrics"

    # Tempo, signature, label
    beat_track.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm), time=0))
    beat_track.append(MetaMessage('text', text=f"Tempo: {tempo_bpm:.1f} BPM", time=0))
    beat_track.append(MetaMessage('text', text="Time Signature: Unknown", time=0))

    # ─────────────────────────────────────────────────────────────────────────────
    # MIDI Structure - Channels & CC #s
    #
    # Within each track, messages are sent to specific MIDI channels (0–15), with custom use:
    #
    #   Channel 12: Static LUFS (rsgain)        — CC #20
    #   Channel 13: Static LUFS (pyloudnorm)    — CC #20
    #   Channel 14: Time-varying metadata
    #       • Loudness curve                    — CC #20
    #       • Beat strength                     — CC #21
    #       • Beat salience (PLP)               — CC #22
    #       • Chord label markers               — CC #29
    #       • Chord roman numeral markers       — CC #30
    #
    # ─────────────────────────────────────────────────────────────────────────────
    # MIDI Track Summary
    #
    #   Track 0: Tempo + Beats
    #     - Channel 14, CC #21 → Beat Strength
    #     - Channel 14, CC #22 → Beat Salience (PLP)
    #
    #   Track 1: Loudness
    #     - Channel 12, CC #20 → rsgain static LUFS
    #     - Channel 13, CC #20 → pyloudnorm static LUFS
    #     - Channel 14, CC #20 → Loudness curve
    #
    #   Track 2: Chords
    #     - Channel 14, CC #29 → Chord label marker
    #     - Channel 14, CC #30 → Roman numeral chord marker
    #
    # ─────────────────────────────────────────────────────────────────────────────


    def sec2tick(s):
        return int((s / (bpm2tempo(tempo_bpm)/1e6)) * mid.ticks_per_beat)

    # Beat & PLP Events (track 0)
    events = []
    if len(beat_strengths):
        max_s = np.max(beat_strengths)
        str_vals = np.clip(beat_strengths/max_s*127, 0, 127).astype(int)
    else:
        str_vals = []

    agg_times, agg_vals = [], []
    end_t = plp_times[-1] if len(plp_times) else 0
    w = 0.0
    while w < end_t:
        mask = (plp_times >= w) & (plp_times < w + PLP_AGG_WINDOW_S)
        if mask.any():
            agg_times.append(w + PLP_AGG_WINDOW_S/2)
            agg_vals.append(plp_curve[mask].max())
        w += PLP_AGG_WINDOW_S
    amp = np.max(plp_curve) if plp_curve.size else 1.0
    step = 127/(PLP_QUANT_BINS-1)
    qvals = [int(round((v/amp*127)/step)*step) for v in agg_vals]

    for t, s in zip(beat_times, str_vals):
        events.append((t, 'marker', None))
        events.append((t, 'strength', int(s)))
    for t, v in zip(agg_times, qvals):
        events.append((t, 'plp', v))
    events.sort(key=lambda x: x[0])

    last_tick = 0
    for t, kind, val in events:
        tick = sec2tick(t)
        delta = tick - last_tick
        if kind == 'marker':
            beat_track.append(MetaMessage('marker', text='Beat', time=delta))
        elif kind == 'strength':
            beat_track.append(Message('control_change', control=21, value=val, channel=14, time=delta))
        else:
            beat_track.append(Message('control_change', control=22, value=val, channel=14, time=delta))
        last_tick = tick

    # Dynamic Loudness Curve (track 1, channel 14)
    last_tick = 0
    for ts, cc in loudness_curve:
        tick = sec2tick(ts)
        delta = max(1, tick - last_tick)
        loud_track.append(Message('control_change', control=20, value=cc, channel=14, time=delta))
        last_tick = tick

    # Static pyloudnorm LUFS (track 1, channel 13)
    if pyloudnorm_lufs is not None:
        cc_val = int(np.clip(abs(pyloudnorm_lufs), 0, 127))
        loud_track.append(Message('control_change', control=20, value=cc_val, channel=13, time=0))
        loud_track.append(MetaMessage('text', text=f"pyloudnorm LUFS: {pyloudnorm_lufs:.1f}", time=0))

    # Static rsgain LUFS (track 1, channel 12)
    if rsgain_lufs is not None:
        cc_val = int(np.clip(abs(rsgain_lufs), 0, 127))
        loud_track.append(Message('control_change', control=20, value=cc_val, channel=12, time=0))
        loud_track.append(MetaMessage('text', text=f"rsgain LUFS: {rsgain_lufs:.1f}", time=0))

    # Chord Detection (track 2, channel 14)
    cord_track.append(MetaMessage('text', text=f"Key: {key_str}", time=0))
    last_tick = 0
    for s, e, ch in chord_segments:
        tick = sec2tick(s)
        delta = tick - last_tick

        # Optional: add a marker on channel 14
        cord_track.append(Message('control_change', control=29, value=0, channel=14, time=delta))

        # Human-readable chord label
        cord_track.append(MetaMessage('text', text=f"Chord: {ch}", time=0))

        last_tick = tick

    # Roman Numeral Chord Analysis (track 2, channel 14)
    tonal = get_music21_key(key_str) if key_str != "Unknown" else None
    if tonal:
        last_tick = 0
        for s, e, ch in chord_segments:
            tick = sec2tick(s)
            delta = tick - last_tick
            roman_str = chord_to_roman(ch, tonal)

            # Channel 14 marker for Roman numeral
            cord_track.append(Message('control_change', control=30, value=0, channel=14, time=delta))

            # Human-readable Roman numeral
            cord_track.append(MetaMessage('text', text=f"Roman: {roman_str}", time=0))

            last_tick = tick

    # annotate if we have PANNs data
    if speech_regions is not None:
        annotate_midi_lyrics(
          lyrx_track,
          speech_regions,
          applause_regions or [],
          char_dist or {"female":0,"male":0,"other":0},
          temp_dist or {"female":0,"male":0,"other":0},
          sec2tick
        )

    # Save MIDI File
    mid.save(output_path)
    print(f"MIDI saved to: {output_path}")


# --- MIDI inspection & parsing ---

def inspect_midi(midi_file):
    mid = MidiFile(midi_file)
    tb = mid.ticks_per_beat     # ticks per beat (e.g. 480)
    tempo = 500000              # default μsec/beat, will be updated

    print(f"--- MIDI File: {midi_file} ---")
    print(f"Ticks per beat: {tb}")

    # Containers for speech/applause segments
    speech_segments   = []
    applause_segments = []
    current_speech    = None
    current_applause  = None

    for idx, tr in enumerate(mid.tracks):
        print(f"\n-- Track {idx} - {tr.name} --")
        tick_acc = 0

        for msg in tr:
            tick_acc += msg.time
            # update tempo if present
            if msg.type == 'set_tempo':
                tempo = msg.tempo

            # compute wall‐clock seconds
            secs = (tick_acc / tb) * (tempo / 1_000_000)

            # 1) Text events → print + segment tracking
            if msg.type == 'text':
                print(f"{msg!r}    ({secs:.2f} s)")

                if msg.text == 'Speech Start':
                    current_speech = secs
                elif msg.text == 'Speech End' and current_speech is not None:
                    speech_segments.append((current_speech, secs))
                    current_speech = None

                if msg.text == 'Applause Start':
                    current_applause = secs
                elif msg.text == 'Applause End' and current_applause is not None:
                    applause_segments.append((current_applause, secs))
                    current_applause = None

            # 2) Control‐change markers → print + secs + annotation
            elif msg.type == 'control_change' and msg.control in (20,29,30):
                # base print
                print(f"{msg!r}    ({secs:.2f} s)")

                # your existing CC annotations
                if msg.control == 20 and msg.channel == 13:
                    print(f"    (pyloudnorm static LUFS: -{msg.value})")
                elif msg.control == 20 and msg.channel == 12:
                    print(f"    (rsgain static LUFS: -{msg.value})")
                elif msg.control == 29 and msg.channel == 14:
                    print("    (Chord label marker)")
                elif msg.control == 30 and msg.channel == 14:
                    print("    (Roman numeral marker)")

            # 3) All other messages → print raw
            else:
                print(msg)

    # After all tracks, dump the consolidated segment lists
    print("\nSpeech segments (s):")
    for start, end in speech_segments:
        print(f"  ({start:.2f}, {end:.2f})")

    print("\nApplause segments (s):")
    for start, end in applause_segments:
        print(f"  ({start:.2f}, {end:.2f})")


def parse_midi_metadata(midi_file):
    mid = MidiFile(midi_file)
    tb = mid.ticks_per_beat
    tempo = 500000

    beat_times, beat_strengths, plp_times, plp_vals = [], [], [], []
    loud_times, loud_vals = [], []
    chord_labels = []        # [(time_in_sec, "C:maj")]
    roman_numerals = []      # [(time_in_sec, "IV")]
    speech_segs   = []
    applause_segs = []
    current = None

    pyloudnorm_lufs = None
    rsgain_lufs = None

    # Track 0: beat and PLP
    tick_acc = 0
    for msg in mid.tracks[0]:
        tick_acc += msg.time
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif msg.type == 'marker' and getattr(msg, 'text', '') == 'Beat':
            beat_times.append(tick_acc / tb * (tempo / 1e6))
        elif msg.type == 'control_change':
            t = tick_acc / tb * (tempo / 1e6)
            if msg.control == 21:
                beat_strengths.append((t, msg.value))
            elif msg.control == 22:
                plp_times.append(t)
                plp_vals.append(msg.value)

    # Track 1: Loudness
    tick_acc = 0
    for msg in mid.tracks[1]:
        tick_acc += msg.time
        if msg.type == 'control_change' and msg.control == 20:
            if msg.channel == 14:
                t = tick_acc / tb * (tempo / 1e6)
                loud_times.append(t)
                loud_vals.append(msg.value)
            elif msg.channel == 13:
                pyloudnorm_lufs = -float(msg.value)
            elif msg.channel == 12:
                rsgain_lufs = -float(msg.value)

    # Track 2: Chords + Roman numerals
    tick_acc = 0
    context = None  # 'chord' or 'roman'
    for msg in mid.tracks[2]:
        tick_acc += msg.time
        t = tick_acc / tb * (tempo / 1e6)

        if msg.type == 'control_change':
            if msg.channel == 14 and msg.control == 29:
                context = 'chord'
            elif msg.channel == 14 and msg.control == 30:
                context = 'roman'
        elif msg.type == 'text':
            if context == 'chord' and msg.text.startswith("Chord:"):
                label = msg.text.replace("Chord:", "").strip()
                chord_labels.append((t, label))
                context = None
            elif context == 'roman' and msg.text.startswith("Roman:"):
                rn = msg.text.replace("Roman:", "").strip()
                roman_numerals.append((t, rn))
                context = None

    # Track 3: Lyrics + Vocals
    tick_acc = 0
    current_speech  = None
    current_applause = None
    for msg in mid.tracks[3]:
        tick_acc += msg.time
        t = tick_acc / tb * (tempo/1e6)

        if msg.type == 'text':
            # speech
            if msg.text == 'Speech Start':
                current_speech = t
            elif msg.text == 'Speech End' and current_speech is not None:
                speech_segs.append((current_speech, t))
                current_speech = None

            # applause
            if msg.text == 'Applause Start':
                current_applause = t
            elif msg.text == 'Applause End' and current_applause is not None:
                applause_segs.append((current_applause, t))
                current_applause = None


    return {
        'tempo': tempo,
        'beat_times': beat_times,
        'beat_strengths': beat_strengths,
        'plp_times': plp_times,
        'plp_vals': plp_vals,
        'loud_times': loud_times,
        'loud_vals': loud_vals,
        'pyloudnorm_lufs': pyloudnorm_lufs,
        'rsgain_lufs': rsgain_lufs,
        'chord_labels': chord_labels,
        'roman_numerals': roman_numerals,
        'speech_segments': speech_segs,
        'applause_segments': applause_segs
    }


def inspect_stems_if_requested(audio_file, inspect_mode):
    if audio_file:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        audio_dir = os.path.dirname(os.path.abspath(audio_file))
        stem_dir  = os.path.join(audio_dir,  f"stems_{base_name}")
        if os.path.isdir(stem_dir):
            stem_files = sorted(f for f in os.listdir(stem_dir) if f.endswith('.mid'))
            for mid_name in stem_files:
                stem_mid = os.path.join(stem_dir, mid_name)
                stem_audio = stem_mid.replace('.mid', '.m4a')
                if not os.path.exists(stem_audio):
                    stem_audio = None
                print(f"\nInspecting stem: {os.path.basename(stem_mid)}")
                if inspect_mode == 'console':
                    inspect_midi(stem_mid)
                elif inspect_mode == 'plot':
                    plot_midi_with_optional_audio(stem_mid, stem_audio)
        else:
            print(f"Stem folder not found: {stem_dir}")


# --- Plotting ---

def plot_midi_with_optional_audio(midi_file, audio_file=None):
    """
    Unified plot of MIDI-derived beats, PLP salience, loudness (MIDI),
    pyloudnorm & rsgain replay‑gain, plus optional audio waveform & spectrogram.
    """
    # ─── 1) Parse MIDI metadata ────────────────────────────────────────────────
    meta = parse_midi_metadata(midi_file)
    bpm = tempo2bpm(meta['tempo'])  # for the legend
    speech_segs   = meta['speech_segments']
    applause_segs = meta['applause_segments']

    # ─── 2) Decode audio ───────────────────────────────────────────
    if audio_file:
        y, sr = analyze_audio(audio_file)
        t_audio = np.linspace(0, len(y)/sr, len(y))
        # compute loudness via pyloudnorm
        meter = pyln.Meter(sr)
        track_lufs = meter.integrated_loudness(y)
        target_lufs = -18.0
        replay_gain = target_lufs - track_lufs
        rg_cc = int(np.clip(abs(track_lufs), 0, 127))
        # locate rsgain
        possible = [shutil.which("rsgain"), "/usr/local/bin/rsgain",
                    "/opt/homebrew/bin/rsgain", "/usr/bin/rsgain"]
        rg_cmd = next((p for p in possible if p and os.path.isfile(p)), None)
        cli_line = None
        if rg_cmd:
            try:
                proc = subprocess.run(
                    [rg_cmd, "custom", "--tagmode=s", "-O", audio_file],
                    capture_output=True, text=True, check=True
                )
                for line in proc.stdout.splitlines():
                    if os.path.basename(audio_file) in line:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            lufs_cli = float(parts[1].replace(" LU","").strip())
                            gain_cli = target_lufs - lufs_cli
                            rg_cli_cc = int(np.clip(abs(lufs_cli), 0, 127))
                            cli_line = (rg_cli_cc, lufs_cli, gain_cli)
                        break
            except Exception as e:
                print("rsgain scan failed (safe to ignore):", e)
    else:
        # placeholders so references below won’t error
        y = None; sr = None
        track_lufs = meta.get("pyloudnorm_lufs")
        rg_cc = int(np.clip(abs(track_lufs), 0, 127)) if track_lufs is not None else None
        lufs_cli = meta.get("rsgain_lufs")
        rg_cli_cc = int(np.clip(abs(lufs_cli), 0, 127)) if lufs_cli is not None else None
        replay_gain = -18.0 - track_lufs if track_lufs is not None else None
        cli_line = (rg_cli_cc, lufs_cli, -18.0 - lufs_cli) if lufs_cli is not None else None

    # ─── 3) Create figure layout ────────────────────────────────────────────────
    # Two-row grid if audio; one-row if MIDI-only
    rows = 2 if audio_file else 1
    fig = plt.figure(figsize=(12, 8.75 if audio_file else 6.75))
    gs = gridspec.GridSpec(rows, 1, height_ratios=[2]*(rows-1) + [1], hspace=0.005)
    ax_main = fig.add_subplot(gs[0])

    # ─── 4) Plot audio waveform ────────────────────────────────────────────────
    plot_elements = []
    plot_labels = []

    ax_main.set_ylim(-1, 1)
    ax_main.set_ylabel("Amplitude")

    if audio_file:
        waveform, = ax_main.plot(t_audio, y, label='Audio Waveform', color='blue')
        plot_elements.append(waveform)
        plot_labels.append("Audio Waveform")
        ax_main.set_xlim(0, len(y) / sr)
        ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        ax_main.set_xlabel('Time (s)')
        max_time = max(
            max(meta['beat_times'], default=0),
            max(meta['loud_times'], default=0),
            max(meta['plp_times'], default=0)
        )
        ax_main.set_xlim(0, max_time)

    # ─── 5) Plot beat markers & strengths ─────────────────────────────────────
    str_vals = [v for _,v in meta['beat_strengths']]
    str_norm = (np.array(str_vals)/127) if str_vals else np.ones(len(meta['beat_times']))
    for i,bt in enumerate(meta['beat_times']):
        ax_main.vlines(bt, -1, 1, linewidth=0.1, color='gray', zorder=1)
        if i < len(str_norm):
            h = str_norm[i] * 2
            ax_main.vlines(bt, -h/2, h/2, linewidth=1, color='red', alpha=0.9,
                           zorder=3, label='Beat Strength (Tempo)' if i==0 else None)
    # Add dummy handle for beat strength to appear in the legend
    beat_strength_legend = plt.Line2D([], [], color='red', linewidth=1, label='Beat Strength (Tempo)')
    plot_elements.append(beat_strength_legend)
    plot_labels.append("Beat Strength (Tempo)")

    # ─── 6) Plot PLP salience ───────────────────────────────────────────────────
    plp_line, = ax_main.plot(meta['plp_times'], [v/127 for v in meta['plp_vals']],
                             label='Beat Salience (PLP)', color='magenta', linewidth=1.5, alpha=0.6, zorder=2)

    # ─── Speech & Applause Shading (only on PLP) ───────────────────────────
    speech_plotted = False
    applause_plotted = False

    plot_elements.append(plp_line)
    plot_labels.append("Beat Salience (PLP)")

    # ─── 7) Secondary axis: loudness + replay‑gain ─────────────────────────────
    ax2 = ax_main.twinx()
    # Add a blank spacer in the legend for visual separation
    if audio_file:
        spacer = plt.Line2D([], [], color='none', label='')
        plot_elements.append(spacer)
        plot_labels.append('')
    loud_line, = ax2.plot(meta['loud_times'], meta['loud_vals'], label='Loudness', color='limegreen', linewidth=1, alpha=0.9)
    plot_elements.append(loud_line)
    plot_labels.append("Loudness")

    if track_lufs is not None:
        lufs_line = ax2.axhline(rg_cc, color='black', linestyle='--', linewidth=1,
                                label=f"pyloudnorm ({track_lufs:.1f} LUFS → {replay_gain:.1f} dB)")
        plot_elements.append(lufs_line)
        plot_labels.append(f"pyloudnorm ({track_lufs:.1f} LUFS → {replay_gain:.1f} dB)")

    if cli_line:
        cc_cli, lufs_cli, gain_cli = cli_line
        rsgain_line = ax2.axhline(cc_cli, color='orange', linestyle=':', linewidth=2,
                                  label=f"rsgain ({lufs_cli:.1f} LUFS → {gain_cli:.1f} dB)")
        plot_elements.append(rsgain_line)
        plot_labels.append(f"rsgain ({lufs_cli:.1f} LUFS → {gain_cli:.1f} dB)")

    ax2.set_ylim(127, 0)
    ticks = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 127]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"-{t}" for t in ticks])
    ax2.set_ylabel("Loudness (LUFS)")

    # ─── 8) Legend ──────────────────────────────────────────────────
    tempo_handle = plt.Line2D([], [], color='none')
    tempo_label = f"Tempo = {bpm:.1f} BPM"

    # Insert tempo legend after Audio Waveform if found
    try:
        idx = plot_labels.index("Audio Waveform")
    except ValueError:
        idx = -1
    plot_elements.insert(idx + 1, tempo_handle)
    plot_labels.insert(idx + 1, tempo_label)

    # Reorganize legend entries
    legend_order = [
        "Audio Waveform",
        "",  # spacer
        "Loudness",
        f"Tempo = {bpm:.1f} BPM",
        "Beat Strength (Tempo)",
        "Beat Salience (PLP)",
    ]

    # Append optional LUFS lines if they exist
    if any("pyloudnorm" in lbl for lbl in plot_labels):
        legend_order.append(next(lbl for lbl in plot_labels if "pyloudnorm" in lbl))
    if any("rsgain" in lbl for lbl in plot_labels):
        legend_order.append(next(lbl for lbl in plot_labels if "rsgain" in lbl))

    # Rebuild the elements list in new order
    reordered_elements = []
    reordered_labels = []

    for label in legend_order:
        if label in plot_labels:
            idx = plot_labels.index(label)
            reordered_elements.append(plot_elements[idx])
            reordered_labels.append(plot_labels[idx])

    legend = ax_main.legend(
        reordered_elements, reordered_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.3,
        ncol=4,
        frameon=False
    )

    # ─── 9) Add filename title above legend ─────────────────────────────────
    raw_name = os.path.basename(audio_file or midi_file)
    title_str = re.sub(r'^.*?\[', '[', raw_name).rsplit('.', 1)[0]  # keep from "[" onward, strip extension
    fig.suptitle(title_str, fontsize=14, y=0.975, fontweight='bold')

    # ─── 10) Spectrogram ─────────────────────────────────────────────
    if audio_file:
        ax_spec = fig.add_subplot(gs[1], sharex=ax_main)
        ax_spec.xaxis.set_major_locator(ax_main.xaxis.get_major_locator())
        ax_spec.xaxis.set_major_formatter(ax_main.xaxis.get_major_formatter())
        S = librosa.feature.melspectrogram(y=y, sr=sr,
                                           n_fft=2048, hop_length=512,
                                           n_mels=128, fmax=sr/2)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=512,
                                 x_axis='time', y_axis='mel',
                                 fmin=20, fmax=sr/2,
                                 ax=ax_spec, cmap='magma')
        ax_spec.set_xlabel('Time (s)')
        ax_spec.set_ylabel('Frequency (Hz)')
        #ax_spec.set_title('Mel Spectrogram', x=0.05, y=-0.2, ha='left')
        plt.setp(ax_spec.get_xticklabels(), visible=True)

    if audio_file:
        plt.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.07)
    else:
        plt.subplots_adjust(left=0.07, right=0.90, top=0.84, bottom=0.07)

    plt.show(block=False)
    plt.pause(0.5)

    # ─── 11) Speech/Singing & Applause Probability Plots ─────────────────────────────

    if audio_file:
        # if the AUDIO you’re plotting lives in a “stems_*” folder, skip it
        audio_parent = os.path.basename(os.path.dirname(os.path.abspath(audio_file)))
        if not audio_parent.startswith("stems_"):
            # and only if this MIDI is exactly the mix’s own .mid
            mix_mid = os.path.splitext(os.path.basename(audio_file))[0] + ".mid"
            if os.path.basename(midi_file) == mix_mid:
                # now plot both speech/singing & applause probability figures
                model, label_map = load_panns_model()
                plot_speech_singing_probs(audio_file, model, label_map)
                plot_applause_probs   (audio_file, model, label_map)
                # small pause so the GUI can update, non‐blocking
                plt.show(block=False)
                plt.pause(0.5)


def plot_stem_overlay(audio_file=None, midi_file=None):
    """
    Overlay the per-stem loudness & beat-salience plots, adding
    speech/applause shading and tempo legends to both figures.
    """
    if not audio_file and not midi_file:
        print("No input file provided for stem overlay plot.")
        return

    # Determine base name & stem folder
    input_file = audio_file or midi_file
    base_name  = os.path.splitext(os.path.basename(input_file))[0]
    base_dir   = os.path.dirname(os.path.abspath(input_file))
    stem_dir   = os.path.join(base_dir, f"stems_{base_name}")
    if not os.path.isdir(stem_dir):
        print(f"Stem folder not found: {stem_dir}")
        return

    # Load speech/applause from the main MIDI
    if midi_file:
        main_midi = midi_file
    else:
        main_midi = os.path.splitext(audio_file)[0] + '.mid'
    meta_main     = parse_midi_metadata(main_midi)
    speech_segs   = meta_main.get('speech_segments', [])
    applause_segs = meta_main.get('applause_segments', [])
    bpm           = tempo2bpm(meta_main['tempo'])

    # Load audio & spectrogram if available
    if audio_file:
        y, sr = analyze_audio(audio_file)
        duration_sec = len(y) / sr
        S = librosa.feature.melspectrogram(y=y, sr=sr,
                                           n_fft=2048, hop_length=512,
                                           n_mels=128, fmax=sr/2)
        S_db = librosa.power_to_db(S, ref=np.max)
    else:
        # estimate duration from MIDI only
        duration_sec = 0
        for mid_name in os.listdir(stem_dir):
            if not mid_name.endswith('.mid'): continue
            meta = parse_midi_metadata(os.path.join(stem_dir, mid_name))
            tmax = max(
                max(meta['beat_times'],   default=0),
                max(meta['loud_times'],   default=0),
                max(meta['plp_times'],    default=0)
            )
            duration_sec = max(duration_sec, tmax)
        S_db = None; sr = None

    stem_files  = sorted(f for f in os.listdir(stem_dir) if f.endswith('.mid'))
    stem_colors = dict(vocals='crimson', drums='royalblue',
                       bass='seagreen',   other='darkorange')

    # ──────────── Figure 1: Loudness ────────────
    fig1 = plt.figure(figsize=(12, 8.75 if audio_file else 6.75))
    gs1  = gridspec.GridSpec(2 if audio_file else 1, 1,
                             height_ratios=[2,1] if audio_file else [2],
                             hspace=0.005)
    ax_loud = fig1.add_subplot(gs1[0])
    ax_lufs = ax_loud.twinx()

    plot_elements, plot_labels = [], []

    # plot each stem's loudness
    for i, mid_name in enumerate(stem_files):
        stem_type = next((k for k in stem_colors if k in mid_name.lower()), None)
        color     = stem_colors.get(stem_type, 'gray')
        meta      = parse_midi_metadata(os.path.join(stem_dir, mid_name))

        loud_vals = 1.0 - np.array(meta['loud_vals'])/127.0
        line, = ax_loud.plot(meta['loud_times'], loud_vals,
                             '-', label=f"{stem_type} - Loud",
                             color=color, linewidth=1, alpha=0.6, zorder=3)
        ax_loud.fill_between(meta['loud_times'], loud_vals,
                             color=color, alpha=0.25, zorder=2)

        plot_elements.append(line)
        plot_labels.append(stem_type)

    # add speech/applause shading behind
    speech_done, applause_done = False, False
    for start, end in speech_segs:
        span = ax_loud.axvspan(start, end,
                              color='magenta', alpha=0.2, zorder=0)
        if not speech_done:
            plot_elements.insert(0, span)
            plot_labels.insert(0, 'Speech')
            speech_done = True
    for start, end in applause_segs:
        span = ax_loud.axvspan(start, end,
                              color='gray', alpha=0.2, zorder=0)
        if not applause_done:
            plot_elements.insert(0, span)
            plot_labels.insert(0, 'Applause')
            applause_done = True

    # add tempo handle at top of legend
    tempo_handle = plt.Line2D([], [], color='none')
    plot_elements.insert(0, tempo_handle)
    plot_labels.insert(0, f"Tempo = {bpm:.1f} BPM")

    # clean up axes
    ax_loud.set_xlim(0, duration_sec)
    ax_loud.set_ylim(0, 1)
    ax_loud.set_yticks([]); ax_loud.set_ylabel('')
    ax_loud.tick_params(axis='x', bottom=False, top=False, labelbottom=False)

    ax_lufs.set_ylim(128, -1)
    ticks = list(range(0,128,12)) + [127]
    ax_lufs.set_yticks(ticks)
    ax_lufs.set_yticklabels([f"-{t}" for t in ticks])
    ax_lufs.set_ylabel("Loudness (LUFS)")

    ax_loud.legend(plot_elements, plot_labels,
                   loc='lower center', bbox_to_anchor=(0.5,1.02),
                   ncol=7, frameon=False)

    # spectrogram under if audio
    if S_db is not None:
        ax_spec1 = fig1.add_subplot(gs1[1], sharex=ax_loud)
        librosa.display.specshow(S_db, sr=sr, hop_length=512,
                                 x_axis='time', y_axis='mel',
                                 cmap='magma', ax=ax_spec1)
        ax_spec1.set_xlabel("Time (s)")
        ax_spec1.set_ylabel("Frequency (Hz)")

    # title & layout
    raw_name  = os.path.basename(audio_file or midi_file)
    title_str = re.sub(r'^.*?\[', '[', raw_name).rsplit('.',1)[0]
    fig1.suptitle(f"{title_str} – Loudness",
                  fontsize=14, y=0.975, fontweight='bold')
    plt.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.07)


    # ──────────── Figure 2: Beat Salience (PLP) ────────────
    fig2 = plt.figure(figsize=(12, 8.75 if audio_file else 6.75))
    gs2  = gridspec.GridSpec(2 if audio_file else 1, 1,
                             height_ratios=[2,1] if audio_file else [2],
                             hspace=0.005)
    ax_plp = fig2.add_subplot(gs2[0])

    plot_elements_plp, plot_labels_plp = [], []

    # plot each stem's PLP curve
    for mid_name in stem_files:
        stem_type = next((k for k in stem_colors if k in mid_name.lower()), None)
        color     = stem_colors.get(stem_type, 'gray')
        meta      = parse_midi_metadata(os.path.join(stem_dir, mid_name))

        plp_vals = np.array(meta['plp_vals'])/127.0
        line, = ax_plp.plot(meta['plp_times'], plp_vals,
                            '-', label=stem_type,
                            color=color, alpha=0.6)
        plot_elements_plp.append(line)
        plot_labels_plp.append(stem_type)

    # build custom legend entries: tempo, shading, then stems
    handles, labels = [], []
    # tempo
    handles.append(tempo_handle)
    labels.append(f"Tempo = {bpm:.1f} BPM")
    # shading
    speech_done, applause_done = False, False
    for start, end in speech_segs:
        span = ax_plp.axvspan(start, end,
                             color='magenta', alpha=0.2, zorder=1)
        if not speech_done:
            handles.append(span)
            labels.append('Speech')
            speech_done = True
    for start, end in applause_segs:
        span = ax_plp.axvspan(start, end,
                             color='gray', alpha=0.2, zorder=0)
        if not applause_done:
            handles.append(span)
            labels.append('Applause')
            applause_done = True
    # stems
    handles.extend(plot_elements_plp)
    labels .extend(plot_labels_plp)

    ax_plp.legend(handles, labels,
                  loc='lower center', bbox_to_anchor=(0.5,1.02),
                  ncol=7, frameon=False)

    # Hide PLP x‐axis labels if spectrogram is below
    if S_db is not None:
        ax_plp.tick_params(axis='x', bottom=False, labelbottom=False)

    # spectrogram under if audio
    if S_db is not None:
        ax_spec2 = fig2.add_subplot(gs2[1], sharex=ax_plp)
        librosa.display.specshow(S_db, sr=sr, hop_length=512,
                                 x_axis='time', y_axis='mel',
                                 cmap='magma', ax=ax_spec2)
        ax_spec2.set_xlabel("Time (s)")
        ax_spec2.set_ylabel("Frequency (Hz)")

    fig2.suptitle(f"{title_str} – Beat Salience (PLP)",
                  fontsize=14, y=0.975, fontweight='bold')
    plt.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.07)

    plt.show(block=False)



# --- Speech vs Singing Detection ---

# Constants
SAMPLE_RATE      = 16000  # Hz required by CNN14
WINDOW_LEN       = 2.0    # seconds
HOP_LEN          = 1.0    # seconds (50% overlap)
THRESH_SPEECH    = 0.60
THRESH_SINGING   = 0.20
THRESH_APPLAUSE  = 0.1
APPLAUSE_TAGS    = ["Applause", "Cheering", "Crowd", "Chatter", "Laughter"]
SINGING_TAGS_OTHER = ["Child singing", "Choir"]


def _get_clipwise_output(model_output):
    """
    Given model_output from Cnn14.forward, return the
    clipwise prediction Tensor of shape [batch, classes].
    """
    if isinstance(model_output, (list, tuple)):
        # old API: (clipwise_output, embedding)
        return model_output[0]
    elif isinstance(model_output, dict):
        # new API: {'clipwise_output': Tensor, ...}
        if 'clipwise_output' in model_output:
            return model_output['clipwise_output']
        # fallback: first tensor-like value
        for v in model_output.values():
            if torch.is_tensor(v):
                return v
    raise RuntimeError(f"Unrecognized model output format: {type(model_output)}")


def load_panns_model(
    model_path: str = None,
    label_csv:   str = None
):
    """
    Load PANNs CNN14 model and AudioSet label map.
    Returns:
      model: PyTorch model in eval mode
      label_map: dict of label->index
    """
    if model_path is None:
        model_path = os.path.join(HERE, "Cnn14_16k_mAP=0.438.pth")
    if label_csv is None:
        label_csv  = os.path.join(HERE, "class_labels_indices.csv")
    # Load labels
    labels = {}
    with open(label_csv, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            idx = int(row[0])
            # Some CSVs use different column orders
            name = row[2] if len(row) > 2 else row[1]
            labels[idx] = name
    label_map = {name: idx for idx, name in labels.items()}

    # Instantiate model
    from models import Cnn14
    model = Cnn14(
        sample_rate=SAMPLE_RATE,
        window_size=512,
        hop_size=160,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=len(labels)
    )
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.spectrogram_extractor.stft.pad_mode = 'constant' # force zero‑pad instead of reflect so 2‑D inputs work
    return model, label_map


def _segment_generator(waveform: torch.Tensor, sr: int):
    """
    Yields (start_s, end_s, chunk_tensor) for sliding windows.
    """
    win = int(WINDOW_LEN * sr)
    hop = int(HOP_LEN * sr)
    length = waveform.shape[-1]
    for start in range(0, length - win + 1, hop):
        end = start + win
        chunk = waveform[start:end].unsqueeze(0)  # [1, num_samples]
        yield start/sr, end/sr, chunk


def detect_spoken_regions(audio_path: str, model, label_map: dict):
    """
    Returns list of (start_s, end_s) where speech dominates.
    """
    y, sr = analyze_audio(audio_path, sr=SAMPLE_RATE)
    #print(f"[PANNs] Loaded {audio_path} at {sr} Hz, shape: {y.shape}")
    waveform = torch.from_numpy(y).float()

    # Label indices
    speech_idx = label_map.get("Speech")
    male_idx   = label_map.get("Male singing")
    female_idx = label_map.get("Female singing")
    other_idxs = [label_map[tag] for tag in SINGING_TAGS_OTHER if tag in label_map]

    regions = []
    current = None
    for start, end, chunk in _segment_generator(waveform, SAMPLE_RATE):
        with torch.no_grad():
            raw = model(chunk)
            clipwise = _get_clipwise_output(raw)
        scores = clipwise[0]
        p_speech = scores[speech_idx].item()
        p_male   = scores[male_idx].item()
        p_female = scores[female_idx].item()
        p_other  = sum(scores[i].item() for i in other_idxs)

        is_speech = (p_speech >= THRESH_SPEECH
                     and max(p_male, p_female, p_other) < THRESH_SINGING)
        if is_speech:
            if current is None:
                current = [start, end]
            else:
                current[1] = end
        else:
            if current is not None:
                regions.append((current[0], current[1]))
                current = None
    if current is not None:
        regions.append((current[0], current[1]))
    return regions


def detect_applause_regions(audio_path: str, model, label_map: dict):
    """
    Returns list of (start_s, end_s) where audience applause detected.
    """
    y, sr = analyze_audio(audio_path, sr=SAMPLE_RATE)
    #print(f"[PANNs] Loaded {audio_path} at {sr} Hz, shape: {y.shape}")
    waveform = torch.from_numpy(y).float()

    applause_idxs = [label_map[tag] for tag in APPLAUSE_TAGS if tag in label_map]
    regions = []
    current = None
    for start, end, chunk in _segment_generator(waveform, SAMPLE_RATE):
        with torch.no_grad():
            raw = model(chunk)
            clipwise = _get_clipwise_output(raw)
        scores = clipwise[0]
        p_max = max(scores[i].item() for i in applause_idxs)
        if p_max >= THRESH_APPLAUSE:
            if current is None:
                current = [start, end]
            else:
                current[1] = end
        else:
            if current is not None:
                regions.append((current[0], current[1]))
                current = None
    if current is not None:
        regions.append((current[0], current[1]))
    return regions


def get_singing_character_distribution(audio_path: str, model, label_map: dict):
    """
    Returns dict of aggregate singing contributions: female, male, other (%).
    """
    y, sr = analyze_audio(audio_path, sr=SAMPLE_RATE)
    waveform = torch.from_numpy(y).float()

    female_idx = label_map.get("Female singing")
    male_idx   = label_map.get("Male singing")
    other_idxs = [label_map[tag] for tag in SINGING_TAGS_OTHER if tag in label_map]

    agg = {"female": 0.0, "male": 0.0, "other": 0.0}
    count = 0
    for _, _, chunk in _segment_generator(waveform, SAMPLE_RATE):
        with torch.no_grad():
            raw = model(chunk)
            clipwise = _get_clipwise_output(raw)
        scores = clipwise[0]
        agg["female"] += scores[female_idx].item()
        agg["male"]   += scores[male_idx].item()
        agg["other"]  += sum(scores[i].item() for i in other_idxs)
        count += 1
    total = sum(agg.values())
    if total > 0:
        for k in agg:
            agg[k] = round(100 * agg[k] / total)
    return agg


def get_singing_temporal_distribution(audio_path: str, model, label_map: dict):
    """
    Returns dict of % of windows dominated by each singing type.
    """
    y, sr = analyze_audio(audio_path, sr=SAMPLE_RATE)
    waveform = torch.from_numpy(y).float()

    female_idx = label_map.get("Female singing")
    male_idx   = label_map.get("Male singing")
    other_idxs = [label_map[tag] for tag in SINGING_TAGS_OTHER if tag in label_map]

    counts = {"female": 0, "male": 0, "other": 0}
    total = 0
    for _, _, chunk in _segment_generator(waveform, SAMPLE_RATE):
        with torch.no_grad():
            raw = model(chunk)
            clipwise = _get_clipwise_output(raw)
        scores = clipwise[0]
        scores = {
            "female": scores[female_idx].item(),
            "male":   scores[male_idx].item(),
            "other":  sum(scores[i].item() for i in other_idxs)
        }
        winner = max(scores, key=scores.get)
        counts[winner] += 1
        total += 1
    if total > 0:
        for k in counts:
            counts[k] = round(100 * counts[k] / total)
    return counts


def annotate_midi_lyrics(lyrx_track, speech_regions, applause_regions,
                         char_dist, temp_dist, sec2tick):
    """
    Appends MetaText events to a MIDI lyrics track:
      - Singing Character + Temporal distribution at tick 0
      - Speech Start/End
      - Applause Start/End
    """
    # ─── 1) Character & Temporal distributions at t=0 ────────────────
    lyrx_track.append(MetaMessage('text', text='Singing Character Distribution:', time=0))
    lyrx_track.append(MetaMessage('text', text=f"- Female: {char_dist['female']}%", time=0))
    lyrx_track.append(MetaMessage('text', text=f"- Male:   {char_dist['male']}%", time=0))
    lyrx_track.append(MetaMessage('text', text=f"- Other:  {char_dist['other']}%", time=0))
    lyrx_track.append(MetaMessage('text', text='Singing Temporal Distribution:', time=0))
    lyrx_track.append(MetaMessage('text', text=f"- Female: {temp_dist['female']}%", time=0))
    lyrx_track.append(MetaMessage('text', text=f"- Male:   {temp_dist['male']}%", time=0))
    lyrx_track.append(MetaMessage('text', text=f"- Other:  {temp_dist['other']}%", time=0))

    # ─── 2) Build a combined, sorted list of all events ──────────────
    events = []
    for start, end in speech_regions:
        events.append((start, 'Speech Start'))
        events.append((end,   'Speech End'))
    for start, end in applause_regions:
        events.append((start, 'Applause Start'))
        events.append((end,   'Applause End'))

    # sort by time
    events.sort(key=lambda x: x[0])

    # ─── 3) Emit them in order, with correct delta times ──────────────
    last_tick = 0
    for event_time, label in events:
        tick = sec2tick(event_time)
        delta = max(0, tick - last_tick)
        lyrx_track.append(MetaMessage('text', text=label, time=delta))
        last_tick = tick


def plot_speech_singing_probs(audio_file, model, label_map):
    # Load 16kHz sample rate for model
    y_model, sr_model = analyze_audio(audio_file, sr=SAMPLE_RATE)
    waveform = torch.from_numpy(y_model).float()

    # Load full-rate for plotting
    y, sr = analyze_audio(audio_file, sr=None)
    duration_sec = len(y) / sr

    times = []
    p_speech_list = []
    p_male_list = []
    p_female_list = []
    p_other_list = []

    other_idxs = [label_map[tag] for tag in SINGING_TAGS_OTHER if tag in label_map]

    for start, end, chunk in _segment_generator(waveform, SAMPLE_RATE):
        with torch.no_grad():
            raw = model(chunk)
            clipwise = _get_clipwise_output(raw)
        scores = clipwise[0]

        times.append((start + end) / 2)  # midpoint of window
        p_speech_list.append(scores[label_map["Speech"]].item())
        p_male_list.append(scores[label_map["Male singing"]].item())
        p_female_list.append(scores[label_map["Female singing"]].item())
        p_other_list.append(sum(scores[i].item() for i in other_idxs))

    # 2) Create figure & axes
    fig, axs = plt.subplots(
        3, 1,
        figsize=(12, 8.75),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 2, 1], 'hspace':0.02}
    )

    # add the suptitle exactly like your other plots
    raw_name = os.path.basename(audio_file)
    title_str = re.sub(r'^.*?\[', '[', raw_name).rsplit('.', 1)[0]
    fig.suptitle(title_str, fontsize=14, y=0.975, fontweight='bold')

    # compute total duration for x-limits
    duration_sec = len(y) / sr

    # 3) Waveform
    t_audio = np.linspace(0, duration_sec, len(y))
    axs[0].plot(t_audio, y, color='blue')
    axs[0].set_ylim(-1, 1)
    axs[0].set_yticks([])
    axs[0].set_ylabel("Amplitude")
    #axs[0].set_title("Waveform")
    axs[0].set_xlim(0, duration_sec)

    # 4) Speech & Singing Probabilities
    axs[1].plot(times, p_speech_list, label="Speech", color='magenta')
    axs[1].plot(times, p_male_list, label="Male Singing", color='tab:red')
    axs[1].plot(times, p_female_list, label="Female Singing", color='tab:blue')
    axs[1].plot(times, p_other_list, label="Other Singing", color='tab:purple')
    #axs[1].axhline(THRESH_SPEECH, color='blue', linestyle='--', alpha=0.5)
    #axs[1].axhline(THRESH_SINGING, color='red', linestyle='--', alpha=0.5)
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)
    axs[1].legend(ncol=6, frameon=False)
    #axs[1].set_title("Speech & Singing Probabilities")
    axs[1].set_xlim(0, duration_sec)

    # 5) Mel spectrogram
    axs[2].xaxis.set_major_locator(axs[0].xaxis.get_major_locator())
    axs[2].xaxis.set_major_formatter(axs[0].xaxis.get_major_formatter())

    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=2048, hop_length=512,
        n_mels=128, fmax=sr/2
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(
        S_db, sr=sr, hop_length=512,
        x_axis='time', y_axis='mel',
        fmin=20, fmax=sr/2,
        ax=axs[2], cmap='magma'
    )
    axs[2].set_ylabel('Frequency (Hz)')
    axs[2].set_xlabel('Time (s)')
    plt.setp(axs[2].get_xticklabels(), visible=True)
    axs[2].set_xlim(0, duration_sec)

    # 6) Tidy up & show
    plt.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.07)
    plt.show(block=False)
    plt.pause(0.5)


def plot_applause_probs(audio_file: str, model, label_map: dict):
    """
    Plot:
      1) Full-rate waveform (no y-ticks)
      2) Applause probability over time (max over applause tags)
      3) Mel spectrogram
    """
    # --- load for model at 16 kHz, and full for plotting ---
    y_model, sr_model = analyze_audio(audio_file, sr=SAMPLE_RATE)
    waveform = torch.from_numpy(y_model).float()

    y_full, sr_full = analyze_audio(audio_file, sr=None)
    duration_sec = len(y_full) / sr_full

    # --- compute per‐window applause probability ---
    applause_idxs = [label_map[tag] for tag in APPLAUSE_TAGS if tag in label_map]
    times = []
    p_applause = []

    for start, end, chunk in _segment_generator(waveform, SAMPLE_RATE):
        with torch.no_grad():
            raw      = model(chunk)
            clipwise = _get_clipwise_output(raw)[0]
        times.append((start + end) / 2)
        # max over all “audience” tags
        p_applause.append(max(clipwise[i].item() for i in applause_idxs))

    # --- make the 3-row figure ---
    fig, axs = plt.subplots(
        3, 1,
        figsize=(12, 8.75),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 2, 1], 'hspace': 0.02}
    )

    # suptitle to match your other plots
    raw_name = os.path.basename(audio_file)
    title_str = re.sub(r'^.*?\[', '[', raw_name).rsplit('.',1)[0]
    fig.suptitle(title_str, fontsize=14, y=0.975, fontweight='bold')

    # 1) waveform (no y-ticks)
    t_full = np.linspace(0, duration_sec, len(y_full))
    axs[0].plot(t_full, y_full, color='blue')
    axs[0].set_ylim(-1, 1)
    axs[0].set_yticks([])
    axs[0].set_xlim(0, duration_sec)

    # 2) applause probability
    axs[1].plot(times, p_applause, label="Applause Prob", color='tab:orange')
    axs[1].axhline(THRESH_APPLAUSE, color='tab:orange', linestyle='--', alpha=0.5,
                   label=f"Threshold = {THRESH_APPLAUSE:.2f}")
    axs[1].set_ylabel("P(applause)")
    axs[1].set_ylim(0, 1)
    axs[1].legend(ncol=2, frameon=False)
    axs[1].set_xlim(0, duration_sec)

    # 3) mel spectrogram
    axs[2].xaxis.set_major_locator(axs[0].xaxis.get_major_locator())
    axs[2].xaxis.set_major_formatter(axs[0].xaxis.get_major_formatter())
    S = librosa.feature.melspectrogram(
        y=y_full, sr=sr_full,
        n_fft=2048, hop_length=512,
        n_mels=128, fmax=sr_full/2
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(
        S_db, sr=sr_full, hop_length=512,
        x_axis='time', y_axis='mel', fmin=20, fmax=sr_full/2,
        ax=axs[2], cmap='magma'
    )
    axs[2].set_ylabel('Frequency (Hz)')
    axs[2].set_xlabel('Time (s)')
    plt.setp(axs[2].get_xticklabels(), visible=True)
    axs[2].set_xlim(0, duration_sec)

    # final layout tweak
    plt.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.07)
    plt.show(block=False)
    plt.pause(0.5)



# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(description='Generate or inspect metadata MIDI files')
    parser.add_argument('files', nargs='+', help='Input .m4a and/or .mid files')
    parser.add_argument('-inspect', choices=['console','plot'], help='Inspect mode: console or plot')
    parser.add_argument('-stems', action='store_true', help='Generate and convert Demucs stems to ALAC')
    args = parser.parse_args()
    

    # Load PANNs model for speech/applause & singing breakdowns
    model, label_map = load_panns_model()

    audio_file = next((f for f in args.files if f.lower().endswith('.m4a')), None)
    midi_file  = next((f for f in args.files if f.lower().endswith(('.mid','.midi'))), None)

    # Audio-only: generate MIDI
    if audio_file and not midi_file and not args.inspect:
        base, _ = os.path.splitext(audio_file)
        out = f"{base}.mid"
        print("Analyzing audio...")
        y, sr = analyze_audio(audio_file)

        # Compute LUFS for MIDI metadata
        meter = pyln.Meter(sr)
        track_lufs = meter.integrated_loudness(y)

        target_lufs = -18.0
        replay_gain = target_lufs - track_lufs
        rg_cc = int(np.clip(abs(track_lufs), 0, 127))

        # Try rsgain
        possible = [shutil.which("rsgain"), "/usr/local/bin/rsgain",
                    "/opt/homebrew/bin/rsgain", "/usr/bin/rsgain"]
        rg_cmd = next((p for p in possible if p and os.path.isfile(p)), None)
        cli_line = None
        lufs_cli = None
        if rg_cmd:
            try:
                proc = subprocess.run(
                    [rg_cmd, "custom", "--tagmode=s", "-O", audio_file],
                    capture_output=True, text=True, check=True
                )
                for line in proc.stdout.splitlines():
                    if os.path.basename(audio_file) in line:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            lufs_cli = float(parts[1].replace(" LU", "").strip())
                            gain_cli = target_lufs - lufs_cli
                            rg_cli_cc = int(np.clip(abs(lufs_cli), 0, 127))
                            cli_line = (rg_cli_cc, lufs_cli, gain_cli)
                        break
            except Exception as e:
                print("rsgain scan failed (safe to ignore):", e)

        print("Computing loudness curve...")
        lc = compute_loudness_curve(y, sr)

        print("Detecting beats & tempo...")
        tempo, bt, bs = detect_beats(y, sr)
        print(f"Detected tempo: {tempo:.1f} BPM")

        # print("Detecting key (initial)...")
        # key_str = detect_key(audio_file, use_stems=False)
        # print(f"Initial detected key: {key_str}")

        # print("Detecting chords (initial)...")
        # chords = detect_chords(audio_file, use_stems=False)
        # print(f"Detected {len(chords)} chord segments")

        env = librosa.onset.onset_strength(y=y, sr=sr)
        plp = librosa.beat.plp(onset_envelope=env, sr=sr)
        pt = librosa.times_like(plp, sr=sr)

        speech_regions   = None
        applause_regions = None
        char_dist        = None
        temp_dist        = None

        # --- 1) Run Demucs if requested ---
        demucs_ok = False
        if args.stems:
            print("Running Demucs for stem separation...\n")
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            audio_dir  = os.path.dirname(os.path.abspath(audio_file))
            output_dir = os.path.join(audio_dir, f"stems_{base_name}")
            temp_input_copy = os.path.join(audio_dir, "demucs_temp_input.m4a")

            try:
                shutil.copy2(audio_file, temp_input_copy)
                subprocess.run(["demucs", temp_input_copy, "-o", audio_dir, "-n", "htdemucs"], check=True)

                demucs_subdir = os.path.join(audio_dir, "htdemucs", "demucs_temp_input")
                if os.path.exists(demucs_subdir):
                    os.makedirs(output_dir, exist_ok=True)
                    for stem_name in ["vocals", "drums", "bass", "other"]:
                        src = os.path.join(demucs_subdir, f"{stem_name}.wav")
                        dst_alac = os.path.join(output_dir, f"{base_name} - {stem_name}.m4a")
                        if os.path.exists(src):
                            subprocess.run(["ffmpeg", "-y", "-i", src, "-c:a", "alac", dst_alac],
                                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            os.remove(src)

                            # Generate per-stem MIDI (simplified, no key/chords/PANNs)
                            stem_y, stem_sr = analyze_audio(dst_alac)
                            stem_lufs = pyln.Meter(stem_sr).integrated_loudness(stem_y)
                            stem_lc = compute_loudness_curve(stem_y, stem_sr)
                            stem_tempo, stem_bt, stem_bs = detect_beats(stem_y, stem_sr)
                            stem_env = librosa.onset.onset_strength(y=stem_y, sr=stem_sr)
                            stem_plp = librosa.beat.plp(onset_envelope=stem_env, sr=stem_sr)
                            stem_pt = librosa.times_like(stem_plp, sr=stem_sr)

                            stem_mid_out = os.path.join(output_dir, f"{base_name} - {stem_name}.mid")
                            create_midi(dst_alac, stem_lc, stem_bt, stem_tempo, "Unknown", [],
                                        stem_mid_out, stem_bs, stem_pt, stem_plp, pyloudnorm_lufs=stem_lufs)
                    demucs_ok = True
                else:
                    print("Demucs output not found.")
            except Exception as e:
                print("Demucs failed:", e)
            finally:
                if os.path.exists(temp_input_copy):
                    os.remove(temp_input_copy)
                demucs_temp_dir = os.path.join(os.getcwd(), "htdemucs", "demucs_temp_input")
                if os.path.exists(demucs_temp_dir):
                    shutil.rmtree(demucs_temp_dir)

        # --- 2) PANNs Analysis (only if Demucs succeeded) ---
        speech_regions = []
        applause_regions = []
        char_dist = {"female":0, "male":0, "other":0}
        temp_dist = {"female":0, "male":0, "other":0}

        if demucs_ok:
            print("\nRunning PANNs analysis...")
            speech_regions   = detect_spoken_regions(audio_file, model, label_map)
            applause_regions = detect_applause_regions(audio_file, model, label_map)
            char_dist        = get_singing_character_distribution(audio_file, model, label_map)
            temp_dist        = get_singing_temporal_distribution(audio_file, model, label_map)

        # --- 3) Key & Chord Detection ---
        print("Detecting key...")
        key_str = detect_key(audio_file, use_stems=args.stems)
        print(f"Detected key: {key_str}")

        chords = []
        if MADMOM_AVAILABLE:
            print("Detecting chords...")
            chords = detect_chords(audio_file, use_stems=args.stems)
            print(f"Detected {len(chords)} chord segments\n")
        else:
            print("Madmom not available. Skipping chord detection.")

        # --- 4) Final MIDI Creation ---
        print("Creating final metadata MIDI file...")
        create_midi(
            audio_file, lc, bt, tempo, key_str, chords, out,
            bs, pt, plp,
            pyloudnorm_lufs=track_lufs,
            rsgain_lufs=(lufs_cli if lufs_cli is not None else None),
            speech_regions=speech_regions,
            applause_regions=applause_regions,
            char_dist=char_dist,
            temp_dist=temp_dist
        )

        # Show individual stem plots if requested
        if args.stems and args.inspect == 'plot':
            inspect_stems_if_requested(audio_file, 'plot')

            # Then plot the aggregate stem overlay
            if audio_file:
                plot_stem_overlay(audio_file)

        # Show all plots if in plot mode
        if args.inspect == 'plot':
            plt.show()

        return

    # MIDI-only, console inspect
    if midi_file and args.inspect == 'console' and not audio_file:
        inspect_midi(midi_file)
        return

    # MIDI-only, plot only beats/loudness
    if midi_file and args.inspect == 'plot' and not audio_file:
        plot_midi_with_optional_audio(midi_file, None)
        if args.stems:
            base_name = os.path.splitext(os.path.basename(midi_file))[0]
            midi_dir = os.path.dirname(os.path.abspath(midi_file))
            stem_dir = os.path.join(midi_dir, f"stems_{base_name}")
            if os.path.isdir(stem_dir):
                stem_mid_files = sorted(f for f in os.listdir(stem_dir) if f.endswith('.mid'))
                for stem_mid_name in stem_mid_files:
                    stem_mid_path = os.path.join(stem_dir, stem_mid_name)
                    print(f"\nInspecting stem: {stem_mid_name}")
                    # Always pass None for audio_file in MIDI-only mode
                    plot_midi_with_optional_audio(stem_mid_path, None)

                # Now try to render the aggregate overlay plot using MIDI only
                print("\nGenerating aggregate stem overlay plot (MIDI-only mode)...")
                try:
                    plot_stem_overlay(None, midi_file=midi_file)
                except Exception as e:
                    print(f"Failed to generate aggregate stem overlay plot: {e}")
            else:
                print(f"Stem folder not found: {stem_dir}")
        plt.show()
        return

    # Audio+MIDI, console inspect
    if midi_file and args.inspect == 'console' and audio_file:
        inspect_midi(midi_file)
        if args.stems:
            inspect_stems_if_requested(audio_file, 'console')
        return

    # Audio+MIDI, full overlay plot
    if midi_file and args.inspect == 'plot' and audio_file:
        plot_midi_with_optional_audio(midi_file, audio_file)
        if args.stems:
            inspect_stems_if_requested(audio_file, 'plot')
            plot_stem_overlay(audio_file, midi_file)
        plt.show()
        return

    print("No valid action: provide only .m4a to generate MIDI, or use -inspect with .mid")

if __name__ == '__main__':
    main()
