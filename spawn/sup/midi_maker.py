#!/usr/bin/env python3
"""
Dependencies:
  pip install librosa pyloudnorm mido music21 Cython numpy wheel demucs torchaudio
  pip install --no-use-pep517 madmom
  conda install -c conda-forge essentia
  brew install libsndfile
  pip install soundfile
"""

import sys
import types

# inject a dummy soundfile module so that `import soundfile` never fails
#sys.modules['soundfile'] = types.ModuleType('soundfile')

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
import subprocess
import shutil
import re
#import tempfile
#import uuid

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

# Optional external analysis tools
try:
    from essentia.standard import MonoLoader, KeyExtractor
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

try:
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False

try:
    from music21 import roman, key as m21key, chord as m21chord
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

# PLP constants
PLP_AGG_WINDOW_S = 0.01
PLP_QUANT_BINS   = 64


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


def detect_key(file_path):
    if not ESSENTIA_AVAILABLE:
        return "Unknown"
    try:
        audio = MonoLoader(filename=file_path)()
        key, scale, _ = KeyExtractor()(audio)
        return f"{key} {scale}"
    except:
        return "Unknown"


def detect_chords(file_path):
    if not MADMOM_AVAILABLE:
        return []
    feats = CNNChordFeatureProcessor()(file_path)
    return CRFChordRecognitionProcessor()([feats])


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
    rsgain_lufs=None
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

    # Save MIDI File
    mid.save(output_path)
    print(f"MIDI saved to: {output_path}")


# --- MIDI inspection & parsing ---

def inspect_midi(midi_file):
    mid = MidiFile(midi_file)
    print(f"--- MIDI File: {midi_file} ---")
    print(f"Ticks per beat: {mid.ticks_per_beat}")
    for idx, tr in enumerate(mid.tracks):
        print(f"\n-- Track {idx} - {tr.name} --")
        for msg in tr:
            print(msg)
            # Annotate LUFS values
            if msg.type == 'control_change' and msg.control == 20:
                if msg.channel == 13:
                    print(f"  (pyloudnorm static LUFS: -{msg.value})")
                elif msg.channel == 12:
                    print(f"  (rsgain static LUFS: -{msg.value})")
            # Annotate chord/roman channel markers
            elif msg.type == 'control_change' and msg.control == 29 and msg.channel == 14:
                print("  (Chord label marker)")
            elif msg.type == 'control_change' and msg.control == 30 and msg.channel == 14:
                print("  (Roman numeral marker)")


def parse_midi_metadata(midi_file):
    mid = MidiFile(midi_file)
    tb = mid.ticks_per_beat
    tempo = 500000

    beat_times, beat_strengths, plp_times, plp_vals = [], [], [], []
    loud_times, loud_vals = [], []
    chord_labels = []        # [(time_in_sec, "C:maj")]
    roman_numerals = []      # [(time_in_sec, "IV")]

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
            elif msg.control == 23 and msg.channel == 14:
                pyloudnorm_lufs = -float(msg.value)
            elif msg.control == 24 and msg.channel == 14:
                rsgain_lufs = -float(msg.value)

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
        'roman_numerals': roman_numerals
    }


def inspect_stems_if_requested(audio_file, inspect_mode):
    if audio_file:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        stem_dir = os.path.join(os.getcwd(), f"stems_{base_name}")
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
        ax_main.vlines(bt, -1, 1, linewidth=0.1, color='grey', zorder=1)
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


def plot_stem_overlay(audio_file=None, midi_file=None):

    if not audio_file and not midi_file:
        print("No input file provided for stem overlay plot.")
        return

    # Determine base name from whichever is provided
    input_file = audio_file or midi_file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    stem_dir = os.path.join(os.getcwd(), f"stems_{base_name}")
    if not os.path.isdir(stem_dir):
        print(f"Stem folder not found: {stem_dir}")
        return

    stem_files = sorted(f for f in os.listdir(stem_dir) if f.endswith('.mid'))
    stem_colors = {
        'vocals': 'crimson',
        'drums': 'royalblue',
        'bass': 'seagreen',
        'other': 'darkorange',
    }

    if audio_file:
        y, sr = analyze_audio(audio_file)
        duration_sec = len(y) / sr
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
                                           n_mels=128, fmax=sr/2)
        S_db = librosa.power_to_db(S, ref=np.max)
    else:
        # Estimate duration from MIDI if no audio
        duration_sec = 0
        for mid_name in stem_files:
            meta = parse_midi_metadata(os.path.join(stem_dir, mid_name))
            max_time = max(
                max(meta['beat_times'], default=0),
                max(meta['loud_times'], default=0),
                max(meta['plp_times'], default=0)
            )
            duration_sec = max(duration_sec, max_time)
        S_db = None
        sr = None

    # ───────────── Figure 1: Loudness ─────────────
    fig1 = plt.figure(figsize=(12, 8.75 if audio_file else 6.75))
    gs1 = gridspec.GridSpec(2 if audio_file else 1, 1,
                        height_ratios=[2, 1] if audio_file else [2],
                        hspace=0.005)
    ax_loud = fig1.add_subplot(gs1[0])
    ax_lufs = ax_loud.twinx()  # Right-hand LUFS axis

    plot_elements = []
    plot_labels = []
    bpm = None

    for i, mid_name in enumerate(stem_files):
        stem_type = next((k for k in stem_colors if k in mid_name.lower()), None)
        color = stem_colors.get(stem_type, 'gray')
        stem_path = os.path.join(stem_dir, mid_name)
        meta = parse_midi_metadata(stem_path)

        if i == 0 and meta.get("tempo"):
            bpm = tempo2bpm(meta["tempo"])

        loud_vals = 1.0 - (np.array(meta['loud_vals']) / 127.0)
        line_loud, = ax_loud.plot(meta['loud_times'], loud_vals, '-', label=f"{stem_type} - Loud",
                                  color=color, linewidth=1, alpha=0.6, zorder=3)
        ax_loud.fill_between(meta['loud_times'], loud_vals, color=color, alpha=0.25, zorder=2)

        plot_elements.append(line_loud)
        plot_labels.append(f"{stem_type}")

    # Remove left-hand y-axis (Amplitude) from loudness plot
    ax_loud.set_yticks([])
    ax_loud.set_ylabel("")
    ax_loud.set_xlim(0, duration_sec)
    ax_loud.set_ylim(0, 1)
    ax_loud.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Right-hand LUFS axis
    ax_lufs.set_ylim(128, -1)
    ticks = list(range(0, 128, 12)) + [127]
    ticks = sorted(set(ticks))
    ax_lufs.set_yticks(ticks)
    ax_lufs.set_yticklabels([f"-{t}" for t in ticks])
    ax_lufs.set_ylabel("Loudness (LUFS)")

    if bpm:
        tempo_handle = plt.Line2D([], [], color='none')
        plot_elements.insert(0, tempo_handle)
        plot_labels.insert(0, f"Tempo = {bpm:.1f} BPM")

    ax_loud.legend(plot_elements, plot_labels, loc='lower center',
                   bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)

    # Add spectrogram under loudness
    if S_db is not None:
        ax_spec1 = fig1.add_subplot(gs1[1], sharex=ax_loud)
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel',
                                 cmap='magma', ax=ax_spec1)
        ax_spec1.set_xlabel("Time (s)")
        ax_spec1.set_ylabel("Frequency (Hz)")

    raw_name = os.path.basename(audio_file or midi_file)
    title_str = re.sub(r'^.*?\[', '[', raw_name).rsplit('.', 1)[0]
    fig1.suptitle(f"{title_str} – Loudness", fontsize=14, y=0.975, fontweight='bold')
    plt.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.07)

    # ───────────── Figure 2: Beat Salience ─────────────
    fig2 = plt.figure(figsize=(12, 8.75 if audio_file else 6.75))
    gs2 = gridspec.GridSpec(2 if audio_file else 1, 1,
                        height_ratios=[2, 1] if audio_file else [2],
                        hspace=0.005)
    ax_plp = fig2.add_subplot(gs2[0])

    plot_elements_plp = []
    plot_labels_plp = []

    for mid_name in stem_files:
        stem_type = next((k for k in stem_colors if k in mid_name.lower()), None)
        color = stem_colors.get(stem_type, 'gray')
        stem_path = os.path.join(stem_dir, mid_name)
        meta = parse_midi_metadata(stem_path)

        plp_vals = np.array(meta['plp_vals']) / 127.0
        line_plp, = ax_plp.plot(meta['plp_times'], plp_vals, '-',
                                label=f"{stem_type}", color=color, alpha=0.6)
        plot_elements_plp.append(line_plp)
        plot_labels_plp.append(f"{stem_type}")

    ax_plp.set_ylabel("Amplitude")
    ax_plp.set_xlim(0, duration_sec)
    ax_plp.set_ylim(0, 1)
    ax_plp.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_plp.legend(plot_elements_plp, plot_labels_plp,
                  loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=4, frameon=False)

    if S_db is not None:
        ax_spec2 = fig2.add_subplot(gs2[1], sharex=ax_plp)
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel',
                                 cmap='magma', ax=ax_spec2)
        ax_spec2.set_xlabel("Time (s)")
        ax_spec2.set_ylabel("Frequency (Hz)")

    fig2.suptitle(f"{title_str} – Beat Salience (PLP)", fontsize=14, y=0.975, fontweight='bold')
    plt.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.07)

    plt.show()



# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(description='Generate or inspect metadata MIDI files')
    parser.add_argument('files', nargs='+', help='Input .m4a and/or .mid files')
    parser.add_argument('-inspect', choices=['console','plot'], help='Inspect mode: console or plot')
    parser.add_argument('-stems', action='store_true', help='Generate and convert Demucs stems to ALAC')
    args = parser.parse_args()

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
        print("Detecting key...")
        key_str = detect_key(audio_file)
        print(f"Detected key: {key_str}")
        print("Detecting chords...")
        chords = detect_chords(audio_file)
        print(f"Detected {len(chords)} chord segments")
        env = librosa.onset.onset_strength(y=y, sr=sr)
        plp = librosa.beat.plp(onset_envelope=env, sr=sr)
        pt = librosa.times_like(plp, sr=sr)

        # Run Demucs to separate stems
        if args.stems:
            print("Running Demucs for stem separation...")
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_dir = os.path.join(os.getcwd(), f"stems_{base_name}")
            temp_input_copy = os.path.join(os.getcwd(), "demucs_temp_input.m4a")

            try:
                shutil.copy2(audio_file, temp_input_copy)

                subprocess.run(
                    ["demucs", temp_input_copy, "-o", os.getcwd(), "-n", "htdemucs"],
                    check=True
                )

                demucs_subdir = os.path.join(os.getcwd(), "htdemucs", "demucs_temp_input")
                if os.path.exists(demucs_subdir):
                    os.makedirs(output_dir, exist_ok=True)
                    for stem_name in ["vocals", "drums", "bass", "other"]:
                        src = os.path.join(demucs_subdir, f"{stem_name}.wav")
                        base_file_name = os.path.splitext(os.path.basename(audio_file))[0]
                        stem_prefix = f"{base_file_name} - {stem_name}"
                        dst_alac = os.path.join(output_dir, f"{stem_prefix}.m4a")
                        if os.path.exists(src):
                            try:
                                subprocess.run([
                                    "ffmpeg", "-y", "-i", src, "-c:a", "alac", dst_alac
                                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                if os.path.exists(dst_alac):
                                    os.remove(src)

                                # Generate MIDI metadata for the stem
                                try:
                                    stem_y, stem_sr = analyze_audio(dst_alac)
                                    stem_lufs = pyln.Meter(stem_sr).integrated_loudness(stem_y)
                                    stem_lc = compute_loudness_curve(stem_y, stem_sr)
                                    stem_tempo, stem_bt, stem_bs = detect_beats(stem_y, stem_sr)
                                    stem_env = librosa.onset.onset_strength(y=stem_y, sr=stem_sr)
                                    stem_plp = librosa.beat.plp(onset_envelope=stem_env, sr=stem_sr)
                                    stem_pt = librosa.times_like(stem_plp, sr=stem_sr)

                                    stem_mid_out = os.path.join(output_dir, f"{stem_prefix}.mid")

                                    create_midi(
                                        dst_alac,
                                        stem_lc,
                                        stem_bt,
                                        stem_tempo,
                                        key_str,
                                        chords,
                                        stem_mid_out,
                                        stem_bs,
                                        stem_pt,
                                        stem_plp,
                                        pyloudnorm_lufs=stem_lufs,
                                        rsgain_lufs=None  # rsgain isn't re-run for stems
                                    )
                                except Exception as e:
                                    print(f"⚠️ Failed to generate MIDI for stem {stem_name}: {e}")

                            except subprocess.CalledProcessError:
                                print(f"⚠️ Failed to convert {stem_name}.wav to ALAC.")
                    print(f"Stems copied to: {output_dir}")
                else:
                    print("Demucs output not found. Skipping stem copy.")

            except Exception as e:
                print("Demucs stem separation failed:", e)

            finally:
                if os.path.exists(temp_input_copy):
                    os.remove(temp_input_copy)

                # Clean up Demucs output temp folder
                demucs_temp_dir = os.path.join(os.getcwd(), "htdemucs", "demucs_temp_input")
                try:
                    if os.path.exists(demucs_temp_dir):
                        shutil.rmtree(demucs_temp_dir)
                    # Optionally remove empty parent folder if it's now empty
                    parent = os.path.dirname(demucs_temp_dir)
                    if os.path.exists(parent) and not os.listdir(parent):
                        os.rmdir(parent)
                except Exception as cleanup_err:
                    print("Warning: Failed to clean up temp Demucs folder:", cleanup_err)

        # Create MIDI file
        print("Creating MIDI...")
        create_midi(audio_file, lc, bt, tempo, key_str, chords, out, bs, pt, plp,
                    pyloudnorm_lufs=track_lufs,
                    rsgain_lufs=lufs_cli if lufs_cli is not None else None)

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
            stem_dir = os.path.join(os.getcwd(), f"stems_{base_name}")
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
            plot_stem_overlay(audio_file)
        plt.show()
        return

    print("No valid action: provide only .m4a to generate MIDI, or use -inspect with .mid")

if __name__ == '__main__':
    main()
