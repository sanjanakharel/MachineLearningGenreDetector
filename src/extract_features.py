"""Extract our own Librosa features from each audio file into data/processed/features_ours.csv.

    python -m src.extract_features
    python -m src.extract_features --limit 200   # quick test on a few files
"""
import argparse

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .data import audio_path, load_tracks


def summarize(name, matrix):
    """Mean and std of each row of a (n_features, n_frames) matrix."""
    matrix = np.atleast_2d(matrix)
    out = {}
    for i, row in enumerate(matrix):
        out[f"{name}_{i}_mean"] = float(row.mean())
        out[f"{name}_{i}_std"] = float(row.std())
    return out


def extract(y, sr):
    feats = {}
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    feats["tempo"] = float(np.atleast_1d(tempo)[0])
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    feats["onset_rate"] = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(y) / sr)
    feats.update(summarize("onset_strength", onset_env))

    harmonic, percussive = librosa.effects.hpss(y)
    feats["harmonic_energy"] = float(np.mean(harmonic ** 2))
    feats["percussive_energy"] = float(np.mean(percussive ** 2))

    feats.update(summarize("mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)))
    feats.update(summarize("chroma", librosa.feature.chroma_stft(y=y, sr=sr)))
    feats.update(summarize("rms", librosa.feature.rms(y=y)))
    feats.update(summarize("centroid", librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats.update(summarize("bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feats.update(summarize("rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feats.update(summarize("contrast", librosa.feature.spectral_contrast(y=y, sr=sr)))
    feats.update(summarize("flatness", librosa.feature.spectral_flatness(y=y)))
    feats.update(summarize("zcr", librosa.feature.zero_crossing_rate(y)))
    feats.update(summarize("tonnetz", librosa.feature.tonnetz(y=harmonic, sr=sr)))
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tracks")
    args = parser.parse_args()

    track_ids = load_tracks().index[: args.limit]
    rows, failed = {}, []
    for tid in tqdm(track_ids):
        try:
            y, sr = librosa.load(audio_path(tid), sr=config.SAMPLE_RATE, duration=config.CLIP_SECONDS)
            rows[tid] = extract(y, sr)
        except Exception:
            failed.append(tid)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.DATA_PROCESSED / "features_ours.csv"
    pd.DataFrame.from_dict(rows, orient="index").rename_axis("track_id").to_csv(out)
    print(f"Saved {len(rows)} tracks to {out} ({len(failed)} failed to load)")


if __name__ == "__main__":
    main()
