"""Helpers for loading the FMA metadata, precomputed features, and audio file paths."""
import pandas as pd

from . import config

SUBSET_ORDER = ["small", "medium", "large"]


def load_tracks():
    """Load tracks.csv (two-level column header) restricted to our subset."""
    tracks = pd.read_csv(config.METADATA_DIR / "tracks.csv", index_col=0, header=[0, 1])
    subset = pd.Categorical(tracks[("set", "subset")], categories=SUBSET_ORDER, ordered=True)
    tracks = tracks[subset <= config.SUBSET]
    return tracks[tracks[("track", "genre_top")].notna()]


def load_precomputed_features():
    """Load FMA's precomputed Librosa features (features.csv, three-level column header)."""
    return pd.read_csv(config.METADATA_DIR / "features.csv", index_col=0, header=[0, 1, 2])


def get_labels(tracks):
    """Series mapping track_id -> top-level genre name."""
    return tracks[("track", "genre_top")]


def get_split_ids(tracks):
    """Return (train_ids, val_ids, test_ids) using FMA's official split."""
    split = tracks[("set", "split")]
    return (
        split.index[split == "training"],
        split.index[split == "validation"],
        split.index[split == "test"],
    )


def audio_path(track_id):
    """fma_medium/000/000002.mp3 style path for a track id."""
    tid = f"{track_id:06d}"
    return config.AUDIO_DIR / tid[:3] / f"{tid}.mp3"
