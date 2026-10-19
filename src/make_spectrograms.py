"""Convert each audio clip to a log-mel spectrogram saved as data/processed/spectrograms/<track_id>.npy.

    python -m src.make_spectrograms
    python -m src.make_spectrograms --limit 200
"""
import argparse

import librosa
import numpy as np
from tqdm import tqdm

from . import config
from .data import audio_path, load_tracks

SPEC_DIR = config.DATA_PROCESSED / "spectrograms"


def mel_spectrogram(y, sr):
    target_len = sr * config.CLIP_SECONDS
    y = librosa.util.fix_length(y, size=target_len)  # pad/trim so every spectrogram is the same size
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    track_ids = load_tracks().index[: args.limit]
    failed = 0
    for tid in tqdm(track_ids):
        out = SPEC_DIR / f"{tid}.npy"
        if out.exists():
            continue
        try:
            y, sr = librosa.load(audio_path(tid), sr=config.SAMPLE_RATE, duration=config.CLIP_SECONDS)
            np.save(out, mel_spectrogram(y, sr))
        except Exception:
            failed += 1
    print(f"Done. {failed} files failed to load.")


if __name__ == "__main__":
    main()
