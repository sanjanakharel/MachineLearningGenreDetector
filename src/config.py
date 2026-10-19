"""Shared paths and settings. Change things here instead of hard-coding them in scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
METADATA_DIR = DATA_RAW / "fma_metadata"
AUDIO_DIR = DATA_RAW / "fma_medium"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SUBSET = "medium"

# Audio settings
SAMPLE_RATE = 22050
CLIP_SECONDS = 30
N_MELS = 128

RANDOM_SEED = 42
