# Machine Learning Genre Detector

COSC 4368 – Fundamentals of Artificial Intelligence (University of Houston), final project.

**Team:** Sanjana Kharel, Kevin Ngo, Ayman Almasree, Hoang Phi Pham

We classify the genre of short music clips from the [FMA Medium](https://github.com/mdeff/fma) dataset
(25,000 30-second clips, 16 genres). We compare two approaches:

1. **Traditional ML.** Each clip becomes one row of numerical audio features (MFCCs, chroma, spectral
   centroid, tempo, and so on). We train Logistic Regression, Random Forest, SVM and XGBoost on those rows.
2. **Deep learning.** Each clip becomes a mel spectrogram, which goes into a ResNet18 CNN (PyTorch),
   trained with cross-entropy loss and SGD.

We evaluate both on a held-out test set using accuracy, per-genre precision, recall and F1, a confusion matrix,
and training and validation loss curves. We also try different regularization techniques
(L1/L2, weight decay, dropout, early stopping, data augmentation).

## Project structure

```
data/
  raw/          # FMA downloads go here (not committed: too large)
  processed/    # extracted features / spectrograms (not committed)
docs/           # proposal, dataset links, notes
notebooks/      # Jupyter notebooks for exploration
src/            # shared Python code
  config.py             # paths and settings
  data.py               # loading FMA metadata, features, splits, audio paths
  evaluate.py           # metrics + confusion matrix plots
  train_baseline.py     # Step 1: models on FMA's precomputed features
  extract_features.py   # Step 2: our own Librosa feature extraction
  make_spectrograms.py  # Step 3a: audio -> mel spectrograms
  train_cnn.py          # Step 3b: ResNet18 on spectrograms
models/         # saved trained models (not committed)
results/        # plots and metrics we want to keep
```

## Setup

```bash
git clone https://github.com/sanjanakharel/MachineLearningGenreDetector.git
cd MachineLearningGenreDetector
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To decode MP3 files, Librosa needs **ffmpeg**. Install it with `brew install ffmpeg` on macOS,
`sudo apt install ffmpeg` on Linux, or download it from ffmpeg.org on Windows.

Then download the data. See [data/README.md](data/README.md).

## Running

Run every command from the repo root.

```bash
# 1. Quick baseline using FMA's precomputed features (needs only fma_metadata)
python -m src.train_baseline --model logreg      # or: rf, svm, xgb, all

# 2. Our own Librosa features (needs fma_medium audio)
python -m src.extract_features
python -m src.train_baseline --model all --features ours

# 3. CNN on mel spectrograms
python -m src.make_spectrograms
python -m src.train_cnn --epochs 20
```

Results such as confusion matrices, loss curves and metric reports are saved in `results/`.

## How we work together

- **Don't push straight to `main`.** Create a branch for your task:
  ```bash
  git checkout main && git pull
  git checkout -b kevin/cnn-dropout
  # ...make changes...
  git add . && git commit -m "Add dropout to CNN"
  git push -u origin kevin/cnn-dropout
  ```
  Then open a Pull Request on GitHub and ask a teammate to look at it before merging.
- Run `git pull` on `main` often so you don't drift out of date.
- **Never commit datasets or model files.** `.gitignore` already blocks them. If you need to share a large
  file, use Google Drive and add the link to `docs/datasets.md`.
- Put experiments in `notebooks/` with your name in the filename, for example `notebooks/sanjana_eda.ipynb`, so we
  don't overwrite each other's work. Move code that works and that others need into `src/`.
- Use GitHub Issues to track tasks and assign them to people.
