# Data

The data files are too large to store in git. Each person downloads them locally into `data/raw/`.

## Download

From https://github.com/mdeff/fma:

| File | Size | Needed for |
|------|------|-----------|
| `fma_metadata.zip` | ~342 MB | Everything (labels, splits, precomputed features) |
| `fma_medium.zip`   | ~22 GB  | Our own feature extraction and the CNN |

```bash
cd data/raw
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_medium.zip
unzip fma_metadata.zip
unzip fma_medium.zip
```

Afterward the folder should look like this:

```
data/raw/
  fma_metadata/
    tracks.csv
    features.csv
    genres.csv
    ...
  fma_medium/
    000/000002.mp3
    ...
```

Download only `fma_metadata` first. It's enough to run the baseline (`python -m src.train_baseline`).

**Note:** a few FMA MP3 files are corrupted or empty. The scripts skip any file they fail to load.
