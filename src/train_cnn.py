"""Train a ResNet18 CNN on mel spectrograms (run src.make_spectrograms first).

    python -m src.train_cnn --epochs 20
    python -m src.train_cnn --epochs 20 --pretrained --dropout 0.5 --weight-decay 5e-4 --augment
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from . import config
from .data import get_labels, get_split_ids, load_tracks
from .evaluate import evaluate, plot_loss_curves
from .make_spectrograms import SPEC_DIR


class SpectrogramDataset(Dataset):
    def __init__(self, track_ids, labels, augment=False):
        self.track_ids = list(track_ids)
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, i):
        tid = self.track_ids[i]
        spec = np.load(SPEC_DIR / f"{tid}.npy").astype(np.float32)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        if self.augment:
            spec = spec_augment(spec)
        x = torch.from_numpy(spec).unsqueeze(0).repeat(3, 1, 1)  # ResNet expects 3 channels
        return x, self.labels[tid]


def spec_augment(spec, max_freq=16, max_time=40):
    """Data augmentation (a form of regularization): blank out a random band of frequencies and of time."""
    spec = spec.copy()
    f = np.random.randint(0, max_freq)
    f0 = np.random.randint(0, spec.shape[0] - f)
    spec[f0:f0 + f, :] = 0
    t = np.random.randint(0, max_time)
    t0 = np.random.randint(0, spec.shape[1] - t)
    spec[:, t0:t0 + t] = 0
    return spec


def build_model(n_classes, pretrained, dropout):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, n_classes))
    return model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.append(out.argmax(1).cpu())
            targets.append(y.cpu())
    return total_loss / len(loader.dataset), torch.cat(preds).numpy(), torch.cat(targets).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    parser.add_argument("--pretrained", action="store_true", help="start from ImageNet weights")
    parser.add_argument("--augment", action="store_true", help="SpecAugment-style masking")
    parser.add_argument("--name", default="resnet18")
    args = parser.parse_args()

    torch.manual_seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tracks = load_tracks()
    available = {int(p.stem) for p in SPEC_DIR.glob("*.npy")}
    tracks = tracks[tracks.index.isin(available)]
    genres = get_labels(tracks)
    encoder = LabelEncoder().fit(genres)
    class_names = list(encoder.classes_)
    labels = dict(zip(genres.index, encoder.transform(genres)))

    train_ids, val_ids, test_ids = get_split_ids(tracks)
    loader = lambda ids, aug, shuffle: DataLoader(SpectrogramDataset(ids, labels, aug),
                                                  batch_size=args.batch_size, shuffle=shuffle, num_workers=2)
    train_loader = loader(train_ids, args.augment, True)
    val_loader = loader(val_ids, False, False)
    test_loader = loader(test_ids, False, False)
    print(f"train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}, {len(class_names)} genres")

    model = build_model(len(class_names), args.pretrained, args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config.MODELS_DIR.mkdir(exist_ok=True)
    best_path = config.MODELS_DIR / f"{args.name}.pt"
    best_val, bad_epochs = float("inf"), 0
    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_pred, val_true = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val acc {(val_pred == val_true).mean():.4f}")

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    plot_loss_curves(train_losses, val_losses, args.name)
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_pred, test_true = run_epoch(model, test_loader, loss_fn, device)
    evaluate(test_true, test_pred, class_names, f"{args.name}_test")


if __name__ == "__main__":
    main()
