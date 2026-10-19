"""Evaluation helpers: metrics report, confusion matrix, loss curves."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import config


def evaluate(y_true, y_pred, class_names, name):
    """Print and save accuracy / precision / recall / F1, and save a confusion matrix plot."""
    config.RESULTS_DIR.mkdir(exist_ok=True)
    labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)
    print(f"\n=== {name} ===\nAccuracy: {acc:.4f}\n{report}")

    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0, output_dict=True
    )
    with open(config.RESULTS_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} — accuracy {acc:.3f}")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close()
    return acc


def plot_loss_curves(train_losses, val_losses, name):
    config.RESULTS_DIR.mkdir(exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{name} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / f"{name}_loss.png", dpi=150)
    plt.close()
