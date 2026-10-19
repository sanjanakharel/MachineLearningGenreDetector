"""Train traditional ML models on tabular audio features.

    python -m src.train_baseline --model logreg              # FMA precomputed features
    python -m src.train_baseline --model all --features ours # our Librosa features
"""
import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config
from .data import get_labels, get_split_ids, load_precomputed_features, load_tracks
from .evaluate import evaluate


def build_model(name):
    # Regularization knobs to experiment with are noted on each model.
    if name == "logreg":
        # C = inverse regularization strength; try penalty="l1" with solver="saga"
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", max_iter=2000))
    if name == "rf":
        # max_depth / min_samples_leaf limit overfitting
        return RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1,
                                      n_jobs=-1, random_state=config.RANDOM_SEED)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf"))
    if name == "xgb":
        # reg_lambda (L2), reg_alpha (L1), max_depth, subsample
        return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                             tree_method="hist", n_jobs=-1, random_state=config.RANDOM_SEED)
    raise ValueError(f"Unknown model: {name}")


def load_feature_table(which):
    if which == "fma":
        features = load_precomputed_features()
        features.columns = ["_".join(map(str, c)) for c in features.columns]
        return features
    return pd.read_csv(config.DATA_PROCESSED / "features_ours.csv", index_col=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf", "svm", "xgb", "all"])
    parser.add_argument("--features", default="fma", choices=["fma", "ours"])
    args = parser.parse_args()

    tracks = load_tracks()
    features = load_feature_table(args.features)
    labels = get_labels(tracks)

    encoder = LabelEncoder().fit(labels)
    class_names = list(encoder.classes_)

    def split(ids):
        ids = ids.intersection(features.index)
        return features.loc[ids].values, encoder.transform(labels.loc[ids])

    train_ids, val_ids, test_ids = get_split_ids(tracks)
    X_train, y_train = split(train_ids)
    X_val, y_val = split(val_ids)
    X_test, y_test = split(test_ids)
    print(f"train {X_train.shape}, val {X_val.shape}, test {X_test.shape}, {len(class_names)} genres")

    names = ["logreg", "rf", "svm", "xgb"] if args.model == "all" else [args.model]
    config.MODELS_DIR.mkdir(exist_ok=True)
    for name in names:
        print(f"\nTraining {name}...")
        model = build_model(name)
        model.fit(X_train, y_train)
        tag = f"{name}_{args.features}"
        evaluate(y_val, model.predict(X_val), class_names, f"{tag}_val")
        evaluate(y_test, model.predict(X_test), class_names, f"{tag}_test")
        joblib.dump(model, config.MODELS_DIR / f"{tag}.joblib")


if __name__ == "__main__":
    main()
