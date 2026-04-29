import os
import json
import glob
from typing import List, Tuple

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extractor import extract_from_bytes


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_image_files(root: str) -> List[Tuple[str, str]]:
    """
    Return list of (filepath, label) for images under root where structure is root/label/*.ext
    """
    items: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    labels = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    for label in labels:
        pattern = os.path.join(root, label, "**", "*")
        for fp in glob.glob(pattern, recursive=True):
            ext = os.path.splitext(fp)[1].lower()
            if ext in IMAGE_EXTS and os.path.isfile(fp):
                items.append((fp, label))
    if not items:
        raise RuntimeError(f"No images found in {root}")
    return items


def train_and_save(dataset_dir: str = "data/train", model_path: str = "models/model.joblib") -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    items = _iter_image_files(dataset_dir)
    labels_list = sorted({label for _, label in items})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels_list)}

    X: List[np.ndarray] = []
    y: List[int] = []

    for fp, label in items:
        with open(fp, "rb") as f:
            b = f.read()
        feats = extract_from_bytes(b)
        X.append(feats)
        y.append(label_to_idx[label])

    X_arr = np.vstack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="auto",
                ),
            ),
        ]
    )
    pipeline.fit(X_arr, y_arr)

    artifact = {
        "pipeline": pipeline,
        "labels": labels_list,
        "feature": {
            "type": "hog+colorhist",
            "image_size": [256, 256],
            "hog": {"orientations": 9, "ppc": [16, 16], "cpb": [2, 2]},
            "color_hist_bins": 32,
        },
    }

    dump(artifact, model_path)
    print(f"Model saved to {model_path} with labels: {labels_list}")


if __name__ == "__main__":
    train_and_save()
