"""Linear probes and collapse diagnostics after a finished run."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
from torch_geometric.loader import DataLoader

from core.trainer import collapse_diagnostics


@torch.no_grad()
def collect_features(model, dataset, batch_size: int, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        encoded = model.encode(batch)
        features.append(encoded.detach().cpu().numpy())
        labels.append(batch.y.detach().cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def probe_regression(model, train_set, test_set, batch_size: int, device, tag: str) -> dict:
    x_train, y_train = collect_features(model, train_set, batch_size, device)
    x_test, y_test = collect_features(model, test_set, batch_size, device)
    print("Data shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    collapse = collapse_diagnostics(x_test, None, tag=tag)
    regressor = Ridge()
    regressor.fit(x_train, y_train)
    predictions = regressor.predict(x_test)
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(regressor.score(x_train, y_train))
    print(f"Train R2.: {r2}")
    print(f"MAE.: {mae}")
    return {"mae": mae, "r2": r2, "collapse": collapse}


def probe_classification(model, train_set, test_set, batch_size: int, device, tag: str) -> dict:
    x_train, y_train = collect_features(model, train_set, batch_size, device)
    x_test, y_test = collect_features(model, test_set, batch_size, device)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    print("Data shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    collapse = collapse_diagnostics(x_test, y_test, tag=tag)
    classifier = LogisticRegression(max_iter=10000, class_weight="balanced", n_jobs=-1)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    raw_acc = float(accuracy_score(y_test, predictions))
    balanced = float(balanced_accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
    print(
        f"[PROBE] raw_acc={raw_acc:.4f} | balanced_acc={balanced:.4f} "
        f"| macro_F1={macro_f1:.4f} | weighted_F1={weighted_f1:.4f}"
    )
    return {
        "raw_acc": raw_acc,
        "balanced_acc": balanced,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "collapse": collapse,
    }
