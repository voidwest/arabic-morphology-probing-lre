from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _validated_labels(
    labels: Sequence[str],
    train_indices: Sequence[int],
    evaluation_indices: Sequence[int],
) -> np.ndarray:
    """Return string labels without learning vocabulary from evaluation data."""
    relevant = sorted(set(int(index) for index in evaluation_indices))
    if any(index < 0 or index >= len(labels) for index in relevant):
        raise IndexError("label index outside feature rows")
    # RidgeClassifier accepts string targets directly. Avoiding LabelEncoder
    # removes any reason to inspect development/test class vocabularies.
    return np.asarray([str(value) for value in labels], dtype=object)


def _probe(alpha: float = 1.0) -> Pipeline:
    # This is the estimator used by the existing paper evaluator.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeClassifier(alpha=float(alpha))),
    ])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def majority_accuracy(train_labels: Sequence[str], test_labels: Sequence[str]) -> float:
    if not train_labels or not test_labels:
        return 0.0
    counts = Counter(str(label) for label in train_labels)
    largest = max(counts.values())
    # Lexical tie-breaking is explicit, deterministic, and independent of row order.
    majority = min(label for label, count in counts.items() if count == largest)
    return sum(label == majority for label in test_labels) / len(test_labels)


def evaluate_probe(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    train_indices: Sequence[int],
    dev_indices: Sequence[int],
    test_indices: Sequence[int],
    probe_seeds: Sequence[int],
) -> dict[str, Any]:
    if features.ndim != 3:
        raise ValueError("features must have shape [examples, layers, hidden_dim]")
    if len(features) != len(labels):
        raise ValueError("feature and label lengths differ")
    if not train_indices or not dev_indices or not test_indices:
        raise ValueError("train, dev, and test must all be non-empty")
    train = np.asarray(train_indices, dtype=int)
    dev = np.asarray(dev_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    y = _validated_labels(
        labels, train_indices, [*train_indices, *dev_indices, *test_indices]
    )
    if len(np.unique(y[train])) < 2:
        raise ValueError("probe training split has fewer than two classes")
    seed_results: list[dict[str, Any]] = []
    # RidgeClassifier is deterministic. Seeds are recorded and repeated only when explicitly configured.
    for seed in probe_seeds:
        dev_by_layer: list[dict[str, float]] = []
        for layer in range(features.shape[1]):
            estimator = _probe()
            estimator.fit(features[train, layer, :], y[train])
            accuracy, macro_f1 = _metrics(y[dev], estimator.predict(features[dev, layer, :]))
            dev_by_layer.append({"accuracy": accuracy, "macro_f1": macro_f1})
        selected_layer = max(
            range(features.shape[1]),
            key=lambda layer: (dev_by_layer[layer]["accuracy"], dev_by_layer[layer]["macro_f1"], -layer),
        )
        estimator = _probe()
        train_dev = np.concatenate([train, dev])
        estimator.fit(features[train_dev, selected_layer, :], y[train_dev])
        test_accuracy, test_macro_f1 = _metrics(
            y[test], estimator.predict(features[test, selected_layer, :])
        )
        seed_results.append({
            "probe_seed": int(seed),
            "selected_layer": selected_layer,
            "dev_layer_metrics": dev_by_layer,
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
        })
    accuracies = [item["test_accuracy"] for item in seed_results]
    macro_f1s = [item["test_macro_f1"] for item in seed_results]
    return {
        "estimator": "StandardScaler + RidgeClassifier(alpha=1.0)",
        "seed_handling": "deterministic estimator; configured seeds are provenance/repetition seeds",
        "seed_results": seed_results,
        "test_accuracy_mean": mean(accuracies),
        "test_accuracy_std": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "test_macro_f1_mean": mean(macro_f1s),
        "test_macro_f1_std": pstdev(macro_f1s) if len(macro_f1s) > 1 else 0.0,
    }
