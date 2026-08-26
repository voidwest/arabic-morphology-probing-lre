from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .probes import _validated_labels


STIMULUS_CLUSTER_FIELDS = (
    "surface_dediac",
    "surface",
    "lemma",
    "root",
    "abstract_pattern",
)


def stimulus_cluster_id(row: dict[str, Any]) -> str:
    """Hash the exact dataset fields that determine the frozen rendered prompt."""
    payload = [str(row.get(field, "")) for field in STIMULUS_CLUSTER_FIELDS]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ProbeFit:
    alpha: float
    selected_layer: int
    test_accuracy: float
    test_macro_f1: float
    test_predictions: list[str]
    dev_layer_metrics: list[dict[str, float]]


def evaluate_alpha_grid(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    train_indices: Sequence[int],
    dev_indices: Sequence[int],
    test_indices: Sequence[int],
    alphas: Sequence[float],
) -> list[ProbeFit]:
    """Run the frozen dev-layer-selection protocol independently per alpha."""
    if features.ndim != 3:
        raise ValueError("features must have shape [examples, layers, hidden_dim]")
    if len(features) != len(labels):
        raise ValueError("feature and label lengths differ")
    if not train_indices or not dev_indices or not test_indices:
        raise ValueError("train, dev, and test must all be non-empty")
    alpha_values = [float(alpha) for alpha in alphas]
    if not alpha_values or len(set(alpha_values)) != len(alpha_values):
        raise ValueError("alphas must be a non-empty unique sequence")
    if any(alpha <= 0 for alpha in alpha_values):
        raise ValueError("alphas must be positive")

    train = np.asarray(train_indices, dtype=int)
    dev = np.asarray(dev_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    y = _validated_labels(
        labels, train_indices, [*train_indices, *dev_indices, *test_indices]
    )
    if len(np.unique(y[train])) < 2:
        raise ValueError("probe training split has fewer than two classes")

    dev_metrics: dict[float, list[dict[str, float]]] = {
        alpha: [] for alpha in alpha_values
    }
    for layer in range(features.shape[1]):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train, layer, :])
        x_dev = scaler.transform(features[dev, layer, :])
        for alpha in alpha_values:
            estimator = RidgeClassifier(alpha=alpha)
            estimator.fit(x_train, y[train])
            prediction = estimator.predict(x_dev)
            dev_metrics[alpha].append({
                "accuracy": float(accuracy_score(y[dev], prediction)),
                "macro_f1": float(
                    f1_score(y[dev], prediction, average="macro", zero_division=0)
                ),
            })

    output: list[ProbeFit] = []
    train_dev = np.concatenate([train, dev])
    for alpha in alpha_values:
        metrics = dev_metrics[alpha]
        selected_layer = max(
            range(features.shape[1]),
            key=lambda layer: (
                metrics[layer]["accuracy"],
                metrics[layer]["macro_f1"],
                -layer,
            ),
        )
        scaler = StandardScaler()
        x_train_dev = scaler.fit_transform(features[train_dev, selected_layer, :])
        x_test = scaler.transform(features[test, selected_layer, :])
        estimator = RidgeClassifier(alpha=alpha)
        estimator.fit(x_train_dev, y[train_dev])
        prediction = estimator.predict(x_test)
        output.append(ProbeFit(
            alpha=alpha,
            selected_layer=selected_layer,
            test_accuracy=float(accuracy_score(y[test], prediction)),
            test_macro_f1=float(
                    f1_score(y[test], prediction, average="macro", zero_division=0)
            ),
            test_predictions=[str(value) for value in prediction],
            dev_layer_metrics=metrics,
        ))
    return output


PAIR_FIELDS = ("model", "task", "split", "example_id")
LOCATIONS = ("prompt_final", "target_final_subtoken")


def validate_and_pair_predictions(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate exact prompt/target pairing and return one record per example."""
    by_location: dict[str, dict[tuple[str, str, str, str], dict[str, Any]]] = {
        location: {} for location in LOCATIONS
    }
    seen_full: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        location = str(row.get("representation_location", ""))
        if location not in by_location:
            raise ValueError(f"unexpected representation location: {location!r}")
        missing = [
            field for field in (*PAIR_FIELDS, "gold_label", "predicted_label")
            if row.get(field) in (None, "")
        ]
        if missing:
            raise ValueError(f"prediction row missing required fields: {missing}")
        key = tuple(str(row[field]) for field in PAIR_FIELDS)
        full_key = (*key, location)
        if full_key in seen_full:
            raise ValueError(f"duplicate prediction row: {full_key}")
        seen_full.add(full_key)
        by_location[location][key] = row

    prompt_keys = set(by_location["prompt_final"])
    target_keys = set(by_location["target_final_subtoken"])
    if prompt_keys != target_keys:
        missing_target = sorted(prompt_keys - target_keys)
        missing_prompt = sorted(target_keys - prompt_keys)
        raise ValueError(
            "prediction pairing mismatch: "
            f"missing_target={missing_target[:5]} missing_prompt={missing_prompt[:5]}"
        )
    paired: list[dict[str, Any]] = []
    for key in sorted(prompt_keys):
        prompt = by_location["prompt_final"][key]
        target = by_location["target_final_subtoken"][key]
        if str(prompt["gold_label"]) != str(target["gold_label"]):
            raise ValueError(
                f"mismatched gold labels for {key}: "
                f"{prompt['gold_label']!r} != {target['gold_label']!r}"
            )
        prompt_cluster = prompt.get("stimulus_cluster_id")
        target_cluster = target.get("stimulus_cluster_id")
        if prompt_cluster != target_cluster:
            raise ValueError(
                f"mismatched stimulus clusters for {key}: "
                f"{prompt_cluster!r} != {target_cluster!r}"
            )
        item = {
            **dict(zip(PAIR_FIELDS, key, strict=True)),
            "gold_label": str(prompt["gold_label"]),
            "prompt_prediction": str(prompt["predicted_label"]),
            "target_prediction": str(target["predicted_label"]),
        }
        if prompt_cluster not in (None, ""):
            item["stimulus_cluster_id"] = str(prompt_cluster)
        paired.append(item)
    return paired


def stratified_bootstrap_samples(
    gold: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if len(gold) == 0:
        raise ValueError("gold labels must not be empty")
    labels = np.asarray(list(gold), dtype=object)
    rng = np.random.default_rng(int(seed))
    groups = [
        np.flatnonzero(labels == label)
        for label in sorted(set(str(value) for value in labels))
    ]
    sampled = np.concatenate([
        rng.choice(indices, size=(n_bootstrap, len(indices)), replace=True)
        for indices in groups
    ], axis=1)
    expected = Counter(str(value) for value in labels)
    for label, indices in zip(sorted(expected), groups, strict=True):
        if len(indices) != expected[label]:
            raise AssertionError("stratification group count mismatch")
    return sampled


def _macro_f1_replicates(
    gold: np.ndarray,
    prediction: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    values = np.zeros(samples.shape[0], dtype=np.float64)
    classes = sorted(set(str(value) for value in gold))
    for label in classes:
        gold_positive = gold == label
        pred_positive = prediction == label
        tp = (gold_positive & pred_positive)[samples].sum(axis=1)
        fp = ((~gold_positive) & pred_positive)[samples].sum(axis=1)
        fn = (gold_positive & (~pred_positive))[samples].sum(axis=1)
        denominator = 2 * tp + fp + fn
        values += np.divide(
            2 * tp,
            denominator,
            out=np.zeros_like(tp, dtype=np.float64),
            where=denominator != 0,
        )
    return values / len(classes)


def paired_bootstrap_deltas(
    paired_rows: Sequence[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    if not paired_rows:
        raise ValueError("paired rows must not be empty")
    gold = np.asarray([str(row["gold_label"]) for row in paired_rows], dtype=object)
    prompt = np.asarray(
        [str(row["prompt_prediction"]) for row in paired_rows], dtype=object
    )
    target = np.asarray(
        [str(row["target_prediction"]) for row in paired_rows], dtype=object
    )
    samples = stratified_bootstrap_samples(
        gold, n_bootstrap=n_bootstrap, seed=seed
    )
    prompt_correct = prompt == gold
    target_correct = target == gold
    accuracy_prompt_reps = prompt_correct[samples].mean(axis=1)
    accuracy_target_reps = target_correct[samples].mean(axis=1)
    macro_prompt_reps = _macro_f1_replicates(gold, prompt, samples)
    macro_target_reps = _macro_f1_replicates(gold, target, samples)

    observed = {
        "accuracy": (
            float(accuracy_score(gold, prompt)),
            float(accuracy_score(gold, target)),
            accuracy_target_reps - accuracy_prompt_reps,
        ),
        "macro_f1": (
            float(f1_score(gold, prompt, average="macro", zero_division=0)),
            float(f1_score(gold, target, average="macro", zero_division=0)),
            macro_target_reps - macro_prompt_reps,
        ),
    }
    output: list[dict[str, float | int | str]] = []
    for metric, (prompt_value, target_value, replicates) in observed.items():
        lower, upper = np.percentile(replicates, [2.5, 97.5])
        output.append({
            "metric": metric,
            "prompt_value": prompt_value,
            "target_value": target_value,
            "delta": target_value - prompt_value,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "bootstrap_se": float(np.std(replicates, ddof=1)),
            "proportion_delta_gt_zero": float(np.mean(replicates > 0)),
            "n_test": len(paired_rows),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(seed),
        })
    return output


def _cluster_bootstrap_weights(
    gold: np.ndarray,
    cluster_ids: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    """Return row weights after resampling exact-stimulus clusters by class.

    Each replicate samples the same number of clusters within every gold class.
    All occurrences of a sampled cluster receive the cluster multiplicity as
    their weight.  This preserves the occurrence-weighted point estimand while
    treating repeated identical stimuli as one independent sampling unit.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if len(gold) == 0 or len(cluster_ids) != len(gold):
        raise ValueError("gold labels and cluster IDs must be non-empty and aligned")

    cluster_rows: dict[str, list[int]] = defaultdict(list)
    for row_index, cluster_id in enumerate(cluster_ids):
        value = str(cluster_id)
        if not value:
            raise ValueError("stimulus cluster IDs must not be empty")
        cluster_rows[value].append(row_index)
    ordered_clusters = sorted(cluster_rows)
    cluster_position = {value: index for index, value in enumerate(ordered_clusters)}
    row_cluster = np.asarray(
        [cluster_position[str(value)] for value in cluster_ids], dtype=int
    )
    cluster_gold: list[str] = []
    max_cluster_size = 0
    for cluster_id in ordered_clusters:
        rows = cluster_rows[cluster_id]
        labels = {str(gold[index]) for index in rows}
        if len(labels) != 1:
            raise ValueError(
                f"stimulus cluster {cluster_id!r} has conflicting gold labels: "
                f"{sorted(labels)}"
            )
        cluster_gold.append(next(iter(labels)))
        max_cluster_size = max(max_cluster_size, len(rows))

    rng = np.random.default_rng(int(seed))
    cluster_weights = np.zeros(
        (n_bootstrap, len(ordered_clusters)), dtype=np.uint16
    )
    replicate_rows = np.arange(n_bootstrap)[:, None]
    for label in sorted(set(cluster_gold)):
        positions = np.flatnonzero(np.asarray(cluster_gold, dtype=object) == label)
        draws = rng.choice(positions, size=(n_bootstrap, len(positions)), replace=True)
        np.add.at(cluster_weights, (replicate_rows, draws), 1)
    return cluster_weights[:, row_cluster], len(ordered_clusters), max_cluster_size


def _weighted_macro_f1_replicates(
    gold: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.zeros(weights.shape[0], dtype=np.float64)
    classes = sorted(set(str(value) for value in gold))
    for label in classes:
        gold_positive = gold == label
        pred_positive = prediction == label
        tp = weights[:, gold_positive & pred_positive].sum(axis=1)
        fp = weights[:, (~gold_positive) & pred_positive].sum(axis=1)
        fn = weights[:, gold_positive & (~pred_positive)].sum(axis=1)
        denominator = 2 * tp + fp + fn
        values += np.divide(
            2 * tp,
            denominator,
            out=np.zeros_like(tp, dtype=np.float64),
            where=denominator != 0,
        )
    return values / len(classes)


def paired_cluster_bootstrap_deltas(
    paired_rows: Sequence[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    cluster_field: str = "stimulus_cluster_id",
) -> list[dict[str, float | int | str]]:
    """Paired bootstrap over exact-stimulus clusters, stratified by gold class."""
    if not paired_rows:
        raise ValueError("paired rows must not be empty")
    missing = [
        str(row.get("example_id", index))
        for index, row in enumerate(paired_rows)
        if row.get(cluster_field) in (None, "")
    ]
    if missing:
        raise ValueError(
            f"paired rows missing {cluster_field}: {missing[:5]}"
        )
    gold = np.asarray([str(row["gold_label"]) for row in paired_rows], dtype=object)
    prompt = np.asarray(
        [str(row["prompt_prediction"]) for row in paired_rows], dtype=object
    )
    target = np.asarray(
        [str(row["target_prediction"]) for row in paired_rows], dtype=object
    )
    weights, n_clusters, max_cluster_size = _cluster_bootstrap_weights(
        gold,
        [str(row[cluster_field]) for row in paired_rows],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    denominators = weights.sum(axis=1)
    if np.any(denominators == 0):
        raise AssertionError("cluster bootstrap produced an empty replicate")
    prompt_correct = prompt == gold
    target_correct = target == gold
    prompt_accuracy = weights[:, prompt_correct].sum(axis=1) / denominators
    target_accuracy = weights[:, target_correct].sum(axis=1) / denominators
    prompt_macro = _weighted_macro_f1_replicates(gold, prompt, weights)
    target_macro = _weighted_macro_f1_replicates(gold, target, weights)

    observed = {
        "accuracy": (
            float(accuracy_score(gold, prompt)),
            float(accuracy_score(gold, target)),
            target_accuracy - prompt_accuracy,
        ),
        "macro_f1": (
            float(f1_score(gold, prompt, average="macro", zero_division=0)),
            float(f1_score(gold, target, average="macro", zero_division=0)),
            target_macro - prompt_macro,
        ),
    }
    output: list[dict[str, float | int | str]] = []
    for metric, (prompt_value, target_value, replicates) in observed.items():
        lower, upper = np.percentile(replicates, [2.5, 97.5])
        output.append({
            "metric": metric,
            "prompt_value": prompt_value,
            "target_value": target_value,
            "delta": target_value - prompt_value,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "bootstrap_se": float(np.std(replicates, ddof=1)),
            "proportion_delta_gt_zero": float(np.mean(replicates > 0)),
            "n_test": len(paired_rows),
            "n_clusters": n_clusters,
            "duplicate_rows": len(paired_rows) - n_clusters,
            "max_cluster_size": max_cluster_size,
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(seed),
            "sampling_unit": "exact rendered prompt cluster",
            "point_estimand": "occurrence-weighted test performance",
        })
    return output


def group_prediction_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["task"]), str(row["split"]))].append(row)
    return dict(grouped)
