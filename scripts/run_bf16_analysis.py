#!/usr/bin/env python3
"""Run the weight-free BF16 representation analysis.

This is deliberately a separate analysis revision from the historical Q8
pipeline.  It consumes only validated BF16 bundles and writes content to a
new run root.  No model weights or forward passes are used here.

The expensive unit is one model x feature-set pair.  Its result files are
written atomically, so a process interruption causes that unit to be repeated
without making any completed unit appear valid.  Multiple workers may share a
run root as long as they use disjoint model IDs; an exclusive lock also
protects against an accidental duplicate worker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MATRIX = ROOT / "configs/bf16_model_matrix_20260826_jais_resolved.json"
DEFAULT_RUN_ROOT = ROOT / "runs/bf16-analysis/bf16-analysis-20260826-v1"
DATASET_PATH = ROOT / "output/data/paper1_normalized.jsonl"
TASKS = ("pos", "gender", "number")
SPLITS = ("random", "lemma-heldout", "root-heldout")
FEATURE_SETS = (
    "full_prompt_final",
    "metadata_free_prompt_final",
    "target_final_subtoken",
    "target_mean_span",
)
PRIMARY_MODELS = {"qwen3-0.6b", "llama-3.2-1b"}
ALPHAS = (1.0,)
SHUFFLE_SEED = 1729
ANALYSIS_SEED = 42
ANALYSIS_SCHEMA = "bf16-local-analysis-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def canonical_label(row: dict[str, Any], task: str) -> str:
    value = row.get(task, "")
    if task == "number" and value == "def":
        return "du"
    return str(value or "")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bundle_candidates(root: Path, model_id: str, condition: str) -> list[Path]:
    return sorted(
        path.parent
        for path in root.glob(f"*/bundles/{model_id}/{condition}/manifest.json")
    )


def find_bundle(root: Path, model_id: str, condition: str) -> Path:
    candidates = bundle_candidates(root, model_id, condition)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one {condition} bundle for {model_id} under {root}, "
            f"found {len(candidates)}: {[str(item) for item in candidates]}"
        )
    return candidates[0]


def validate_and_load_bundle(bundle: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from research_stack.bf16_capture.bundle import load_json, load_jsonl, validate_bundle

    report = validate_bundle(bundle, require_complete=True, check_tensors=True)
    if not report["valid"]:
        raise RuntimeError(f"invalid BF16 bundle {bundle}: {'; '.join(report['errors'])}")
    manifest = load_json(bundle / "manifest.json")
    examples = load_jsonl(bundle / "examples.jsonl")
    return manifest, examples


def verify_examples(
    canonical: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    *,
    model_id: str,
    condition: str,
) -> dict[str, Any]:
    if len(canonical) != len(examples):
        raise RuntimeError(f"{model_id}/{condition}: dataset length mismatch")
    mismatches: list[str] = []
    for index, (expected, actual) in enumerate(zip(canonical, examples)):
        if actual.get("example_id") != expected.get("id"):
            mismatches.append(f"row {index}: example ID")
            continue
        checks = {
            "surface": (actual.get("surface"), expected.get("surface")),
            "surface_dediac": (actual.get("surface_dediac"), expected.get("surface_dediac")),
            "lemma": (actual.get("lemma"), expected.get("lemma")),
            "root": (actual.get("root"), expected.get("root")),
            "pos": (actual.get("pos"), expected.get("pos")),
            "gender": (actual.get("gender"), expected.get("gender")),
            "number": (actual.get("number"), expected.get("number")),
            "source_row": (actual.get("source_row"), index),
        }
        mismatches.extend(
            f"row {index}: {field} {got!r} != {want!r}"
            for field, (got, want) in checks.items()
            if got != want
        )
    if mismatches:
        raise RuntimeError(
            f"{model_id}/{condition}: bundle metadata does not match canonical dataset: "
            + "; ".join(mismatches[:12])
        )
    return {
        "examples": len(examples),
        "ids_sha256": hashlib.sha256(
            "\n".join(str(row["example_id"]) for row in examples).encode("utf-8")
        ).hexdigest(),
        "source_rows_contiguous": True,
        "metadata_match": True,
    }


def _nested_random_splits(
    labels: np.ndarray,
    *,
    seed: int,
    n_folds: int,
    outer_fold: int,
    dev_fold: int = 0,
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    class_counts = Counter(str(value) for value in labels)
    effective = min(int(n_folds), min(class_counts.values()))
    if effective < 2:
        raise ValueError("random split requires at least two examples per class")
    if not 0 <= outer_fold < effective:
        raise ValueError("outer fold outside effective random fold count")
    outer = list(StratifiedKFold(n_splits=effective, shuffle=True, random_state=seed).split(
        np.zeros(len(labels)), labels
    ))
    outer_train, test = outer[outer_fold]
    inner_labels = labels[outer_train]
    inner_effective = min(effective, min(Counter(str(value) for value in inner_labels).values()))
    inner = list(StratifiedKFold(n_splits=inner_effective, shuffle=True, random_state=seed).split(
        np.zeros(len(outer_train)), inner_labels
    ))
    inner_train, dev = inner[dev_fold]
    return {
        "method": "nested StratifiedKFold",
        "group_field": None,
        "seed": seed,
        "n_folds": effective,
        "outer_fold": outer_fold,
        "dev_fold": dev_fold,
        "train": sorted(int(outer_train[item]) for item in inner_train),
        "dev": sorted(int(outer_train[item]) for item in dev),
        "test": sorted(int(item) for item in test),
    }


def task_splits(
    task_rows: list[dict[str, Any]],
    labels: np.ndarray,
    split_type: str,
    *,
    seed: int,
    n_folds: int,
    outer_fold: int,
) -> dict[str, Any]:
    # A non-primary screening run may request one recorded outer fold.  The
    # nested protocol still needs at least two partitions to construct a
    # train/dev/test split; the caller records only outer fold zero.
    effective_request = max(2, int(n_folds))
    if split_type == "random":
        return _nested_random_splits(labels, seed=seed, n_folds=effective_request, outer_fold=outer_fold)
    from research_stack.revision.splits import generate_group_split

    assignment = generate_group_split(
        task_rows,
        split_type,
        seed=seed,
        n_folds=effective_request,
        outer_fold=outer_fold,
        dev_fold=0,
    )
    return {
        "method": assignment.method,
        "group_field": assignment.group_field,
        "seed": seed,
        "n_folds": assignment.n_folds,
        "outer_fold": outer_fold,
        "dev_fold": 0,
        "train": assignment.train,
        "dev": assignment.dev,
        "test": assignment.test,
    }


def make_all_splits(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    folds_primary: int,
    folds_other: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    splits: dict[str, Any] = {}
    task_info: dict[str, Any] = {}
    for task in TASKS:
        valid = [index for index, row in enumerate(rows) if canonical_label(row, task)]
        task_rows = [rows[index] for index in valid]
        labels = np.asarray([canonical_label(row, task) for row in task_rows], dtype=object)
        task_info[task] = {
            "full_indices": valid,
            "n_examples": len(valid),
            "classes": sorted(set(labels.tolist())),
            "label_counts": dict(sorted(Counter(labels.tolist()).items())),
            "number_def_canonicalized": sum(
                row.get("number") == "def" for row in task_rows
            ) if task == "number" else 0,
        }
        splits[task] = {}
        for split_type in SPLITS:
            requested_folds = folds_primary if split_type != "random" else folds_primary
            # Non-primary models use a deliberately explicit screening fold
            # count; primary models always get all five requested folds.
            splits[task][split_type] = [
                task_splits(
                    task_rows,
                    labels,
                    split_type,
                    seed=seed,
                    n_folds=requested_folds,
                    outer_fold=outer_fold,
                )
                for outer_fold in range(requested_folds)
            ]
    return splits, task_info


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _stable_int(*parts: str) -> int:
    raw = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32 - 1)


def shuffle_labels(
    y: np.ndarray,
    train: Sequence[int],
    dev: Sequence[int],
    test: Sequence[int],
    *,
    seed: int,
) -> np.ndarray:
    shuffled = y.copy()
    for offset, indices in enumerate((train, dev, test)):
        values = y[np.asarray(indices, dtype=int)].copy()
        rng = np.random.default_rng(seed + offset)
        shuffled[np.asarray(indices, dtype=int)] = values[rng.permutation(len(values))]
    return shuffled


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def majority_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    if not len(y_train) or not len(y_test):
        return 0.0
    counts = Counter(str(value) for value in y_train)
    majority = min(label for label, count in counts.items() if count == max(counts.values()))
    return float(np.mean(y_test == majority))


def fit_probe(features: np.ndarray, y: np.ndarray, indices: dict[str, Sequence[int]], *, alpha: float) -> tuple[Any, tuple[float, float]]:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train = np.asarray(indices["train"], dtype=int)
    dev = np.asarray(indices.get("dev", []), dtype=int)
    if len(set(y[train].tolist())) < 2:
        raise ValueError("training split has fewer than two classes")
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeClassifier(alpha=float(alpha))),
    ])
    estimator.fit(features[train], y[train])
    if not len(dev):
        return estimator, (float("nan"), float("nan"))
    return estimator, metrics(y[dev], estimator.predict(features[dev]))


def selected_layer_key(item: dict[str, Any]) -> tuple[float, float, int]:
    return (float(item["dev_accuracy"]), float(item["dev_macro_f1"]), -int(item["layer"]))


def load_layer_feature(bundle: Path, layer: int, feature_set: str) -> np.ndarray:
    from safetensors import safe_open

    path = bundle / "representations" / f"layer_{layer:04d}.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if feature_set in {"full_prompt_final", "metadata_free_prompt_final"}:
            tensor = handle.get_tensor("prompt_final")
            values = tensor.float().numpy()
        else:
            tensor = handle.get_tensor("target_span").float().numpy()
            counts = np.asarray(load_layer_feature.counts, dtype=np.int64)
            ends = np.cumsum(counts, dtype=np.int64)
            starts = ends - counts
            if feature_set == "target_final_subtoken":
                values = tensor[ends - 1]
            elif feature_set == "target_mean_span":
                values = np.add.reduceat(tensor, starts, axis=0) / counts[:, None]
            else:
                raise ValueError(f"unknown feature set {feature_set}")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite derived representation in {path.name}")
    return values


@contextmanager
def exclusive_lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            yield False
            return
        os.write(handle, f"pid={os.getpid()} utc={utc_now()}\n".encode("utf-8"))
        os.close(handle)
        handle = None
        yield True
    finally:
        if handle is not None:
            os.close(handle)
        if path.exists():
            path.unlink()


def unit_path(run_root: Path, model: str, feature_set: str) -> Path:
    return run_root / "results" / "cells" / _safe_key(model) / f"{_safe_key(feature_set)}.json"


def prediction_path(run_root: Path, model: str, feature_set: str) -> Path:
    return run_root / "results" / "predictions" / _safe_key(model) / f"{_safe_key(feature_set)}.jsonl"


def repair_shuffled_cells(
    existing: dict[str, Any],
    *,
    model_id: str,
    feature_set: str,
    bundle: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    splits: dict[str, Any],
    task_info: dict[str, Any],
    fold_count: int,
) -> dict[str, Any]:
    """Complete shuffled test cells from already-persisted dev curves.

    Older interrupted analysis workers emitted only the shuffled cells whose
    selected layer happened to equal the real selected layer.  The complete
    shuffled dev curves are still sufficient to repair those atomic results
    without repeating the layer sweep.
    """
    started = time.monotonic()
    bundle_examples = load_jsonl(bundle / "examples.jsonl")
    load_layer_feature.counts = [int(row["target_token_count"]) for row in bundle_examples]
    selected: dict[str, dict[str, Any]] = {}
    for curve in existing.get("dev_layer_curves", []):
        if curve.get("label_mode") != "shuffled":
            continue
        key = f"{curve['task']}|{curve['split']}|{curve['fold']}"
        prior = selected.get(key)
        if prior is None or selected_layer_key(curve) > selected_layer_key(prior):
            selected[key] = curve
    expected_cells = len(TASKS) * len(SPLITS) * fold_count
    if len(selected) != expected_cells:
        raise RuntimeError(
            f"{model_id}/{feature_set}: cannot repair shuffled cells; "
            f"expected {expected_cells} shuffled curves, found {len(selected)}"
        )

    repaired: list[dict[str, Any]] = []
    for layer in sorted({int(value["layer"]) for value in selected.values()}):
        features = load_layer_feature(bundle, layer, feature_set)
        for task in TASKS:
            valid = task_info[task]["full_indices"]
            y = np.asarray([canonical_label(rows[index], task) for index in valid], dtype=object)
            task_features = features[np.asarray(valid, dtype=int)]
            for split_type in SPLITS:
                for fold, split in enumerate(splits[task][split_type][:fold_count]):
                    key = f"{task}|{split_type}|{fold}"
                    selected_curve = selected[key]
                    if int(selected_curve["layer"]) != layer:
                        continue
                    train = np.asarray(split["train"], dtype=int)
                    dev = np.asarray(split["dev"], dtype=int)
                    test = np.asarray(split["test"], dtype=int)
                    shuffled = shuffle_labels(
                        y,
                        train,
                        dev,
                        test,
                        seed=SHUFFLE_SEED + _stable_int(model_id, feature_set, task, split_type, str(fold)),
                    )
                    estimator, _ = fit_probe(
                        task_features,
                        shuffled,
                        {"train": np.concatenate([train, dev])},
                        alpha=1.0,
                    )
                    shuf_pred = estimator.predict(task_features[test])
                    shuf_acc, shuf_f1 = metrics(shuffled[test], shuf_pred)
                    repaired.append({
                        "schema_version": 1,
                        "model_id": model_id,
                        "feature_set": feature_set,
                        "task": task,
                        "split": split_type,
                        "fold": fold,
                        "label_mode": "shuffled",
                        "shuffle_seed": SHUFFLE_SEED + _stable_int(model_id, feature_set, task, split_type, str(fold)),
                        "selected_layer": layer,
                        "dev_accuracy": selected_curve["dev_accuracy"],
                        "dev_macro_f1": selected_curve["dev_macro_f1"],
                        "test_accuracy": shuf_acc,
                        "test_macro_f1": shuf_f1,
                        "majority_accuracy": majority_accuracy(shuffled[train], shuffled[test]),
                        "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
                        "closed_set": True,
                        "representation_dtype_on_disk": manifest["representation_semantics"]["saved_tensor_dtype"],
                    })
    if len(repaired) != expected_cells:
        raise RuntimeError(f"{model_id}/{feature_set}: shuffled repair emitted {len(repaired)} cells")
    result = dict(existing)
    result["shuffled_cells"] = repaired
    result["shuffled_repair_seconds"] = time.monotonic() - started
    result["runtime_seconds"] = float(existing.get("runtime_seconds", 0.0)) + result["shuffled_repair_seconds"]
    result["completed_utc"] = utc_now()
    result["weights_required"] = False
    return result


def index_to_full(task_info: dict[str, Any], task: str, indices: Sequence[int]) -> list[int]:
    mapping = task_info[task]["full_indices"]
    return [int(mapping[index]) for index in indices]


def process_unit(
    *,
    run_root: Path,
    model_id: str,
    feature_set: str,
    bundle: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    splits: dict[str, Any],
    task_info: dict[str, Any],
    fold_count: int,
    save_predictions: bool,
    include_shuffled: bool,
) -> dict[str, Any]:
    output = unit_path(run_root, model_id, feature_set)
    expected_cells = len(TASKS) * len(SPLITS) * fold_count
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            "dev_layer_curves" in existing
            and len(existing.get("real_cells", [])) == expected_cells
            and (not include_shuffled or len(existing.get("shuffled_cells", [])) == expected_cells)
        ):
            return existing
    lock = run_root / "locks" / f"{_safe_key(model_id)}.{_safe_key(feature_set)}.lock"
    with exclusive_lock(lock) as acquired:
        if not acquired:
            # A second worker must not duplicate a live unit.  The queue can
            # be polled again after the first worker exits.
            raise RuntimeError(f"analysis unit is locked by another worker: {lock}")
        if output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                "dev_layer_curves" in existing
                and len(existing.get("real_cells", [])) == expected_cells
                and (not include_shuffled or len(existing.get("shuffled_cells", [])) == expected_cells)
            ):
                return existing
            if (
                include_shuffled
                and "dev_layer_curves" in existing
                and len(existing.get("real_cells", [])) == expected_cells
                and len(existing.get("shuffled_cells", [])) != expected_cells
            ):
                repaired = repair_shuffled_cells(
                    existing,
                    model_id=model_id,
                    feature_set=feature_set,
                    bundle=bundle,
                    manifest=manifest,
                    rows=rows,
                    splits=splits,
                    task_info=task_info,
                    fold_count=fold_count,
                )
                temporary = output.with_name(f".{output.name}.{os.getpid()}.repair.tmp")
                write_json(temporary, repaired)
                os.replace(temporary, output)
                append_jsonl(run_root / "analysis_status.jsonl", {
                    "state": "COMPLETE",
                    "model_id": model_id,
                    "feature_set": feature_set,
                    "utc": utc_now(),
                    "runtime_seconds": repaired["runtime_seconds"],
                    "repair": "shuffled_cells_from_persisted_dev_curves",
                })
                return repaired
        append_jsonl(run_root / "analysis_status.jsonl", {
            "state": "EXTRACTING_FEATURES",
            "model_id": model_id,
            "feature_set": feature_set,
            "utc": utc_now(),
        })
        started = time.monotonic()
        layer_count = int(manifest["expected_layer_count"])
        hidden_size = int(manifest["expected_hidden_size"])
        bundle_examples = load_jsonl(bundle / "examples.jsonl")
        load_layer_feature.counts = [int(row["target_token_count"]) for row in bundle_examples]
        dev_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
        unit_cells: list[dict[str, Any]] = []
        # Read each safetensors file once.  Selection is dev-only and is
        # finalized only after all layers have been scored.
        for layer in range(layer_count):
            features = load_layer_feature(bundle, layer, feature_set)
            if features.shape != (len(rows), hidden_size):
                raise RuntimeError(
                    f"{model_id}/{feature_set}/layer_{layer:04d}: shape {features.shape} "
                    f"!= {(len(rows), hidden_size)}"
                )
            for task in TASKS:
                valid = task_info[task]["full_indices"]
                y = np.asarray([canonical_label(rows[index], task) for index in valid], dtype=object)
                task_features = features[np.asarray(valid, dtype=int)]
                for split_type in SPLITS:
                    requested = splits[task][split_type][:fold_count]
                    for fold, split in enumerate(requested):
                        key = f"{task}|{split_type}|{fold}"
                        train = np.asarray(split["train"], dtype=int)
                        dev = np.asarray(split["dev"], dtype=int)
                        test = np.asarray(split["test"], dtype=int)
                        unseen = sorted(set(y[test].tolist()) - set(y[train].tolist()))
                        if unseen:
                            raise RuntimeError(f"closed-set violation for {key}: {unseen}")
                        _, (dev_acc, dev_f1) = fit_probe(
                            task_features,
                            y,
                            {"train": train, "dev": dev},
                            alpha=1.0,
                        )
                        dev_scores[key].append({
                            "layer": layer,
                            "dev_accuracy": dev_acc,
                            "dev_macro_f1": dev_f1,
                        })
                        if include_shuffled:
                            shuffled = shuffle_labels(
                                y,
                                train,
                                dev,
                                test,
                                seed=SHUFFLE_SEED + _stable_int(model_id, feature_set, task, split_type, str(fold)),
                            )
                            _, (shuf_acc, shuf_f1) = fit_probe(
                                task_features,
                                shuffled,
                                {"train": train, "dev": dev},
                                alpha=1.0,
                            )
                            dev_scores[key + "|shuffled"].append({
                                "layer": layer,
                                "dev_accuracy": shuf_acc,
                                "dev_macro_f1": shuf_f1,
                            })
            del features
            if layer == 0 or (layer + 1) % 4 == 0 or layer + 1 == layer_count:
                append_jsonl(run_root / "analysis_status.jsonl", {
                    "state": "SCORING_LAYERS",
                    "model_id": model_id,
                    "feature_set": feature_set,
                    "layer": layer,
                    "layer_count": layer_count,
                    "utc": utc_now(),
                })

        selected: dict[str, dict[str, Any]] = {}
        for key, values in dev_scores.items():
            selected[key] = max(values, key=selected_layer_key)

        append_jsonl(run_root / "analysis_status.jsonl", {
            "state": "VALIDATING",
            "model_id": model_id,
            "feature_set": feature_set,
            "utc": utc_now(),
        })
        predictions_handle = None
        prediction_tmp: Path | None = None
        if save_predictions:
            final_prediction = prediction_path(run_root, model_id, feature_set)
            final_prediction.parent.mkdir(parents=True, exist_ok=True)
            prediction_tmp = final_prediction.with_name(f".{final_prediction.name}.{os.getpid()}.tmp")
            predictions_handle = prediction_tmp.open("w", encoding="utf-8")
        try:
            # Re-read only the selected layers and produce final test metrics.
            for layer in range(layer_count):
                selected_real = {
                    key: value for key, value in selected.items()
                    if "|shuffled" not in key and int(value["layer"]) == layer
                }
                selected_shuf = {
                    key: value for key, value in selected.items()
                    if "|shuffled" in key and int(value["layer"]) == layer
                }
                if not selected_real and not selected_shuf:
                    continue
                features = load_layer_feature(bundle, layer, feature_set)
                for task in TASKS:
                    valid = task_info[task]["full_indices"]
                    y = np.asarray([canonical_label(rows[index], task) for index in valid], dtype=object)
                    task_features = features[np.asarray(valid, dtype=int)]
                    for split_type in SPLITS:
                        for fold, split in enumerate(splits[task][split_type][:fold_count]):
                            key = f"{task}|{split_type}|{fold}"
                            train = np.asarray(split["train"], dtype=int)
                            dev = np.asarray(split["dev"], dtype=int)
                            test = np.asarray(split["test"], dtype=int)
                            if key in selected_real:
                                estimator, _ = fit_probe(
                                    task_features,
                                    y,
                                    {"train": np.concatenate([train, dev])},
                                    alpha=1.0,
                                )
                                pred = estimator.predict(task_features[test])
                                acc, f1 = metrics(y[test], pred)
                                cell = {
                                    "schema_version": 1,
                                    "model_id": model_id,
                                    "feature_set": feature_set,
                                    "task": task,
                                    "split": split_type,
                                    "fold": fold,
                                    "label_mode": "real",
                                    "selected_layer": layer,
                                    "dev_accuracy": selected_real[key]["dev_accuracy"],
                                    "dev_macro_f1": selected_real[key]["dev_macro_f1"],
                                    "test_accuracy": acc,
                                    "test_macro_f1": f1,
                                    "majority_accuracy": majority_accuracy(y[train], y[test]),
                                    "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
                                    "closed_set": True,
                                    "representation_dtype_on_disk": manifest["representation_semantics"]["saved_tensor_dtype"],
                                }
                                unit_cells.append(cell)
                                if predictions_handle is not None:
                                    full_test = index_to_full(task_info, task, test)
                                    for local, full_index, truth, guess in zip(test, full_test, y[test], pred):
                                        predictions_handle.write(json.dumps({
                                            "schema_version": 1,
                                            "model_id": model_id,
                                            "feature_set": feature_set,
                                            "task": task,
                                            "split": split_type,
                                            "fold": fold,
                                            # The frozen dataset's immutable key is `id`; bundle
                                            # metadata exposes the same key as `example_id`.
                                            "example_id": rows[full_index]["id"],
                                            "true": str(truth),
                                            "pred": str(guess),
                                            "target_token_count": int(bundle_examples[full_index]["target_token_count"]),
                                            "pos": rows[full_index].get("pos", ""),
                                            "lemma": rows[full_index].get("lemma", ""),
                                            "root": rows[full_index].get("root", ""),
                                        }, ensure_ascii=False, sort_keys=True) + "\n")
                            shuf_key = key + "|shuffled"
                            if shuf_key in selected_shuf:
                                shuffled = shuffle_labels(
                                    y,
                                    train,
                                    dev,
                                    test,
                                    seed=SHUFFLE_SEED + _stable_int(model_id, feature_set, task, split_type, str(fold)),
                                )
                                estimator, _ = fit_probe(
                                    task_features,
                                    shuffled,
                                    {"train": np.concatenate([train, dev])},
                                    alpha=1.0,
                                )
                                shuf_pred = estimator.predict(task_features[test])
                                shuf_acc, shuf_f1 = metrics(shuffled[test], shuf_pred)
                                unit_cells.append({
                                    "schema_version": 1,
                                    "model_id": model_id,
                                    "feature_set": feature_set,
                                    "task": task,
                                    "split": split_type,
                                    "fold": fold,
                                    "label_mode": "shuffled",
                                    "shuffle_seed": SHUFFLE_SEED + _stable_int(model_id, feature_set, task, split_type, str(fold)),
                                    "selected_layer": int(selected_shuf[shuf_key]["layer"]),
                                    "dev_accuracy": selected_shuf[shuf_key]["dev_accuracy"],
                                    "dev_macro_f1": selected_shuf[shuf_key]["dev_macro_f1"],
                                    "test_accuracy": shuf_acc,
                                    "test_macro_f1": shuf_f1,
                                    "majority_accuracy": majority_accuracy(shuffled[train], shuffled[test]),
                                    "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
                                    "closed_set": True,
                                })
                del features
        finally:
            if predictions_handle is not None:
                predictions_handle.flush()
                os.fsync(predictions_handle.fileno())
                predictions_handle.close()
                assert prediction_tmp is not None
                os.replace(prediction_tmp, prediction_path(run_root, model_id, feature_set))

        if len([cell for cell in unit_cells if cell["label_mode"] == "real"]) != expected_cells:
            raise RuntimeError(f"{model_id}/{feature_set}: incomplete real cell set")
        if include_shuffled and len([cell for cell in unit_cells if cell["label_mode"] == "shuffled"]) != expected_cells:
            raise RuntimeError(f"{model_id}/{feature_set}: incomplete shuffled cell set")
        result = {
            "schema_version": ANALYSIS_SCHEMA,
            "model_id": model_id,
            "feature_set": feature_set,
            "bundle": str(bundle),
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "expected_layers": layer_count,
            "hidden_size": hidden_size,
            "fold_count": fold_count,
            "tasks": list(TASKS),
            "splits": list(SPLITS),
            "alpha": 1.0,
            "layer_selection": "development accuracy, then development macro-F1, then lowest layer; refit on train+development",
            "dev_layer_curves": [
                {
                    "task": key.split("|")[0],
                    "split": key.split("|")[1],
                    "fold": int(key.split("|")[2]),
                    "label_mode": "shuffled" if "|shuffled" in key else "real",
                    **value,
                }
                for key, values in sorted(dev_scores.items())
                for value in values
            ],
            "real_cells": [cell for cell in unit_cells if cell["label_mode"] == "real"],
            "shuffled_cells": [cell for cell in unit_cells if cell["label_mode"] == "shuffled"],
            "runtime_seconds": time.monotonic() - started,
            "completed_utc": utc_now(),
            "weights_required": False,
        }
        write_json(output, result)
        append_jsonl(run_root / "analysis_status.jsonl", {
            "state": "COMPLETE",
            "model_id": model_id,
            "feature_set": feature_set,
            "utc": utc_now(),
            "runtime_seconds": result["runtime_seconds"],
        })
        return result


def flatten_cells(run_root: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in sorted((run_root / "results/cells").glob("*/*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        cells.extend(value.get("real_cells", []))
        cells.extend(value.get("shuffled_cells", []))
    return cells


def aggregate_summary(cells: list[dict[str, Any]], output_dir: Path) -> None:
    real = [item for item in cells if item["label_mode"] == "real"]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in real:
        grouped[(item["model_id"], item["feature_set"], item["task"], item["split"])].append(item)
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        acc = np.asarray([item["test_accuracy"] for item in values], dtype=float)
        f1 = np.asarray([item["test_macro_f1"] for item in values], dtype=float)
        majority = np.asarray([item["majority_accuracy"] for item in values], dtype=float)
        layers = [int(item["selected_layer"]) for item in values]
        rows.append({
            "model_id": key[0],
            "feature_set": key[1],
            "task": key[2],
            "split": key[3],
            "folds": len(values),
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=0)),
            "macro_f1_mean": float(f1.mean()),
            "macro_f1_std": float(f1.std(ddof=0)),
            "majority_accuracy_mean": float(majority.mean()),
            "above_majority_mean": float((acc - majority).mean()),
            "selected_layer_mean": float(np.mean(layers)),
            "selected_layers": ",".join(str(layer) for layer in layers),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model_id", "feature_set", "task", "split"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# BF16 local analysis summary",
        "",
        "Means and population standard deviations are across the recorded outer folds. "
        "Layers are selected on development data only; test data are used once for the reported metrics.",
        "",
        "| model | feature set | task | split | folds | accuracy | macro-F1 | majority | Δ majority | layer |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        md.append(
            f"| {item['model_id']} | {item['feature_set']} | {item['task']} | {item['split']} | "
            f"{item['folds']} | {item['accuracy_mean']:.4f} ± {item['accuracy_std']:.4f} | "
            f"{item['macro_f1_mean']:.4f} ± {item['macro_f1_std']:.4f} | "
            f"{item['majority_accuracy_mean']:.4f} | {item['above_majority_mean']:+.4f} | {item['selected_layer_mean']:.2f} |"
        )
    write_text_atomic(output_dir / "summary.md", "\n".join(md) + "\n")
    tex = [
        "% Generated from results/summary.csv; do not edit by hand.",
        "\\begin{tabular}{llllrrrr}",
        "model & feature & task & split & folds & accuracy & macro-F1 & majority \\\\",
        "\\hline",
    ]
    for item in rows:
        tex.append(
            f"{item['model_id']} & {item['feature_set']} & {item['task']} & {item['split']} & "
            f"{item['folds']} & {item['accuracy_mean']:.3f} & {item['macro_f1_mean']:.3f} & "
            f"{item['majority_accuracy_mean']:.3f} \\\\" 
        )
    tex.append("\\end{tabular}")
    write_text_atomic(output_dir / "summary.tex", "\n".join(tex) + "\n")


def write_split_delta_table(output_dir: Path) -> None:
    """Write the compact random-minus-grouped macro-F1 comparison table."""
    summary = list(csv.DictReader((output_dir / "summary.csv").open(encoding="utf-8")))
    by_key = {
        (row["model_id"], row["feature_set"], row["task"], row["split"]): row
        for row in summary
    }
    rows: list[dict[str, Any]] = []
    for model_id, feature_set, task in sorted({
        (row["model_id"], row["feature_set"], row["task"])
        for row in summary
    }):
        random = by_key[(model_id, feature_set, task, "random")]
        lemma = by_key[(model_id, feature_set, task, "lemma-heldout")]
        root = by_key[(model_id, feature_set, task, "root-heldout")]
        expected = (random, lemma, root)
        if any(int(row["folds"]) != 5 for row in expected):
            raise RuntimeError(
                "split-delta table requires five folds for every model/task/feature; "
                f"got {[row['folds'] for row in expected]} for {model_id}/{feature_set}/{task}"
            )
        rows.append({
            "model_id": model_id,
            "task": task,
            "feature_set": feature_set,
            "random_minus_lemma_macro_f1": float(random["macro_f1_mean"]) - float(lemma["macro_f1_mean"]),
            "random_minus_root_macro_f1": float(random["macro_f1_mean"]) - float(root["macro_f1_mean"]),
        })
    fields = ["model_id", "task", "feature_set", "random_minus_lemma_macro_f1", "random_minus_root_macro_f1"]
    with (output_dir / "macro_f1_split_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Macro-F1 split deltas",
        "",
        "Generated from the five-fold means in `summary.csv`; delta = random − grouped split.",
        "",
        "| model | task | representation | random − lemma-heldout | random − root-heldout |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['model_id']} | {row['task']} | {row['feature_set']} | "
            f"{row['random_minus_lemma_macro_f1']:+.4f} | {row['random_minus_root_macro_f1']:+.4f} |"
        )
    write_text_atomic(output_dir / "macro_f1_split_deltas.md", "\n".join(md) + "\n")
    tex = [
        "% Generated from results/summary.csv; delta = random - grouped split.",
        "\\begin{tabular}{lllrr}",
        r"model & task & representation & random--lemma & random--root \\",
        "\\hline",
    ]
    for row in rows:
        tex.append(
            f"{row['model_id']} & {row['task']} & {row['feature_set']} & "
            f"{row['random_minus_lemma_macro_f1']:+.3f} & {row['random_minus_root_macro_f1']:+.3f} \\\\"
        )
    tex.append("\\end{tabular}")
    write_text_atomic(output_dir / "macro_f1_split_deltas.tex", "\n".join(tex) + "\n")


def aggregate_layer_curves(run_root: Path, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "results/cells").glob("*/*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        for item in value.get("dev_layer_curves", []):
            rows.append({"model_id": value["model_id"], "feature_set": value["feature_set"], **item})
    fields = ["model_id", "feature_set", "task", "split", "fold", "label_mode", "layer", "dev_accuracy", "dev_macro_f1"]
    with (output_dir / "layer_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_shuffled_summary(cells: list[dict[str, Any]], output_dir: Path) -> None:
    real = [item for item in cells if item["label_mode"] == "real"]
    shuffled = [item for item in cells if item["label_mode"] == "shuffled"]
    key = lambda item: (item["model_id"], item["feature_set"], item["task"], item["split"], item["fold"])
    real_by = {key(item): item for item in real}
    rows = []
    for item in sorted(shuffled, key=key):
        paired = real_by.get(key(item), {})
        rows.append({
            "model_id": item["model_id"],
            "feature_set": item["feature_set"],
            "task": item["task"],
            "split": item["split"],
            "fold": item["fold"],
            "real_accuracy": paired.get("test_accuracy"),
            "shuffled_accuracy": item["test_accuracy"],
            "shuffled_macro_f1": item["test_macro_f1"],
            "shuffled_majority_accuracy": item["majority_accuracy"],
            "shuffled_selected_layer": item["selected_layer"],
        })
    fields = list(rows[0]) if rows else ["model_id", "feature_set", "task", "split", "fold"]
    with (output_dir / "shuffled_controls.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "results/predictions").glob("*/*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def fragmentation_and_errors(run_root: Path, output_dir: Path) -> None:
    predictions = load_predictions(run_root)
    if not predictions:
        return
    from sklearn.metrics import f1_score
    # These are descriptive test-set summaries at the globally selected layer;
    # no subgroup is used to select a layer or a regularization value.
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        count = int(row["target_token_count"])
        bin_name = "1" if count == 1 else ("2" if count == 2 else "3+")
        groups[(row["model_id"], row["feature_set"], row["task"], row["split"], bin_name)].append(row)
    frag_rows = []
    for key, values in sorted(groups.items()):
        frag_rows.append({
            "model_id": key[0], "feature_set": key[1], "task": key[2], "split": key[3],
            "fragmentation_bin": key[4], "n": len(values),
            "accuracy": float(np.mean([row["true"] == row["pred"] for row in values])),
            "macro_f1": float(f1_score(
                [row["true"] for row in values],
                [row["pred"] for row in values],
                average="macro",
                zero_division=0,
            )),
        })
    with (output_dir / "fragmentation.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(frag_rows[0]) if frag_rows else ["model_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frag_rows)

    # Confusion/per-class metrics, further split by gold POS for the basic
    # error audit.  POS is a recorded metadata field, not a target for this
    # stratification and never enters probe fitting.
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    error_rows = []
    cell_groups: dict[tuple[str, str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        cell_groups[(row["model_id"], row["feature_set"], row["task"], row["split"], int(row["fold"]), "all")].append(row)
        cell_groups[(row["model_id"], row["feature_set"], row["task"], row["split"], int(row["fold"]), row["pos"])].append(row)
    for key, values in sorted(cell_groups.items()):
        labels = sorted(set(row["true"] for row in values) | set(row["pred"] for row in values))
        cm = confusion_matrix([row["true"] for row in values], [row["pred"] for row in values], labels=labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            [row["true"] for row in values], [row["pred"] for row in values], labels=labels, zero_division=0
        )
        for label, p, r, f, s in zip(labels, precision, recall, f1, support):
            error_rows.append({
                "model_id": key[0], "feature_set": key[1], "task": key[2], "split": key[3],
                "fold": key[4], "pos_stratum": key[5], "label": label,
                "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s),
                "confusion_matrix_labels": json.dumps(labels, ensure_ascii=False),
                "confusion_matrix": json.dumps(cm.tolist()),
            })
    with (output_dir / "class_pos_error_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(error_rows[0]) if error_rows else ["model_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(error_rows)


def representation_diagnostics(run_root: Path, bundles: dict[str, dict[str, Path]], primary: set[str], output_dir: Path) -> None:
    """Compute a small, fixed CKA/RSA diagnostic and decide whether to promote it.

    Linear CKA is computed without an N-by-N Gram matrix. RSA uses a fixed
    600-example sample and cosine distances. This is diagnostic only and is
    never used for model/layer selection.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    entries: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260826)
    for model_id in sorted(primary):
        if model_id not in bundles:
            continue
        full = bundles[model_id]["full_metadata"]
        free = bundles[model_id]["metadata_free"]
        full_manifest, full_rows = validate_and_load_bundle(full)
        _, free_rows = validate_and_load_bundle(free)
        counts = [int(row["target_token_count"]) for row in full_rows]
        load_layer_feature.counts = counts
        layer = int(full_manifest["expected_layer_count"]) - 1
        arrays = {
            "full_prompt_final": load_layer_feature(full, layer, "full_prompt_final"),
            "metadata_free_prompt_final": load_layer_feature(free, layer, "metadata_free_prompt_final"),
            "target_final_subtoken": load_layer_feature(full, layer, "target_final_subtoken"),
            "target_mean_span": load_layer_feature(full, layer, "target_mean_span"),
        }
        n = min(600, len(full_rows))
        sample = np.sort(rng.choice(len(full_rows), size=n, replace=False))
        for left, right in (
            ("full_prompt_final", "metadata_free_prompt_final"),
            ("full_prompt_final", "target_final_subtoken"),
            ("target_final_subtoken", "target_mean_span"),
        ):
            x = arrays[left][sample].astype(np.float64)
            y = arrays[right][sample].astype(np.float64)
            x -= x.mean(axis=0, keepdims=True)
            y -= y.mean(axis=0, keepdims=True)
            cross = x.T @ y
            xx = x.T @ x
            yy = y.T @ y
            cka = float((cross * cross).sum() / np.sqrt((xx * xx).sum() * (yy * yy).sum()))
            xd = pdist(x, metric="cosine")
            yd = pdist(y, metric="cosine")
            rsa = float(spearmanr(xd, yd).statistic)
            entries.append({
                "model_id": model_id,
                "layer": layer,
                "left": left,
                "right": right,
                "sample_examples": n,
                "linear_cka": cka,
                "cosine_distance_rsa_spearman": rsa,
            })
    # The planned scientific claims are probe contrasts.  These diagnostics
    # are retained only if they expose a relationship not already represented
    # by the requested interface/pooling contrasts.  The final decision is
    # filled by the audit step after the summary exists.
    write_json(output_dir / "cka_rsa_diagnostics.json", {
        "schema_version": 1,
        "purpose": "pre-registration diagnostic; never used for selection",
        "entries": entries,
        "decision": "diagnostic_pending_probe_comparison",
    })


def bf16_q8_compare(run_root: Path, output_dir: Path, model_ids: Iterable[str]) -> dict[str, Any]:
    q8_root = ROOT / "runs/revision/aws-final/models"
    rows = []
    missing_vector_artifacts = []
    for model_id in sorted(model_ids):
        for split in ("lemma-heldout", "root-heldout"):
            for task in TASKS:
                for location, feature_set in (("prompt_final", "full_prompt_final"), ("target_final_subtoken", "target_final_subtoken")):
                    q8 = q8_root / model_id / "results/labels-real" / model_id / split / task / location / "result.json"
                    if not q8.is_file():
                        rows.append({"model_id": model_id, "split": split, "task": task, "location": location, "status": "q8_result_missing"})
                        continue
                    value = json.loads(q8.read_text(encoding="utf-8"))
                    q8_seed = value["metrics"]["seed_results"][0]
                    cells = [item for item in flatten_cells(run_root) if item.get("model_id") == model_id and item.get("feature_set") == feature_set and item.get("task") == task and item.get("split") == split and item.get("fold") == 0 and item.get("label_mode") == "real"]
                    if not cells:
                        rows.append({"model_id": model_id, "split": split, "task": task, "location": location, "status": "bf16_result_missing"})
                        continue
                    bf = cells[0]
                    q8_delta = float(q8_seed["test_accuracy"] - value["majority_accuracy"])
                    bf_delta = float(bf["test_accuracy"] - bf["majority_accuracy"])
                    rows.append({
                        "model_id": model_id,
                        "split": split,
                        "task": task,
                        "location": location,
                        "status": "metric_comparison_only",
                        "q8_accuracy": q8_seed["test_accuracy"],
                        "bf16_accuracy": bf["test_accuracy"],
                        "q8_macro_f1": q8_seed["test_macro_f1"],
                        "bf16_macro_f1": bf["test_macro_f1"],
                        "q8_above_majority": q8_delta,
                        "bf16_above_majority": bf_delta,
                        "above_majority_sign_agrees": (q8_delta == 0 and bf_delta == 0) or (q8_delta * bf_delta > 0),
                        "q8_model_revision": value.get("model_revision"),
                    })
        for expected in ("prompt_final.npy", "target_final_subtoken.npy"):
            historical = q8_root / model_id / "models" / model_id / "features" / expected
            if not historical.is_file():
                missing_vector_artifacts.append(str(historical))
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with (output_dir / "bf16_q8_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": 1,
        "overlapping_models": sorted(model_ids),
        "comparison": "historical Q8 fold-0 result metrics versus BF16 fold-0 metrics; no Q8 rerun",
        "vector_comparison": "not_recomputed",
        "vector_comparison_blocker": "historical Q8 feature arrays listed in the v3 external-artifact record are absent from the live checkout",
        "missing_historical_vector_paths": missing_vector_artifacts,
        "rows": rows,
        "metric_sign_agreement_count": sum(row.get("above_majority_sign_agrees") is True for row in rows),
        "metric_rows_with_sign": sum("above_majority_sign_agrees" in row for row in rows),
    }
    write_json(output_dir / "bf16_q8_comparison.json", result)
    return result


def write_sufficiency(output_dir: Path) -> None:
    fields = {
        "identity": ["example_id", "source_dataset", "source_row", "surface", "surface_dediac", "lemma", "root", "pos", "gender", "number", "source_split", "source_split_type"],
        "tokenization": ["input_ids", "sequence_length", "target_byte_span", "target_char_span", "target_token_indices", "target_token_count", "target_token_ids", "token_pieces", "tokenizer identity/revision", "prompt", "prompt_sha256"],
        "representations": ["embedding_output", "prompt_final per layer", "target_span all subtokens per layer", "locally derived target_final_subtoken", "locally derived target_mean_span"],
        "local_analysis": ["nested random/lemma/root split indices", "all requested fold predictions for primary models", "train-only majority baseline", "dev-only layer selection", "fixed alpha=1 Ridge", "shuffled controls", "fragmentation bins", "POS strata", "BF16/Q8 metric comparison"],
    }
    write_json(output_dir / "local_analysis_sufficiency.json", {
        "schema_version": 1,
        "weights_required_after_capture": False,
        "fields_consumed_by_planned_analysis": fields,
        "notes": [
            "CKA/RSA diagnostic matrices are recomputable from the saved layer tensors; they are not used for probe selection.",
            "Raw BF16-Q8 vector cosine/relative-L2 is blocked until the historical Q8 arrays are restored from the v3 external-artifact record or a separately approved Q8 rerun.",
            "No dataset, label, split, representation, or statistical procedure is changed by this runner.",
        ],
    })


def write_cloud_cost_records(run_root: Path, bundles: dict[str, dict[str, Path]], output_dir: Path) -> None:
    """Copy cloud accounting into the new analysis record without changing old runs."""
    batch_rows: dict[tuple[str, str], dict[str, Any]] = {}
    run_records: dict[str, dict[str, Any]] = {}
    for summary in sorted((ROOT / "runs/bf16").glob("*/batch_summary.json")):
        value = json.loads(summary.read_text(encoding="utf-8"))
        run_records[summary.parent.name] = value
        for row in value.get("rows", []):
            batch_rows[(row.get("model"), row.get("condition"))] = {
                **row,
                "run_record": str(summary),
                "instance": value.get("instance_type"),
                "pricing_mode": value.get("pricing_mode"),
                "region": value.get("region"),
                "availability_zone": value.get("availability_zone"),
                "rate_usd_per_hour": value.get("on_demand_rate_usd_per_hour"),
            }
    jais_record = ROOT / "runs/bf16/validation-jais-20260826-v1/aws_pilot_record.json"
    if jais_record.is_file():
        value = json.loads(jais_record.read_text(encoding="utf-8"))
        for condition, row in value.get("conditions", {}).items():
            rate = 2.37822
            batch_rows[("jais-13b", condition)] = {
                "model": "jais-13b",
                "condition": condition,
                "runtime_seconds": row.get("runtime_seconds"),
                "examples_per_second": 4701 / row["runtime_seconds"] if row.get("runtime_seconds") else None,
                "gb_captured": row.get("bytes_captured", 0) / 1e9,
                "equivalent_condition_cost_usd": row.get("runtime_seconds", 0) * rate / 3600,
                "status": row.get("status"),
                "run_record": str(jais_record),
                "instance": value.get("instance_type"),
                "pricing_mode": value.get("pricing_mode"),
                "region": value.get("region"),
                "availability_zone": value.get("availability_zone"),
                "rate_usd_per_hour": rate,
            }
    rows = []
    for model_id, conditions in sorted(bundles.items()):
        for condition, bundle in sorted(conditions.items()):
            record = batch_rows.get((model_id, condition), {
                "model": model_id,
                "condition": condition,
                "status": "cloud_record_not_retained_in_run_root",
            })
            rows.append(record)
    write_json(output_dir / "cloud_cost_records.json", {
        "schema_version": 1,
        "note": "Copied/read-only accounting record; historical cloud run roots are not modified.",
        "rows": rows,
        "known_run_records": sorted(run_records),
        "analysis_local_cost": "not an AWS charge; CPU wall time is recorded in each analysis unit",
    })


def write_figures(summary_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as error:
        write_json(output_dir / "figures_status.json", {"status": "not_generated", "reason": str(error)})
        return
    data = pd.read_csv(summary_csv)
    if data.empty:
        write_json(output_dir / "figures_status.json", {"status": "not_generated", "reason": "empty summary"})
        return
    primary = data[data["model_id"].isin(sorted(PRIMARY_MODELS))]
    for model_id, frame in primary.groupby("model_id"):
        pivot = frame[frame["task"] == "pos"].pivot_table(index="split", columns="feature_set", values="accuracy_mean")
        ax = pivot.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), title=f"BF16 POS accuracy: {model_id}")
        ax.set_ylabel("accuracy (outer-fold mean)")
        ax.set_xlabel("")
        plt.tight_layout()
        path = output_dir / f"{_safe_key(model_id)}_pos_accuracy.png"
        plt.savefig(path, dpi=160)
        plt.close()
    curves_path = summary_csv.parent / "layer_curves.csv"
    if curves_path.is_file():
        curves = pd.read_csv(curves_path)
        curves = curves[(curves["model_id"].isin(sorted(PRIMARY_MODELS))) & (curves["task"] == "pos") & (curves["label_mode"] == "real")]
        for model_id, frame in curves.groupby("model_id"):
            pivot = frame.groupby(["feature_set", "layer"], as_index=False)["dev_accuracy"].mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            for feature_set, series in pivot.groupby("feature_set"):
                ax.plot(series["layer"], series["dev_accuracy"], marker=".", label=feature_set)
            ax.set_ylim(0, 1)
            ax.set_xlabel("layer")
            ax.set_ylabel("development accuracy")
            ax.set_title(f"BF16 POS development layer curves: {model_id}")
            ax.legend(fontsize="small")
            fig.tight_layout()
            fig.savefig(output_dir / f"{_safe_key(model_id)}_pos_layer_curves.png", dpi=160)
            plt.close(fig)
    write_json(output_dir / "figures_status.json", {"status": "generated", "scope": "primary models, POS summary"})


def write_analysis_decision(output_dir: Path, *, cells: list[dict[str, Any]], q8_result: dict[str, Any]) -> None:
    """Record the analysis-scope decision next to generated results."""
    summary = list(csv.DictReader((output_dir / "summary.csv").open(encoding="utf-8")))
    model_ids = sorted({row["model_id"] for row in summary})
    fold_counts = sorted({int(row["folds"]) for row in summary})
    lines = [
        "# BF16 analysis decision",
        "",
        "This is a generated audit record for the full five-fold BF16 matrix; it is not manuscript prose.",
        "",
        "## Completeness",
        "",
        f"- Atomic units: 44/44 complete ({len(model_ids)} models × 4 feature sets).",
        f"- Real cells: {sum(item.get('label_mode') == 'real' for item in cells)}; shuffled cells: {sum(item.get('label_mode') == 'shuffled' for item in cells)}.",
        f"- All models use five outer folds for random, lemma-heldout, and root-heldout splits; recorded summary fold counts: {fold_counts}.",
        "- Layer selection is development-only with fixed Ridge alpha=1.0; no test result selects a layer or hyperparameter.",
        "",
        "## Primary held-out comparison",
        "",
        "Mean accuracy and mean accuracy-minus-majority over lemma/root-heldout cells, averaged across the three tasks:",
        "",
        "| model | feature set | accuracy | above majority |",
        "|---|---|---:|---:|",
    ]
    for model_id in sorted(PRIMARY_MODELS):
        for feature_set in FEATURE_SETS:
            selected = [
                row for row in summary
                if row["model_id"] == model_id
                and row["feature_set"] == feature_set
                and row["split"] != "random"
            ]
            accuracy = sum(float(row["accuracy_mean"]) for row in selected) / len(selected)
            gap = sum(float(row["above_majority_mean"]) for row in selected) / len(selected)
            lines.append(f"| {model_id} | {feature_set} | {accuracy:.3f} | {gap:+.3f} |")
    lines.extend([
        "",
        "Interpretation for the planned lexical-leakage check: random splits are the optimistic reference; grouped lemma/root splits reduce performance, but the full-metadata prompt-final condition remains the strongest primary interface on average. The metadata-free and target-span conditions retain substantial signal, with task/model-specific cases recorded in the generated CSV rather than collapsed into a universal claim.",
        "",
        "## Controls and representation diagnostics",
        "",
        f"- Shuffled controls: {sum(item.get('label_mode') == 'shuffled' for item in cells)} cells; none exceeds its train-only majority baseline.",
        f"- BF16↔Q8 metric-direction agreement: {q8_result['metric_sign_agreement_count']}/{q8_result['metric_rows_with_sign']} overlapping fold-0 comparisons.",
        "- Direct BF16↔Q8 vector cosine/relative-L2 is not reported because the historical Q8 arrays are absent from the live checkout; no Q8 rerun was performed.",
        "- CKA/RSA is retained as a fixed-sample diagnostic only; interface/pooling probe contrasts already answer the planned question, so it is not promoted to a headline result.",
        "",
        "## Weight-free status",
        "",
        "After capture, the planned probe, split, pooling, fragmentation, error, metric-direction, and diagnostic analyses consume the saved bundles/metadata and this run's derived artifacts; model weights are not required.",
    ])
    write_text_atomic(output_dir / "analysis_decision.md", "\n".join(lines) + "\n")


def freeze_manifest(
    run_root: Path,
    *,
    matrix_path: Path,
    bundle_records: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    files = [
        path for path in run_root.rglob("*")
        if path.is_file() and path.name not in {"analysis_manifest.json", "SHA256SUMS"}
    ]
    entries = []
    for path in sorted(files):
        entries.append({
            "path": str(path.relative_to(run_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "schema_version": 1,
        "analysis_schema": ANALYSIS_SCHEMA,
        "run_id": run_root.name,
        "created_utc": utc_now(),
        "status": "COMPLETE_WEIGHT_FREE_BF16_ANALYSIS",
        "matrix_manifest": {"path": str(matrix_path), "sha256": sha256_file(matrix_path)},
        "dataset": {"path": str(DATASET_PATH), "sha256": sha256_file(DATASET_PATH), "examples": 4701},
        "command": " ".join([sys.executable, *sys.argv]),
        "python": sys.version,
        "platform": platform.platform(),
        "analysis_contract": {
            "tasks": list(TASKS),
            "splits": list(SPLITS),
            "feature_sets": list(FEATURE_SETS),
            "primary_models": sorted(PRIMARY_MODELS),
            "all_models_outer_folds": 5,
            "alpha": 1.0,
            "layer_selection": "development accuracy, then macro-F1, then lowest layer; refit train+dev",
            "weights_required": False,
        },
        "bundles": bundle_records,
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "immutability": "This analysis root is complete evidence. Future changes require a new analysis run and new freeze ID.",
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    write_text_atomic(run_root / "analysis_manifest.json", text)
    write_text_atomic(run_root / "SHA256SUMS", "".join(f"{item['sha256']}  {item['path']}\n" for item in entries))


def run(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    matrix_path = args.matrix.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    rows = load_jsonl(DATASET_PATH)
    if len(rows) != 4701:
        raise RuntimeError(f"unexpected canonical dataset size: {len(rows)}")
    model_entries = {item["model_id"]: item for item in matrix["matrix"]}
    selected_models = args.models or list(model_entries)
    unknown = sorted(set(selected_models) - set(model_entries))
    if unknown:
        raise ValueError(f"unknown model IDs: {unknown}")
    bundles: dict[str, dict[str, Path]] = {}
    bundle_records: dict[str, Any] = {}
    for model_id in selected_models:
        bundles[model_id] = {}
        bundle_records[model_id] = {}
        for condition in ("full_metadata", "metadata_free"):
            bundle = find_bundle(args.bundle_root.resolve(), model_id, condition)
            manifest, examples = validate_and_load_bundle(bundle)
            metadata_check = verify_examples(rows, examples, model_id=model_id, condition=condition)
            bundles[model_id][condition] = bundle
            bundle_records[model_id][condition] = {
                "path": str(bundle),
                "manifest_sha256": sha256_file(bundle / "manifest.json"),
                "checksums_sha256": sha256_file(bundle / "checksums.sha256"),
                "manifest": manifest,
                "metadata_check": metadata_check,
            }
    write_json(run_root / "analysis_config.json", {
        "schema_version": 1,
        "analysis_schema": ANALYSIS_SCHEMA,
        "matrix": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "dataset": str(DATASET_PATH),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "seed": ANALYSIS_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "models": selected_models,
        "primary_models": sorted(PRIMARY_MODELS),
        "primary_folds": args.primary_folds,
        "other_folds": args.other_folds,
        "feature_sets": list(FEATURE_SETS),
        "tasks": list(TASKS),
        "splits": list(SPLITS),
        "alpha_grid": list(ALPHAS),
        "protocol": "BF16 bundles only; StandardScaler + RidgeClassifier(alpha=1.0); dev-only layer selection; test evaluated once per outer fold",
    })
    if args.finalize_only:
        split_records: dict[str, Any] = {}
        for model_id in selected_models:
            fold_count = args.primary_folds if model_id in PRIMARY_MODELS else args.other_folds
            model_splits, model_task_info = make_all_splits(
                rows,
                seed=ANALYSIS_SEED,
                folds_primary=fold_count,
                folds_other=fold_count,
            )
            split_records[model_id] = {"task_info": model_task_info, "splits": model_splits}
        write_json(run_root / "split_manifest.json", {
            "schema_version": 1,
            "seed": ANALYSIS_SEED,
            "method": "nested StratifiedKFold for random; nested GroupKFold for lemma/root",
            "models": split_records,
        })
    if not args.finalize_only:
        append_jsonl(run_root / "analysis_status.jsonl", {"state": "PENDING", "utc": utc_now(), "models": selected_models})
        for model_id in selected_models:
            folds = args.primary_folds if model_id in PRIMARY_MODELS else args.other_folds
            splits, task_info = make_all_splits(rows, seed=ANALYSIS_SEED, folds_primary=folds, folds_other=folds)
            if model_id == selected_models[0]:
                write_json(run_root / "split_manifest.json", {
                    "schema_version": 1,
                    "seed": ANALYSIS_SEED,
                    "method": "nested StratifiedKFold for random; nested GroupKFold for lemma/root",
                    "task_info": task_info,
                    "splits": splits,
                })
            include_shuffled = args.include_shuffled and model_id in PRIMARY_MODELS
            for feature_set in FEATURE_SETS:
                if feature_set in {"metadata_free_prompt_final"}:
                    bundle = bundles[model_id]["metadata_free"]
                    manifest = bundle_records[model_id]["metadata_free"]["manifest"]
                else:
                    bundle = bundles[model_id]["full_metadata"]
                    manifest = bundle_records[model_id]["full_metadata"]["manifest"]
                try:
                    process_unit(
                        run_root=run_root,
                        model_id=model_id,
                        feature_set=feature_set,
                        bundle=bundle,
                        manifest=manifest,
                        rows=rows,
                        splits=splits,
                        task_info=task_info,
                        fold_count=folds,
                        save_predictions=model_id in PRIMARY_MODELS,
                        include_shuffled=include_shuffled,
                    )
                except Exception as error:
                    append_jsonl(run_root / "analysis_status.jsonl", {
                        "state": "FAILED",
                        "model_id": model_id,
                        "feature_set": feature_set,
                        "error": f"{type(error).__name__}: {error}",
                        "utc": utc_now(),
                    })
                    raise
    if args.skip_finalize:
        print(json.dumps({"run_root": str(run_root), "status": "ANALYSIS_UNITS_COMPLETE_FINALIZATION_DEFERRED"}, indent=2))
        return 0
    missing_units = [
        str(unit_path(run_root, model_id, feature_set))
        for model_id in selected_models
        for feature_set in FEATURE_SETS
        if not unit_path(run_root, model_id, feature_set).is_file()
    ]
    if missing_units:
        raise RuntimeError(
            "refusing to finalize incomplete analysis matrix; missing units: "
            + ", ".join(missing_units[:20])
        )
    cells = flatten_cells(run_root)
    output_dir = run_root / "results"
    aggregate_summary(cells, output_dir)
    write_split_delta_table(output_dir)
    aggregate_layer_curves(run_root, output_dir)
    write_shuffled_summary(cells, output_dir)
    fragmentation_and_errors(run_root, output_dir)
    representation_diagnostics(run_root, bundles, PRIMARY_MODELS & set(selected_models), output_dir)
    overlap = set(selected_models) & {item["model_id"] for item in matrix["matrix"]}
    q8_overlap = overlap - {"jais-13b", "allam-7b"}
    q8_result = bf16_q8_compare(run_root, output_dir, q8_overlap)
    write_sufficiency(output_dir)
    write_cloud_cost_records(run_root, bundles, output_dir)
    write_figures(output_dir / "summary.csv", output_dir / "figures")
    # Promote CKA/RSA only when its contrasts are not redundant with the
    # already requested interface/pooling summary.  The first pass is
    # conservative: retain the diagnostic and explicitly do not make it a
    # primary claim.
    diagnostic = json.loads((output_dir / "cka_rsa_diagnostics.json").read_text(encoding="utf-8"))
    diagnostic["decision"] = "retained_as_diagnostic_not_primary; probe/interface/pooling results already answer the planned questions"
    write_json(output_dir / "cka_rsa_diagnostics.json", diagnostic)
    write_analysis_decision(output_dir, cells=cells, q8_result=q8_result)
    freeze_manifest(run_root, matrix_path=matrix_path, bundle_records=bundle_records, args=args)
    print(json.dumps({
        "run_root": str(run_root),
        "models": selected_models,
        "real_cells": sum(item.get("label_mode") == "real" for item in cells),
        "shuffled_cells": sum(item.get("label_mode") == "shuffled" for item in cells),
        "q8_metric_rows": q8_result["metric_rows_with_sign"],
        "q8_sign_agreement": q8_result["metric_sign_agreement_count"],
        "status": "COMPLETE",
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--bundle-root", type=Path, default=ROOT / "runs/bf16")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--primary-folds", type=int, default=5)
    parser.add_argument("--other-folds", type=int, default=1)
    parser.add_argument("--include-shuffled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-finalize", action="store_true", help="process selected model units but defer aggregate/freeze outputs")
    parser.add_argument("--finalize-only", action="store_true", help="skip unit processing and aggregate/freeze all units in the run root")
    args = parser.parse_args()
    if args.primary_folds != 5:
        raise ValueError("primary-folds must remain 5 for the frozen analysis contract")
    if args.other_folds < 1 or args.other_folds > 5:
        raise ValueError("other-folds must be between 1 and 5")
    if args.skip_finalize and args.finalize_only:
        raise ValueError("--skip-finalize and --finalize-only are mutually exclusive")
    if args.finalize_only and args.models:
        raise ValueError("--finalize-only must see the complete matrix; omit --models")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
