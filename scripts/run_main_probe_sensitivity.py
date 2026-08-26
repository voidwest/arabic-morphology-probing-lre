#!/usr/bin/env python3
"""Reconstruct primary predictions and run frozen-main alpha sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from research_stack.revision.io import load_jsonl
from research_stack.revision.labels import task_labels
from research_stack.revision.secondary_analysis import evaluate_alpha_grid, stimulus_cluster_id


MODELS = ("qwen3-0.6b", "llama-3.2-1b")
TASKS = ("pos", "gender", "number")
SPLITS = ("lemma-heldout", "root-heldout")
LOCATIONS = ("prompt_final", "target_final_subtoken")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _labels(rows: list[dict[str, Any]], task: str) -> list[str]:
    return task_labels(rows, task)


def _float_equal(first: float, second: float) -> bool:
    return abs(float(first) - float(second)) <= 1e-15


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Alpha sensitivity: main analysis",
        "",
        "The primary setting remains `alpha = 1.0`. Each alpha independently uses "
        "development-set layer selection; no alpha is selected on test.",
        "",
        "| Model | Task | Split | Location | Alpha | Layer | Accuracy | Macro-F1 | "
        "Δ Acc. vs 1 | Δ F1 vs 1 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['task']} | {row['split']} | "
            f"{row['representation_location']} | {row['alpha']:g} | "
            f"{row['selected_layer']} | {row['test_accuracy']:.4f} | "
            f"{row['test_macro_f1']:.4f} | "
            f"{row['accuracy_delta_from_alpha_1']:+.4f} | "
            f"{row['macro_f1_delta_from_alpha_1']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def render_tex(rows: list[dict[str, Any]]) -> str:
    # Compact range summary; full 96 rows remain in CSV/JSON/Markdown.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["representation_location"]), []).append(row)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Location & Max $|\Delta|$ acc. & Max $|\Delta|$ F1 & "
        r"Layer changes & Baseline crossings \\",
        r"\midrule",
    ]
    for (model, location), values in grouped.items():
        max_acc = max(abs(row["accuracy_delta_from_alpha_1"]) for row in values)
        max_f1 = max(abs(row["macro_f1_delta_from_alpha_1"]) for row in values)
        layer_changes = sum(
            row["selected_layer"] != row["alpha_1_selected_layer"]
            for row in values if row["alpha"] != 1.0
        )
        crossings = sum(bool(row["majority_crossing_vs_alpha_1"]) for row in values)
        lines.append(
            f"{model.replace('_', r'\_')} & "
            f"{'PF' if location == 'prompt_final' else 'Target'} & "
            f"{max_acc:.4f} & {max_f1:.4f} & {layer_changes}/18 & {crossings} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0]
    )
    args = parser.parse_args()
    if 1.0 not in args.alphas:
        raise ValueError("alpha grid must contain the frozen primary alpha=1.0")

    dataset = load_jsonl(args.dataset)
    dataset_ids = [str(row["id"]) for row in dataset]
    cluster_ids = [stimulus_cluster_id(row) for row in dataset]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("dataset contains duplicate example IDs")
    output_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for model in MODELS:
        model_run = args.run_root / "models" / model
        split_values = {
            split: json.loads(
                (model_run / "splits" / split / "split.json").read_text(encoding="utf-8")
            )
            for split in SPLITS
        }
        for location in LOCATIONS:
            feature_path = (
                model_run / "models" / model / "features" / f"{location}.npy"
            )
            if not feature_path.is_file():
                raise FileNotFoundError(feature_path)
            features = np.load(feature_path, mmap_mode="r")
            if features.shape[0] != len(dataset):
                raise ValueError(f"{feature_path}: feature/dataset length mismatch")
            for split in SPLITS:
                split_data = split_values[split]
                base_indices = split_data["indices"]
                if split_data["example_ids"] != {
                    name: [dataset_ids[index] for index in indices]
                    for name, indices in base_indices.items()
                }:
                    raise ValueError(f"{model} {split}: split example IDs do not align")
                for task in TASKS:
                    labels = _labels(dataset, task)
                    indices = {
                        name: [
                            index for index in values
                            if labels[index] != "__MISSING__"
                        ]
                        for name, values in base_indices.items()
                    }
                    fits = evaluate_alpha_grid(
                        features,
                        labels,
                        train_indices=indices["train"],
                        dev_indices=indices["dev"],
                        test_indices=indices["test"],
                        alphas=args.alphas,
                    )
                    baseline = next(fit for fit in fits if fit.alpha == 1.0)
                    frozen_path = (
                        model_run / "results" / "labels-real" / model / split /
                        task / location / "result.json"
                    )
                    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                    frozen_seed = frozen["metrics"]["seed_results"][0]
                    checks = {
                        "selected_layer": baseline.selected_layer == frozen_seed["selected_layer"],
                        "test_accuracy": _float_equal(
                            baseline.test_accuracy, frozen_seed["test_accuracy"]
                        ),
                        "test_macro_f1": _float_equal(
                            baseline.test_macro_f1, frozen_seed["test_macro_f1"]
                        ),
                    }
                    if not all(checks.values()):
                        raise ValueError(
                            f"{model}/{task}/{split}/{location}: frozen alpha=1 "
                            f"reproduction failed: {checks}"
                        )
                    validation_rows.append({
                        "model": model,
                        "task": task,
                        "split": split,
                        "representation_location": location,
                        "frozen_result_path": str(frozen_path),
                        "feature_path": str(feature_path),
                        **checks,
                    })
                    for test_index, prediction in zip(
                        indices["test"], baseline.test_predictions, strict=True
                    ):
                        prediction_rows.append({
                            "model": model,
                            "task": task,
                            "split": split,
                            "representation_location": location,
                            "example_id": dataset_ids[test_index],
                            "stimulus_cluster_id": cluster_ids[test_index],
                            "stimulus_cluster_definition": (
                                "SHA-256 of exact frozen prompt fields: "
                                "surface_dediac, surface/target, lemma, root, abstract_pattern"
                            ),
                            "gold_label": labels[test_index],
                            "predicted_label": prediction,
                            "selected_layer": baseline.selected_layer,
                            "alpha": 1.0,
                            "reconstruction_source": "preserved feature tensor + frozen split",
                        })
                    for fit in fits:
                        output_rows.append({
                            "model": model,
                            "task": task,
                            "split": split,
                            "representation_location": location,
                            "alpha": fit.alpha,
                            "selected_layer": fit.selected_layer,
                            "test_accuracy": fit.test_accuracy,
                            "test_macro_f1": fit.test_macro_f1,
                            "accuracy_delta_from_alpha_1":
                                fit.test_accuracy - baseline.test_accuracy,
                            "macro_f1_delta_from_alpha_1":
                                fit.test_macro_f1 - baseline.test_macro_f1,
                            "alpha_1_selected_layer": baseline.selected_layer,
                            "majority_accuracy": frozen["majority_accuracy"],
                            "majority_crossing_vs_alpha_1": (
                                (fit.test_accuracy >= frozen["majority_accuracy"])
                                != (baseline.test_accuracy >= frozen["majority_accuracy"])
                            ),
                        })

    expected_predictions = sum(
        2 * (941 if task == "pos" else 941)
        for _model in MODELS for _split in SPLITS for task in TASKS
    )
    if len(prediction_rows) != expected_predictions:
        raise AssertionError(
            f"expected {expected_predictions} prediction rows, got {len(prediction_rows)}"
        )
    if len(output_rows) != 24 * len(args.alphas):
        raise AssertionError("alpha sensitivity grid is incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "main_alpha1_test_predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)
    fields = [
        "model", "task", "split", "representation_location", "alpha",
        "selected_layer", "test_accuracy", "test_macro_f1",
        "accuracy_delta_from_alpha_1", "macro_f1_delta_from_alpha_1",
        "alpha_1_selected_layer", "majority_accuracy",
        "majority_crossing_vs_alpha_1",
    ]
    with (args.output_dir / "alpha_sensitivity_main.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    try:
        prediction_file = str(predictions_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        prediction_file = str(predictions_path.resolve())
    write_json(args.output_dir / "alpha_sensitivity_main.json", {
        "schema_version": 1,
        "primary_alpha": 1.0,
        "alphas": [float(value) for value in args.alphas],
        "test_selection": False,
        "layer_selection": "independent development selection for each alpha",
        "prediction_file": prediction_file,
        "label_aliases": {"number": {"def": "du"}},
        "stimulus_cluster_definition": (
            "SHA-256 of exact frozen prompt fields: surface_dediac, surface/target, "
            "lemma, root, abstract_pattern"
        ),
        "frozen_reproduction": validation_rows,
        "rows": output_rows,
    })
    (args.output_dir / "alpha_sensitivity_main.md").write_text(
        render_markdown(output_rows), encoding="utf-8"
    )
    (args.output_dir / "alpha_sensitivity_main.tex").write_text(
        render_tex(output_rows), encoding="utf-8"
    )
    print(
        f"wrote {len(prediction_rows)} primary predictions and "
        f"{len(output_rows)} alpha rows to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
