#!/usr/bin/env python3
"""Paired stratified bootstrap for prompt-vs-target representation deltas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from research_stack.revision.io import load_jsonl
from research_stack.revision.secondary_analysis import (
    group_prediction_rows,
    paired_cluster_bootstrap_deltas,
    paired_bootstrap_deltas,
    stimulus_cluster_id,
    validate_and_pair_predictions,
)


def cell_seed(seed: int, key: tuple[str, str, str]) -> int:
    digest = hashlib.sha256(
        f"{seed}|{'|'.join(key)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Paired representation-location bootstrap",
        "",
        "Delta is `target_final_subtoken - prompt_final`. Intervals are percentile "
        "95% intervals from a gold-class-stratified paired bootstrap over exact "
        "rendered-prompt clusters.",
        "",
        "| Model | Task | Split | Metric | Prompt | Target | Delta | 95% CI | "
        "SE | P(Δ>0) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['task']} | {row['split']} | {row['metric']} | "
            f"{row['prompt_value']:.4f} | {row['target_value']:.4f} | "
            f"{row['delta']:+.4f} | [{row['ci_lower']:+.4f}, "
            f"{row['ci_upper']:+.4f}] | {row['bootstrap_se']:.4f} | "
            f"{row['proportion_delta_gt_zero']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def render_tex(rows: list[dict[str, Any]]) -> str:
    lookup = {
        (row["model"], row["task"], row["split"], row["metric"]): row
        for row in rows
    }
    lines = [
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Model & Task & Split & $\Delta$ accuracy [95\% CI] & "
        r"$\Delta$ macro-F1 [95\% CI] \\",
        r"\midrule",
    ]
    keys = sorted({(row["model"], row["task"], row["split"]) for row in rows})
    for model, task, split in keys:
        accuracy = lookup[(model, task, split, "accuracy")]
        macro = lookup[(model, task, split, "macro_f1")]
        lines.append(
            f"{model.replace('_', r'\_')} & {task.title()} & "
            f"{'Lemma' if split.startswith('lemma') else 'Root'} & "
            f"{accuracy['delta']:+.4f} [{accuracy['ci_lower']:+.4f}, "
            f"{accuracy['ci_upper']:+.4f}] & "
            f"{macro['delta']:+.4f} [{macro['ci_lower']:+.4f}, "
            f"{macro['ci_upper']:+.4f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("output/data/paper1_normalized.jsonl"),
        help="frozen dataset used to verify/backfill exact-prompt cluster IDs",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    raw_rows = load_jsonl(args.predictions)
    dataset = load_jsonl(args.dataset)
    clusters_by_id = {str(row["id"]): stimulus_cluster_id(row) for row in dataset}
    if len(clusters_by_id) != len(dataset):
        raise ValueError("dataset contains duplicate identifiers")
    for row in raw_rows:
        example_id = str(row["example_id"])
        if example_id not in clusters_by_id:
            raise ValueError(f"prediction references unknown dataset identifier {example_id!r}")
        expected_cluster = clusters_by_id[example_id]
        present_cluster = row.get("stimulus_cluster_id")
        if present_cluster is not None and str(present_cluster) != expected_cluster:
            raise ValueError(
                f"prediction cluster mismatch for {example_id!r}: "
                f"expected {expected_cluster}, got {present_cluster}"
            )
        row["stimulus_cluster_id"] = expected_cluster
    grouped = group_prediction_rows(raw_rows)
    expected = {
        (model, task, split)
        for model in ("qwen3-0.6b", "llama-3.2-1b")
        for task in ("pos", "gender", "number")
        for split in ("lemma-heldout", "root-heldout")
    }
    if set(grouped) != expected:
        raise ValueError(
            f"prediction comparison grid mismatch: "
            f"missing={sorted(expected - set(grouped))} "
            f"extra={sorted(set(grouped) - expected)}"
        )

    output: list[dict[str, Any]] = []
    pairing: list[dict[str, Any]] = []
    for key in sorted(grouped):
        paired = validate_and_pair_predictions(grouped[key])
        counts = dict(sorted(Counter(row["gold_label"] for row in paired).items()))
        derived_seed = cell_seed(args.seed, key)
        metrics = paired_cluster_bootstrap_deltas(
            paired, n_bootstrap=args.n_bootstrap, seed=derived_seed
        )
        row_sensitivity = paired_bootstrap_deltas(
            paired, n_bootstrap=args.n_bootstrap, seed=derived_seed
        )
        for metric in metrics:
            output.append({
                "model": key[0],
                "task": key[1],
                "split": key[2],
                **metric,
                "cell_bootstrap_seed": derived_seed,
            })
        cluster_counts = Counter(
            str(row["stimulus_cluster_id"]) for row in paired
        )
        class_cluster_counts = Counter()
        seen_clusters: set[str] = set()
        for row in paired:
            cluster_id = str(row["stimulus_cluster_id"])
            if cluster_id not in seen_clusters:
                class_cluster_counts[row["gold_label"]] += 1
                seen_clusters.add(cluster_id)
        pairing.append({
            "model": key[0],
            "task": key[1],
            "split": key[2],
            "n_test": len(paired),
            "gold_class_counts": counts,
            "identical_example_sets": True,
            "identical_gold_labels": True,
            "missing_examples": 0,
            "duplicate_identifiers": 0,
            "n_stimulus_clusters": len(cluster_counts),
            "duplicate_rows": len(paired) - len(cluster_counts),
            "max_cluster_size": max(cluster_counts.values()),
            "gold_class_cluster_counts": dict(sorted(class_cluster_counts.items())),
            "stratification_preserves_cluster_counts": True,
            "row_bootstrap_sensitivity": row_sensitivity,
        })

    fields = [
        "model", "task", "split", "metric", "prompt_value", "target_value",
        "delta", "ci_lower", "ci_upper", "bootstrap_se",
        "proportion_delta_gt_zero", "n_test", "n_bootstrap",
        "n_clusters", "duplicate_rows", "max_cluster_size", "sampling_unit",
        "point_estimand", "bootstrap_seed", "cell_bootstrap_seed",
    ]
    for path in (
        args.output_csv, args.output_json, args.output_tex, args.output_md
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    args.output_json.write_text(
        json.dumps({
            "schema_version": 2,
            "delta_definition": "target_final_subtoken - prompt_final",
            "method": (
                "paired percentile bootstrap over exact rendered-prompt clusters, "
                "stratified by gold class"
            ),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.seed,
            "input_predictions": str(args.predictions),
            "cluster_source_dataset": str(args.dataset),
            "pairing_validation": pairing,
            "conditional_uncertainty": [
                "trained probe",
                "selected layer",
                "frozen train/dev/test split",
                "fixed model representations",
                "occurrence-weighted test-set estimand",
            ],
            "not_included": [
                "probe retraining",
                "alternative GroupKFold partitions",
                "layer-selection instability",
                "model seeds",
                "alternate prompts",
                "alternate datasets",
                "alternative definitions of a dependent stimulus cluster",
            ],
            "rows": output,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_tex.write_text(render_tex(output), encoding="utf-8")
    args.output_md.write_text(render_markdown(output), encoding="utf-8")
    print(f"wrote {len(output)} metric rows for {len(pairing)} paired comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
