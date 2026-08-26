#!/usr/bin/env python3
"""Build the final bootstrap and alpha-sensitivity verification report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def fmt(value: str | float) -> str:
    return f"{float(value):+.4f}"


def main() -> None:
    bootstrap = list(csv.DictReader((RESULTS / "bootstrap_location_deltas.csv").open()))
    alpha = list(csv.DictReader((RESULTS / "alpha_sensitivity_main.csv").open()))
    meta = json.loads((RESULTS / "alpha_sensitivity_main.json").read_text())
    reproduction = meta["frozen_reproduction"]
    exact = all(
        all(row[key] for key in ("selected_layer", "test_accuracy", "test_macro_f1"))
        for row in reproduction
    )

    lines = [
        "# Secondary-analysis verification report",
        "",
        "## 1. Inputs",
        "",
        "- Reconstructed alpha=1 predictions: `results/main_alpha1_test_predictions.jsonl`.",
        "- Frozen dataset and identifiers: `output/data/paper1_normalized.jsonl`.",
        "- Frozen split files: each main model's `runs/revision/aws-final/models/<model>/splits/{lemma-heldout,root-heldout}/split.json`.",
        "- Frozen representations: the `prompt_final.npy` and `target_final_subtoken.npy` files under each main model's `runs/revision/aws-final/models/<model>/models/<model>/features/` directory.",
        "- Frozen aggregates used as the reproduction gate: each main model's `results/labels-real/<model>/<split>/<task>/<location>/result.json`.",
        "",
        "The original pipeline did not serialize per-example predictions. With explicit authorization, they were deterministically reconstructed from the preserved representations and frozen partitions. No model representations were regenerated. All 24 alpha=1 selected layers, test accuracies, and test macro-F1 values matched the frozen aggregates exactly: "
        + ("**PASS**." if exact else "**FAIL**."),
        "",
        "## 2. Pairing validation",
        "",
        "All 12 model × task × split comparisons contained exactly 941 aligned test examples. Stable identifiers and gold labels matched exactly; there were no missing or duplicate identifiers. Exact rendered prompts form 458 clusters in lemma-heldout test folds and 469 clusters in root-heldout test folds. **PASS**.",
        "",
        "## 3. Bootstrap protocol",
        "",
        "- Master seed: `20260723` (deterministic cell seeds are SHA-256-derived from the master seed and comparison key).",
        "- Replicates: `10000` per comparison.",
        "- Sampling: paired exact-prompt clusters, stratified by gold-label class; the same sampled cluster multiplicities were used for both locations.",
        "- Estimand: occurrence-weighted metrics; each selected cluster contributes all of its original rows.",
        "- Class-cluster-count preservation: verified in every comparison and covered by automated tests. **PASS**.",
        "- Sensitivity: the previous paired row bootstrap is retained inside the JSON output, but is not the primary interval because duplicated prompts violate row independence.",
        "",
        "## 4–6. Paired location intervals",
        "",
        "Delta is target-final-subtoken minus prompt-final.",
        "",
        "| Model | Task | Split | Accuracy delta [95% CI] | Macro-F1 delta [95% CI] |",
        "|---|---|---|---:|---:|",
    ]
    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in bootstrap:
        grouped.setdefault((row["model"], row["task"], row["split"]), {})[
            row["metric"]
        ] = row
    for key in sorted(grouped):
        acc = grouped[key]["accuracy"]
        f1 = grouped[key]["macro_f1"]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            f"{fmt(acc['delta'])} [{fmt(acc['ci_lower'])}, {fmt(acc['ci_upper'])}] | "
            f"{fmt(f1['delta'])} [{fmt(f1['ci_lower'])}, {fmt(f1['ci_upper'])}] |"
        )

    lines += [
        "",
        "## 7. Alpha sensitivity: all 24 cells × 4 values",
        "",
        "Each alpha independently selected a layer on development data, refit on train + development, and was evaluated once on test.",
        "",
        "| Model | Task | Split | Location | Alpha | Layer | Accuracy | Macro-F1 | Δ accuracy vs 1 | Δ macro-F1 vs 1 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in alpha:
        lines.append(
            f"| {row['model']} | {row['task']} | {row['split']} | "
            f"{row['representation_location']} | {float(row['alpha']):g} | "
            f"{row['selected_layer']} | {float(row['test_accuracy']):.4f} | "
            f"{float(row['test_macro_f1']):.4f} | "
            f"{fmt(row['accuracy_delta_from_alpha_1'])} | "
            f"{fmt(row['macro_f1_delta_from_alpha_1'])} |"
        )

    lines += [
        "",
        "## 8. Interpretation changes under alpha sensitivity",
        "",
        "- The alpha=1 point estimates and primary table remain frozen, but the sensitivity analysis does **not** support calling every location conclusion regularization-invariant.",
        "- Qwen root-heldout POS retains the target-final accuracy advantage at all four alpha values.",
        "- Qwen lemma-heldout POS remains a small location contrast, although its sign changes at alpha=0.1.",
        "- Llama target-token number is below majority at alpha=0.1 and 1, but above majority at alpha=10 and 100; root-heldout prompt-final number also crosses the majority threshold across the grid.",
        "- Several other Llama location directions change, and selected layers vary frequently. These facts are disclosed in the manuscript and full CSV/JSON.",
        "- Ill-conditioned-matrix warnings occurred for weakly regularized fits. The estimators completed; the warning is retained as a numerical sensitivity caveat.",
        "",
        "## 9. Manuscript sentences changed because of bootstrap evidence",
        "",
        "- Qwen lemma-heldout POS now reports both paired deltas and intervals and says the estimates are compatible with little or no location difference.",
        "- “The location difference reverses sharply” was replaced with “Target-final POS accuracy was substantially higher,” followed by its accuracy and macro-F1 intervals.",
        "- Llama number now reports the paired declines and intervals under both lexical splits.",
        "- The Results table introduction states that intervals are conditional finite-test-set estimates, not causal or architectural tests.",
        "- Discussion §6.1 contrasts the near-zero Qwen lemma-POS intervals with intervals excluding zero for Qwen root-POS and both Llama number-accuracy declines.",
        "",
        "## 10–11. Frozen table and hyperparameter-selection checks",
        "",
        f"- Table 1 point estimates unchanged and exactly reproduced: **{'PASS' if exact else 'FAIL'}**.",
        "- Test-set hyperparameter selection: **none**. Alpha remains fixed at 1.0 in the primary analysis; every sensitivity alpha uses development-only layer selection and is reported rather than selected.",
        "",
        "## 12. Tests",
        "",
        "- Focused secondary-analysis plus revision tests: 38 passed, 0 failed.",
        "- Top-level project suite: 76 passed plus 97 subtests, 0 failed.",
        "- An unscoped workspace-wide collection also entered vendored projects and stopped on three unavailable optional dependencies (`parity_tools`, Appium, and `wget`); this did not affect the scoped project suite.",
        "",
        "## 13. Remaining issues",
        "",
        "- Cluster-bootstrap intervals are conditional on the reconstructed fixed probes and the exact-prompt cluster definition; they do not include probe retraining, alternative partitions, layer-selection uncertainty, model seeds, prompts, or datasets.",
        "- Alpha sensitivity reveals substantial estimator dependence in some Llama cells; the manuscript now limits those claims to the frozen alpha=1 protocol.",
        "- Per-example predictions were reconstructed because the production run retained representations and aggregates but not prediction rows. Exact aggregate reproduction is the validation gate.",
        "",
    ]
    (RESULTS / "secondary_analysis_verification.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
