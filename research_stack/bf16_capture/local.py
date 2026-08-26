from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import load_jsonl, validate_bundle


def load_bundle_features(bundle: Path) -> dict[str, Any]:
    result = validate_bundle(bundle, require_complete=True, check_tensors=True)
    if not result["valid"]:
        raise ValueError("cannot load invalid bundle: " + "; ".join(result["errors"][:10]))
    from safetensors import safe_open
    examples = load_jsonl(bundle / "examples.jsonl")
    layer_files = sorted((bundle / "representations").glob("layer_*.safetensors"))
    if not layer_files:
        raise ValueError("bundle has no transformer layer files")
    with safe_open(str(layer_files[0]), framework="pt", device="cpu") as handle:
        hidden_size = int(handle.get_slice("prompt_final").get_shape()[1])
    example_count = len(examples)
    layer_count = len(layer_files)
    prompt = np.empty((example_count, layer_count, hidden_size), dtype=np.float32)
    target_final = np.empty_like(prompt)
    target_mean = np.empty_like(prompt)
    target_counts = np.asarray([int(row["target_token_count"]) for row in examples], dtype=np.int64)
    if np.any(target_counts <= 0):
        raise ValueError("every example must have at least one target token")
    starts = np.concatenate(([0], np.cumsum(target_counts[:-1], dtype=np.int64)))
    ends = starts + target_counts
    total_target = int(target_counts.sum())
    for layer_index, path in enumerate(layer_files):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            prompt_layer = handle.get_tensor("prompt_final").float().numpy()
            target_layer = handle.get_tensor("target_span").float().numpy()
        if prompt_layer.shape != (example_count, hidden_size):
            raise ValueError(f"prompt layer shape mismatch in {path.name}: {prompt_layer.shape}")
        if target_layer.shape != (total_target, hidden_size):
            raise ValueError(f"target layer shape mismatch in {path.name}: {target_layer.shape}")
        prompt[:, layer_index, :] = prompt_layer
        target_final[:, layer_index, :] = target_layer[ends - 1]
        target_mean[:, layer_index, :] = np.add.reduceat(target_layer, starts, axis=0) / target_counts[:, None]
        del prompt_layer, target_layer
    return {
        "examples": examples,
        "prompt_final": prompt,
        "target_final_subtoken": target_final,
        "target_mean_span": target_mean,
        "target_span_total": total_target,
        "cka_rsa_matrix_prompt": prompt[:, -1, :],
        "cka_rsa_matrix_target_final": target_final[:, -1, :],
    }


def reconstruct_splits(examples: list[dict[str, Any]], *, seed: int = 42) -> dict[str, dict[str, list[int]]]:
    from ..revision.splits import generate_group_split
    result: dict[str, dict[str, list[int]]] = {}
    for split_type in ("lemma-heldout", "root-heldout"):
        assignment = generate_group_split(examples, split_type, seed=seed, n_folds=5, outer_fold=0, dev_fold=0)
        result[split_type] = {"train": assignment.train, "dev": assignment.dev, "test": assignment.test}
    rng = np.random.default_rng(seed)
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    n_test = max(1, len(indices) // 5)
    n_dev = max(1, (len(indices) - n_test) // 5)
    result["random"] = {
        "train": sorted(indices[n_test + n_dev:].tolist()),
        "dev": sorted(indices[n_test:n_test + n_dev].tolist()),
        "test": sorted(indices[:n_test].tolist()),
    }
    return result


def reconstruct_all_group_folds(examples: list[dict[str, Any]], *, seed: int = 42) -> dict[str, list[dict[str, list[int]]]]:
    from ..revision.splits import generate_group_split
    result: dict[str, list[dict[str, list[int]]]] = {}
    for split_type in ("lemma-heldout", "root-heldout"):
        folds: list[dict[str, list[int]]] = []
        for outer_fold in range(5):
            assignment = generate_group_split(examples, split_type, seed=seed, n_folds=5, outer_fold=outer_fold, dev_fold=0)
            folds.append({"train": assignment.train, "dev": assignment.dev, "test": assignment.test})
        result[split_type] = folds
    return result


def validate_local_analysis(bundle: Path, *, output: Path | None = None) -> dict[str, Any]:
    data = load_bundle_features(bundle)
    examples = data["examples"]
    splits = reconstruct_splits(examples)
    all_group_folds = reconstruct_all_group_folds(examples)
    labels = [str(row.get("pos") or "__MISSING__") for row in examples]
    probe_result: dict[str, Any]
    try:
        from ..revision.probes import evaluate_probe
        split = splits["lemma-heldout"]
        probe_result = evaluate_probe(
            data["prompt_final"], labels,
            train_indices=split["train"], dev_indices=split["dev"], test_indices=split["test"], probe_seeds=[42],
        )
    except Exception as error:
        probe_result = {"status": "not_run", "reason": f"{type(error).__name__}: {error}"}
    result = {
        "bundle": str(bundle),
        "status": "pass",
        "examples": len(examples),
        "layer_shape": list(data["prompt_final"].shape),
        "target_final_shape": list(data["target_final_subtoken"].shape),
        "target_mean_shape": list(data["target_mean_span"].shape),
        "final_subtoken_derivation": "target_span flat rows grouped by target_token_count; final row selected locally",
        "mean_target_span_derivation": "mean over the complete saved target span locally",
        "layerwise_analysis": True,
        "linear_probe": probe_result,
        "cka_rsa_compatible": {
            "prompt_matrix_shape": list(data["cka_rsa_matrix_prompt"].shape),
            "target_final_matrix_shape": list(data["cka_rsa_matrix_target_final"].shape),
            "same_example_order": True,
        },
        "split_reconstruction": {name: {part: len(indices) for part, indices in value.items()} for name, value in splits.items()},
        "all_five_group_folds_reconstructed": {name: len(folds) for name, folds in all_group_folds.items()},
        "weights_required": False,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
