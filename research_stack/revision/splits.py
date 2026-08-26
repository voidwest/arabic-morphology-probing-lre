from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class SplitAssignment:
    split_type: str
    seed: int
    group_field: str
    train: list[int]
    dev: list[int]
    test: list[int]
    n_folds: int
    outer_fold: int
    dev_fold: int
    method: str = "nested GroupKFold"

    def as_dict(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        def ids(indices: Sequence[int]) -> list[str]:
            return [str(rows[index].get("id", index)) for index in indices]

        return {
            "schema_version": 1,
            "split_type": self.split_type,
            "seed": self.seed,
            "seed_role": "run provenance only; sklearn GroupKFold is deterministic and unshuffled",
            "group_field": self.group_field,
            "method": self.method,
            "source_logic": "scripts/eval_paper1.py: closed_set_splits / GroupKFold",
            "n_folds": self.n_folds,
            "outer_fold": self.outer_fold,
            "dev_fold": self.dev_fold,
            "sizes": {"train": len(self.train), "dev": len(self.dev), "test": len(self.test)},
            "indices": {"train": self.train, "dev": self.dev, "test": self.test},
            "example_ids": {"train": ids(self.train), "dev": ids(self.dev), "test": ids(self.test)},
        }


def load_split(path: str, rows: Sequence[dict[str, Any]]) -> SplitAssignment:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    indices = value["indices"]
    assignment = SplitAssignment(
        split_type=str(value["split_type"]),
        seed=int(value["seed"]),
        group_field=str(value["group_field"]),
        train=[int(item) for item in indices["train"]],
        dev=[int(item) for item in indices["dev"]],
        test=[int(item) for item in indices["test"]],
        n_folds=int(value["n_folds"]),
        outer_fold=int(value["outer_fold"]),
        dev_fold=int(value["dev_fold"]),
        method=str(value["method"]),
    )
    expected_ids = assignment.as_dict(rows)["example_ids"]
    if expected_ids != value.get("example_ids"):
        raise ValueError(f"{path}: split example IDs do not match the dataset/order")
    return assignment


def _groups(rows: Sequence[dict[str, Any]], field: str) -> list[str]:
    output: list[str] = []
    for index, row in enumerate(rows):
        raw = row.get(field)
        output.append(str(raw) if raw not in (None, "") else f"__empty_{index}")
    return output


def generate_group_split(
    rows: Sequence[dict[str, Any]],
    split_type: str,
    *,
    seed: int,
    n_folds: int = 5,
    outer_fold: int = 0,
    dev_fold: int = 0,
) -> SplitAssignment:
    """Create train/dev/test using the paper's deterministic GroupKFold logic.

    The selected outer GroupKFold fold is test. A second GroupKFold over the
    outer-training examples selects dev, preserving the same group exclusion
    rule while adding the dev set needed for revision-run layer selection.
    """
    field_by_type = {"lemma-heldout": "lemma", "root-heldout": "root"}
    if split_type not in field_by_type:
        raise ValueError(f"unsupported leakage-aware split: {split_type}")
    group_field = field_by_type[split_type]
    groups = np.asarray(_groups(rows, group_field), dtype=object)
    unique_groups = np.unique(groups)
    effective_folds = min(int(n_folds), len(unique_groups))
    if effective_folds < 2:
        raise ValueError(f"{split_type} requires at least two distinct groups")
    if not 0 <= outer_fold < effective_folds:
        raise ValueError(f"outer_fold must be in [0, {effective_folds})")
    outer = list(GroupKFold(n_splits=effective_folds).split(
        np.zeros(len(rows)), groups=groups
    ))
    outer_train, test = outer[outer_fold]
    inner_groups = groups[outer_train]
    inner_folds = min(effective_folds, len(np.unique(inner_groups)))
    if inner_folds < 2 or not 0 <= dev_fold < inner_folds:
        raise ValueError("invalid or infeasible dev fold")
    inner = list(GroupKFold(n_splits=inner_folds).split(
        np.zeros(len(outer_train)), groups=inner_groups
    ))
    inner_train, inner_dev = inner[dev_fold]
    train = outer_train[inner_train]
    dev = outer_train[inner_dev]

    partitions = {
        "train": sorted(int(item) for item in train),
        "dev": sorted(int(item) for item in dev),
        "test": sorted(int(item) for item in test),
    }
    group_sets = {
        name: {groups[index] for index in indices}
        for name, indices in partitions.items()
    }
    if any(not values for values in partitions.values()):
        raise ValueError(f"{split_type} produced an empty partition")
    if group_sets["train"] & group_sets["dev"] or \
            group_sets["train"] & group_sets["test"] or \
            group_sets["dev"] & group_sets["test"]:
        raise AssertionError("group leakage detected after nested GroupKFold")
    return SplitAssignment(
        split_type=split_type,
        seed=seed,
        group_field=group_field,
        train=partitions["train"],
        dev=partitions["dev"],
        test=partitions["test"],
        n_folds=effective_folds,
        outer_fold=outer_fold,
        dev_fold=dev_fold,
    )
