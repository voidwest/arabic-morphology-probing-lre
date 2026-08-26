#!/usr/bin/env python3
"""Verify the recovered Paper 1 source, frozen dataset, and provenance claims."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/provenance/paper1_dataset_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(relative_path: str, expected: str) -> dict[str, object]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {relative_path}: expected {expected}, got {actual}")
    return {"path": relative_path, "bytes": path.stat().st_size, "sha256": actual}


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks = []
    upstream = manifest["upstream"]
    for name in ("train", "dev", "test"):
        item = upstream["files"][name]
        checks.append(check_file(item["path"], item["sha256"]))
    checks.append(check_file(upstream["license_path"], upstream["files"]["license"]["sha256"]))
    frozen = manifest["authoritative_frozen_dataset"]
    checks.append(check_file(frozen["path"], frozen["sha256"]))

    historical = manifest["historical_pipeline"]
    checks.append(
        check_file(
            "vendor/ember_dataset_pipeline_1a98982e/scripts/export_camel_disambiguated_padt.py",
            historical["exporter_sha256"],
        )
    )
    checks.append(
        check_file(
            "vendor/ember_dataset_pipeline_1a98982e/src/arabic_morph_dataset/normalize.py",
            historical["normalizer_sha256"],
        )
    )
    checks.append(
        check_file(
            "vendor/ember_dataset_pipeline_1a98982e/src/arabic_morph_dataset/split.py",
            historical["splitter_sha256"],
        )
    )
    checks.append(
        check_file(
            "vendor/ember_dataset_pipeline_1a98982e/configs/arabic_morph_disambig_padt_5000_strict.toml",
            historical["config_sha256"],
        )
    )

    dataset_path = ROOT / frozen["path"]
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 4701:
        raise RuntimeError(f"expected 4,701 rows, found {len(rows)}")
    expected_ids = [f"stim-{index:04d}" for index in range(1, 4702)]
    actual_ids = [row["id"] for row in rows]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("stimulus IDs are not the exact unique sequential frozen IDs")

    expected_counts = {
        "split": {"dev": 705, "test": 705, "train": 3291},
        "pos": {"ADJ": 910, "NOUN": 2957, "VERB": 834},
        "gender": {"": 2, "fem": 1757, "masc": 2942},
        "number": {"": 2, "def": 45, "pl": 700, "sg": 3954},
    }
    actual_counts = {
        field: dict(sorted(Counter(row[field] for row in rows).items()))
        for field in expected_counts
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"dataset count mismatch: {actual_counts}")

    for row in rows:
        if row["gender"] != row["features"].get("gender", ""):
            raise RuntimeError(f"gender adapter mismatch for {row['id']}")
        if row["number"] != row["features"].get("number", ""):
            raise RuntimeError(f"number adapter mismatch for {row['id']}")

    overlaps = {}
    for field in ("lemma", "root"):
        sets = {
            split: {row[field] for row in rows if row["split"] == split}
            for split in ("train", "dev", "test")
        }
        overlaps[field] = {
            "train_dev": len(sets["train"] & sets["dev"]),
            "train_test": len(sets["train"] & sets["test"]),
            "dev_test": len(sets["dev"] & sets["test"]),
        }
        if any(overlaps[field].values()):
            raise RuntimeError(f"nonzero {field} overlap: {overlaps[field]}")

    report = {
        "status": "verified",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "files": checks,
        "records": len(rows),
        "counts": actual_counts,
        "unique_lemmas": len({row["lemma"] for row in rows}),
        "unique_roots": len({row["root"] for row in rows}),
        "unique_surface_forms": len({row["surface_dediac"] for row in rows}),
        "overlaps": overlaps,
        "number_label_alias": {"def": "du", "affected_records": 45},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
