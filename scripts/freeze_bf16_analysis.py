#!/usr/bin/env python3
"""Create an immutable freeze for a completed BF16 local-analysis revision.

Large BF16 bundle tensors remain external to the compact freeze, but every
bundle file is rehashed and recorded here.  The freeze refuses to overwrite
anything and refuses to freeze an incomplete or invalid analysis root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "runs/bf16-analysis/bf16-analysis-20260826-v1"
DEFAULT_FREEZE_ID = "bf16-analysis-20260826-v1"
MATRIX = ROOT / "configs/bf16_model_matrix_20260826_jais_resolved.json"
DATASET = ROOT / "output/data/paper1_normalized.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_verified(source: Path, destination: Path, *, record_path: Path | None = None) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(source)
    shutil.copy2(source, destination)
    copied = sha256_file(destination)
    if copied != digest:
        raise RuntimeError(f"copy verification failed: {source}")
    return {"path": str(record_path or destination), "source": str(source.resolve()), "bytes": source.stat().st_size, "sha256": digest}


def find_bundle(bundle_root: Path, model_id: str, condition: str) -> Path:
    candidates = sorted(bundle_root.glob(f"*/bundles/{model_id}/{condition}"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one bundle for {model_id}/{condition}, found {candidates}")
    return candidates[0]


def bundle_record(bundle: Path) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from research_stack.bf16_capture.bundle import validate_bundle

    validation = validate_bundle(bundle, require_complete=True, check_tensors=True)
    if not validation["valid"]:
        raise RuntimeError(f"cannot freeze invalid bundle {bundle}: {validation['errors']}")
    files = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.name in {"status.json", "sync_receipt.json"}:
            continue
        files.append({
            "path_relative_to_bundle": str(path.relative_to(bundle)),
            "source_path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "path": str(bundle.resolve()),
        "validation": validation,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"freeze contains a symlink: {path}")
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=ROOT / "runs/bf16")
    parser.add_argument("--freeze-id", default=DEFAULT_FREEZE_ID)
    parser.add_argument("--output-root", type=Path, default=ROOT / "freezes")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    destination = output_root / args.freeze_id
    staging = output_root / f".{args.freeze_id}.staging-{os.getpid()}"
    if destination.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite freeze or staging path: {destination}")
    analysis_manifest = run_root / "analysis_manifest.json"
    required_outputs = [
        analysis_manifest,
        run_root / "analysis_config.json",
        run_root / "split_manifest.json",
        run_root / "SHA256SUMS",
        run_root / "results/summary.csv",
        run_root / "results/summary.md",
        run_root / "results/summary.tex",
        run_root / "results/macro_f1_split_deltas.csv",
        run_root / "results/macro_f1_split_deltas.md",
        run_root / "results/macro_f1_split_deltas.tex",
        run_root / "results/layer_curves.csv",
        run_root / "results/shuffled_controls.csv",
        run_root / "results/fragmentation.csv",
        run_root / "results/class_pos_error_analysis.csv",
        run_root / "results/local_analysis_sufficiency.json",
        run_root / "results/cloud_cost_records.json",
        run_root / "results/bf16_q8_comparison.json",
        run_root / "results/cka_rsa_diagnostics.json",
        run_root / "results/analysis_decision.md",
    ]
    for path in required_outputs:
        if not path.is_file():
            raise FileNotFoundError(f"analysis is not finalizable; missing {path}")
    analysis = json.loads(analysis_manifest.read_text(encoding="utf-8"))
    if analysis.get("status") != "COMPLETE_WEIGHT_FREE_BF16_ANALYSIS":
        raise RuntimeError(f"analysis manifest is not complete: {analysis.get('status')}")
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    model_ids = [item["model_id"] for item in matrix["matrix"]]
    external: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "Complete hashes and validation records for external BF16 bundle tensors",
        "bundles": {},
    }
    for model_id in model_ids:
        external["bundles"][model_id] = {}
        for condition in ("full_metadata", "metadata_free"):
            external["bundles"][model_id][condition] = bundle_record(find_bundle(bundle_root, model_id, condition))

    output_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    snapshot = staging / "snapshot"
    snapshot.mkdir()
    inventory = []
    sources = [(MATRIX, Path("configs") / MATRIX.name), (DATASET, Path("dataset") / DATASET.name),
               (ROOT / "scripts/run_bf16_analysis.py", Path("scripts/run_bf16_analysis.py")),
               (ROOT / "scripts/freeze_bf16_analysis.py", Path("scripts/freeze_bf16_analysis.py")),
               (ROOT / "docs/bf16_analysis_runbook.md", Path("docs/bf16_analysis_runbook.md"))]
    for relative in (Path("research_stack/bf16_capture/bundle.py"), Path("research_stack/bf16_capture/local.py"), Path("research_stack/revision/splits.py"), Path("research_stack/revision/probes.py")):
        sources.append((ROOT / relative, relative))
    for source, relative in sources:
        inventory.append(copy_verified(source, snapshot / relative, record_path=relative))
    for source in sorted(run_root.rglob("*")):
        if source.is_file() and not source.is_symlink():
            relative = Path("analysis") / source.relative_to(run_root)
            inventory.append(copy_verified(source, snapshot / relative, record_path=relative))
    write_json(staging / "BF16_EXTERNAL_ARTIFACTS.json", external)
    freeze_manifest = {
        "schema_version": 1,
        "freeze_id": args.freeze_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "IMMUTABLE_COMPLETE_WEIGHT_FREE_BF16_ANALYSIS",
        "analysis_run_root": str(run_root),
        "matrix": {"path": str(MATRIX), "sha256": sha256_file(MATRIX)},
        "dataset": {"path": str(DATASET), "sha256": sha256_file(DATASET), "examples": 4701},
        "analysis_manifest_sha256": sha256_file(analysis_manifest),
        "external_bundle_count": len(model_ids) * 2,
        "external_bundle_bytes": sum(value["total_bytes"] for model in external["bundles"].values() for value in model.values()),
        "external_artifacts": "BF16 bundle files are not duplicated in the compact snapshot; every file is rehashed in BF16_EXTERNAL_ARTIFACTS.json.",
        "inventory": inventory,
        "snapshot_file_count": len(inventory),
        "snapshot_total_bytes": sum(item["bytes"] for item in inventory),
        "weight_free_after_capture": True,
        "immutability": "Never edit or overwrite this freeze. Future analysis or presentation changes require a new revision and freeze.",
    }
    write_json(staging / "FREEZE_MANIFEST.json", freeze_manifest)
    (staging / "README.md").write_text(
        f"# {args.freeze_id}\n\n"
        "Status: FROZEN — DO NOT MODIFY.\n\n"
        "This freeze records the BF16 hidden-state bundles, weight-free local-analysis outputs, "
        "analysis code and split/config manifests, plus complete external hashes for every bundle file.\n\n"
        "Verify with `sha256sum -c META_SHA256SUMS` and `sha256sum -c SHA256SUMS`.\n\n"
        "Historical Q8 and lre-corrected-analysis freezes are untouched. Raw Q8 vector arrays remain "
        "an explicit blocker for direct BF16↔Q8 vector comparison when absent from the live checkout.\n",
        encoding="utf-8",
    )
    (staging / "IMMUTABLE").write_text(f"{args.freeze_id}\nDO NOT MODIFY; CREATE A NEW REVISION.\n", encoding="utf-8")
    snapshot_sums = "".join(f"{item['sha256']}  snapshot/{item['path']}\n" for item in inventory)
    (staging / "SHA256SUMS").write_text(snapshot_sums, encoding="utf-8")
    metadata_names = ["BF16_EXTERNAL_ARTIFACTS.json", "FREEZE_MANIFEST.json", "IMMUTABLE", "README.md", "SHA256SUMS"]
    metadata_sums = "".join(f"{sha256_file(staging / name)}  {name}\n" for name in metadata_names)
    (staging / "META_SHA256SUMS").write_text(metadata_sums, encoding="utf-8")
    staging.rename(destination)
    make_read_only(destination)
    print(json.dumps({
        "freeze_id": args.freeze_id,
        "path": str(destination),
        "snapshot_files": len(inventory),
        "snapshot_bytes": sum(item["bytes"] for item in inventory),
        "external_bundle_count": len(model_ids) * 2,
        "external_bundle_bytes": freeze_manifest["external_bundle_bytes"],
        "status": "FROZEN",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
