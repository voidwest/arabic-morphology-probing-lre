#!/usr/bin/env python3
"""Verify a BF16 analysis freeze and its external complete bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(path: Path, base: Path) -> None:
    completed = subprocess.run(
        ["sha256sum", "-c", path.name],
        cwd=base,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"checksum failure for {path}: {completed.stdout}{completed.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("freeze", type=Path)
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()
    freeze = args.freeze.resolve()
    check_manifest(freeze / "META_SHA256SUMS", freeze)
    check_manifest(freeze / "SHA256SUMS", freeze)
    external = json.loads((freeze / "BF16_EXTERNAL_ARTIFACTS.json").read_text(encoding="utf-8"))
    checked = 0
    if not args.skip_external:
        if str(freeze.parent.parent) not in sys.path:
            sys.path.insert(0, str(freeze.parent.parent))
        from research_stack.bf16_capture.bundle import validate_bundle

        for model_id, conditions in external["bundles"].items():
            for condition, record in conditions.items():
                bundle = Path(record["path"])
                validation = validate_bundle(bundle, require_complete=True, check_tensors=True)
                if not validation["valid"]:
                    raise RuntimeError(f"invalid external bundle {model_id}/{condition}: {validation['errors']}")
                for item in record["files"]:
                    path = Path(item["source_path"])
                    if not path.is_file() or sha256_file(path) != item["sha256"]:
                        raise RuntimeError(f"external hash failure: {path}")
                    checked += 1
    print(json.dumps({
        "freeze": str(freeze),
        "internal_checksums": "verified",
        "external_files_verified": checked,
        "status": "PASS",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
