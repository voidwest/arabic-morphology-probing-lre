from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import BUNDLE_SCHEMA


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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _included_files(bundle: Path) -> list[Path]:
    # Lifecycle state and the post-upload receipt are deliberately excluded:
    # both are mutable across the VALIDATING -> SYNCING -> COMPLETE transition.
    # All scientific metadata, tensors, logs, and validation output remain
    # covered by checksums.sha256.
    excluded = {"checksums.sha256", "COMPLETE", "status.json", "sync_receipt.json"}
    return sorted(
        path for path in bundle.rglob("*")
        if path.is_file() and path.relative_to(bundle).as_posix() not in excluded
    )


def write_checksums(bundle: Path) -> str:
    entries = [f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}" for path in _included_files(bundle)]
    text = "\n".join(entries) + "\n"
    temporary = bundle / ".checksums.sha256.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, bundle / "checksums.sha256")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_checksums(bundle: Path) -> tuple[bool, list[str]]:
    manifest = bundle / "checksums.sha256"
    if not manifest.is_file():
        return False, ["missing checksums.sha256"]
    failures: list[str] = []
    listed: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"checksums.sha256:{line_number}: malformed line")
            continue
        target = (bundle / relative).resolve()
        if bundle.resolve() not in target.parents or not target.is_file():
            failures.append(f"missing:{relative}")
        elif sha256_file(target) != expected:
            failures.append(f"changed:{relative}")
        listed.add(Path(relative).as_posix())
    actual = {path.relative_to(bundle).as_posix() for path in _included_files(bundle)}
    failures.extend(f"unlisted:{relative}" for relative in sorted(actual - listed))
    return not failures, failures


def write_complete_marker(bundle: Path, *, destination: str, checksum_manifest_sha256: str) -> None:
    marker = {
        "schema_version": 1,
        "state": "COMPLETE",
        "completed_utc": utc_now(),
        "bundle_checksums_sha256": checksum_manifest_sha256,
        "destination_verified": destination,
    }
    (bundle / "COMPLETE").write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")


def _finite_tensor(tensor: Any) -> bool:
    import torch
    return bool(torch.isfinite(tensor.float()).all().item())


def validate_bundle(bundle: Path, *, require_complete: bool = False, check_tensors: bool = True) -> dict[str, Any]:
    """Validate a bundle without trusting its completion marker."""
    errors: list[str] = []
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        return {"valid": False, "errors": ["missing manifest.json"], "bundle": str(bundle)}
    try:
        manifest = load_json(manifest_path)
    except Exception as error:
        return {"valid": False, "errors": [f"invalid manifest: {error}"], "bundle": str(bundle)}
    if manifest.get("bundle_schema") != BUNDLE_SCHEMA:
        errors.append("wrong bundle schema")
    checks_ok, check_failures = verify_checksums(bundle)
    if not checks_ok:
        errors.extend(check_failures)
    marker = bundle / "COMPLETE"
    if require_complete and not marker.is_file():
        errors.append("missing COMPLETE marker")
    if marker.is_file():
        try:
            marker_value = load_json(marker)
            if marker_value.get("state") != "COMPLETE":
                errors.append("COMPLETE marker does not declare COMPLETE state")
            recorded_checksums = marker_value.get("bundle_checksums_sha256")
            current_checksums = sha256_file(bundle / "checksums.sha256") if (bundle / "checksums.sha256").is_file() else None
            if not recorded_checksums or recorded_checksums != current_checksums:
                errors.append("COMPLETE marker is not bound to the current checksum manifest")
            if not marker_value.get("destination_verified"):
                errors.append("COMPLETE marker lacks a verified destination")
        except Exception as error:
            errors.append(f"invalid COMPLETE marker: {error}")
    try:
        examples = load_jsonl(bundle / "examples.jsonl")
        alignments = load_jsonl(bundle / "alignment.jsonl")
    except Exception as error:
        examples, alignments = [], []
        errors.append(f"metadata load failed: {error}")
    expected = int(manifest.get("expected_examples", -1))
    if len(examples) != expected:
        errors.append(f"expected {expected} examples, found {len(examples)}")
    if len(alignments) != expected:
        errors.append(f"expected {expected} alignments, found {len(alignments)}")
    ids = [str(row.get("example_id")) for row in examples]
    if len(set(ids)) != len(ids):
        errors.append("duplicate example IDs")
    alignment_ids = [str(row.get("example_id")) for row in alignments]
    if ids != alignment_ids:
        errors.append("alignment IDs/order do not match examples")
    if any(row.get("status") != "aligned" for row in alignments):
        errors.append("one or more alignments are not aligned")
    for index, row in enumerate(examples):
        required = {"example_id", "input_ids", "sequence_length", "target_token_indices", "target_token_count", "prompt", "target_byte_span"}
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"example {index} missing fields: {missing}")
        elif len(row["input_ids"]) != int(row["sequence_length"]):
            errors.append(f"example {index} sequence_length mismatch")
    representation_dir = bundle / "representations"
    layer_count = int(manifest.get("expected_layer_count", -1))
    hidden_size = int(manifest.get("expected_hidden_size", -1))
    expected_files = {"embedding_output.safetensors"} | {f"layer_{layer:04d}.safetensors" for layer in range(layer_count)}
    actual_files = {path.name for path in representation_dir.glob("*.safetensors")}
    if actual_files != expected_files:
        errors.append(f"representation files mismatch: expected {len(expected_files)}, found {len(actual_files)}")
    total_target = sum(int(row.get("target_token_count", 0)) for row in examples)
    tensor_shapes: dict[str, Any] = {}
    if check_tensors and not errors:
        try:
            from safetensors import safe_open
            for filename in sorted(expected_files):
                path = representation_dir / filename
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    keys = sorted(handle.keys())
                    if filename == "embedding_output.safetensors":
                        expected_shapes = {"embedding_output": [expected, hidden_size]}
                    else:
                        expected_shapes = {"prompt_final": [expected, hidden_size], "target_span": [total_target, hidden_size]}
                    if set(keys) != set(expected_shapes):
                        errors.append(f"{filename}: tensor keys {keys} != {sorted(expected_shapes)}")
                    for key, shape in expected_shapes.items():
                        if key in keys:
                            actual_shape = list(handle.get_slice(key).get_shape())
                            tensor_shapes[f"{filename}:{key}"] = actual_shape
                            if actual_shape != shape:
                                errors.append(f"{filename}:{key}: shape {actual_shape} != {shape}")
                            tensor = handle.get_tensor(key)
                            if not _finite_tensor(tensor):
                                errors.append(f"{filename}:{key}: NaN or Inf")
        except Exception as error:
            errors.append(f"tensor validation failed: {type(error).__name__}: {error}")
    return {
        "valid": not errors,
        "errors": errors,
        "bundle": str(bundle),
        "complete_marker": marker.is_file(),
        "expected_examples": expected,
        "successful_alignments": sum(row.get("status") == "aligned" for row in alignments),
        "expected_alignments": expected,
        "total_target_tokens": total_target,
        "tensor_shapes": tensor_shapes,
        "checksums_verified": checks_ok,
    }


def quarantine(path: Path) -> Path:
    """Move an incomplete unit aside; never overwrite a previous attempt."""
    if not path.exists():
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.partial-{stamp}-{os.getpid()}")
    os.replace(path, target)
    return target
