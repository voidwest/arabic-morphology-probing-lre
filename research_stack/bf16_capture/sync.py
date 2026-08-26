from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .bundle import _included_files, sha256_file, utc_now, validate_bundle, verify_checksums


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.copy-{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _prepare_destination(destination: Path) -> Path:
    if destination.exists():
        existing = validate_bundle(destination, require_complete=True, check_tensors=False)
        if existing["valid"]:
            return destination
        stamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.replace(destination, destination.with_name(f"{destination.name}.partial-{stamp}-{os.getpid()}"))
    staging = destination.with_name(f".{destination.name}.sync-{os.getpid()}")
    if staging.exists():
        os.replace(staging, staging.with_name(f"{staging.name}.partial"))
    staging.mkdir(parents=True)
    return staging


def sync_local_bundle(source: Path, destination: Path) -> dict[str, Any]:
    source_ok, failures = verify_checksums(source)
    if not source_ok:
        raise RuntimeError("refusing to sync invalid source bundle: " + "; ".join(failures[:10]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and validate_bundle(destination, require_complete=True, check_tensors=False)["valid"]:
        return {"kind": "local", "destination": str(destination), "already_verified": True, "verified_utc": utc_now()}
    staging = _prepare_destination(destination)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name == "COMPLETE":
            continue
        _copy_file(path, staging / path.relative_to(source))
    copied_ok, copied_failures = verify_checksums(staging)
    if not copied_ok:
        raise RuntimeError("destination checksum verification failed: " + "; ".join(copied_failures[:10]))
    os.replace(staging, destination)
    return {
        "kind": "local",
        "destination": str(destination),
        "source_checksums_sha256": sha256_file(source / "checksums.sha256"),
        "destination_checksums_sha256": sha256_file(destination / "checksums.sha256"),
        "verified_utc": utc_now(),
    }


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def _aws(*args: str) -> dict[str, Any]:
    result = subprocess.run(["aws", *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"aws {' '.join(args[:3])} failed: {result.stderr.strip()}")
    return json.loads(result.stdout) if result.stdout.strip().startswith("{") else {"stdout": result.stdout}


def _remote_sha256(bucket: str, key: str) -> str:
    value = _aws("s3api", "head-object", "--bucket", bucket, "--key", key, "--checksum-mode", "ENABLED")
    encoded = value.get("ChecksumSHA256")
    if not encoded:
        raise RuntimeError(f"S3 object has no stored SHA-256 checksum: s3://{bucket}/{key}")
    if "-" in encoded:
        raise RuntimeError(f"S3 object has a composite checksum, not a full-object checksum: s3://{bucket}/{key}")
    return base64.b64decode(encoded).hex()


def sync_s3_bundle(source: Path, destination: str) -> dict[str, Any]:
    source_ok, failures = verify_checksums(source)
    if not source_ok:
        raise RuntimeError("refusing to sync invalid source bundle: " + "; ".join(failures[:10]))
    bucket, prefix = _s3_parts(destination)
    files = sorted(path for path in source.rglob("*") if path.is_file() and path.name not in {"COMPLETE", "sync_receipt.json", "status.json"})
    # Per-layer safetensors keep objects below the single PUT limit.  A full
    # SHA-256 is requested and verified with HeadObject; ETag is never used as
    # scientific evidence.
    for path in files:
        key = "/".join(item for item in (prefix, path.relative_to(source).as_posix()) if item)
        _aws("s3api", "put-object", "--bucket", bucket, "--key", key, "--body", str(path), "--checksum-algorithm", "SHA256")
        remote = _remote_sha256(bucket, key)
        local = __import__("hashlib").sha256(path.read_bytes()).hexdigest() if path.stat().st_size < 128 * 1024 * 1024 else sha256_file(path)
        if remote != local:
            raise RuntimeError(f"S3 SHA-256 mismatch for {key}: {remote} != {local}")
    checksum_key = "/".join(item for item in (prefix, "checksums.sha256") if item)
    if _remote_sha256(bucket, checksum_key) != sha256_file(source / "checksums.sha256"):
        raise RuntimeError("S3 checksum-manifest verification failed")
    return {
        "kind": "s3",
        "destination": destination,
        "object_count": len(files),
        "checksum_manifest_sha256": sha256_file(source / "checksums.sha256"),
        "verified_utc": utc_now(),
    }


def finalize_marker(source: Path, destination: str) -> None:
    marker = source / "COMPLETE"
    if destination.startswith("s3://"):
        bucket, prefix = _s3_parts(destination)
        status = source / "status.json"
        status_key = "/".join(item for item in (prefix, "status.json") if item)
        _aws("s3api", "put-object", "--bucket", bucket, "--key", status_key, "--body", str(status), "--checksum-algorithm", "SHA256")
        if _remote_sha256(bucket, status_key) != sha256_file(status):
            raise RuntimeError("S3 status verification failed")
        key = "/".join(item for item in (prefix, "COMPLETE") if item)
        _aws("s3api", "put-object", "--bucket", bucket, "--key", key, "--body", str(marker), "--checksum-algorithm", "SHA256")
        if _remote_sha256(bucket, key) != sha256_file(marker):
            raise RuntimeError("S3 COMPLETE marker verification failed")
    else:
        target = Path(destination)
        _copy_file(source / "status.json", target / "status.json")
        _copy_file(marker, target / "COMPLETE")
        if sha256_file(target / "COMPLETE") != sha256_file(marker):
            raise RuntimeError("local COMPLETE marker verification failed")
