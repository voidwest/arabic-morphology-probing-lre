from __future__ import annotations

import csv
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .bundle import load_json, sha256_file, validate_bundle, write_json
from .extractor import InterruptionRequested, extract_one
from .preflight import load_matrix


@contextmanager
def _queue_lock(path: Path):
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _queue_path(run_root: Path) -> Path:
    return run_root / "queue.json"


def init_queue(manifest: dict[str, Any], *, manifest_path: Path, run_root: Path, run_id: str) -> Path:
    path = _queue_path(run_root)
    if path.exists():
        raise FileExistsError(f"queue already exists: {path}")
    run_root.mkdir(parents=True, exist_ok=True)
    items = []
    conditions = manifest["prompt_conditions"]["conditions"]
    for model in manifest["matrix"]:
        for condition in conditions:
            items.append({
                "model_id": model["model_id"],
                "condition_id": condition["id"],
                "resources": {
                    "minimum_instance": model.get("minimum_instance"),
                    "recommended_instance": model.get("recommended_instance"),
                    "expected_vram_gib": model.get("expected_vram_gib"),
                    "expected_approx_bf16_weight_bytes": model.get("expected_approx_bf16_weight_bytes"),
                },
                "state": "PENDING", "attempts": 0, "worker": None,
                "started_utc": None, "ended_utc": None, "error": None,
            })
    write_json(path, {"schema_version": 1, "run_id": run_id, "manifest": str(manifest_path), "status": "PENDING", "items": items})
    return path


def _load_manifest_record(manifest: dict[str, Any], model_id: str) -> dict[str, Any]:
    for model in manifest["matrix"]:
        if model["model_id"] == model_id:
            return model
    raise KeyError(model_id)


def _condition(manifest: dict[str, Any], condition_id: str) -> dict[str, Any]:
    for condition in manifest["prompt_conditions"]["conditions"]:
        if condition["id"] == condition_id:
            return condition
    raise KeyError(condition_id)


def _unit_destination(root: str, model_id: str, condition_id: str) -> str:
    if root.startswith("s3://"):
        return root.rstrip("/") + f"/{model_id}/{condition_id}"
    return str(Path(root) / model_id / condition_id)


def _claim(path: Path, worker: str, retry_failed: bool) -> dict[str, Any] | None:
    with _queue_lock(path):
        queue = load_json(path)
        for item in queue["items"]:
            if item["state"] == "RUNNING":
                # A dead worker cannot leave an atomic bundle complete. It is
                # safe to reclaim only after explicit retry/worker restart.
                continue
            allowed = item["state"] == "PENDING" or (retry_failed and item["state"] in {"FAILED", "INTERRUPTED"})
            if not allowed:
                continue
            item.update({"state": "RUNNING", "worker": worker, "attempts": int(item["attempts"]) + 1, "started_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "error": None})
            write_json(path, queue)
            return dict(item)
    return None


def _finish(path: Path, item: dict[str, Any], state: str, error: str | None = None) -> None:
    with _queue_lock(path):
        queue = load_json(path)
        for value in queue["items"]:
            if value["model_id"] == item["model_id"] and value["condition_id"] == item["condition_id"]:
                value.update({"state": state, "ended_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "error": error})
                break
        states = {value["state"] for value in queue["items"]}
        queue["status"] = "COMPLETE" if states == {"COMPLETE"} else ("RUNNING" if "RUNNING" in states else "PARTIAL")
        write_json(path, queue)


def _reconcile_completed(path: Path, run_root: Path) -> None:
    with _queue_lock(path):
        queue = load_json(path)
        changed = False
        for item in queue["items"]:
            bundle = run_root / "bundles" / item["model_id"] / item["condition_id"]
            if validate_bundle(bundle, require_complete=True, check_tensors=False)["valid"] and item["state"] != "COMPLETE":
                item.update({"state": "COMPLETE", "error": None, "ended_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})
                changed = True
        if changed:
            queue["status"] = "COMPLETE" if all(item["state"] == "COMPLETE" for item in queue["items"]) else "PARTIAL"
            write_json(path, queue)


def run_queue(*, manifest: dict[str, Any], run_root: Path, dataset_rows: list[dict[str, Any]], worker: str, device: str, batch_size: int, sync_destination: str, retry_failed: bool = False, once: bool = False, recover_running: bool = False) -> bool:
    path = _queue_path(run_root)
    if not path.is_file():
        raise FileNotFoundError(f"initialize the queue first: {path}")
    _reconcile_completed(path, run_root)
    if recover_running:
        with _queue_lock(path):
            queue = load_json(path)
            for item in queue["items"]:
                if item["state"] == "RUNNING":
                    item.update({"state": "PENDING", "worker": None, "error": "reclaimed after explicit worker restart"})
            write_json(path, queue)
    while True:
        item = _claim(path, worker, retry_failed)
        if item is None:
            status = queue_status(run_root)
            return not any(state in status["counts"] for state in ("FAILED", "INTERRUPTED"))
        model = _load_manifest_record(manifest, item["model_id"])
        condition = _condition(manifest, item["condition_id"])
        bundle = run_root / "bundles" / item["model_id"] / item["condition_id"]
        try:
            extract_one(manifest=manifest, model_spec=model, condition=condition, rows=dataset_rows, bundle=bundle, run_id=load_json(path)["run_id"], device=device, batch_size=batch_size, sync_destination=_unit_destination(sync_destination, item["model_id"], item["condition_id"]))
            if not validate_bundle(bundle, require_complete=True, check_tensors=False)["valid"]:
                raise RuntimeError("extractor returned without a verified COMPLETE bundle")
            _finish(path, item, "COMPLETE")
        except InterruptionRequested as error:
            _finish(path, item, "INTERRUPTED", str(error))
            if once:
                return False
        except Exception as error:
            _finish(path, item, "FAILED", f"{type(error).__name__}: {error}")
            if once:
                return False
        if once:
            return True


def queue_status(run_root: Path, *, output: Path | None = None) -> dict[str, Any]:
    queue = load_json(_queue_path(run_root))
    counts: dict[str, int] = {}
    for item in queue["items"]:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    result = {"run_id": queue["run_id"], "status": queue["status"], "counts": counts, "items": queue["items"]}
    if output:
        write_json(output, result)
    return result
