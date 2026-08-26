from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from .bundle import load_jsonl, validate_bundle, write_json
from .extractor import extract_one
from .local import validate_local_analysis
from .preflight import load_matrix, run_preflight
from .queue import init_queue, queue_status, run_queue


def _model(manifest: dict, model_id: str) -> dict:
    for value in manifest["matrix"]:
        if value["model_id"] == model_id:
            return value
    raise SystemExit(f"unknown model: {model_id}")


def _condition(manifest: dict, condition_id: str) -> dict:
    for value in manifest["prompt_conditions"]["conditions"]:
        if value["id"] == condition_id:
            return value
    raise SystemExit(f"unknown condition: {condition_id}")


def _unit_destination(root: str, model_id: str, condition_id: str) -> str:
    if root.startswith("s3://"):
        return root.rstrip("/") + f"/{model_id}/{condition_id}"
    return str(Path(root) / model_id / condition_id)


def _sidecar_output(bundle: Path, output: Path | None) -> Path | None:
    """Keep post-hoc reports outside immutable completed bundles."""
    if output is None:
        return None
    bundle = bundle.resolve()
    output = output.resolve()
    if bundle in output.parents:
        raise ValueError(f"post-hoc output must be outside the bundle: {output}")
    return output


def _dataset_path(manifest_path: Path, manifest: dict) -> Path:
    value = Path(manifest["dataset"]["path"])
    return value if value.is_absolute() else (manifest_path.parent.parent / value).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BF16 Hugging Face extraction infrastructure")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--manifest", type=Path, required=True)
    pre.add_argument("--run-root", type=Path, required=True)
    pre.add_argument("--output", type=Path)
    pre.add_argument("--no-hub", action="store_true")
    pre.add_argument("--model-id", action="append", help="pilot preflight for selected model(s); full matrix remains the default")
    qi = sub.add_parser("queue-init")
    qi.add_argument("--manifest", type=Path, required=True)
    qi.add_argument("--run-root", type=Path, required=True)
    qi.add_argument("--run-id", required=True)
    qr = sub.add_parser("queue-run")
    qr.add_argument("--manifest", type=Path, required=True)
    qr.add_argument("--run-root", type=Path, required=True)
    qr.add_argument("--sync-destination", required=True)
    qr.add_argument("--worker", default=os.uname().nodename)
    qr.add_argument("--device", default="cuda:0")
    qr.add_argument("--batch-size", type=int, default=4)
    qr.add_argument("--retry-failed", action="store_true")
    qr.add_argument("--recover-running", action="store_true", help="reclaim RUNNING items only after confirming the old worker is stopped")
    qr.add_argument("--once", action="store_true")
    qs = sub.add_parser("queue-status")
    qs.add_argument("--run-root", type=Path, required=True)
    qs.add_argument("--output", type=Path)
    ex = sub.add_parser("extract")
    ex.add_argument("--manifest", type=Path, required=True)
    ex.add_argument("--run-root", type=Path, required=True)
    ex.add_argument("--model", required=True)
    ex.add_argument("--condition", required=True)
    ex.add_argument("--sync-destination", required=True)
    ex.add_argument("--device", default="cuda:0")
    ex.add_argument("--batch-size", type=int, default=4)
    vb = sub.add_parser("validate-bundle")
    vb.add_argument("--bundle", type=Path, required=True)
    vb.add_argument("--allow-incomplete", action="store_true")
    vb.add_argument("--output", type=Path)
    la = sub.add_parser("local-validate")
    la.add_argument("--bundle", type=Path, required=True)
    la.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        manifest = load_matrix(args.manifest.resolve())
        result = run_preflight(manifest, manifest_path=args.manifest.resolve(), run_root=args.run_root.resolve(), verify_hub=not args.no_hub, selected_models=set(args.model_id) if args.model_id else None)
        output = args.output.resolve() if args.output else args.run_root.resolve() / "preflight.json"
        write_json(output, result)
        print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))
        return 0 if result["status"] == "pass" else 1
    if args.command == "queue-init":
        manifest = load_matrix(args.manifest.resolve())
        print(init_queue(manifest, manifest_path=args.manifest.resolve(), run_root=args.run_root.resolve(), run_id=args.run_id))
        return 0
    if args.command == "queue-run":
        manifest = load_matrix(args.manifest.resolve())
        rows = load_jsonl(_dataset_path(args.manifest.resolve(), manifest))
        os.environ["BF16_COMMAND"] = shlex.join(sys.argv)
        ok = run_queue(manifest=manifest, run_root=args.run_root.resolve(), dataset_rows=rows, worker=args.worker, device=args.device, batch_size=args.batch_size, sync_destination=args.sync_destination, retry_failed=args.retry_failed, once=args.once, recover_running=args.recover_running)
        return 0 if ok else 1
    if args.command == "queue-status":
        result = queue_status(args.run_root.resolve(), output=args.output.resolve() if args.output else None)
        print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))
        return 0
    if args.command == "extract":
        manifest = load_matrix(args.manifest.resolve())
        rows = load_jsonl(_dataset_path(args.manifest.resolve(), manifest))
        os.environ["BF16_COMMAND"] = shlex.join(sys.argv)
        bundle = args.run_root.resolve() / "bundles" / args.model / args.condition
        result = extract_one(manifest=manifest, model_spec=_model(manifest, args.model), condition=_condition(manifest, args.condition), rows=rows, bundle=bundle, run_id=args.run_root.name, device=args.device, batch_size=args.batch_size, sync_destination=_unit_destination(args.sync_destination, args.model, args.condition))
        print(result)
        return 0
    if args.command == "validate-bundle":
        result = validate_bundle(args.bundle.resolve(), require_complete=not args.allow_incomplete, check_tensors=True)
        if args.output:
            write_json(_sidecar_output(args.bundle, args.output), result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.command == "local-validate":
        result = validate_local_analysis(args.bundle.resolve(), output=_sidecar_output(args.bundle, args.output))
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(args.command)
