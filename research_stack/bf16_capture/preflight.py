from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


def load_matrix(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("matrix"), list):
        raise ValueError("invalid BF16 matrix manifest")
    return value


def _config_value(config: Any, key: str) -> Any:
    value = getattr(config, key, None)
    if value is not None:
        return value
    nested = getattr(config, "text_config", None)
    return getattr(nested, key, None) if nested is not None else None


def static_manifest_checks(manifest: dict[str, Any], selected_models: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    models = [model for model in manifest["matrix"] if selected_models is None or model.get("model_id") in selected_models]
    if selected_models is None and len(models) != 11:
        errors.append(f"expected 11 planned model records, found {len(models)}")
    ids = [model.get("model_id") for model in models]
    if len(set(ids)) != len(ids):
        errors.append("duplicate model_id")
    for model in models:
        for field in ("model_id", "hf_repo", "revision", "architecture", "tokenizer_repo", "tokenizer_revision", "config_revision", "requested_inference_dtype", "gated", "trust_remote_code", "expected_hidden_size", "expected_layer_count"):
            if field not in model:
                errors.append(f"{model.get('model_id')}: missing {field}")
        if model.get("requested_inference_dtype") != "bfloat16":
            errors.append(f"{model.get('model_id')}: requested dtype is not bfloat16")
        if model.get("expected_hidden_size") is None or model.get("expected_layer_count") is None:
            errors.append(f"{model.get('model_id')}: dimensions are not frozen")
        if model.get("native_checkpoint_dtype") == "unknown_until_gated_access":
            errors.append(f"{model.get('model_id')}: native checkpoint dtype is not frozen")
    conditions = manifest.get("prompt_conditions", {}).get("conditions", [])
    if {condition.get("id") for condition in conditions} != {"full_metadata", "metadata_free"}:
        errors.append("prompt conditions must be exactly full_metadata and metadata_free")
    if manifest.get("inference_contract", {}).get("allow_cpu_offload") is not False:
        errors.append("CPU offload must be disabled")
    if manifest.get("inference_contract", {}).get("allow_quantization") is not False:
        errors.append("quantization fallback must be disabled")
    allowed_loaders = set(manifest.get("inference_contract", {}).get("allowed_model_loaders", []))
    for model in models:
        if model.get("model_loader", "AutoModelForCausalLM") not in allowed_loaders:
            errors.append(f"{model.get('model_id')}: model loader is not explicitly allowed")
    return errors


def run_preflight(manifest: dict[str, Any], *, manifest_path: Path, run_root: Path, verify_hub: bool = True, selected_models: set[str] | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "status": "pass" if passed else "fail", "detail": detail})

    selected = [model for model in manifest["matrix"] if selected_models is None or model.get("model_id") in selected_models]
    unknown_selected = sorted((selected_models or set()) - {model["model_id"] for model in manifest["matrix"]})
    static_errors = static_manifest_checks(manifest, selected_models)
    static_errors.extend(f"unknown selected model: {model_id}" for model_id in unknown_selected)
    check("manifest-static", not static_errors, "; ".join(static_errors) if static_errors else f"{len(selected)} selected records")
    try:
        import torch
        check("torch-cuda", bool(torch.cuda.is_available()), f"available={torch.cuda.is_available()}; version={torch.__version__}; cuda={torch.version.cuda}")
        check("cuda-bfloat16", bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()), "BF16 capability query")
    except Exception as error:
        check("torch-cuda", False, f"{type(error).__name__}: {error}")
    packages = {}
    for package in ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "numpy", "scikit-learn", "pyarrow", "tiktoken", "sentencepiece"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    required_packages = {"torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "numpy", "scikit-learn", "tiktoken", "sentencepiece"}
    check("bf16-python-environment", all(packages[name] for name in required_packages), json.dumps(packages, sort_keys=True))
    usage = shutil.disk_usage(run_root if run_root.exists() else run_root.parent)
    check("disk-space", usage.free >= 100 * 1024**3, f"free_gib={usage.free / 1024**3:.2f}; required_gib=100")
    check("hf-token-for-gated", bool(os.environ.get("HF_TOKEN")), "token present (value not recorded)") if any(model.get("gated") for model in selected) else None

    hub_reports: list[dict[str, Any]] = []
    if verify_hub:
        try:
            from huggingface_hub import HfApi
            from transformers import AutoConfig, AutoTokenizer
            api = HfApi()
            for model in selected:
                report: dict[str, Any] = {"model_id": model["model_id"], "repo": model["hf_repo"], "revision": model["revision"]}
                try:
                    info = api.model_info(model["hf_repo"], revision=model["revision"], files_metadata=True)
                    report.update({"resolved_sha": info.sha, "gated": info.gated, "private": info.private})
                    if info.sha != model["revision"]:
                        report["error"] = f"resolved SHA {info.sha} differs from manifest"
                    config = AutoConfig.from_pretrained(model["hf_repo"], revision=model["config_revision"], trust_remote_code=bool(model["trust_remote_code"]))
                    architecture = (getattr(config, "architectures", None) or [None])[0]
                    hidden = _config_value(config, "hidden_size")
                    layers = _config_value(config, "num_hidden_layers")
                    native = _config_value(config, "dtype") or _config_value(config, "torch_dtype")
                    report.update({"architecture": architecture, "hidden_size": hidden, "layer_count": layers, "native_dtype": str(native), "hub_files": [{"name": getattr(file, "rfilename", None), "size": getattr(file, "size", None), "blob_id": getattr(file, "blob_id", None)} for file in info.siblings or []]})
                    if architecture != model["architecture"]:
                        report["error"] = f"architecture {architecture} differs from manifest {model['architecture']}"
                    if hidden != model.get("expected_hidden_size") or layers != model.get("expected_layer_count"):
                        report["error"] = f"dimensions {layers}x{hidden} differ from manifest"
                    if model.get("native_checkpoint_dtype") == "bfloat16" and str(native).lower() not in {"bfloat16", "torch.bfloat16"}:
                        report["error"] = f"native dtype {native} is not bfloat16"
                    tokenizer = AutoTokenizer.from_pretrained(model["tokenizer_repo"], revision=model["tokenizer_revision"], use_fast=True, trust_remote_code=bool(model["trust_remote_code"]))
                    report["tokenizer"] = {"class": tokenizer.__class__.__name__, "is_fast": bool(getattr(tokenizer, "is_fast", False)), "commit": getattr(tokenizer, "_commit_hash", None), "vocab_size": getattr(tokenizer, "vocab_size", None)}
                    if not tokenizer.is_fast:
                        report["error"] = "tokenizer is not fast"
                    if report["tokenizer"]["commit"] and report["tokenizer"]["commit"] != model["tokenizer_revision"]:
                        report["error"] = "tokenizer commit differs from manifest"
                except Exception as error:
                    report["error"] = f"{type(error).__name__}: {error}"
                report["status"] = "pass" if "error" not in report else "fail"
                hub_reports.append(report)
                check(f"hub:{model['model_id']}", report["status"] == "pass", json.dumps({k: v for k, v in report.items() if k != "hub_files"}, sort_keys=True))
        except Exception as error:
            check("huggingface-preflight", False, f"{type(error).__name__}: {error}")
    estimated_capture_bytes = 0
    for model in selected:
        if model.get("expected_hidden_size") and model.get("expected_layer_count"):
            # Four target tokens per example is only a sizing guard. The exact
            # total is written after tokenization in each bundle.
            estimated_capture_bytes += 2 * 4701 * int(model["expected_layer_count"]) * int(model["expected_hidden_size"]) * 2 * 5
    check("estimated-capture-space", usage.free >= estimated_capture_bytes * 1.25, f"estimated_gib={estimated_capture_bytes / 1024**3:.2f}; with_margin_gib={estimated_capture_bytes * 1.25 / 1024**3:.2f}")
    result = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_id": manifest.get("manifest_id"),
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
        "packages": packages,
        "python": sys.version,
        "platform": platform.platform(),
        "run_root": str(run_root),
        "estimated_capture_gib": round(estimated_capture_bytes / 1024**3, 3),
        "hub_reports": hub_reports,
        "fallback_policy": manifest["inference_contract"],
    }
    return result
