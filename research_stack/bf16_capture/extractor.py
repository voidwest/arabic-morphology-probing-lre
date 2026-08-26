from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import sys
import threading
import time
import types
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .alignment import align_all, prepare_examples
from .bundle import (
    BUNDLE_SCHEMA,
    load_json,
    quarantine,
    sha256_file,
    utc_now,
    validate_bundle,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from .telemetry import Telemetry


class ExtractionError(RuntimeError):
    pass


class InterruptionRequested(ExtractionError):
    pass


class StopController:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.reason: str | None = None
        self._old_handlers: dict[int, Any] = {}

    def request(self, reason: str) -> None:
        self.reason = reason
        self.event.set()

    def check(self) -> None:
        if self.event.is_set():
            raise InterruptionRequested(self.reason or "interruption requested")

    def install_signals(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signum, _frame: self.request("process signal"))

    def restore_signals(self) -> None:
        for signum, handler in self._old_handlers.items():
            signal.signal(signum, handler)


class SpotPoller:
    """Best-effort IMDSv2 interruption detector; absence of IMDS is normal locally."""
    def __init__(self, controller: StopController, interval: float = 5.0):
        self.controller = controller
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="spot-interruption", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        token: str | None = None
        while not self._stop.wait(self.interval):
            try:
                if token is None:
                    request = urllib.request.Request(
                        "http://169.254.169.254/latest/api/token", method="PUT",
                        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                    )
                    with urllib.request.urlopen(request, timeout=1) as response:
                        token = response.read().decode("ascii")
                request = urllib.request.Request(
                    "http://169.254.169.254/latest/meta-data/spot/instance-action",
                    headers={"X-aws-ec2-metadata-token": token},
                )
                with urllib.request.urlopen(request, timeout=1) as response:
                    value = response.read().decode("utf-8", errors="replace")
                if value.strip():
                    self.controller.request(f"EC2 Spot interruption notice: {value[:300]}")
                    return
            except Exception:
                token = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def _code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return {"kind": "content_hash", "sha256": digest.hexdigest(), "files": [path.name for path in files], "git_commit": None, "git_state": "workspace is not a Git worktree"}


def _nested_config_value(config: Any, key: str) -> Any:
    value = getattr(config, key, None)
    if value is not None:
        return value
    text = getattr(config, "text_config", None)
    return getattr(text, key, None) if text is not None else None


def _first_tensor(value: Any, torch: Any) -> Any | None:
    if torch.is_tensor(value):
        return value if value.ndim >= 3 else None
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item, torch)
            if found is not None:
                return found
    return None


def _find_text_layers(model: Any, expected: int, torch: Any) -> list[Any]:
    import torch.nn as nn
    candidates: list[tuple[int, int, str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) != expected:
            continue
        score = 0
        lowered = name.lower()
        for token, points in (("language", 20), ("text", 20), ("decoder", 10), ("model", 5), ("layers", 2)):
            if token in lowered:
                score += points
        if "vision" in lowered or "audio" in lowered:
            score -= 50
        candidates.append((score, -len(name), name, module))
    if not candidates:
        raise ExtractionError(f"could not find a text decoder ModuleList with {expected} layers")
    candidates.sort(reverse=True, key=lambda value: (value[0], value[1], value[2]))
    return list(candidates[0][3])


def _loader_class(manifest_model: dict[str, Any]) -> Any:
    import transformers
    name = manifest_model.get("model_loader", "AutoModelForCausalLM")
    try:
        return getattr(transformers, name)
    except AttributeError as error:
        raise ExtractionError(f"unsupported explicit model loader {name}") from error


def _install_legacy_jais_transformers_compat(model_spec: dict[str, Any], torch: Any, config: Any | None = None) -> None:
    """Provide import-only pruning helpers removed from modern Transformers."""
    if model_spec.get("architecture") != "JAISLMHeadModel":
        return
    import torch.nn as nn
    from transformers import pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads: Any, n_heads: int, head_size: int, already_pruned_heads: Any) -> tuple[Any, Any]:
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - set(already_pruned_heads)
            for head in heads:
                head -= sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            index = torch.arange(len(mask.view(-1)))[mask.view(-1).eq(1)].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(pytorch_utils, "prune_conv1d_layer"):
        def prune_conv1d_layer(layer: Any, index: Any, dim: int = 1) -> Any:
            index = index.to(layer.weight.device)
            weight = layer.weight.index_select(dim, index).clone().detach()
            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new_layer = pytorch_utils.Conv1D(new_size[1], new_size[0])
            new_layer.weight = nn.Parameter(weight)
            new_layer.bias = nn.Parameter(layer.bias.clone().detach())
            return new_layer.to(layer.weight.device).to(layer.weight.dtype)

        pytorch_utils.prune_conv1d_layer = prune_conv1d_layer

    module_name = "transformers.utils.model_parallel_utils"
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)

        def get_device_map(num_blocks: int, devices: Any) -> dict[Any, list[int]]:
            devices = list(devices)
            if not devices:
                raise ValueError("at least one device is required")
            counts = [num_blocks // len(devices)] * len(devices)
            for index in range(num_blocks % len(devices)):
                counts[index] += 1
            result: dict[Any, list[int]] = {}
            cursor = 0
            for device, count in zip(devices, counts):
                result[device] = list(range(cursor, cursor + count))
                cursor += count
            return result

        def assert_device_map(device_map: Any, num_blocks: int) -> None:
            if device_map is None:
                return
            blocks = sorted(block for block_ids in device_map.values() for block in block_ids)
            if blocks != list(range(num_blocks)):
                raise ValueError(f"device map must cover blocks 0..{num_blocks - 1}")

        module.get_device_map = get_device_map
        module.assert_device_map = assert_device_map
        sys.modules[module_name] = module

    if config is not None:
        legacy_defaults = {
            "add_cross_attention": False,
            "classifier_dropout": None,
            "hidden_dropout": 0.0,
            "num_labels": 2,
            "output_attentions": False,
            "output_hidden_states": False,
            "problem_type": None,
            "use_return_dict": True,
        }
        for name, value in legacy_defaults.items():
            if not hasattr(config, name):
                setattr(config, name, value)


def _resolve_hf_snapshot(repo: str, revision: str, *, materialize: bool = False) -> tuple[str, float, bool]:
    """Resolve one immutable Hub snapshot and report network/cache time.

    A local-only attempt makes cache hits explicit. A cache miss is then
    downloaded at the pinned revision before Transformers is allowed to load
    anything, so model-download and model-load timing are not conflated.
    """
    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("HF_HOME")
    if materialize:
        # Transformers' dynamic-module loader follows relative imports from
        # the resolved blob path.  HF's symlinked cache can therefore turn a
        # sibling import such as configuration_jais.py into
        # <cache>/blobs/configuration_jais.py, which does not exist.  Keep the
        # pinned revision and cache the remote-code snapshot as real files.
        root = Path(cache_dir or (Path.home() / ".cache" / "huggingface"))
        local_dir = root / "materialized" / repo.replace("/", "--") / revision
        if (local_dir / "config.json").is_file():
            return str(local_dir), 0.0, True
        started = time.monotonic()
        path = snapshot_download(
            repo_id=repo,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_files_only=False,
            token=os.environ.get("HF_TOKEN"),
        )
        return str(path), time.monotonic() - started, False
    try:
        path = snapshot_download(repo_id=repo, revision=revision, cache_dir=cache_dir, local_files_only=True)
        return path, 0.0, True
    except Exception:
        started = time.monotonic()
        path = snapshot_download(
            repo_id=repo, revision=revision, cache_dir=cache_dir,
            local_files_only=False, token=os.environ.get("HF_TOKEN"),
        )
        return path, time.monotonic() - started, False


def load_hf_model_and_tokenizer(
    model_spec: dict[str, Any], *, device: str,
    model_snapshot: str | None = None, config_snapshot: str | None = None,
    tokenizer_snapshot: str | None = None,
) -> tuple[Any, Any, dict[str, Any], list[Any], Any]:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise ExtractionError("BF16 extraction requires an available CUDA device; CPU execution is not an allowed fallback")
    if not torch.cuda.is_bf16_supported():
        raise ExtractionError("CUDA device does not report BF16 support")
    revision = model_spec["revision"]
    trust = bool(model_spec.get("trust_remote_code", False))
    repo = model_spec["hf_repo"]
    config_source = config_snapshot or repo
    config_kwargs = {"trust_remote_code": trust}
    if config_snapshot is None:
        config_kwargs["revision"] = model_spec["config_revision"]
    else:
        config_kwargs["local_files_only"] = True
    config = AutoConfig.from_pretrained(config_source, **config_kwargs)
    architecture = (getattr(config, "architectures", None) or [None])[0]
    if architecture != model_spec["architecture"]:
        raise ExtractionError(f"architecture mismatch: resolved {architecture!r}, manifest {model_spec['architecture']!r}")
    hidden = _nested_config_value(config, "hidden_size")
    layers = _nested_config_value(config, "num_hidden_layers")
    if hidden != model_spec.get("expected_hidden_size") or layers != model_spec.get("expected_layer_count"):
        raise ExtractionError(f"dimension mismatch: resolved layers={layers}, hidden={hidden}; manifest expects {model_spec.get('expected_layer_count')} x {model_spec.get('expected_hidden_size')}")
    tokenizer_source = tokenizer_snapshot or model_spec["tokenizer_repo"]
    tokenizer_kwargs = {"use_fast": True, "trust_remote_code": trust}
    if tokenizer_snapshot is None:
        tokenizer_kwargs["revision"] = model_spec["tokenizer_revision"]
    else:
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if not getattr(tokenizer, "is_fast", False):
        raise ExtractionError("a fast tokenizer is required for deterministic offset alignment")
    tokenizer_commit = getattr(tokenizer, "_commit_hash", None)
    if tokenizer_commit and tokenizer_commit != model_spec["tokenizer_revision"]:
        raise ExtractionError(f"tokenizer commit mismatch: {tokenizer_commit} != {model_spec['tokenizer_revision']}")
    loader = _loader_class(model_spec)
    _install_legacy_jais_transformers_compat(model_spec, torch, config)
    model_kwargs = {"config": config, "trust_remote_code": trust, "dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    model_source = model_snapshot or repo
    if model_snapshot is None:
        model_kwargs["revision"] = revision
    else:
        model_kwargs["local_files_only"] = True
    model = loader.from_pretrained(model_source, **model_kwargs)
    if architecture == "JAISLMHeadModel":
        transformer = getattr(model, "transformer", None)
        if transformer is not None and not hasattr(transformer, "get_head_mask"):
            def get_head_mask(self: Any, head_mask: Any, num_hidden_layers: int, is_attention_chunked: bool = False) -> Any:
                if head_mask is not None:
                    if head_mask.dim() == 1:
                        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                    elif head_mask.dim() == 2:
                        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    if is_attention_chunked:
                        head_mask = head_mask.unsqueeze(-1)
                    head_mask = head_mask.to(dtype=self.dtype)
                else:
                    head_mask = [None] * num_hidden_layers
                return head_mask

            transformer.get_head_mask = types.MethodType(get_head_mask, transformer)
    model = model.to(device)
    model.eval()
    if hasattr(model, "hf_device_map") and set(model.hf_device_map.values()) - {device, 0, "cuda:0"}:
        raise ExtractionError(f"device map indicates offload or multi-device placement: {model.hf_device_map}")
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    parameter_dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if parameter_devices != {device}:
        raise ExtractionError(f"model is not entirely on {device}: {sorted(parameter_devices)}")
    if parameter_dtypes != {"torch.bfloat16"}:
        raise ExtractionError(f"model parameters are not exclusively BF16: {sorted(parameter_dtypes)}")
    layers_list = _find_text_layers(model, int(layers), torch)
    embeddings = model.get_input_embeddings()
    resolved = {
        "architecture": architecture,
        "hidden_size": int(hidden),
        "layer_count": int(layers),
        "model_config_commit": getattr(config, "_commit_hash", None),
        "tokenizer_commit": tokenizer_commit,
        "parameter_devices": sorted(parameter_devices),
        "parameter_dtypes": sorted(parameter_dtypes),
        "model_loader": loader.__name__,
        "model_class": model.__class__.__name__,
        "model_snapshot": model_snapshot,
        "config_snapshot": config_snapshot,
        "tokenizer_snapshot": tokenizer_snapshot,
        "transformers_version": getattr(__import__("transformers"), "__version__", None),
        "torch_version": torch.__version__,
    }
    return model, tokenizer, resolved, layers_list, embeddings


def _pad(ids: list[list[int]], pad_id: int, torch: Any, device: str) -> tuple[Any, Any, list[int]]:
    lengths = [len(row) for row in ids]
    maximum = max(lengths)
    input_ids = torch.full((len(ids), maximum), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros((len(ids), maximum), dtype=torch.long, device=device)
    for index, row in enumerate(ids):
        input_ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[index, :len(row)] = 1
    return input_ids, mask, lengths


@contextmanager
def _capture_hooks(model: Any, layers: list[Any], embeddings: Any, torch: Any) -> Iterator[dict[str, Any]]:
    captured: dict[str, Any] = {"embedding": None, "layers": [None for _ in layers]}
    handles = []

    def embedding_hook(_module: Any, _inputs: Any, output: Any) -> None:
        value = _first_tensor(output, torch)
        if value is not None:
            captured["embedding"] = value

    def layer_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = _first_tensor(output, torch)
            if value is not None:
                captured["layers"][index] = value
        return hook

    handles.append(embeddings.register_forward_hook(embedding_hook))
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(layer_hook(index)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _model_forward(model: Any, input_ids: Any, mask: Any, torch: Any) -> None:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True)


def _write_uint16_tensor(handle: Any, tensor: Any, torch: Any) -> None:
    value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16)
    handle.write(value.numpy().tobytes())


def _raw_to_safetensors(raw: Path, output: Path, *, key: str, rows: int, hidden: int, torch: Any) -> None:
    from safetensors.torch import save_file
    expected_bytes = rows * hidden * 2
    if raw.stat().st_size != expected_bytes:
        raise ExtractionError(f"raw representation has {raw.stat().st_size} bytes, expected {expected_bytes}: {raw}")
    mapped = torch.from_file(str(raw), shared=False, size=rows * hidden, dtype=torch.bfloat16).reshape(rows, hidden)
    save_file({key: mapped.clone()}, str(output), metadata={"dtype": "bfloat16", "rows": str(rows), "hidden_size": str(hidden)})
    mapped = None
    if sha256_file(output) == "":  # pragma: no cover - defensive, hash can never be empty
        raise ExtractionError(f"failed to hash {output}")
    raw.unlink()


def _tokenizer_metadata(tokenizer: Any, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": spec["tokenizer_repo"],
        "revision": spec["tokenizer_revision"],
        "commit": getattr(tokenizer, "_commit_hash", None),
        "class": tokenizer.__class__.__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "model_max_length": str(getattr(tokenizer, "model_max_length", None)),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }


def _prepare_work(bundle: Path) -> Path:
    bundle.parent.mkdir(parents=True, exist_ok=True)
    for stale in sorted(bundle.parent.glob(f".{bundle.name}.work-*")):
        quarantine(stale)
    if bundle.exists():
        result = validate_bundle(bundle, require_complete=True, check_tensors=False)
        if result["valid"]:
            raise FileExistsError(f"verified bundle already exists; refusing overwrite: {bundle}")
        quarantine(bundle)
    work = bundle.with_name(f".{bundle.name}.work-{os.getpid()}-{int(time.time())}")
    work.mkdir(parents=True, exist_ok=False)
    return work


def extract_one(
    *, manifest: dict[str, Any], model_spec: dict[str, Any], condition: dict[str, Any],
    rows: list[dict[str, Any]], bundle: Path, run_id: str, device: str = "cuda:0",
    batch_size: int = 4, telemetry_interval: float = 15.0, sync_destination: str | None = None,
) -> Path:
    """Extract one model/condition into a temporary bundle and atomically publish it.

    The function intentionally performs no probing or statistics. A model is
    loaded once for this atomic condition, then weights may be discarded after
    the destination-verified bundle exists.
    """
    work = _prepare_work(bundle)
    controller = StopController()
    controller.install_signals()
    spot = SpotPoller(controller)
    processed = 0
    started = utc_now()
    started_monotonic = time.monotonic()
    download_seconds = 0.0
    model_load_seconds: float | None = None
    extraction_seconds: float | None = None
    validation_seconds: float | None = None
    telemetry: Telemetry | None = None
    status = {"schema_version": 1, "state": "PENDING", "updated_utc": started, "reason": None}

    def set_state(state: str, reason: str | None = None) -> None:
        status.update({"state": state, "updated_utc": utc_now(), "reason": reason})
        write_json(work / "status.json", status)

    try:
        set_state("DOWNLOADING")
        examples = prepare_examples(rows, condition)
        model_snapshot, elapsed, _cache_hit = _resolve_hf_snapshot(
            model_spec["hf_repo"], model_spec["revision"],
            materialize=bool(model_spec.get("trust_remote_code", False)),
        )
        download_seconds += elapsed
        if model_spec.get("config_repo", model_spec["hf_repo"]) == model_spec["hf_repo"] and model_spec["config_revision"] == model_spec["revision"]:
            config_snapshot = model_snapshot
        else:
            config_snapshot, elapsed, _cache_hit = _resolve_hf_snapshot(model_spec.get("config_repo", model_spec["hf_repo"]), model_spec["config_revision"])
            download_seconds += elapsed
        if model_spec["tokenizer_repo"] == model_spec["hf_repo"] and model_spec["tokenizer_revision"] == model_spec["revision"]:
            tokenizer_snapshot = model_snapshot
        else:
            tokenizer_snapshot, elapsed, _cache_hit = _resolve_hf_snapshot(model_spec["tokenizer_repo"], model_spec["tokenizer_revision"])
            download_seconds += elapsed
        model_load_started = time.monotonic()
        model, tokenizer, resolved, layers, embeddings = load_hf_model_and_tokenizer(
            model_spec, device=device, model_snapshot=model_snapshot,
            config_snapshot=config_snapshot, tokenizer_snapshot=tokenizer_snapshot,
        )
        model_load_seconds = time.monotonic() - model_load_started
        audits, enriched = align_all(tokenizer, examples)
        if len(enriched) != len(rows) or any(audit["status"] != "aligned" for audit in audits):
            raise ExtractionError(f"alignment audit failed: {sum(audit['status'] != 'aligned' for audit in audits)} failures")
        context_length = model_spec.get("expected_context_length")
        if context_length is None:
            raise ExtractionError("model context length is not frozen in the matrix")
        too_long = [row["example_id"] for row in enriched if int(row["sequence_length"]) > int(context_length)]
        if too_long:
            raise ExtractionError(f"{len(too_long)} rendered prompts exceed model context length {context_length}")
        expected_examples = len(enriched)
        hidden = int(model_spec["expected_hidden_size"])
        layer_count = int(model_spec["expected_layer_count"])
        if len(layers) != layer_count:
            raise ExtractionError(f"captured {len(layers)} decoder layers, expected {layer_count}")
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else 0
        write_jsonl(work / "examples.jsonl", enriched)
        write_jsonl(work / "alignment.jsonl", audits)
        write_json(work / "model.json", {"manifest_record": model_spec, "resolved_runtime": resolved})
        write_json(work / "tokenizer.json", _tokenizer_metadata(tokenizer, model_spec))
        write_json(work / "prompt.json", {"prompt_version": manifest["prompt_conditions"]["prompt_version"], "condition": condition})
        raw_dir = work / "raw"
        representation_dir = work / "representations"
        raw_dir.mkdir()
        representation_dir.mkdir()
        total_target = sum(int(row["target_token_count"]) for row in enriched)
        raw_prompt = [raw_dir / f"layer_{layer:04d}.prompt.bf16" for layer in range(layer_count)]
        raw_target = [raw_dir / f"layer_{layer:04d}.target.bf16" for layer in range(layer_count)]
        prompt_handles = [path.open("wb") for path in raw_prompt]
        target_handles = [path.open("wb") for path in raw_target]
        embedding_raw = raw_dir / "embedding.bf16"
        embedding_handle = embedding_raw.open("wb")
        set_state("EXTRACTING")
        extraction_started = time.monotonic()
        telemetry = Telemetry(work / "telemetry.jsonl", run_root=work, counter=lambda: processed, interval_seconds=telemetry_interval)
        telemetry.start()
        spot.start()
        with _capture_hooks(model, layers, embeddings, __import__("torch")) as captured:
            import torch
            for begin in range(0, expected_examples, batch_size):
                controller.check()
                batch = enriched[begin:begin + batch_size]
                full_ids = [row["input_ids"] for row in batch]
                full_input, full_mask, full_lengths = _pad(full_ids, int(pad_id), torch, device)
                captured["embedding"] = None
                captured["layers"] = [None for _ in layers]
                _model_forward(model, full_input, full_mask, torch)
                if captured["embedding"] is None or any(value is None for value in captured["layers"]):
                    raise ExtractionError("one or more capture hooks did not receive tensor output")
                for layer, value in enumerate(captured["layers"]):
                    prompt_rows = torch.stack([value[offset, length - 1, :] for offset, length in enumerate(full_lengths)])
                    _write_uint16_tensor(prompt_handles[layer], prompt_rows, torch)
                embedding_rows = torch.stack([captured["embedding"][offset, length - 1, :] for offset, length in enumerate(full_lengths)])
                _write_uint16_tensor(embedding_handle, embedding_rows, torch)

                prefix_ids = [row["input_ids"][:int(row["selected_final_token_index"]) + 1] for row in batch]
                prefix_input, prefix_mask, _prefix_lengths = _pad(prefix_ids, int(pad_id), torch, device)
                captured["embedding"] = None
                captured["layers"] = [None for _ in layers]
                _model_forward(model, prefix_input, prefix_mask, torch)
                if any(value is None for value in captured["layers"]):
                    raise ExtractionError("one or more target capture hooks did not receive tensor output")
                for layer, value in enumerate(captured["layers"]):
                    target_rows = torch.cat([
                        value[offset, [int(position) for position in row["target_token_indices"]], :]
                        for offset, row in enumerate(batch)
                    ], dim=0)
                    _write_uint16_tensor(target_handles[layer], target_rows, torch)
                processed += len(batch)
                controller.check()
        extraction_seconds = time.monotonic() - extraction_started
        for handle in prompt_handles + target_handles + [embedding_handle]:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        # Telemetry must be quiescent before the checksum manifest is written;
        # otherwise a late sample could invalidate the source during sync.
        if telemetry:
            telemetry.stop()
            telemetry = None
        spot.stop()
        set_state("VALIDATING")
        validation_started = time.monotonic()
        _raw_to_safetensors(embedding_raw, representation_dir / "embedding_output.safetensors", key="embedding_output", rows=expected_examples, hidden=hidden, torch=__import__("torch"))
        import torch
        for layer in range(layer_count):
            _raw_to_safetensors(raw_prompt[layer], representation_dir / f"layer_{layer:04d}.safetensors", key="prompt_final", rows=expected_examples, hidden=hidden, torch=torch)
            # Append target to the same layer file by creating a single file from both rows.
            from safetensors.torch import save_file
            target_raw = raw_target[layer]
            if target_raw.stat().st_size != total_target * hidden * 2:
                raise ExtractionError(f"target representation byte count mismatch: {target_raw}")
            target = torch.from_file(str(target_raw), shared=False, size=total_target * hidden, dtype=torch.bfloat16).reshape(total_target, hidden).clone()
            prompt_path = representation_dir / f"layer_{layer:04d}.safetensors"
            from safetensors import safe_open
            prompt = safe_open(str(prompt_path), framework="pt", device="cpu")
            with prompt:
                prompt_tensor = prompt.get_tensor("prompt_final")
            save_file({"prompt_final": prompt_tensor, "target_span": target}, str(prompt_path), metadata={"dtype": "bfloat16", "hidden_size": str(hidden)})
            target_raw.unlink()
        shutil.rmtree(raw_dir, ignore_errors=False)
        manifest_path = work / "manifest.json"
        write_json(manifest_path, {
            "bundle_schema": BUNDLE_SCHEMA,
            "schema_version": 1,
            "run_id": run_id,
            "model_id": model_spec["model_id"],
            "condition_id": condition["id"],
            "model_revision": model_spec["revision"],
            "tokenizer_revision": model_spec["tokenizer_revision"],
            "config_revision": model_spec["config_revision"],
            "requested_dtype": "bfloat16",
            "saved_tensor_dtype": "bfloat16",
            "device": device,
            "expected_examples": expected_examples,
            "successful_alignments": expected_examples,
            "expected_layer_count": layer_count,
            "expected_hidden_size": hidden,
            "total_target_tokens": total_target,
            "prompt_version": manifest["prompt_conditions"]["prompt_version"],
            "condition": condition,
            "representation_semantics": manifest["inference_contract"],
            "dataset": manifest["dataset"],
            "code_identity": _code_identity(),
            "started_utc": started,
            "validated_utc": utc_now(),
            "status": "VALIDATED",
            "command": os.environ.get("BF16_COMMAND", "unknown"),
            "cost": {
                "instance_type": os.environ.get("BF16_INSTANCE_TYPE"),
                "pricing_mode": os.environ.get("BF16_PRICING_MODE"),
                "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
                "availability_zone": os.environ.get("AWS_AVAILABILITY_ZONE"),
                "runtime_seconds": round(time.monotonic() - started_monotonic, 6),
                "model_download_seconds": round(download_seconds, 6),
                "model_load_seconds": round(model_load_seconds, 6) if model_load_seconds is not None else None,
                "extraction_seconds": round(extraction_seconds, 6) if extraction_seconds is not None else None,
                "validation_seconds": round(validation_seconds, 6) if validation_seconds is not None else None,
                "sync_seconds": None,
                "bytes_captured": sum(path.stat().st_size for path in representation_dir.glob("*.safetensors")),
                "examples_per_second": processed / max(time.monotonic() - started_monotonic, 1e-9),
                "hourly_usd": float(os.environ["BF16_EC2_USD_PER_HOUR"]) if os.environ.get("BF16_EC2_USD_PER_HOUR") else None,
                "estimated_cost_usd": ((time.monotonic() - started_monotonic) / 3600.0 * float(os.environ["BF16_EC2_USD_PER_HOUR"])) if os.environ.get("BF16_EC2_USD_PER_HOUR") else None
            },
        })
        write_checksums(work)
        validation = validate_bundle(work, require_complete=False, check_tensors=True)
        write_json(work / "validation.json", validation)
        # validation.json is part of the evidence, so hash it and validate again.
        write_checksums(work)
        validation = validate_bundle(work, require_complete=False, check_tensors=True)
        if not validation["valid"]:
            raise ExtractionError("bundle validation failed: " + "; ".join(validation["errors"][:10]))
        validation_seconds = time.monotonic() - validation_started
        set_state("SYNCING")
        if sync_destination is None:
            raise ExtractionError("a persistent sync destination is required; refusing to declare a local-only bundle complete")
        from .sync import finalize_marker, sync_local_bundle, sync_s3_bundle
        sync_started = time.monotonic()
        if sync_destination.startswith("s3://"):
            receipt = sync_s3_bundle(work, sync_destination)
        else:
            receipt = sync_local_bundle(work, Path(sync_destination))
        receipt["sync_seconds"] = round(time.monotonic() - sync_started, 6)
        write_json(work / "sync_receipt.json", receipt)
        # Receipt is intentionally excluded from the checksum manifest because it
        # is created after destination verification; COMPLETE binds the manifest hash.
        checksum_hash = sha256_file(work / "checksums.sha256")
        from .bundle import write_complete_marker
        set_state("COMPLETE")
        write_complete_marker(work, destination=str(receipt["destination"]), checksum_manifest_sha256=checksum_hash)
        finalize_marker(work, str(receipt["destination"]))
        final_validation = validate_bundle(work, require_complete=True, check_tensors=False)
        if not final_validation["valid"]:
            raise ExtractionError("final bundle validation failed: " + "; ".join(final_validation["errors"][:10]))
        bundle.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work, bundle)
        return bundle
    except InterruptionRequested as error:
        set_state("INTERRUPTED", str(error))
        raise
    except Exception as error:
        try:
            set_state("FAILED", f"{type(error).__name__}: {error}")
        finally:
            raise
    finally:
        if telemetry:
            telemetry.stop()
        spot.stop()
        controller.restore_signals()
