# BF16 Hugging Face extraction runbook

This runbook defines a new extraction revision. It does not modify or reuse
the Q8_0 artifacts, the v1/v2/v3 freezes, `runs/revision/aws-final`,
`runs/revision/aws-handoff`, or the historical extractor archive.

The resolved 11-record manifest is
[`configs/bf16_model_matrix_20260826_jais_resolved.json`](../configs/bf16_model_matrix_20260826_jais_resolved.json).
It contains three Qwen, three Llama, one Gemma, Phi, Mistral, Jais, and ALLaM;
all records use immutable Hub commit SHAs. The 2026 manifest resolves Jais
metadata and records the exact access/preflight state. It is a release
manifest, not evidence that every matrix unit has completed.

## What the audit found

The current package retains the small set of shared analysis primitives needed
by the BF16 path:

* `research_stack/revision/alignment.py` is the semantic reference for a
  structural designated target placeholder, contiguous offset coverage, and
  final overlapping token selection.
* `research_stack/revision/alignment.py` defines structural target-span
  rendering and token alignment.
* `output/data/paper1_normalized.jsonl` and the resolved model matrix define
  the dataset and extraction identity. The dataset is read-only; the raw
  `def` spelling is not rewritten.
* `research_stack/bf16_capture/` performs model loading, capture, validation,
  telemetry, queueing, and synchronization. It preserves the complete target
  span and performs no scientific analysis on the GPU.

## Frozen extraction semantics

The two condition templates are frozen in the matrix:

* `full_metadata`: the existing surface, target, lemma, root, and pattern
  prompt;
* `metadata_free`: the same surface and target fields, with lemma, root, and
  pattern removed.

Chat templates are disabled. The target span comes from exactly one structural
`{target}` placeholder. A fast tokenizer's character offsets must cover the
full target span contiguously. The corresponding UTF-8 byte span, token IDs,
pieces, offsets, target token indices, selected final token, and decoded audit
value are recorded. The decoded span must match the target after only
boundary-whitespace normalization; any other mismatch aborts the atomic unit.

The saved representations are the embedding output at the final complete-prompt
token and every text-decoder block output before final model normalization. For
each prompt condition, `prompt_final` is the final non-padding token of the
complete prompt. The target representation is computed from the causal prefix
ending at the final target-overlapping token; every target-span token state from
that prefix is retained. Final-subtoken and mean-target-span features are local
derivations, not additional forward passes.

The runner refuses FP32, FP16, quantization, CPU offload, multi-device
placement, floating revisions, tokenizer revision drift, non-BF16 parameters,
missing CUDA BF16 support, and non-fast tokenizers.

Jais uses pinned `inceptionai/Jais-13B` remote code whose legacy imports and
configuration defaults are not provided by the installed modern Transformers
release. The extractor contains a narrow, recorded compatibility shim for the
removed pruning/model-parallel helpers and legacy defaults; it does not edit
Hub files or change weights. Jais remains single-device BF16 and fails closed
if any other compatibility or placement assumption is encountered.

## Bundle schema

Each model/condition is an independent unit:

```text
<run-root>/bundles/<model-id>/<condition-id>/
  manifest.json                 # immutable semantic/runtime contract
  model.json                    # manifest record + resolved runtime identity
  tokenizer.json                # tokenizer class, exact revision/commit, IDs
  prompt.json                   # condition/template/version
  examples.jsonl                # identity, labels, prompt, IDs, offsets, spans
  alignment.jsonl               # one deterministic alignment audit per example
  representations/
    embedding_output.safetensors       # [N, H], BF16
    layer_0000.safetensors              # prompt_final [N,H], target_span [T,H]
    ...
  telemetry.jsonl               # GPU/VRAM, host RAM, disk, throughput
  validation.json                # structural/tensor validation result
  status.json                    # lifecycle state; excluded from scientific hash
  checksums.sha256               # SHA-256 for every scientific artifact
  COMPLETE                       # written only after destination verification
```

`N` is the expected example count, `H` is the resolved hidden size, and `T` is
the sum of target token counts. Layer files are intentionally small upload
units; there is no full sequence-by-layer activation dump. `status.json` and
the post-upload `sync_receipt.json` are lifecycle records excluded from the
checksum list because they change during `VALIDATING`, `SYNCING`, and
`COMPLETE`. The marker binds the checksum-manifest hash and verified
destination. A directory without a valid marker is never scientific evidence.

The fields in `examples.jsonl` are sufficient to reconstruct all planned local
analysis inputs:

| Later analysis | Bundle fields |
| --- | --- |
| probes, alpha/layer selection, metrics | `representations/layer_*.safetensors`, labels, IDs |
| final target subtoken | `target_span`, `target_token_count`, span order |
| mean target span | same complete flat `target_span` rows |
| all five folds / random split | `example_id`, `lemma`, `root`, row order, labels |
| majority / shuffled controls | labels and split indices; no weights |
| grouped/paired bootstrap | IDs, exact prompt hashes, labels, both locations |
| per-class / fragmentation | morphology metadata and target token count |
| BF16/Q8 cosine and relative-L2 | same example order and locally loaded arrays |
| CKA / RSA | same-example `[N,H]` matrices from any layer/location |

`research_stack.bf16_capture.local_validate` exercises these derivations and
records `weights_required: false`. It streams one layer file at a time while
constructing the local matrices, so a large target-span capture does not
require holding every layer's flat target tensor in RAM. It is a compatibility
check, not a new scientific analysis. Post-hoc reports must be written beside
the bundle, not inside a completed bundle; the CLI rejects an output path that
would mutate the checksum-covered bundle tree.

## Required environment and preflight

Install the separate extraction environment from
`requirements-bf16.lock.txt`; do not replace the audited analysis environment.
The environment must include the tokenizer/runtime dependencies in the lock
file, including `sentencepiece` and `protobuf`; the CUDA wheel source must
match the selected PyTorch build. Then authenticate to Hugging Face with a
token whose value is never written to logs:

```bash
export HF_TOKEN='...'
export HF_HOME=/mnt/research-bf16/hf-cache
make bf16-preflight \
  BF16_MANIFEST=configs/bf16_model_matrix_20260826_jais_resolved.json \
  BF16_RUN_ROOT=/mnt/research-bf16/runs/<new-unique-run-id>
```

Full preflight resolves every exact revision, checks the Hub SHA, architecture,
hidden size, layer count, native dtype, tokenizer fastness and tokenizer commit,
records Hub file sizes/blob IDs, checks CUDA BF16 support and disk headroom, and
fails if Jais metadata is still unresolved. `--no-hub` is setup-only and never
authorizes an extraction.

The preflight report must be copied into the new run root. Before the first
real model, manually confirm that all gated access requests have been approved,
the resolved Jais config is filled into a new matrix revision, and the report
has no failures.

## AWS recommendation

Use one independent worker per atomic queue slice, initially one worker. The
default is `eu-north-1`, G6e, one NVIDIA L40S, Spot with On-Demand fallback.
AWS lists G6e L40S accelerator memory as 44 GiB per GPU; `g6e.xlarge` has 32
GiB host RAM and 4 vCPUs, `g6e.2xlarge` 64 GiB and 8 vCPUs, and `g6e.4xlarge`
128 GiB and 16 vCPUs. The practical choices are:

| Use | Instance | Reason |
| --- | --- | --- |
| smoke / Qwen 0.6B | `g6e.xlarge` minimum | enough GPU; CPU/tokenization margin is small |
| default matrix worker | `g6e.2xlarge` | 44 GiB GPU, 64 GiB host RAM, 8 vCPUs |
| Jais or high host-memory pressure | `g6e.4xlarge` | same GPU memory, double host RAM/CPU |

The instance size must be selected from the resolved model config and observed
peak VRAM. A 44 GiB GPU should fit the planned BF16 weights up to about 13B,
but the harness must fail before extraction if model placement or peak memory
requires offload. It must not make the model fit by silently changing dtype.

Use a separate gp3 EBS volume, not root disk, for the HF cache and run root.
Start with 300 GiB; use 400 GiB if retaining the full immutable Hub cache and
all bundles locally. The sizing guard is based on four target tokens per example
and includes both conditions; exact bundle sizes are reported after alignment.
The current resolved ten non-Jais rows estimate about 77 GiB of BF16 capture
bytes before temporary/raw-space and safety margin. gp3 provides a sustained
125 MiB/s baseline; provision more throughput only after telemetry shows EBS
write or download pressure.

Keep the HF cache on EBS for restart efficiency, but do not treat cached weights
as evidence. The permanent identity is the exact Hub revision plus the recorded
Hub file sizes/blob IDs and the runtime manifest. Captures should be synced to
S3 immediately after each condition and copied to the local research stack
before terminating the instance.

The runner samples `nvidia-smi`, `/proc/meminfo`, disk usage, processed examples,
and examples/sec. Poor GPU utilization is investigated from telemetry in this
order: tokenizer/CPU, Python batching, tensor serialization, EBS, network, and
unnecessary synchronization.

## Spot and persistence procedure

The worker polls the EC2 IMDSv2 Spot action endpoint and handles SIGTERM/SIGINT.
On a notice it stops before the next batch, flushes telemetry and raw files,
and exits without a marker. AWS documents the interruption warning as a
best-effort two-minute notice; it is not a correctness guarantee. A queue
restart must use `--recover-running` only after confirming the old worker is
gone.

The launch helper defaults to On-Demand. Spot requires an explicit
`BF16_MARKET_MODE=spot` and `BF16_SPOT_MAX_PRICE`; it refuses an implicit
On-Demand fallback. Set `BF16_SPOT_ONDEMAND_FALLBACK=true` only when that
fallback is approved for the run. Spot instances are configured to terminate
on interruption, so the queue retries only the unfinished atomic unit.

Local synchronization copies to a staging directory, fsyncs each file, verifies
the source checksum list against the destination, atomically renames the
destination, and only then copies `COMPLETE`. S3 synchronization uploads each
scientific object with a requested SHA-256 checksum and verifies it via
`HeadObject` checksum mode; ETag is not accepted. A failed or partial transfer
cannot create `COMPLETE`.

```bash
python scripts/bf16_queue.py queue-init \
  --manifest configs/bf16_model_matrix_20260826_jais_resolved.json \
  --run-root /mnt/research-bf16/runs/<new-unique-run-id> \
  --run-id <new-unique-run-id>

python scripts/bf16_queue.py queue-run \
  --manifest configs/bf16_model_matrix_20260826_jais_resolved.json \
  --run-root /mnt/research-bf16/runs/<new-unique-run-id> \
  --sync-destination s3://BUCKET/<new-unique-run-id> \
  --worker worker-a
```

For a local/shared persistent destination, replace the S3 URI with an explicit
directory outside the temporary work path. To divide the queue, run workers
with distinct `--worker` values against the same queue; the file lock makes
claims atomic. Start with one worker, measure real wall time, and add workers
only if the deadline warrants the extra download/storage contention.

Per-unit cost metadata records instance type, pricing mode, region/AZ,
timestamps, runtime, pinned-snapshot download time, model-load time,
extraction time, validation time, and bytes/examples per second. The
post-upload `sync_receipt.json` records synchronization time because it is
written after the checksum-covered manifest. An estimated cost is recorded
only when
`BF16_EC2_USD_PER_HOUR` is explicitly supplied. No price is guessed into the
scientific record.

## First-run gate

Do not release the 22-item matrix queue until this sequence has passed:

1. Qwen3-0.6B full and metadata-free conditions on a real GPU;
2. local bundle validation and destination hash verification;
3. local final-subtoken, mean-span, layer-curve, one Ridge probe, and CKA/RSA
   matrix compatibility checks;
4. Phi and ALLaM cross-family pilots;
5. Jais access/config/architecture validation and one successful specialist
   capture;
6. a small-Llama end-to-end pilot (passed for Llama-3.2-1B and Llama-3.1-8B
   on 2026-08-26);
7. only then enable remaining models.

The real AWS pilot completed on 2026-08-25 for Qwen3-0.6B on a G6e.xlarge
(L40S, eu-north-1a). Both prompt conditions completed independently in about
89 seconds each at about 52.7 examples/sec, produced about 0.84 GiB per
condition, synchronized to a second persistent path, and passed destination
checksum validation. The pilot captured 4,701/4,701 alignments and all 28
layers. Local validation then derived final-subtoken and mean-span matrices,
layer curves, a Ridge probe, all five lemma/root group folds, and CKA/RSA
compatible matrices with `weights_required: false`.

The first full-metadata attempt deliberately exposed a real completion-path
bug: a final telemetry append occurred after validation and invalidated the
checksum before synchronization. It produced no `COMPLETE` marker. The
runner was corrected to stop telemetry before checksumming, and the retry
passed. The warning and failed log are retained beside the pilot evidence.

The pilot proves the Qwen path, not universal architecture compatibility. The
second current-code pilot completed on 2026-08-25 for Phi-3-mini and ALLaM
on the same G6e.xlarge. Both prompt conditions for both models passed remote
bundle validation, persistent-destination validation, and local analysis
validation. Phi produced 4,701 alignments, 32 layers, hidden size 3,072, and
23,375 target-span rows per condition. ALLaM produced 4,701 alignments, 32
layers, hidden size 4,096, and 5,302 target-span rows per condition. The
local reports again reconstruct final-subtoken, mean-span, layer-wise,
five-fold, Ridge, CKA/RSA-compatible analyses without weights. These pilots
also verified the pinned-snapshot download path on first use and the persistent
HF cache on reuse.

The Phi/ALLaM pilots and the current Jais pilot clear the additional-architecture
and Arabic-specialist gates. The Jais run used the resolved 2026 manifest and
completed both conditions on a `g6e.2xlarge` On-Demand L40S worker: 4,701/4,701
alignments, 40 layers, hidden size 5,120, 5,149 target-span rows, checksum
verified, and local analysis for both conditions passed with
`weights_required: false`.

The accessible BF16 batch completed on 2026-08-26 on one On-Demand
`g6e.2xlarge` in `eu-north-1b`: Qwen2.5-1.5B, Qwen3-8B, Llama-3.2-1B, and
Llama-3.1-8B, with both prompt conditions for each model. Every remote bundle
passed validation and every copied local bundle passed checksum, marker,
alignment, and tensor-shape validation. The batch captured 4,701/4,701
alignments per condition; local analysis also passed for Qwen3-8B with
`weights_required: false`. The Llama-3.2-1B and Llama-3.1-8B captures therefore
clear the Llama architecture gate.

The final BF16 batch completed on 2026-08-26 in fresh run root
`runs/bf16/bf16-gemma-mistral-20260826-v1/` after Llama-3.2-3B access was
granted. Gemma-4-E2B-it, Mistral-7B-Instruct-v0.3, and Llama-3.2-3B each
completed both prompt conditions on the same On-Demand `g6e.2xlarge` worker.
All six bundles passed local tensor/alignment validation and independent
SHA-256 verification with 4,701/4,701 alignments. Gemma, Mistral, and Llama
3B full-metadata local proofs reconstructed the five-fold splits, final and
mean target-span derivations, layerwise matrices, a Ridge probe, and
CKA/RSA-compatible matrices with `weights_required: false`. The run recorded
its per-condition timings and cost summary in
`runs/bf16/bf16-gemma-mistral-20260826-v1/batch_summary.json`; the worker,
working volume, temporary SSH rule, and temporary S3 bucket were then removed.

The account currently permits eight running G/VT instances for both On-Demand
and Spot, not the previously requested sixteen. Spot launch attempts for
`g6e.2xlarge` in `eu-north-1a` and `eu-north-1b` returned insufficient capacity;
the On-Demand `eu-north-1b` fallback succeeded. Do not assume parallel Spot
capacity. Start with one worker and use On-Demand only when its cost is
explicitly accepted.

Do not release still-pending matrix units until their exact HF access,
checkpoint identity, and worker/cost plan are explicitly approved. The fixture tests
continue to exercise bundle validation, corruption rejection, marker
semantics, alignment, local derivation, and split reconstruction without
creating research output.

## Failure injection checklist

Run these against a temporary run root, never a frozen or historical path:

* kill during extraction and confirm no `COMPLETE`;
* kill while writing a safetensors file and confirm checksum failure;
* unset `HF_TOKEN` and confirm gated preflight failure;
* use a gated denial and confirm no bundle marker;
* fill or simulate a low-space run root and confirm preflight/validation fail;
* corrupt one tensor, delete one layer, duplicate/reorder an ID, or inject NaN;
* alter tokenizer revision or prompt version and confirm manifest mismatch;
* interrupt S3 upload and confirm no remote `COMPLETE`;
* restart with `--recover-running` and confirm only that atomic unit retries;
* verify an already complete unit is skipped and never overwritten.

## Release decision

For the full 11-model BF16 matrix (Qwen3-0.6B, Qwen2.5-1.5B, Qwen3-8B,
Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Gemma-4-E2B-it, Phi-3-mini,
Mistral-7B-Instruct-v0.3, Jais, and ALLaM), yes: all 22 model-condition
bundles are locally usable for the planned post-extraction representations
and analyses without model weights. There is no remaining cloud-extraction
blocker in the resolved manifest. The AWS capacity limit is known and the
On-Demand path is working; future extraction should only be needed for an
explicitly changed experiment revision or a newly frozen checkpoint.
