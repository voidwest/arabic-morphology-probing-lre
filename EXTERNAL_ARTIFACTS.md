# External and deliberately excluded artifacts

This archive is intentionally limited to material needed for the current LRE
BF16 paper and its reproducibility checks.

## BF16 representation bundles

The complete BF16 hidden-state bundles are approximately 67 GB and remain
outside GitHub. The compact freeze at
`evidence/bf16-analysis-20260826-v2/` contains the analysis outputs, bundle
manifests, validation records, and `BF16_EXTERNAL_ARTIFACTS.json`, which records
the source path, byte count, and SHA-256 for every external bundle file.

The bundles are not required for inspecting the committed compact analysis
state. They are required only to rerun local analyses from raw representations
or to perform a new extraction. Their absence from this repository is not a
claim that the captures were discarded.

## Model weights and caches

No model weights, Hugging Face caches, credentials, or tokens are included.
The pinned model matrix records exact repository revisions, access status,
architectures, dtype contracts, licenses, and resource metadata. Weights are
temporary extraction inputs; the archived scientific evidence is the validated
representation and analysis state.

## Historical Q8 material

The paper's BF16/Q8 result is a metric-direction comparison against the
preserved historical record. Historical Q8 weights, raw run trees, and vector
arrays are outside this curated archive. The compact BF16 freeze retains the
comparison record and its explicit limitation; no Q8 artifact is silently
recreated or relabeled as BF16 evidence.

## Distribution policy for gated and card-restricted derivatives

This public repository does not distribute model weights, model caches,
credentials, or raw hidden-state bundles. For bundles derived from gated or
card-restricted checkpoints, including Meta Llama, Jais, and ALLaM, any separate
raw-bundle request is considered only after the requester independently obtains
authorization for the corresponding checkpoint, accepts its model-card and
license terms, and the proposed transfer is permitted under those terms. No
weights, credentials, or Hugging Face tokens are shared. If those conditions
cannot be confirmed, the raw bundle is not distributed.

## Third-party data

The archived UD Arabic-PADT source and its license are retained under
`data/upstream/UD_Arabic-PADT-r2.17/`. The dataset provenance manifest records
the source-content hashes, transformation lineage, and applicable license.
