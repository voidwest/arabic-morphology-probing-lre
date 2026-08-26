# BF16 local analysis runbook

This runbook describes the weight-free analysis revisions created from the
validated BF16 bundles. It is separate from the immutable Q8 historical run
and from all `lre-corrected-analysis-20260824-v*` freezes.

## Contract

The input is the pinned matrix in
`configs/bf16_model_matrix_20260826_jais_resolved.json` and the 22 complete
bundles under `runs/bf16/`. Every bundle is validated, checksum-verified, and
cross-checked against the 4,701-row frozen dataset before use.

The four feature sets are:

* `full_prompt_final`: full-metadata condition, prompt-final state;
* `metadata_free_prompt_final`: metadata-ablation condition, prompt-final state;
* `target_final_subtoken`: final row of the complete saved target span;
* `target_mean_span`: mean of the complete saved target span.

The target features come from the full-metadata bundle. The target state is
causal and ends at the selected final target-overlapping token; later prompt
fields are not visible to it. The complete target span is retained in every
bundle, so final and mean pooling are local derivations.

For each task, rows with a missing task label are excluded before splitting.
The historical `number=def` spelling is canonicalized to `du` at task load;
the frozen dataset bytes are not changed. Random uses nested five-fold
`StratifiedKFold` with seed 42. Lemma/root use the existing nested five-fold
`GroupKFold` implementation, with dev fold zero. Layer selection is strictly
development-only: maximize dev accuracy, then dev macro-F1, then choose the
lowest layer. The selected layer is refit on train+dev and evaluated once on
test with `StandardScaler + RidgeClassifier(alpha=1.0)`.

The original `bf16-analysis-20260826-v1` freeze used all five outer folds for
the two primary models and one deterministic screen for the other models. The
full-matrix revision uses five outer folds for every model. Primary models
also receive independent within-partition shuffled-label controls.

## Commands

Run one disjoint model worker and defer aggregation:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  /home/west/ember/.venv/bin/python scripts/run_bf16_analysis.py \
  --models qwen3-0.6b --skip-finalize \
  --run-root runs/bf16-analysis/bf16-analysis-20260826-v1
```

Workers may share a run root only with disjoint model IDs. Each model/feature
unit has an exclusive lock and an atomic JSON result. A killed worker leaves
no completed unit marker; rerunning the same command verifies existing unit
files and skips them.

After all 11 models have four unit files, finalize once:

```bash
/home/west/ember/.venv/bin/python scripts/run_bf16_analysis.py \
  --finalize-only \
  --run-root runs/bf16-analysis/bf16-analysis-20260826-v1
```

Finalization refuses to continue if any unit is missing. It writes the
summary, shuffled controls, fragmentation results, class/POS error audit,
figures, BF16/Q8 comparison, sufficiency record, and a content-addressed
analysis manifest.

## Full five-fold matrix

Use a new run root for the complete five-fold matrix. Workers may be split by
disjoint model IDs; `--other-folds 5` is the required setting for the nine
non-primary models:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  /home/west/ember/.venv/bin/python scripts/run_bf16_analysis.py \
  --models qwen3-0.6b llama-3.2-1b qwen2.5-1.5b qwen3-8b \
  llama-3.2-3b llama-3.1-8b gemma-4-2b phi-3-mini mistral-7b \
  jais-13b allam-7b --other-folds 5 --skip-finalize \
  --run-root runs/bf16-analysis/bf16-analysis-20260826-v2
```

The finalizer additionally writes `results/macro_f1_split_deltas.csv` and its
Markdown/TeX renderings. Each row is one model × task × representation, with
`random - lemma-heldout` and `random - root-heldout` macro-F1 deltas computed
from the five outer-fold means.

## Outputs

`results/summary.csv` and its Markdown/TeX renderings are generated from the
cell JSON files. Primary-model test predictions are in
`results/predictions/`; no prediction is used to select a layer. The
fragmentation bins are target token counts `1`, `2`, and `3+`. Error analysis
contains per-class precision/recall/F1 and confusion matrices for all rows and
for each recorded gold-POS stratum.

`results/cka_rsa_diagnostics.json` contains a fixed-sample, final-layer
diagnostic only. It is not used for layer selection, probe fitting, or the
primary scientific claim. It is retained only to document the decision that
the requested interface/pooling probe contrasts already answer the planned
question; CKA/RSA is not promoted to a headline result.

The BF16/Q8 comparison uses preserved historical Q8 result JSONs for the
overlapping models and compares fold-zero metric direction/agreement. Raw
vector cosine and relative-L2 cannot be regenerated in the current checkout:
the v3 external-artifact record lists the historical Q8 feature arrays, but
those arrays are absent locally. This is recorded as a blocker; no Q8 run is
silently repeated and no vector claim is manufactured.

## Freeze

The analysis run root is the source of truth for this revision. Do not write
to any historical Q8 run or v1/v2/v3 freeze. Freeze the BF16 analysis with a
new dedicated freeze helper after the final audit; the large representation
files remain external artifacts identified by their bundle checksums, just as
the historical freeze records large arrays.

```bash
/home/west/ember/.venv/bin/python scripts/freeze_bf16_analysis.py \
  --run-root runs/bf16-analysis/bf16-analysis-20260826-v1 \
  --freeze-id bf16-analysis-20260826-v1

/home/west/ember/.venv/bin/python scripts/verify_bf16_analysis_freeze.py \
  freezes/bf16-analysis-20260826-v1
```
