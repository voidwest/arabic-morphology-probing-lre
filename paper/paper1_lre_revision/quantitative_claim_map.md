# Quantitative claim map

The manuscript uses the BF16 analysis revision
`evidence/bf16-analysis-20260826-v2/snapshot/analysis/`, frozen and independently
verified at `evidence/bf16-analysis-20260826-v2/`. Paths below are relative to
the repository root. Numeric summaries derived from a generated CSV are
recomputable by filtering or aggregating that CSV; no values are estimated.

| Manuscript claim | Source artifact | Verification rule |
|---|---|---|
| 4,701 authoritative examples; 2,297,339 bytes; dataset SHA-256 | `data/provenance/paper1_dataset_manifest.json` | `authoritative_frozen_dataset`; hash and record count |
| 1,519 lemmas; 707 roots; 2,331 surface forms | `data/provenance/paper1_dataset_manifest.json`, `output/data/paper1_normalized.jsonl` | recompute unique fields |
| POS counts 2,957/910/834; gender 2,942/1,757/2 missing; number 3,954/700/45/2 missing | `data/provenance/paper1_dataset_manifest.json` | dataset audit counts; `def` is task-loaded as `du` |
| POS x gender x number descriptive cross-tab | `output/data/paper1_normalized.jsonl` | task-load `number:def -> du`, require all three labels; rendered in `tables/pos_gender_number.tex` |
| PADT source and 5,000-to-4,701 construction | `data/provenance/paper1_dataset_manifest.json`, `docs/paper1_dataset_provenance.md` | transformation stages and hashes |
| 11 models; 22 BF16 condition bundles; 44 analysis units | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/analysis_manifest.json` | 11 bundle model keys x 2 condition manifests; 44 cell-unit files |
| 1,980 real cells and 360 shuffled cells | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/cells/*` and `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/shuffled_controls.csv` | 44 x 3 tasks x 3 splits x 5 folds; two primary-model controls |
| Five outer folds for every model/split/interface | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/summary.csv` | every row has `folds=5`; 396 summary rows |
| All 132 random-minus-grouped delta rows positive | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/macro_f1_split_deltas.csv` | 132 rows; both delta columns > 0 |
| Delta ranges and grand means: lemma +0.0127 to +0.2863, mean +0.1059; root +0.0255 to +0.2796, mean +0.1179 | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/macro_f1_split_deltas.csv` | min/max/mean over 132 rows |
| Model-level and interface-level delta summaries | same delta CSV | grouped means; rendered compact table in manuscript |
| Example cell results and fold variation | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/summary.csv` | select exact model/interface/task/split rows |
| No shuffled control exceeds matching majority baseline | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/shuffled_controls.csv`, analysis audit | compare each shuffled accuracy with train-only majority |
| 10,000-replicate paired uncertainty if discussed | historical `results/bootstrap_location_deltas.{json,csv,md,tex}` | primary bootstrap protocol is documented; not recomputed for all BF16 interfaces |
| CKA/RSA is diagnostic only | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/cka_rsa_diagnostics.json`, `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/analysis_decision.md` | fixed sample; no selection or primary claim |
| Fragmentation bins and POS/class errors | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/fragmentation.csv`, `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/class_pos_error_analysis.csv` | bins 1, 2, 3+; per-class/POS rows |
| Local analyses do not require weights | `evidence/bf16-analysis-20260826-v2/snapshot/analysis/results/local_analysis_sufficiency.json` | `weights_required_after_capture=false` |
| Content-addressed BF16 freeze passed | `evidence/bf16-analysis-20260826-v2/README.md`, verifier output | external bundle files and internal checksums verified |

The paper intentionally avoids inserting exact numbers for analyses whose
artifacts are diagnostic rather than primary, unless the number is needed to
explain a decision. The public source and validation archive is available at
`https://github.com/voidwest/arabic-morphology-probing-lre`; remaining
provenance limitations are stated in the manuscript and dataset provenance
record.
