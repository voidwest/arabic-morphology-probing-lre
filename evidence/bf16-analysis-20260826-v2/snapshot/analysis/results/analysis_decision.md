# BF16 analysis decision

This is a generated audit record for the full five-fold BF16 matrix; it is not manuscript prose.

## Completeness

- Atomic units: 44/44 complete (11 models × 4 feature sets).
- Real cells: 1980; shuffled cells: 360.
- All models use five outer folds for random, lemma-heldout, and root-heldout splits; recorded summary fold counts: [5].
- Layer selection is development-only with fixed Ridge alpha=1.0; no test result selects a layer or hyperparameter.

## Primary held-out comparison

Mean accuracy and mean accuracy-minus-majority over lemma/root-heldout cells, averaged across the three tasks:

| model | feature set | accuracy | above majority |
|---|---|---:|---:|
| llama-3.2-1b | full_prompt_final | 0.875 | +0.177 |
| llama-3.2-1b | metadata_free_prompt_final | 0.793 | +0.094 |
| llama-3.2-1b | target_final_subtoken | 0.735 | +0.036 |
| llama-3.2-1b | target_mean_span | 0.722 | +0.023 |
| qwen3-0.6b | full_prompt_final | 0.910 | +0.212 |
| qwen3-0.6b | metadata_free_prompt_final | 0.859 | +0.160 |
| qwen3-0.6b | target_final_subtoken | 0.874 | +0.175 |
| qwen3-0.6b | target_mean_span | 0.865 | +0.166 |

Interpretation for the planned lexical-leakage check: random splits are the optimistic reference; grouped lemma/root splits reduce performance, but the full-metadata prompt-final condition remains the strongest primary interface on average. The metadata-free and target-span conditions retain substantial signal, with task/model-specific cases recorded in the generated CSV rather than collapsed into a universal claim.

## Controls and representation diagnostics

- Shuffled controls: 360 cells; none exceeds its train-only majority baseline.
- BF16↔Q8 metric-direction agreement: 105/108 overlapping fold-0 comparisons.
- Direct BF16↔Q8 vector cosine/relative-L2 is not reported because the historical Q8 arrays are absent from the live checkout; no Q8 rerun was performed.
- CKA/RSA is retained as a fixed-sample diagnostic only; interface/pooling probe contrasts already answer the planned question, so it is not promoted to a headline result.

## Weight-free status

After capture, the planned probe, split, pooling, fragmentation, error, metric-direction, and diagnostic analyses consume the saved bundles/metadata and this run's derived artifacts; model weights are not required.
