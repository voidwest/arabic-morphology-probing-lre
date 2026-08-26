# Secondary-analysis verification report

## 1. Inputs

- Reconstructed alpha=1 predictions: `results/main_alpha1_test_predictions.jsonl`.
- Frozen dataset and identifiers: `output/data/paper1_normalized.jsonl`.
- Frozen split files: each main model's `runs/revision/aws-final/models/<model>/splits/{lemma-heldout,root-heldout}/split.json`.
- Frozen representations: the `prompt_final.npy` and `target_final_subtoken.npy` files under each main model's `runs/revision/aws-final/models/<model>/models/<model>/features/` directory.
- Frozen aggregates used as the reproduction gate: each main model's `results/labels-real/<model>/<split>/<task>/<location>/result.json`.

The original pipeline did not serialize per-example predictions. With explicit authorization, they were deterministically reconstructed from the preserved representations and frozen partitions. No model representations were regenerated. All 24 alpha=1 selected layers, test accuracies, and test macro-F1 values matched the frozen aggregates exactly: **PASS**.

## 2. Pairing validation

All 12 model × task × split comparisons contained exactly 941 aligned test examples. Stable identifiers and gold labels matched exactly; there were no missing or duplicate identifiers. Exact rendered prompts form 458 clusters in lemma-heldout test folds and 469 clusters in root-heldout test folds. **PASS**.

## 3. Bootstrap protocol

- Master seed: `20260723` (deterministic cell seeds are SHA-256-derived from the master seed and comparison key).
- Replicates: `10000` per comparison.
- Sampling: paired exact-prompt clusters, stratified by gold-label class; the same sampled cluster multiplicities were used for both locations.
- Estimand: occurrence-weighted metrics; each selected cluster contributes all of its original rows.
- Class-cluster-count preservation: verified in every comparison and covered by automated tests. **PASS**.
- Sensitivity: the previous paired row bootstrap is retained inside the JSON output, but is not the primary interval because duplicated prompts violate row independence.

## 4–6. Paired location intervals

Delta is target-final-subtoken minus prompt-final.

| Model | Task | Split | Accuracy delta [95% CI] | Macro-F1 delta [95% CI] |
|---|---|---|---:|---:|
| llama-3.2-1b | gender | lemma-heldout | -0.1180 [-0.2195, -0.0029] | -0.1265 [-0.2294, -0.0137] |
| llama-3.2-1b | gender | root-heldout | -0.0606 [-0.1270, +0.0071] | -0.0642 [-0.1369, +0.0087] |
| llama-3.2-1b | number | lemma-heldout | -0.1668 [-0.2592, -0.0838] | -0.2502 [-0.3550, -0.1100] |
| llama-3.2-1b | number | root-heldout | -0.0914 [-0.1401, -0.0427] | -0.0475 [-0.1060, +0.0184] |
| llama-3.2-1b | pos | lemma-heldout | -0.0499 [-0.1196, +0.0235] | -0.1012 [-0.1734, -0.0293] |
| llama-3.2-1b | pos | root-heldout | -0.0956 [-0.1626, -0.0289] | -0.1032 [-0.1736, -0.0334] |
| qwen3-0.6b | gender | lemma-heldout | -0.0446 [-0.0881, -0.0033] | -0.0466 [-0.0927, -0.0030] |
| qwen3-0.6b | gender | root-heldout | +0.0542 [-0.0063, +0.1164] | +0.0498 [-0.0106, +0.1113] |
| qwen3-0.6b | number | lemma-heldout | +0.0096 [-0.0184, +0.0384] | +0.0647 [-0.0438, +0.1971] |
| qwen3-0.6b | number | root-heldout | +0.0255 [-0.0022, +0.0540] | +0.2003 [+0.0203, +0.3416] |
| qwen3-0.6b | pos | lemma-heldout | -0.0011 [-0.0569, +0.0593] | +0.0026 [-0.0666, +0.0758] |
| qwen3-0.6b | pos | root-heldout | +0.1169 [+0.0489, +0.1841] | +0.1691 [+0.0934, +0.2402] |

## 7. Alpha sensitivity: all 24 cells × 4 values

Each alpha independently selected a layer on development data, refit on train + development, and was evaluated once on test.

| Model | Task | Split | Location | Alpha | Layer | Accuracy | Macro-F1 | Δ accuracy vs 1 | Δ macro-F1 vs 1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b | pos | lemma-heldout | prompt_final | 0.1 | 12 | 0.8395 | 0.7910 | -0.0298 | -0.0402 |
| qwen3-0.6b | pos | lemma-heldout | prompt_final | 1 | 12 | 0.8693 | 0.8312 | +0.0000 | +0.0000 |
| qwen3-0.6b | pos | lemma-heldout | prompt_final | 10 | 17 | 0.9001 | 0.8743 | +0.0308 | +0.0431 |
| qwen3-0.6b | pos | lemma-heldout | prompt_final | 100 | 10 | 0.9065 | 0.8780 | +0.0372 | +0.0468 |
| qwen3-0.6b | gender | lemma-heldout | prompt_final | 0.1 | 24 | 0.8587 | 0.8530 | -0.0712 | -0.0715 |
| qwen3-0.6b | gender | lemma-heldout | prompt_final | 1 | 1 | 0.9299 | 0.9246 | +0.0000 | +0.0000 |
| qwen3-0.6b | gender | lemma-heldout | prompt_final | 10 | 3 | 0.9469 | 0.9426 | +0.0170 | +0.0180 |
| qwen3-0.6b | gender | lemma-heldout | prompt_final | 100 | 3 | 0.9543 | 0.9503 | +0.0244 | +0.0258 |
| qwen3-0.6b | number | lemma-heldout | prompt_final | 0.1 | 17 | 0.9384 | 0.7772 | +0.0021 | -0.0040 |
| qwen3-0.6b | number | lemma-heldout | prompt_final | 1 | 24 | 0.9362 | 0.7813 | +0.0000 | +0.0000 |
| qwen3-0.6b | number | lemma-heldout | prompt_final | 10 | 13 | 0.9416 | 0.7816 | +0.0053 | +0.0004 |
| qwen3-0.6b | number | lemma-heldout | prompt_final | 100 | 24 | 0.9532 | 0.7191 | +0.0170 | -0.0621 |
| qwen3-0.6b | pos | root-heldout | prompt_final | 0.1 | 13 | 0.7216 | 0.6667 | -0.0032 | -0.0027 |
| qwen3-0.6b | pos | root-heldout | prompt_final | 1 | 13 | 0.7248 | 0.6695 | +0.0000 | +0.0000 |
| qwen3-0.6b | pos | root-heldout | prompt_final | 10 | 9 | 0.8151 | 0.8188 | +0.0903 | +0.1493 |
| qwen3-0.6b | pos | root-heldout | prompt_final | 100 | 11 | 0.7853 | 0.7822 | +0.0606 | +0.1128 |
| qwen3-0.6b | gender | root-heldout | prompt_final | 0.1 | 3 | 0.7991 | 0.7980 | -0.0064 | -0.0063 |
| qwen3-0.6b | gender | root-heldout | prompt_final | 1 | 3 | 0.8055 | 0.8043 | +0.0000 | +0.0000 |
| qwen3-0.6b | gender | root-heldout | prompt_final | 10 | 24 | 0.8799 | 0.8740 | +0.0744 | +0.0697 |
| qwen3-0.6b | gender | root-heldout | prompt_final | 100 | 5 | 0.8767 | 0.8740 | +0.0712 | +0.0697 |
| qwen3-0.6b | number | root-heldout | prompt_final | 0.1 | 17 | 0.9341 | 0.6650 | +0.0011 | +0.0963 |
| qwen3-0.6b | number | root-heldout | prompt_final | 1 | 17 | 0.9330 | 0.5687 | +0.0000 | +0.0000 |
| qwen3-0.6b | number | root-heldout | prompt_final | 10 | 19 | 0.9766 | 0.7283 | +0.0436 | +0.1596 |
| qwen3-0.6b | number | root-heldout | prompt_final | 100 | 22 | 0.9309 | 0.5730 | -0.0021 | +0.0043 |
| qwen3-0.6b | pos | lemma-heldout | target_final_subtoken | 0.1 | 18 | 0.8597 | 0.8221 | -0.0085 | -0.0116 |
| qwen3-0.6b | pos | lemma-heldout | target_final_subtoken | 1 | 18 | 0.8682 | 0.8337 | +0.0000 | +0.0000 |
| qwen3-0.6b | pos | lemma-heldout | target_final_subtoken | 10 | 19 | 0.8799 | 0.8382 | +0.0117 | +0.0044 |
| qwen3-0.6b | pos | lemma-heldout | target_final_subtoken | 100 | 20 | 0.9044 | 0.8752 | +0.0361 | +0.0414 |
| qwen3-0.6b | gender | lemma-heldout | target_final_subtoken | 0.1 | 18 | 0.8512 | 0.8429 | -0.0340 | -0.0351 |
| qwen3-0.6b | gender | lemma-heldout | target_final_subtoken | 1 | 17 | 0.8852 | 0.8780 | +0.0000 | +0.0000 |
| qwen3-0.6b | gender | lemma-heldout | target_final_subtoken | 10 | 17 | 0.8969 | 0.8904 | +0.0117 | +0.0124 |
| qwen3-0.6b | gender | lemma-heldout | target_final_subtoken | 100 | 17 | 0.8990 | 0.8907 | +0.0138 | +0.0127 |
| qwen3-0.6b | number | lemma-heldout | target_final_subtoken | 0.1 | 5 | 0.9384 | 0.7972 | -0.0074 | -0.0488 |
| qwen3-0.6b | number | lemma-heldout | target_final_subtoken | 1 | 11 | 0.9458 | 0.8460 | +0.0000 | +0.0000 |
| qwen3-0.6b | number | lemma-heldout | target_final_subtoken | 10 | 12 | 0.9458 | 0.8436 | +0.0000 | -0.0023 |
| qwen3-0.6b | number | lemma-heldout | target_final_subtoken | 100 | 17 | 0.9702 | 0.9009 | +0.0244 | +0.0549 |
| qwen3-0.6b | pos | root-heldout | target_final_subtoken | 0.1 | 11 | 0.8395 | 0.8347 | -0.0021 | -0.0039 |
| qwen3-0.6b | pos | root-heldout | target_final_subtoken | 1 | 11 | 0.8417 | 0.8386 | +0.0000 | +0.0000 |
| qwen3-0.6b | pos | root-heldout | target_final_subtoken | 10 | 14 | 0.8502 | 0.8452 | +0.0085 | +0.0066 |
| qwen3-0.6b | pos | root-heldout | target_final_subtoken | 100 | 21 | 0.8990 | 0.8994 | +0.0574 | +0.0608 |
| qwen3-0.6b | gender | root-heldout | target_final_subtoken | 0.1 | 0 | 0.8193 | 0.8130 | -0.0404 | -0.0411 |
| qwen3-0.6b | gender | root-heldout | target_final_subtoken | 1 | 17 | 0.8597 | 0.8541 | +0.0000 | +0.0000 |
| qwen3-0.6b | gender | root-heldout | target_final_subtoken | 10 | 17 | 0.9086 | 0.9040 | +0.0489 | +0.0499 |
| qwen3-0.6b | gender | root-heldout | target_final_subtoken | 100 | 23 | 0.9129 | 0.9079 | +0.0531 | +0.0538 |
| qwen3-0.6b | number | root-heldout | target_final_subtoken | 0.1 | 16 | 0.9575 | 0.7676 | -0.0011 | -0.0015 |
| qwen3-0.6b | number | root-heldout | target_final_subtoken | 1 | 16 | 0.9586 | 0.7691 | +0.0000 | +0.0000 |
| qwen3-0.6b | number | root-heldout | target_final_subtoken | 10 | 20 | 0.9532 | 0.8144 | -0.0053 | +0.0453 |
| qwen3-0.6b | number | root-heldout | target_final_subtoken | 100 | 11 | 0.9702 | 0.7872 | +0.0117 | +0.0182 |
| llama-3.2-1b | pos | lemma-heldout | prompt_final | 0.1 | 6 | 0.6674 | 0.6359 | -0.0871 | -0.0966 |
| llama-3.2-1b | pos | lemma-heldout | prompt_final | 1 | 10 | 0.7545 | 0.7325 | +0.0000 | +0.0000 |
| llama-3.2-1b | pos | lemma-heldout | prompt_final | 10 | 10 | 0.8278 | 0.7987 | +0.0733 | +0.0662 |
| llama-3.2-1b | pos | lemma-heldout | prompt_final | 100 | 6 | 0.9224 | 0.9063 | +0.1679 | +0.1738 |
| llama-3.2-1b | gender | lemma-heldout | prompt_final | 0.1 | 11 | 0.8108 | 0.7983 | -0.0733 | -0.0797 |
| llama-3.2-1b | gender | lemma-heldout | prompt_final | 1 | 0 | 0.8842 | 0.8780 | +0.0000 | +0.0000 |
| llama-3.2-1b | gender | lemma-heldout | prompt_final | 10 | 10 | 0.9405 | 0.9353 | +0.0563 | +0.0573 |
| llama-3.2-1b | gender | lemma-heldout | prompt_final | 100 | 5 | 0.9681 | 0.9655 | +0.0840 | +0.0875 |
| llama-3.2-1b | number | lemma-heldout | prompt_final | 0.1 | 6 | 0.8077 | 0.5946 | -0.0999 | -0.1874 |
| llama-3.2-1b | number | lemma-heldout | prompt_final | 1 | 7 | 0.9075 | 0.7820 | +0.0000 | +0.0000 |
| llama-3.2-1b | number | lemma-heldout | prompt_final | 10 | 10 | 0.9607 | 0.8075 | +0.0531 | +0.0255 |
| llama-3.2-1b | number | lemma-heldout | prompt_final | 100 | 10 | 0.9564 | 0.8302 | +0.0489 | +0.0482 |
| llama-3.2-1b | pos | root-heldout | prompt_final | 0.1 | 9 | 0.6939 | 0.6851 | -0.0755 | -0.0869 |
| llama-3.2-1b | pos | root-heldout | prompt_final | 1 | 9 | 0.7694 | 0.7721 | +0.0000 | +0.0000 |
| llama-3.2-1b | pos | root-heldout | prompt_final | 10 | 7 | 0.8406 | 0.8465 | +0.0712 | +0.0744 |
| llama-3.2-1b | pos | root-heldout | prompt_final | 100 | 8 | 0.8130 | 0.8115 | +0.0436 | +0.0395 |
| llama-3.2-1b | gender | root-heldout | prompt_final | 0.1 | 13 | 0.6929 | 0.6923 | -0.1180 | -0.1012 |
| llama-3.2-1b | gender | root-heldout | prompt_final | 1 | 1 | 0.8108 | 0.7935 | +0.0000 | +0.0000 |
| llama-3.2-1b | gender | root-heldout | prompt_final | 10 | 1 | 0.8937 | 0.8835 | +0.0829 | +0.0900 |
| llama-3.2-1b | gender | root-heldout | prompt_final | 100 | 3 | 0.8778 | 0.8749 | +0.0670 | +0.0815 |
| llama-3.2-1b | number | root-heldout | prompt_final | 0.1 | 7 | 0.8151 | 0.5229 | -0.0903 | -0.0145 |
| llama-3.2-1b | number | root-heldout | prompt_final | 1 | 6 | 0.9054 | 0.5374 | +0.0000 | +0.0000 |
| llama-3.2-1b | number | root-heldout | prompt_final | 10 | 12 | 0.8672 | 0.5104 | -0.0383 | -0.0269 |
| llama-3.2-1b | number | root-heldout | prompt_final | 100 | 6 | 0.9745 | 0.7937 | +0.0691 | +0.2563 |
| llama-3.2-1b | pos | lemma-heldout | target_final_subtoken | 0.1 | 7 | 0.6716 | 0.5944 | -0.0329 | -0.0369 |
| llama-3.2-1b | pos | lemma-heldout | target_final_subtoken | 1 | 7 | 0.7046 | 0.6313 | +0.0000 | +0.0000 |
| llama-3.2-1b | pos | lemma-heldout | target_final_subtoken | 10 | 8 | 0.8172 | 0.7727 | +0.1126 | +0.1414 |
| llama-3.2-1b | pos | lemma-heldout | target_final_subtoken | 100 | 8 | 0.8820 | 0.8536 | +0.1775 | +0.2223 |
| llama-3.2-1b | gender | lemma-heldout | target_final_subtoken | 0.1 | 12 | 0.6493 | 0.6398 | -0.1169 | -0.1117 |
| llama-3.2-1b | gender | lemma-heldout | target_final_subtoken | 1 | 12 | 0.7662 | 0.7515 | +0.0000 | +0.0000 |
| llama-3.2-1b | gender | lemma-heldout | target_final_subtoken | 10 | 7 | 0.8321 | 0.8183 | +0.0659 | +0.0668 |
| llama-3.2-1b | gender | lemma-heldout | target_final_subtoken | 100 | 13 | 0.9203 | 0.9141 | +0.1541 | +0.1627 |
| llama-3.2-1b | number | lemma-heldout | target_final_subtoken | 0.1 | 14 | 0.6780 | 0.4755 | -0.0627 | -0.0563 |
| llama-3.2-1b | number | lemma-heldout | target_final_subtoken | 1 | 14 | 0.7407 | 0.5318 | +0.0000 | +0.0000 |
| llama-3.2-1b | number | lemma-heldout | target_final_subtoken | 10 | 14 | 0.9054 | 0.8194 | +0.1647 | +0.2876 |
| llama-3.2-1b | number | lemma-heldout | target_final_subtoken | 100 | 9 | 0.9724 | 0.9056 | +0.2317 | +0.3738 |
| llama-3.2-1b | pos | root-heldout | target_final_subtoken | 0.1 | 3 | 0.5717 | 0.5518 | -0.1020 | -0.1170 |
| llama-3.2-1b | pos | root-heldout | target_final_subtoken | 1 | 9 | 0.6738 | 0.6688 | +0.0000 | +0.0000 |
| llama-3.2-1b | pos | root-heldout | target_final_subtoken | 10 | 13 | 0.8162 | 0.8046 | +0.1424 | +0.1357 |
| llama-3.2-1b | pos | root-heldout | target_final_subtoken | 100 | 15 | 0.8480 | 0.8485 | +0.1743 | +0.1797 |
| llama-3.2-1b | gender | root-heldout | target_final_subtoken | 0.1 | 12 | 0.6950 | 0.6763 | -0.0553 | -0.0529 |
| llama-3.2-1b | gender | root-heldout | target_final_subtoken | 1 | 12 | 0.7503 | 0.7293 | +0.0000 | +0.0000 |
| llama-3.2-1b | gender | root-heldout | target_final_subtoken | 10 | 12 | 0.8757 | 0.8685 | +0.1254 | +0.1392 |
| llama-3.2-1b | gender | root-heldout | target_final_subtoken | 100 | 14 | 0.8948 | 0.8885 | +0.1445 | +0.1592 |
| llama-3.2-1b | number | root-heldout | target_final_subtoken | 0.1 | 4 | 0.7396 | 0.4451 | -0.0744 | -0.0448 |
| llama-3.2-1b | number | root-heldout | target_final_subtoken | 1 | 4 | 0.8140 | 0.4899 | +0.0000 | +0.0000 |
| llama-3.2-1b | number | root-heldout | target_final_subtoken | 10 | 4 | 0.8810 | 0.5847 | +0.0670 | +0.0948 |
| llama-3.2-1b | number | root-heldout | target_final_subtoken | 100 | 14 | 0.9766 | 0.7982 | +0.1626 | +0.3083 |

## 8. Interpretation changes under alpha sensitivity

- The alpha=1 point estimates and primary table remain frozen, but the sensitivity analysis does **not** support calling every location conclusion regularization-invariant.
- Qwen root-heldout POS retains the target-final accuracy advantage at all four alpha values.
- Qwen lemma-heldout POS remains a small location contrast, although its sign changes at alpha=0.1.
- Llama target-token number is below majority at alpha=0.1 and 1, but above majority at alpha=10 and 100; root-heldout prompt-final number also crosses the majority threshold across the grid.
- Several other Llama location directions change, and selected layers vary frequently. These facts are disclosed in the manuscript and full CSV/JSON.
- Ill-conditioned-matrix warnings occurred for weakly regularized fits. The estimators completed; the warning is retained as a numerical sensitivity caveat.

## 9. Manuscript sentences changed because of bootstrap evidence

- Qwen lemma-heldout POS now reports both paired deltas and intervals and says the estimates are compatible with little or no location difference.
- “The location difference reverses sharply” was replaced with “Target-final POS accuracy was substantially higher,” followed by its accuracy and macro-F1 intervals.
- Llama number now reports the paired declines and intervals under both lexical splits.
- The Results table introduction states that intervals are conditional finite-test-set estimates, not causal or architectural tests.
- Discussion §6.1 contrasts the near-zero Qwen lemma-POS intervals with intervals excluding zero for Qwen root-POS and both Llama number-accuracy declines.

## 10–11. Frozen table and hyperparameter-selection checks

- Table 1 point estimates unchanged and exactly reproduced: **PASS**.
- Test-set hyperparameter selection: **none**. Alpha remains fixed at 1.0 in the primary analysis; every sensitivity alpha uses development-only layer selection and is reported rather than selected.

## 12. Tests

- Focused secondary-analysis plus revision tests: 38 passed, 0 failed.
- Top-level project suite: 76 passed plus 97 subtests, 0 failed.
- An unscoped workspace-wide collection also entered vendored projects and stopped on three unavailable optional dependencies (`parity_tools`, Appium, and `wget`); this did not affect the scoped project suite.

## 13. Remaining issues

- Cluster-bootstrap intervals are conditional on the reconstructed fixed probes and the exact-prompt cluster definition; they do not include probe retraining, alternative partitions, layer-selection uncertainty, model seeds, prompts, or datasets.
- Alpha sensitivity reveals substantial estimator dependence in some Llama cells; the manuscript now limits those claims to the frozen alpha=1 protocol.
- Per-example predictions were reconstructed because the production run retained representations and aggregates but not prediction rows. Exact aggregate reproduction is the validation gate.
