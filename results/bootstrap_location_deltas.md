# Paired representation-location bootstrap

Delta is `target_final_subtoken - prompt_final`. Intervals are percentile 95% intervals from a gold-class-stratified paired bootstrap over exact rendered-prompt clusters.

| Model | Task | Split | Metric | Prompt | Target | Delta | 95% CI | SE | P(Δ>0) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| llama-3.2-1b | gender | lemma-heldout | accuracy | 0.8842 | 0.7662 | -0.1180 | [-0.2195, -0.0029] | 0.0551 | 0.0215 |
| llama-3.2-1b | gender | lemma-heldout | macro_f1 | 0.8780 | 0.7515 | -0.1265 | [-0.2294, -0.0137] | 0.0548 | 0.0141 |
| llama-3.2-1b | gender | root-heldout | accuracy | 0.8108 | 0.7503 | -0.0606 | [-0.1270, +0.0071] | 0.0341 | 0.0385 |
| llama-3.2-1b | gender | root-heldout | macro_f1 | 0.7935 | 0.7293 | -0.0642 | [-0.1369, +0.0087] | 0.0370 | 0.0428 |
| llama-3.2-1b | number | lemma-heldout | accuracy | 0.9075 | 0.7407 | -0.1668 | [-0.2592, -0.0838] | 0.0448 | 0.0000 |
| llama-3.2-1b | number | lemma-heldout | macro_f1 | 0.7820 | 0.5318 | -0.2502 | [-0.3550, -0.1100] | 0.0622 | 0.0005 |
| llama-3.2-1b | number | root-heldout | accuracy | 0.9054 | 0.8140 | -0.0914 | [-0.1401, -0.0427] | 0.0251 | 0.0001 |
| llama-3.2-1b | number | root-heldout | macro_f1 | 0.5374 | 0.4899 | -0.0475 | [-0.1060, +0.0184] | 0.0315 | 0.0716 |
| llama-3.2-1b | pos | lemma-heldout | accuracy | 0.7545 | 0.7046 | -0.0499 | [-0.1196, +0.0235] | 0.0363 | 0.0825 |
| llama-3.2-1b | pos | lemma-heldout | macro_f1 | 0.7325 | 0.6313 | -0.1012 | [-0.1734, -0.0293] | 0.0364 | 0.0029 |
| llama-3.2-1b | pos | root-heldout | accuracy | 0.7694 | 0.6738 | -0.0956 | [-0.1626, -0.0289] | 0.0343 | 0.0013 |
| llama-3.2-1b | pos | root-heldout | macro_f1 | 0.7721 | 0.6688 | -0.1032 | [-0.1736, -0.0334] | 0.0358 | 0.0014 |
| qwen3-0.6b | gender | lemma-heldout | accuracy | 0.9299 | 0.8852 | -0.0446 | [-0.0881, -0.0033] | 0.0216 | 0.0143 |
| qwen3-0.6b | gender | lemma-heldout | macro_f1 | 0.9246 | 0.8780 | -0.0466 | [-0.0927, -0.0030] | 0.0228 | 0.0169 |
| qwen3-0.6b | gender | root-heldout | accuracy | 0.8055 | 0.8597 | +0.0542 | [-0.0063, +0.1164] | 0.0313 | 0.9565 |
| qwen3-0.6b | gender | root-heldout | macro_f1 | 0.8043 | 0.8541 | +0.0498 | [-0.0106, +0.1113] | 0.0311 | 0.9443 |
| qwen3-0.6b | number | lemma-heldout | accuracy | 0.9362 | 0.9458 | +0.0096 | [-0.0184, +0.0384] | 0.0145 | 0.7327 |
| qwen3-0.6b | number | lemma-heldout | macro_f1 | 0.7813 | 0.8460 | +0.0647 | [-0.0438, +0.1971] | 0.0628 | 0.8559 |
| qwen3-0.6b | number | root-heldout | accuracy | 0.9330 | 0.9586 | +0.0255 | [-0.0022, +0.0540] | 0.0143 | 0.9620 |
| qwen3-0.6b | number | root-heldout | macro_f1 | 0.5687 | 0.7691 | +0.2003 | [+0.0203, +0.3416] | 0.0830 | 0.9938 |
| qwen3-0.6b | pos | lemma-heldout | accuracy | 0.8693 | 0.8682 | -0.0011 | [-0.0569, +0.0593] | 0.0294 | 0.4507 |
| qwen3-0.6b | pos | lemma-heldout | macro_f1 | 0.8312 | 0.8337 | +0.0026 | [-0.0666, +0.0758] | 0.0361 | 0.4885 |
| qwen3-0.6b | pos | root-heldout | accuracy | 0.7248 | 0.8417 | +0.1169 | [+0.0489, +0.1841] | 0.0346 | 0.9998 |
| qwen3-0.6b | pos | root-heldout | macro_f1 | 0.6695 | 0.8386 | +0.1691 | [+0.0934, +0.2402] | 0.0378 | 1.0000 |
