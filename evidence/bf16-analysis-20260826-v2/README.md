# bf16-analysis-20260826-v2

Status: FROZEN — DO NOT MODIFY.

This freeze records the BF16 hidden-state bundles, weight-free local-analysis outputs, analysis code and split/config manifests, plus complete external hashes for every bundle file.

Verify with `sha256sum -c META_SHA256SUMS` and `sha256sum -c SHA256SUMS`.

Historical Q8 and lre-corrected-analysis freezes are untouched. Raw Q8 vector arrays remain an explicit blocker for direct BF16↔Q8 vector comparison when absent from the live checkout.
