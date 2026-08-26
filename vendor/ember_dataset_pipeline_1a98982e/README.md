# Historical Paper 1 dataset pipeline snapshot

This directory contains an exact source snapshot from the local Ember
repository at commit:

```text
1a98982e3e21c1e4e98e86025bd2d823d3ff0574
```

The snapshot is vendored to preserve the executable behavior that generated
the frozen Paper 1 dataset. Do not silently modernize these files. Corrections
belong in the current `research-stack` wrapper or analysis code and must remain
distinguishable from the historical transformation.

Included paths:

- `scripts/export_camel_disambiguated_padt.py`
- `scripts/arabic_morph_dataset.py`
- `src/arabic_morph_dataset/`
- `configs/arabic_morph_disambig_padt_5000_strict.toml`

The authoritative file hashes, upstream inputs, dependency versions, known
number-label quirk, and rebuild instructions are recorded in:

- `data/provenance/paper1_dataset_manifest.json`
- `docs/paper1_dataset_provenance.md`
- `scripts/data/rebuild_paper1_dataset.py`

The original config retains paths relative to the Ember repository because it
is an archival source file. The rebuild wrapper supplies the paths in the
current repository and verifies the exact output hashes.
