# Arabic morphology probing: LRE reproducibility archive

This public repository contains the source, configuration, data provenance,
validated compact evidence, extraction code, analysis code, and manuscript for
the LRE study. The current scientific state is the completed BF16 analysis
freeze:

```text
evidence/bf16-analysis-20260826-v2/
```

The current manuscript is the official Springer Nature journal source and
compiled PDF:

```text
paper/paper1_lre_revision/main.tex
paper/paper1_lre_revision/main.pdf
```

The source paper uses the eleven-model BF16 matrix, three tasks, three split
conditions, four representation interfaces, five outer folds, development-only
layer selection, fixed Ridge alpha 1.0, and the recorded controls and
diagnostics. The compact freeze contains the generated result cells, split
manifests, predictions where applicable, validation records, and checksums.

## Reproduction

The committed compact evidence can be checked without model weights or GPU
inference:

```bash
python3 -m pip install -r requirements-bf16.lock.txt
make data-verify
make freeze-verify
make test
```

The manuscript can be compiled from its directory:

```bash
cd paper/paper1_lre_revision
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The cloud extraction and weight-free local-analysis commands are documented in
[`docs/bf16_extraction_runbook.md`](docs/bf16_extraction_runbook.md) and
[`docs/bf16_analysis_runbook.md`](docs/bf16_analysis_runbook.md). A new
extraction requires access to the pinned Hugging Face checkpoints and the raw
BF16 bundles. The completed compact analysis state itself requires neither.

## Data and model access

The normalized stimulus file and the archived UD Arabic-PADT source are
distributed under the licenses recorded in `data/provenance/` and
`data/upstream/`. Model weights, Hugging Face caches, credentials, and raw
hidden-state bundles are excluded. The exact external bundle files and hashes,
and the policy for gated or card-restricted model derivatives, are documented
in [`EXTERNAL_ARTIFACTS.md`](EXTERNAL_ARTIFACTS.md).

## License

Original code in this repository is released under the MIT License; see
[`LICENSE`](LICENSE). Third-party data, model metadata, model-derived artifacts,
and bundled Springer or upstream files retain their own applicable licenses
and access conditions.

## Scope and hygiene

The archive contains only material needed to inspect, validate, reproduce, or
submit the current BF16 paper state. It contains no model weights, cloud
credentials, Hugging Face tokens, agent/session logs, historical TACL working
trees, unrelated benchmark projects, or obsolete Q8 execution code.
