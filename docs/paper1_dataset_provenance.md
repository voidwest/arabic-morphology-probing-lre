# Paper 1 dataset provenance and exact rebuild

## Bottom line

The Paper 1 dataset is recoverable from an official upstream source and the
complete transformation code is now in this repository.

The authoritative frozen experiment input is:

```text
output/data/paper1_normalized.jsonl
records: 4,701
bytes:   2,297,339
SHA-256: 9098be4e60390f50686a4cee277b0cb4ab6a1e4bee73aeddd77106f9bdb7d719
```

The exact source **content** has been identified. The original unversioned copy
at `/home/west/ember/data/ud/UD_Arabic-PADT/ar_padt-ud-train.conllu` has SHA-256
`f0b56962...d65e85`. That hash matches the official UD Arabic-PADT training file
in both release `r2.17` (commit `e9744f8e...`) and `r2.18` (commit
`5b1ad7e8...`). Release `r2.16` differs. Because the original local directory
did not retain Git metadata, it is impossible to prove whether its label was
`r2.17` or `r2.18`. This repository pins the earliest matching release,
`r2.17`, while recording the content hash as the authoritative identity. The
two matching releases produce the same Paper 1 input because their relevant
training files are byte-identical.

The machine-readable record is
[`data/provenance/paper1_dataset_manifest.json`](../data/provenance/paper1_dataset_manifest.json).

## Upstream acquisition

The source is the official [UD Arabic-PADT repository](https://github.com/UniversalDependencies/UD_Arabic-PADT).
The archived checkout is under `data/upstream/UD_Arabic-PADT-r2.17/`.

```bash
git clone --depth 1 --branch r2.17 \
  https://github.com/UniversalDependencies/UD_Arabic-PADT.git \
  data/upstream/UD_Arabic-PADT-r2.17

sha256sum data/upstream/UD_Arabic-PADT-r2.17/ar_padt-ud-train.conllu
# f0b56962b340f5325bb5d293f47717522e706112c19029e82c225355c6d65e85
```

Only `ar_padt-ud-train.conllu` enters the Paper 1 build. The upstream `dev` and
`test` files are retained for completeness but are not used. “Train,” “dev,”
and “test” in Paper 1 refer to newly constructed morphology-probe partitions,
not the original UD partitions.

Arabic-PADT is derived from the Prague Arabic Dependency Treebank. Its bundled
README describes it as predominantly Modern Standard Arabic newswire and
documents the annotation provenance. The upstream files are distributed under
CC BY-NC-SA 3.0; see
[`data/upstream/UD_Arabic-PADT-r2.17/LICENSE.txt`](../data/upstream/UD_Arabic-PADT-r2.17/LICENSE.txt)
and preserve the upstream attribution and license when distributing the data
or processed derivatives. This paragraph records the artifact's license; it is
not legal advice.

## CAMeL disambiguation

The build uses:

- `camel-tools==1.6.0`;
- MLE disambiguator `calima-msa-r13`, package
  `disambig-mle-calima-msa-r13==0.2.5`;
- morphology database `morphology-db-msa-r13==0.4.0`;
- `top=1` analysis;
- full-sentence disambiguation;
- PADT source-UPOS eligibility restricted to `NOUN`, `VERB`, and `ADJ`;
- source order, stopping after 5,000 eligible records.

The exact model/database file and package-archive hashes are in the manifest.
Install the pinned software and data with:

```bash
python -m venv .venv-paper1-data
.venv-paper1-data/bin/python -m pip install 'camel-tools==1.6.0'
.venv-paper1-data/bin/camel_data -i disambig-mle-calima-msa-r13
```

The exporter consumes 330 PADT sentence units and sees 11,437 tokens before
writing 5,000 records. The exact intermediate hash is:

```text
056e13b3778d58149f8eca4c2df9fd975d933bbc1a5ea73b4690b4af0a3ba42c
```

The current hardened exporter in the old Ember tree was also tested. It
produced identical linguistic content in all 5,000 rows; its only byte-level
difference was a renamed provenance string. The vendored historical exporter
is used here because it reproduces the frozen bytes, including the original
underscore spelling of that string.

## Filtering, normalization, and splitting

The exact historical code snapshot is vendored at
`vendor/ember_dataset_pipeline_1a98982e/`, from Ember commit
`1a98982e3e21c1e4e98e86025bd2d823d3ff0574`.

The pipeline then:

1. normalizes CAMeL fields and values;
2. requires nonempty lemma, root, abstract pattern, and concrete pattern;
3. drops explicitly ambiguous records;
4. keeps normalized POS `NOUN`, `VERB`, or `ADJ` (299 other records are
   removed, leaving 4,701);
5. constructs a `root_heldout` split with seed 17 and ratios 0.70/0.15/0.15;
6. sorts probe rows deterministically;
7. appends sequential `stim-NNNN` identifiers plus top-level gender and number
   fields.

The historical `root_heldout` implementation groups by both lemma and root.
The resulting counts are train 3,291, dev 705, and test 705, with zero direct
lemma or root overlap across every pair of partitions.

## Important historical label quirk

The frozen file contains 45 number labels spelled `def`. These records are
CAMeL `num=d`, meaning **dual**, not definiteness. The historical normalizer
used one global alias table and mapped `d` to `def`, a mapping intended for the
state/definiteness feature.

This is not hidden or silently rewritten:

- the authoritative frozen JSONL retains `def`, preserving the exact input
  used in the experiment;
- current analysis code maps `number:def` to the semantic class `du` when the
  dataset is loaded;
- the manuscript's number classes (`sg`, `pl`, `du`) describe the semantic
  classes after that explicit task-specific alias;
- 45 of 4,699 number-labeled records are affected.

## Exact rebuild

From the repository root, with the pinned CAMeL environment active:

```bash
python scripts/data/rebuild_paper1_dataset.py \
  --output-dir output/data/paper1_dataset_rebuild
```

The script refuses to reuse an existing output directory, verifies upstream
and CAMeL artifact hashes before processing, verifies every authoritative stage
hash afterward, and does not touch the frozen dataset. To rebuild downstream
steps from a previously verified CAMeL intermediate (this mode does not
require CAMeL Tools or its model files):

```bash
python scripts/data/rebuild_paper1_dataset.py \
  --intermediate /path/to/camel_disambig_msa_padt_5000.jsonl \
  --output-dir output/data/paper1_dataset_rebuild_from_intermediate
```

Only after all hashes pass can the result be installed explicitly:

```bash
python scripts/data/rebuild_paper1_dataset.py \
  --output-dir output/data/paper1_dataset_rebuild_install \
  --install
```

Expected authoritative stage hashes:

| Stage | SHA-256 |
|---|---|
| Official PADT train input | `f0b56962b340f5325bb5d293f47717522e706112c19029e82c225355c6d65e85` |
| 5,000-row CAMeL JSONL | `056e13b3778d58149f8eca4c2df9fd975d933bbc1a5ea73b4690b4af0a3ba42c` |
| 4,701-row canonical JSONL | `2a6ff31ce0e77e22b0360d22a7f5f0ec36630957a91ac694c0722fdad7b941fc` |
| 4,701-row probe JSONL | `0082ef10554f94996e4faf34cb703a39e373fe8a49697a8534cdf085cf6b9708` |
| Paper 1 JSONL | `9098be4e60390f50686a4cee277b0cb4ab6a1e4bee73aeddd77106f9bdb7d719` |

## What was independently checked

- The recovered upstream train file is byte-identical to the old unversioned
  source file.
- A fresh CAMeL run reproduced every linguistic field in all 5,000 rows.
- Using the historical provenance-string spelling reproduced the intermediate
  byte-for-byte.
- The historical normalization/filter/split pipeline reproduced the canonical
  dataset byte-for-byte.
- The paper adapter transformation was independently reconstructed from the
  historical probe output and reproduces the frozen Paper 1 JSONL byte-for-byte.
- Dataset counts, class counts, unique lemma/root counts, and cross-partition
  intersections were recomputed independently.

## Artifact identity versus semantic corrections

Do not edit `output/data/paper1_normalized.jsonl` to “clean up” labels or JSON
formatting. Its hash is part of the scientific record. Corrections such as
`number:def -> number:du` belong in task-specific loading code and must remain
documented. A future newly versioned dataset may encode `du` directly, but it
would be a derivative artifact with a new hash, not the frozen experiment
input.
