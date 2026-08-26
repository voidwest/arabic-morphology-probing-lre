#!/usr/bin/env python3
"""Rebuild the exact Paper 1 morphology dataset from UD Arabic-PADT.

This deliberately vendors and executes the historical dataset code used for
the frozen experiment.  It verifies every authoritative input and output hash
and never replaces the frozen dataset unless ``--install`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "ember_dataset_pipeline_1a98982e"
sys.path.insert(0, str(VENDOR / "src"))

from arabic_morph_dataset.exporters import make_probe_records, make_sft_examples  # noqa: E402
from arabic_morph_dataset.filters import apply_filters  # noqa: E402
from arabic_morph_dataset.io import (  # noqa: E402
    read_raw_records,
    write_json,
    write_jsonl,
    write_morph_records,
)
from arabic_morph_dataset.normalize import normalize_records  # noqa: E402
from arabic_morph_dataset.report import make_summary_report  # noqa: E402
from arabic_morph_dataset.split import split_records  # noqa: E402
from arabic_morph_dataset.stats import dataset_stats  # noqa: E402
from arabic_morph_dataset.validate import (  # noqa: E402
    validate_canonical,
    validate_probe_records,
    validate_sft_examples,
)

_exporter_spec = importlib.util.spec_from_file_location(
    "paper1_historical_camel_exporter",
    VENDOR / "scripts/export_camel_disambiguated_padt.py",
)
if _exporter_spec is None or _exporter_spec.loader is None:
    raise RuntimeError("could not load the vendored historical CAMeL exporter")
_exporter = importlib.util.module_from_spec(_exporter_spec)
sys.modules[_exporter_spec.name] = _exporter
_exporter_spec.loader.exec_module(_exporter)
export_sentences = _exporter.export_sentences
iter_conllu = _exporter.iter_conllu


EXPECTED = {
    "padt_train": "f0b56962b340f5325bb5d293f47717522e706112c19029e82c225355c6d65e85",
    "camel_model": "e7d79e7744101b6933f9a4b41063d4b9934488fc8c8b6130a24467e0f0dfe9af",
    "camel_morphology_db": "195bc25a333237a2126470da888d7936b59ed3729f9210e0a4194ba43497dd70",
    "camel_jsonl": "056e13b3778d58149f8eca4c2df9fd975d933bbc1a5ea73b4690b4af0a3ba42c",
    "camel_report": "077398963106922ffd9a22560ec1ec1cd73d347a802c138dfb4fff42d2f090d6",
    "canonical": "2a6ff31ce0e77e22b0360d22a7f5f0ec36630957a91ac694c0722fdad7b941fc",
    "probes": "0082ef10554f94996e4faf34cb703a39e373fe8a49697a8534cdf085cf6b9708",
    "paper1": "9098be4e60390f50686a4cee277b0cb4ab6a1e4bee73aeddd77106f9bdb7d719",
}

SOURCE_NAME = "camel_tools_disambig_mle_calima_msa_r13"
FILTERS = {
    "drop_missing_root": True,
    "require_abstract_pattern": True,
    "require_concrete_pattern": True,
    "drop_missing_lemma": True,
    "drop_ambiguous": True,
    "pos_allowlist": ["NOUN", "VERB", "ADJ"],
    "min_examples_per_root": 1,
    "min_examples_per_pattern": 1,
    "max_examples_per_root": 0,
    "max_examples_per_pattern": 0,
}
RATIOS = {"train": 0.7, "dev": 0.15, "test": 0.15}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{description} hash mismatch: expected {expected}, got {actual} ({path})"
        )
    return actual


def verify_camel_environment(camel_data: Path) -> dict[str, str]:
    version = importlib.metadata.version("camel-tools")
    if version != "1.6.0":
        raise RuntimeError(f"camel-tools must be 1.6.0, found {version}")
    model = camel_data / "data" / "disambig_mle" / "calima-msa-r13" / "model.json"
    morphology_db = camel_data / "data" / "morphology_db" / "calima-msa-r13" / "morphology.db"
    require_hash(model, EXPECTED["camel_model"], "CAMeL MLE model")
    require_hash(morphology_db, EXPECTED["camel_morphology_db"], "CAMeL morphology database")
    return {
        "camel_tools_version": version,
        "camel_data_root": str(camel_data.resolve()),
        "disambiguator_model_path": str(model.resolve()),
        "disambiguator_model_sha256": EXPECTED["camel_model"],
        "morphology_database_path": str(morphology_db.resolve()),
        "morphology_database_sha256": EXPECTED["camel_morphology_db"],
    }


def write_paper1_jsonl(probe_rows: list[dict], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, source_row in enumerate(probe_rows, start=1):
            # The historical adapter read the sort_keys=True probes JSONL back
            # from disk before appending these fields, so preserve that exact
            # insertion order for a byte-identical artifact.
            row = dict(sorted(source_row.items()))
            row["id"] = f"stim-{index:04d}"
            row["gender"] = row["features"].get("gender", "")
            row["number"] = row["features"].get("number", "")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def install_atomically(source: Path, destination: Path) -> str | None:
    previous = sha256_file(destination) if destination.is_file() else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.install-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    require_hash(destination, EXPECTED["paper1"], "installed Paper 1 dataset")
    return previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        type=Path,
        default=ROOT / "data/upstream/UD_Arabic-PADT-r2.17/ar_padt-ud-train.conllu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/data/paper1_dataset_rebuild",
    )
    parser.add_argument(
        "--camel-data",
        type=Path,
        default=Path.home() / ".camel_tools",
        help="CAMeL Tools data root containing data/disambig_mle and data/morphology_db",
    )
    parser.add_argument(
        "--intermediate",
        type=Path,
        help="reuse an already verified 5,000-row CAMeL JSONL instead of rerunning CAMeL",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="atomically replace output/data/paper1_normalized.jsonl after all hashes pass",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing output directory: {output_dir}; choose a new --output-dir"
        )
    output_dir.mkdir(parents=True)

    upstream = args.upstream.resolve()
    require_hash(upstream, EXPECTED["padt_train"], "UD Arabic-PADT r2.17 train file")

    camel_jsonl = output_dir / "camel_disambig_msa_padt_5000.jsonl"
    camel_report = output_dir / "camel_disambig_msa_padt_5000_report.json"
    if args.intermediate:
        require_hash(args.intermediate.resolve(), EXPECTED["camel_jsonl"], "CAMeL intermediate")
        shutil.copyfile(args.intermediate.resolve(), camel_jsonl)
        environment = {
            "camel_environment_verification": "not_required_for_verified_intermediate"
        }
        export_mode = "verified_existing_intermediate"
    else:
        environment = verify_camel_environment(args.camel_data.resolve())
        report = export_sentences(
            sentences=iter_conllu(upstream),
            output=camel_jsonl,
            report_path=camel_report,
            model_name="calima-msa-r13",
            top=1,
            limit_sentences=None,
            limit_records=5000,
            require_source_upos=True,
        )
        if report["records_written"] != 5000:
            raise RuntimeError(f"expected 5,000 CAMeL records, got {report['records_written']}")
        require_hash(camel_report, EXPECTED["camel_report"], "historical CAMeL export report")
        export_mode = "recomputed_from_padt"
    require_hash(camel_jsonl, EXPECTED["camel_jsonl"], "CAMeL intermediate")

    raw = read_raw_records(camel_jsonl)
    records, ingest_report = normalize_records(raw, SOURCE_NAME)
    records, filter_report = apply_filters(records, FILTERS)
    split_records_out, split_report = split_records(records, "root_heldout", 17, RATIOS)
    probe_rows = make_probe_records(split_records_out, "root_heldout")
    sft_rows = make_sft_examples(
        split_records_out, ["analyze_form", "root_pattern", "feature_bundle"]
    )
    validation_report = {
        "canonical": validate_canonical(split_records_out, "root_heldout"),
        "sft": validate_sft_examples(sft_rows),
        "probes": validate_probe_records(probe_rows),
    }
    validation_report["passed"] = all(
        item["passed"] for item in validation_report.values() if isinstance(item, dict)
    )
    if not validation_report["passed"]:
        raise RuntimeError("historical pipeline validation failed")

    canonical_path = output_dir / "canonical.jsonl"
    probes_path = output_dir / "probes.jsonl"
    paper1_path = output_dir / "paper1_normalized.jsonl"
    write_morph_records(canonical_path, split_records_out)
    write_jsonl(probes_path, probe_rows)
    write_jsonl(output_dir / "sft.jsonl", sft_rows)
    write_json(output_dir / "ingest_report.json", ingest_report)
    write_json(output_dir / "filter_report.json", filter_report)
    write_json(output_dir / "split_report.json", split_report)
    write_json(output_dir / "stats.json", dataset_stats(split_records_out))
    write_json(
        output_dir / "summary_report.json",
        make_summary_report(records, filter_report, 17, RATIOS),
    )
    write_json(output_dir / "validation.json", validation_report)
    write_paper1_jsonl(probe_rows, paper1_path)

    require_hash(canonical_path, EXPECTED["canonical"], "canonical dataset")
    require_hash(probes_path, EXPECTED["probes"], "probe dataset")
    require_hash(paper1_path, EXPECTED["paper1"], "Paper 1 frozen dataset")

    counts = {
        "records": len(probe_rows),
        "splits": dict(sorted(Counter(row["split"] for row in probe_rows).items())),
        "pos": dict(sorted(Counter(row["pos"] for row in probe_rows).items())),
        "gender": dict(
            sorted(Counter(row["features"].get("gender", "") for row in probe_rows).items())
        ),
        "number_historical_labels": dict(
            sorted(Counter(row["features"].get("number", "") for row in probe_rows).items())
        ),
        "unique_lemmas": len({row["lemma"] for row in probe_rows}),
        "unique_roots": len({row["root"] for row in probe_rows}),
        "unique_surface_forms": len({row["surface_dediac"] for row in probe_rows}),
    }

    installed_previous_hash = None
    if args.install:
        installed_previous_hash = install_atomically(
            paper1_path, ROOT / "output/data/paper1_normalized.jsonl"
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified_exact_rebuild",
        "export_mode": export_mode,
        "source": {
            "path": str(upstream),
            "sha256": EXPECTED["padt_train"],
            "ud_release": "r2.17",
            "git_commit": "e9744f8efb593ce746c46df3edfbfb102957b38a",
        },
        "historical_pipeline": {
            "path": str(VENDOR),
            "ember_commit": "1a98982e3e21c1e4e98e86025bd2d823d3ff0574",
            "seed": 17,
            "split_strategy": "root_heldout",
            "split_ratios": RATIOS,
            "source_name": SOURCE_NAME,
        },
        "environment": environment,
        "counts": counts,
        "artifacts": {
            "camel_jsonl": {"path": str(camel_jsonl), "sha256": EXPECTED["camel_jsonl"]},
            "canonical": {"path": str(canonical_path), "sha256": EXPECTED["canonical"]},
            "probes": {"path": str(probes_path), "sha256": EXPECTED["probes"]},
            "paper1": {"path": str(paper1_path), "sha256": EXPECTED["paper1"]},
        },
        "installed": args.install,
        "installed_previous_sha256": installed_previous_hash,
    }
    write_json(output_dir / "rebuild_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
