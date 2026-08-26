from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from research_stack.bf16_capture.alignment import align_all, prepare_examples
from research_stack.bf16_capture.bundle import (
    validate_bundle,
    write_checksums,
    write_complete_marker,
    write_json,
    write_jsonl,
)
from research_stack.bf16_capture.cli import main as bf16_cli
from research_stack.bf16_capture.local import load_bundle_features, validate_local_analysis
from research_stack.bf16_capture.preflight import load_matrix, static_manifest_checks
from research_stack.bf16_capture.sync import finalize_marker, sync_local_bundle


class FakeFastTokenizer:
    is_fast = True
    vocab_size = 100
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        tokens = []
        offsets = []
        for index, word in enumerate(text.split()):
            start = text.index(word, offsets[-1][1] if offsets else 0)
            tokens.append(index + 10)
            offsets.append([start, start + len(word)])
        return {"input_ids": tokens, "offset_mapping": offsets, "attention_mask": [1] * len(tokens)}

    def convert_ids_to_tokens(self, ids):
        return [str(value) for value in ids]

    def decode(self, ids, **kwargs):
        return "كتاب"


def _fixture_bundle(path: Path, n: int = 30, layers: int = 3, hidden: int = 8) -> Path:
    examples = []
    alignments = []
    total_target = 0
    for index in range(n):
        length = 1 + index % 2
        lemma = f"lemma-{index // 3}"
        root = f"root-{index // 5}"
        examples.append({
            "example_id": f"stim-{index:04d}", "input_ids": [1, 10 + index, 11], "sequence_length": 3,
            "target_token_indices": [1] if length == 1 else [1, 2], "target_token_count": length,
            "prompt": f"Token {index}", "target_byte_span": [6, 7], "lemma": lemma, "root": root,
            "pos": "NOUN" if index % 2 else "VERB", "gender": "masc", "number": "sg",
        })
        alignments.append({
            "example_id": f"stim-{index:04d}", "target_text": "x", "target_char_span": [6, 7],
            "target_byte_span": [6, 7], "target_token_indices": examples[-1]["target_token_indices"],
            "target_token_span": [1, 1 + length], "selected_final_token_index": 1 + length - 1,
            "target_token_count": length, "token_pieces": ["x"], "status": "aligned", "reason": None,
        })
        total_target += length
    path.mkdir(parents=True)
    write_jsonl(path / "examples.jsonl", examples)
    write_jsonl(path / "alignment.jsonl", alignments)
    write_json(path / "model.json", {"model": "fixture"})
    write_json(path / "tokenizer.json", {"tokenizer": "fixture"})
    write_json(path / "prompt.json", {"condition": "fixture"})
    write_json(path / "status.json", {"state": "VALIDATING"})
    representation = path / "representations"
    representation.mkdir()
    generator = torch.Generator().manual_seed(17)
    save_file({"embedding_output": torch.randn(n, hidden, generator=generator).bfloat16()}, str(representation / "embedding_output.safetensors"))
    for layer in range(layers):
        save_file({
            "prompt_final": torch.randn(n, hidden, generator=generator).bfloat16(),
            "target_span": torch.randn(total_target, hidden, generator=generator).bfloat16(),
        }, str(representation / f"layer_{layer:04d}.safetensors"))
    write_json(path / "manifest.json", {
        "bundle_schema": "bf16-hidden-state-bundle-v1", "schema_version": 1, "expected_examples": n,
        "expected_layer_count": layers, "expected_hidden_size": hidden, "total_target_tokens": total_target,
        "successful_alignments": n,
    })
    checksum = write_checksums(path)
    write_complete_marker(path, destination="fixture-destination", checksum_manifest_sha256=checksum)
    return path


class TestBF16Manifest(unittest.TestCase):
    def test_resolved_matrix_is_pinned_and_satisfies_hard_gates(self):
        manifest = load_matrix(Path("configs/bf16_model_matrix_20260826_jais_resolved.json"))
        self.assertEqual(len(manifest["matrix"]), 11)
        self.assertEqual(static_manifest_checks(manifest), [])
        self.assertFalse(manifest["inference_contract"]["allow_cpu_offload"])
        self.assertFalse(manifest["inference_contract"]["allow_quantization"])


class TestBF16Bundle(unittest.TestCase):
    def test_complete_bundle_validates_and_supports_local_derivations(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _fixture_bundle(Path(directory) / "bundle")
            result = validate_bundle(bundle, require_complete=True, check_tensors=True)
            self.assertTrue(result["valid"], result)
            local = validate_local_analysis(bundle)
            self.assertEqual(local["status"], "pass")
            self.assertEqual(local["layer_shape"], [30, 3, 8])
            self.assertEqual(local["target_final_shape"], [30, 3, 8])
            self.assertFalse(local["weights_required"])
            self.assertEqual(set(local["split_reconstruction"]), {"lemma-heldout", "root-heldout", "random"})

    def test_corruption_and_missing_completion_never_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _fixture_bundle(Path(directory) / "bundle")
            path = bundle / "representations" / "layer_0000.safetensors"
            original = path.read_bytes()
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            self.assertFalse(validate_bundle(bundle, require_complete=True, check_tensors=False)["valid"])
            path.write_bytes(original)
            (bundle / "COMPLETE").unlink()
            self.assertTrue(validate_bundle(bundle, require_complete=False, check_tensors=False)["valid"])
            self.assertFalse(validate_bundle(bundle, require_complete=True, check_tensors=False)["valid"])

    def test_completion_marker_binds_checksum_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _fixture_bundle(Path(directory) / "bundle")
            marker = json.loads((bundle / "COMPLETE").read_text())
            marker["bundle_checksums_sha256"] = "0" * 64
            (bundle / "COMPLETE").write_text(json.dumps(marker) + "\n")
            result = validate_bundle(bundle, require_complete=True, check_tensors=False)
            self.assertFalse(result["valid"])
            self.assertIn("not bound", " ".join(result["errors"]))

    def test_local_sync_is_verified_before_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _fixture_bundle(root / "source")
            destination_root = root / "persistent"
            receipt = sync_local_bundle(source, destination_root / "qwen" / "full")
            destination = Path(receipt["destination"])
            self.assertTrue(validate_bundle(destination, require_complete=False, check_tensors=False)["valid"])
            self.assertFalse((destination / "COMPLETE").exists())
            finalize_marker(source, str(destination))
            self.assertTrue(validate_bundle(destination, require_complete=True, check_tensors=False)["valid"])

    def test_posthoc_reports_cannot_mutate_completed_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _fixture_bundle(Path(directory) / "bundle")
            with self.assertRaises(ValueError):
                bf16_cli(["local-validate", "--bundle", str(bundle), "--output", str(bundle / "report.json")])
            with self.assertRaises(ValueError):
                bf16_cli(["validate-bundle", "--bundle", str(bundle), "--output", str(bundle / "report.json")])

    def test_posthoc_report_creates_external_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = _fixture_bundle(root / "bundle")
            output = root / "reports" / "nested" / "local.json"
            bf16_cli(["local-validate", "--bundle", str(bundle), "--output", str(output)])
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text())["weights_required"], False)

    def test_alignment_uses_offsets_and_records_all_target_tokens(self):
        rows = [{"id": "x", "surface": "كتاب", "surface_dediac": "كتاب", "lemma": "كتب", "root": "ك.ت.ب", "abstract_pattern": "p"}]
        examples = prepare_examples(rows, {"template": "Surface {target}", "target_field": "surface", "target_placeholder": "target"})
        audits, enriched = align_all(FakeFastTokenizer(), examples)
        self.assertEqual(audits[0]["status"], "aligned")
        self.assertEqual(audits[0]["target_token_indices"], [1])
        self.assertEqual(enriched[0]["target_byte_span"], [8, 16])


if __name__ == "__main__":
    unittest.main()
