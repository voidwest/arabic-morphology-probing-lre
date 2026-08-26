from __future__ import annotations

from typing import Any, Sequence

from ..revision.alignment import TargetSpanError, render_with_target_span


def _byte_offset(text: str, char_offset: int) -> int:
    return len(text[:char_offset].encode("utf-8"))


def prepare_examples(rows: Sequence[dict[str, Any]], prompt: dict[str, Any]) -> list[dict[str, Any]]:
    """Render a condition with structural target spans; never search for text."""
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        rendered = render_with_target_span(
            prompt["template"], row,
            target_field=prompt.get("target_field", "surface"),
            target_placeholder=prompt.get("target_placeholder", "target"),
        )
        prepared.append({
            "example_index": index,
            "example_id": str(row.get("id", index)),
            "source_row": index,
            "source_dataset": "output/data/paper1_normalized.jsonl",
            "surface": row.get("surface"),
            "surface_dediac": row.get("surface_dediac"),
            "normalized_form": row.get("surface_dediac"),
            "lemma": row.get("lemma"),
            "root": row.get("root"),
            "pos": row.get("pos"),
            "gender": row.get("gender"),
            "number": row.get("number"),
            "source": row.get("source"),
            "source_split": row.get("split"),
            "source_split_type": row.get("split_type"),
            "prompt": rendered.text,
            "prompt_sha256": __import__("hashlib").sha256(rendered.text.encode("utf-8")).hexdigest(),
            "target_text": rendered.target_text,
            "target_char_span": [rendered.char_start, rendered.char_end],
            "target_byte_span": [
                _byte_offset(rendered.text, rendered.char_start),
                _byte_offset(rendered.text, rendered.char_end),
            ],
        })
    return prepared


def _alignment_failure(example: dict[str, Any], reason: str, status: str = "failed") -> dict[str, Any]:
    return {
        "example_id": example["example_id"],
        "target_text": example["target_text"],
        "target_char_span": example["target_char_span"],
        "target_byte_span": example["target_byte_span"],
        "target_token_indices": [],
        "target_token_span": None,
        "selected_final_token_index": None,
        "target_token_count": 0,
        "token_pieces": [],
        "decoded_target_text": None,
        "decoded_target_match": None,
        "status": status,
        "reason": reason,
    }


def align_tokenizer(tokenizer: Any, example: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tokenize and structurally align one rendered prompt.

    Fast-tokenizer character offsets are checked against the renderer span. The
    UTF-8 byte span is retained as the canonical structural identity; offsets
    are not inferred from row order, ``[-1]``, or text search.
    """
    if not getattr(tokenizer, "is_fast", False):
        return _alignment_failure(example, "a fast tokenizer with offset_mapping is required"), example
    try:
        encoded = tokenizer(
            example["prompt"], add_special_tokens=True, truncation=False,
            return_offsets_mapping=True, return_attention_mask=True,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        offsets = [[int(pair[0]), int(pair[1])] for pair in encoded["offset_mapping"]]
    except Exception as error:  # tokenizer-specific errors must be recorded
        return _alignment_failure(example, f"tokenization failed: {type(error).__name__}: {error}"), example
    if len(ids) != len(offsets):
        return _alignment_failure(example, "token IDs and offset metadata differ in length"), example
    start, end = [int(value) for value in example["target_char_span"]]
    overlap: list[int] = []
    for index, (token_start, token_end) in enumerate(offsets):
        if token_end > start and token_start < end and token_end > token_start:
            overlap.append(index)
    if not overlap:
        return _alignment_failure(example, "no token overlaps target character span"), example
    if overlap != list(range(overlap[0], overlap[-1] + 1)):
        return _alignment_failure(example, "overlapping tokens are non-contiguous", "ambiguous"), example
    cursor = start
    for index in overlap:
        token_start, token_end = offsets[index]
        if token_start > cursor:
            return _alignment_failure(example, "token offsets do not completely cover target span", "ambiguous"), example
        cursor = max(cursor, token_end)
    if cursor != end:
        return _alignment_failure(example, "token offsets do not completely cover target span", "ambiguous"), example
    pieces = [str(value) for value in tokenizer.convert_ids_to_tokens(ids)]
    target_ids = ids[overlap[0]:overlap[-1] + 1]
    try:
        decoded = tokenizer.decode(
            target_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        decoded = None
    decoded_target_match = decoded is not None and decoded.strip() == example["target_text"].strip()
    if not decoded_target_match:
        return _alignment_failure(
            example,
            "decoded target-token span does not match target text after boundary-whitespace normalization",
            "ambiguous",
        ), example
    audit = {
        "example_id": example["example_id"],
        "target_text": example["target_text"],
        "target_char_span": example["target_char_span"],
        "target_byte_span": example["target_byte_span"],
        "target_token_indices": overlap,
        "target_token_span": [overlap[0], overlap[-1] + 1],
        "selected_final_token_index": overlap[-1],
        "target_token_count": len(overlap),
        "token_pieces": pieces,
        "target_token_ids": target_ids,
        "decoded_target_text": decoded,
        "decoded_target_match": decoded_target_match,
        "offsets": offsets,
        "input_ids": ids,
        "sequence_length": len(ids),
        "status": "aligned",
        "reason": None,
    }
    enriched = dict(example)
    enriched.update({
        "input_ids": ids,
        "sequence_length": len(ids),
        "target_token_indices": overlap,
        "target_token_count": len(overlap),
        "selected_final_token_index": overlap[-1],
        "target_token_ids": target_ids,
        "token_pieces": pieces,
    })
    return audit, enriched


def align_all(tokenizer: Any, examples: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audits: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for example in examples:
        audit, value = align_tokenizer(tokenizer, example)
        audits.append(audit)
        enriched.append(value if audit["status"] == "aligned" else dict(example))
    return audits, enriched
