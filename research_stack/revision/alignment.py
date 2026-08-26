from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, Mapping, Sequence


class TargetSpanError(ValueError):
    pass


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    target_text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class AlignmentRecord:
    example_id: str
    target_text: str
    target_char_span: list[int]
    target_token_span: list[int] | None
    selected_token_index: int | None
    selected_token_text: str | None
    selected_token_id: int | None
    target_token_length: int
    status: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_with_target_span(
    template: str,
    row: Mapping[str, Any],
    *,
    target_field: str,
    target_placeholder: str = "target",
) -> RenderedPrompt:
    target = row.get(target_field)
    if target is None or str(target) == "":
        raise TargetSpanError(f"missing target field {target_field!r}")
    target_text = str(target)
    formatter = string.Formatter()
    pieces: list[str] = []
    target_start: int | None = None
    target_count = 0
    length = 0
    for literal, field, format_spec, conversion in formatter.parse(template):
        pieces.append(literal)
        length += len(literal)
        if field is None:
            continue
        if field == target_placeholder:
            value = target_text
            target_start = length
            target_count += 1
        else:
            if field not in row:
                raise TargetSpanError(f"prompt field {field!r} is missing")
            value = row[field]
        if conversion:
            value = formatter.convert_field(value, conversion)
        rendered = formatter.format_field(value, format_spec)
        pieces.append(rendered)
        length += len(rendered)
    if target_count != 1 or target_start is None:
        raise TargetSpanError(
            f"prompt must contain exactly one {{{target_placeholder}}} placeholder; found {target_count}"
        )
    text = "".join(pieces)
    target_end = target_start + len(target_text)
    if text[target_start:target_end] != target_text:
        raise AssertionError("rendered target span does not round-trip")
    return RenderedPrompt(text, target_text, target_start, target_end)


def align_offsets(
    *,
    example_id: str,
    target_text: str,
    char_span: tuple[int, int],
    offsets: Sequence[Sequence[int]],
    token_ids: Sequence[int],
    token_texts: Sequence[str],
) -> AlignmentRecord:
    if not (len(offsets) == len(token_ids) == len(token_texts)):
        return AlignmentRecord(
            example_id, target_text, list(char_span), None, None, None, None, 0,
            "failed", "token metadata lengths differ",
        )
    start, end = char_span
    if start < 0 or end <= start:
        return AlignmentRecord(
            example_id, target_text, list(char_span), None, None, None, None, 0,
            "failed", "invalid target character span",
        )
    overlapping: list[int] = []
    for index, offset in enumerate(offsets):
        if len(offset) != 2:
            return AlignmentRecord(
                example_id, target_text, list(char_span), None, None, None, None, 0,
                "failed", f"token {index} has invalid offset metadata",
            )
        token_start, token_end = int(offset[0]), int(offset[1])
        if token_end > start and token_start < end and token_end > token_start:
            overlapping.append(index)
    if not overlapping:
        return AlignmentRecord(
            example_id, target_text, list(char_span), None, None, None, None, 0,
            "failed", "no token overlaps target character span",
        )
    if overlapping != list(range(overlapping[0], overlapping[-1] + 1)):
        return AlignmentRecord(
            example_id, target_text, list(char_span), [overlapping[0], overlapping[-1] + 1],
            None, None, None, len(overlapping), "ambiguous", "overlapping tokens are non-contiguous",
        )
    clipped = sorted(
        (max(int(offsets[index][0]), start), min(int(offsets[index][1]), end))
        for index in overlapping
    )
    cursor = start
    has_gap = False
    for interval_start, interval_end in clipped:
        if interval_start > cursor:
            has_gap = True
            break
        cursor = max(cursor, interval_end)
    if has_gap or cursor != end:
        return AlignmentRecord(
            example_id, target_text, list(char_span), [overlapping[0], overlapping[-1] + 1],
            None, None, None, len(overlapping), "ambiguous", "token offsets do not cover the target span",
        )
    selected = overlapping[-1]
    return AlignmentRecord(
        example_id=example_id,
        target_text=target_text,
        target_char_span=list(char_span),
        target_token_span=[overlapping[0], overlapping[-1] + 1],
        selected_token_index=selected,
        selected_token_text=str(token_texts[selected]),
        selected_token_id=int(token_ids[selected]),
        target_token_length=len(overlapping),
        status="aligned",
    )


class FixtureTokenizer:
    """Deterministic offset tokenizer used only by the non-research smoke backend."""

    _word = re.compile(r"\S+")

    def encode(self, text: str) -> tuple[list[int], list[str], list[list[int]]]:
        texts: list[str] = []
        offsets: list[list[int]] = []
        for match in self._word.finditer(text):
            word = match.group()
            chunk_size = 3 if any(ord(char) > 127 for char in word) else len(word)
            for local in range(0, len(word), max(chunk_size, 1)):
                piece = word[local:local + chunk_size]
                texts.append(piece)
                offsets.append([match.start() + local, match.start() + local + len(piece)])
        ids = [sum(piece.encode("utf-8")) % 32000 for piece in texts]
        return ids, texts, offsets


def tokenization_report(records: Sequence[AlignmentRecord], example_limit: int = 5) -> dict[str, Any]:
    aligned = [record for record in records if record.status == "aligned"]
    lengths = [record.target_token_length for record in aligned]
    reasons: dict[str, int] = {}
    for record in records:
        if record.status != "aligned":
            reason = record.reason or record.status
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "total_examples": len(records),
        "aligned_examples": len(aligned),
        "ambiguous_alignment_count": sum(r.status == "ambiguous" for r in records),
        "failed_alignment_count": sum(r.status == "failed" for r in records),
        "single_token_target_count": sum(length == 1 for length in lengths),
        "multi_token_target_count": sum(length > 1 for length in lengths),
        "mean_target_token_length": round(mean(lengths), 4) if lengths else None,
        "median_target_token_length": median(lengths) if lengths else None,
        "max_target_token_length": max(lengths) if lengths else None,
        "dropped_example_reasons": reasons,
        "success_examples": [r.to_dict() for r in aligned[:example_limit]],
        "failure_examples": [r.to_dict() for r in records if r.status != "aligned"][:example_limit],
    }
