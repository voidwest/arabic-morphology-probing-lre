from __future__ import annotations

from typing import Any, Mapping, Sequence


# The frozen July 2026 artifacts used ``def`` for the dual number class.  That
# spelling came from an upstream normalization error; it is a label alias, not
# a fourth class.  Canonicalize it at the analysis boundary so historical
# artifacts remain readable while all newly emitted labels use ``du``.
TASK_LABEL_ALIASES: dict[str, dict[str, str]] = {
    "number": {"def": "du"},
}


def canonicalize_task_label(task: str, value: object) -> str:
    label = str(value)
    return TASK_LABEL_ALIASES.get(task, {}).get(label, label)


def task_labels(
    rows: Sequence[Mapping[str, Any]],
    task: str,
    *,
    missing: str = "__MISSING__",
) -> list[str]:
    output: list[str] = []
    for row in rows:
        value = row.get(task)
        output.append(
            missing
            if value in (None, "")
            else canonicalize_task_label(task, value)
        )
    return output
