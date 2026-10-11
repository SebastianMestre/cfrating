from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

DEFAULT_OUTCOME_Y: dict[str, float | None] = {"solved": 1.0, "failed": 0.0, "none": 0.0}

_OUTCOMES = frozenset({"solved", "failed", "none"})
_LABELS = {"0": 0.0, "1": 1.0, "ignore": None}


def parse_outcome_y(text: str) -> dict[str, float | None]:
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("invalid outcome map")
    table: dict[str, float | None] = {}
    for part in text.split(","):
        piece = part.strip()
        if "=" not in piece:
            raise ValueError("invalid outcome map")
        key, value = piece.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in _OUTCOMES or key in table or value not in _LABELS:
            raise ValueError("invalid outcome map")
        table[key] = _LABELS[value]
    if set(table) != _OUTCOMES:
        raise ValueError("invalid outcome map")
    return table


def apply_outcome_y(
    outcomes: np.ndarray,
    table: Mapping[str, float | None],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = np.asarray(outcomes)
    if values.ndim != 1:
        raise ValueError("outcomes must be one-dimensional")
    items = values.tolist()
    y = np.empty(len(items), dtype=np.float64)
    keep = np.empty(len(items), dtype=np.bool_)
    for index, outcome in enumerate(items):
        try:
            label = table[outcome]
        except (KeyError, TypeError):
            raise ValueError(f"unknown outcome: {outcome!r}") from None
        if label is None:
            y[index] = 0.0
            keep[index] = False
            continue
        y[index] = float(label)
        keep[index] = True
    return y, keep


def outcome_y_record(table: Mapping[str, float | None]) -> dict[str, int | None]:
    def encode(value: float | None) -> int | None:
        if value is None:
            return None
        if value == 0:
            return 0
        if value == 1:
            return 1
        raise ValueError("outcome label must be 0, 1, or ignore")

    return {
        "y_solved": encode(table["solved"]),
        "y_failed": encode(table["failed"]),
        "y_none": encode(table["none"]),
    }
