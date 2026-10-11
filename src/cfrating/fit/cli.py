import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd

from cfrating.fit.labels import (
    DEFAULT_OUTCOME_Y,
    apply_outcome_y,
    outcome_y_record,
    parse_outcome_y,
)
from cfrating.fit.logistic import LOCATION_PRIOR_SD, LOG_SCALE_PRIOR_SD, fit_outcomes

_REQUIRED = (
    "handle",
    "contest_id",
    "problem_index",
    "problem_name",
    "problem_url",
    "outcome",
)


class FitError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        raise SystemExit(1)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="cfrating-fit")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--y")
    return parser


def _status(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _read_frame(path: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (
        OSError,
        UnicodeError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise FitError(1, "unreadable file") from exc
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise FitError(1, "missing columns: " + ", ".join(missing))
    return frame.loc[:, list(_REQUIRED)].copy()


def _contest_id(text: str) -> int:
    if text == "" or text[0] == "+" or "_" in text:
        raise ValueError
    if text[0] == "-":
        digits = text[1:]
    else:
        digits = text
    if digits == "" or not digits.isdecimal():
        raise ValueError
    return int(text)


def _table_from_arg(text: str | None) -> dict[str, float | None]:
    if text is None:
        return dict(DEFAULT_OUTCOME_Y)
    try:
        return parse_outcome_y(text)
    except ValueError as exc:
        raise FitError(1, str(exc)) from exc


def _prepare(frame: pd.DataFrame, table: Mapping[str, float | None]) -> pd.DataFrame:
    if frame.empty:
        raise FitError(2, "zero rows")
    contest_ids: list[int] = []
    for text in frame["contest_id"].tolist():
        try:
            contest_ids.append(_contest_id(str(text)))
        except ValueError as exc:
            raise FitError(1, "invalid contest_id") from exc
    frame = frame.copy()
    frame["contest_id"] = contest_ids
    for column in ("handle", "problem_index", "problem_name", "problem_url", "outcome"):
        frame[column] = frame[column].map(str)
    if frame.duplicated(["handle", "contest_id", "problem_index"]).any():
        raise FitError(1, "duplicate (handle, contest_id, problem_index)")
    grouped = frame.groupby(["contest_id", "problem_index"], sort=False)
    names = grouped["problem_name"].nunique()
    urls = grouped["problem_url"].nunique()
    if bool((names > 1).any() or (urls > 1).any()):
        raise FitError(1, "conflicting problem name or url")
    try:
        _, keep = apply_outcome_y(np.asarray(frame["outcome"].tolist(), dtype=object), table)
    except ValueError as exc:
        raise FitError(1, str(exc)) from exc
    if int(frame["handle"].nunique()) < 2:
        raise FitError(2, "fewer than 2 distinct handles")
    kept = frame.iloc[np.flatnonzero(keep)].copy()
    if kept.empty:
        raise FitError(2, "zero rows")
    if int(kept["handle"].nunique()) < 2:
        raise FitError(2, "fewer than 2 distinct handles")
    if int(kept.groupby(["contest_id", "problem_index"], sort=False).ngroups) == 0:
        raise FitError(2, "zero problems")
    return kept


def _finite(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FitError(1, "non-finite parameter")
    return number


def _scale(value: float) -> float:
    number = _finite(value)
    if not number > 0.0:
        raise FitError(1, "scale must be positive")
    return number


def _model(table: Mapping[str, float | None]) -> dict[str, Any]:
    recorded = outcome_y_record(table)
    return {
        "name": "logistic",
        "formula": "1 / (1 + exp(-(skill - location) / scale))",
        "y_solved": recorded["y_solved"],
        "y_failed": recorded["y_failed"],
        "y_none": recorded["y_none"],
        "skill_mean": 0.0,
        "skill_std": 1.0,
        "skill_std_kind": "population",
        "location_prior_sd": LOCATION_PRIOR_SD,
        "log_scale_prior_sd": LOG_SCALE_PRIOR_SD,
    }


def _document(frame: pd.DataFrame, table: Mapping[str, float | None]) -> dict[str, Any]:
    handles = sorted({str(handle) for handle in frame["handle"].tolist()})
    handle_pos = {handle: index for index, handle in enumerate(handles)}
    meta: dict[tuple[int, str], tuple[str, str]] = {}
    contest_ids = [int(value) for value in frame["contest_id"].tolist()]
    problem_indexes = [str(value) for value in frame["problem_index"].tolist()]
    problem_names = [str(value) for value in frame["problem_name"].tolist()]
    problem_urls = [str(value) for value in frame["problem_url"].tolist()]
    for contest_id, problem_index, problem_name, problem_url in zip(
        contest_ids, problem_indexes, problem_names, problem_urls, strict=True
    ):
        meta[(contest_id, problem_index)] = (problem_name, problem_url)
    problem_keys = sorted(meta)
    problem_pos = {key: index for index, key in enumerate(problem_keys)}
    user_index = np.array(
        [handle_pos[str(handle)] for handle in frame["handle"].tolist()],
        dtype=np.intp,
    )
    problem_index = np.array(
        [problem_pos[(contest_id, index)] for contest_id, index in zip(contest_ids, problem_indexes, strict=True)],
        dtype=np.intp,
    )
    outcomes = np.asarray(frame["outcome"].tolist(), dtype=object)
    try:
        skills, locations, scales = fit_outcomes(
            outcomes,
            user_index,
            problem_index,
            len(handles),
            len(problem_keys),
            table,
        )
    except (RuntimeError, ValueError) as exc:
        raise FitError(1, str(exc)) from exc
    users = [
        {"handle": handle, "skill": _finite(skills[index])}
        for index, handle in enumerate(handles)
    ]
    problems = []
    for index, key in enumerate(problem_keys):
        contest_id, index_text = key
        problem_name, problem_url = meta[key]
        problems.append(
            {
                "contest_id": int(contest_id),
                "problem_index": index_text,
                "problem_name": problem_name,
                "problem_url": problem_url,
                "location": _finite(locations[index]),
                "scale": _scale(scales[index]),
            }
        )
    return {
        "format": "cfrating.fit",
        "version": 1,
        "model": _model(table),
        "users": users,
        "problems": problems,
    }


def _write(path: str, document: Mapping[str, Any]) -> None:
    try:
        payload = json.dumps(document, indent=2, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise FitError(1, "could not encode fit.json") from exc
    try:
        Path(path).write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        raise FitError(1, "could not write output") from exc


def run(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return _status(exc)
    try:
        table = _table_from_arg(args.y)
        frame = _prepare(_read_frame(args.input), table)
        _write(args.output, _document(frame, table))
    except FitError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))
