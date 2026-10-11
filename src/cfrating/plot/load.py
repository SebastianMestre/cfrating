import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FitFileError(Exception):
    pass


@dataclass(frozen=True)
class User:
    handle: str
    skill: float


@dataclass(frozen=True)
class Problem:
    name: str
    url: str
    location: float
    scale: float


@dataclass(frozen=True)
class Fit:
    users: tuple[User, ...]
    problems: tuple[Problem, ...]


def load_fit(path: Path) -> Fit:
    data = _read_json(path)
    if data.get("format") != "cfrating.fit":
        raise FitFileError("format is not cfrating.fit")
    version = data.get("version")
    if type(version) is not int or version != 1:
        raise FitFileError("version is not 1")
    model = data.get("model")
    if not isinstance(model, dict) or model.get("name") != "logistic":
        raise FitFileError("model.name is not logistic")
    _check_outcomes(model)
    return Fit(users=_users(data.get("users")), problems=_problems(data.get("problems")))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FitFileError(f"file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FitFileError(f"invalid JSON: {exc}") from exc
    except OSError as exc:
        raise FitFileError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FitFileError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise FitFileError("fit file must be a JSON object")
    return data


def _check_outcomes(model: dict[str, Any]) -> None:
    for key in ("y_solved", "y_failed", "y_none"):
        if key not in model:
            continue
        value = model[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FitFileError(f"{key} must be a number or null")


def _users(value: object) -> tuple[User, ...]:
    if not isinstance(value, list):
        raise FitFileError("users must be an array")
    if len(value) == 0:
        raise FitFileError("users is empty")
    users: list[User] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise FitFileError(f"users[{index}] must be an object")
        handle = _string(item.get("handle"), f"users[{index}].handle")
        skill = _finite(item.get("skill"), f"users[{index}].skill")
        users.append(User(handle=handle, skill=skill))
    return tuple(users)


def _problems(value: object) -> tuple[Problem, ...]:
    if not isinstance(value, list):
        raise FitFileError("problems must be an array")
    if len(value) == 0:
        raise FitFileError("problems is empty")
    problems: list[Problem] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise FitFileError(f"problems[{index}] must be an object")
        prefix = f"problems[{index}]"
        problems.append(
            Problem(
                name=_string(item.get("problem_name"), f"{prefix}.problem_name"),
                url=_string(item.get("problem_url"), f"{prefix}.problem_url"),
                location=_finite(item.get("location"), f"{prefix}.location"),
                scale=_positive_scale(item.get("scale"), f"{prefix}.scale"),
            )
        )
    return tuple(problems)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise FitFileError(f"{label} must be a string")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FitFileError(f"{label} is not finite")
    try:
        number = float(value)
    except OverflowError as exc:
        raise FitFileError(f"{label} is not finite") from exc
    if not math.isfinite(number):
        raise FitFileError(f"{label} is not finite")
    return number


def _positive_scale(value: object, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise FitFileError(f"{label} must be finite and greater than 0")
    return number
