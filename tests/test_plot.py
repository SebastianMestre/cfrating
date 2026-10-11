import matplotlib

matplotlib.use("Agg")

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from cfrating.plot.cli import main, run
from cfrating.plot.figure import build_figure, hit_test
from cfrating.plot.load import Fit, Problem, User

_URL = "https://codeforces.com/contest/1/problem/A"


def _fit() -> Fit:
    return Fit(
        users=(User("alice", 0.0),),
        problems=(Problem("Sum", _URL, 0.0, 1.0),),
    )


def _payload() -> dict[str, object]:
    return {
        "format": "cfrating.fit",
        "version": 1,
        "model": {
            "name": "logistic",
            "formula": "1 / (1 + exp(-(skill - location) / scale))",
            "y_solved": 1,
            "y_failed": 0,
            "y_none": None,
        },
        "users": [{"handle": "alice", "skill": 0.0}],
        "problems": [
            {
                "contest_id": 1,
                "problem_index": "A",
                "problem_name": "Sum",
                "problem_url": _URL,
                "location": 0.0,
                "scale": 1.0,
            }
        ],
    }


def _write(directory: Path, payload: dict[str, object]) -> Path:
    path = directory / "fit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _display(ax: plt.Axes, x: float, y: float) -> tuple[float, float]:
    point = ax.transData.transform((x, y))
    return float(point[0]), float(point[1])


def test_hover_selects_user_at_skill_line() -> None:
    fig = build_figure(_fit())
    try:
        fig.canvas.draw()
        px, py = _display(fig.axes[0], 0.0, 0.5)
        text = hit_test(fig, px, py)
        assert text is not None
        assert "alice" in text
    finally:
        plt.close(fig)


def test_hover_selects_problem_away_from_user() -> None:
    fig = build_figure(_fit())
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        x = 1.2
        y = 1.0 / (1.0 + math.exp(-x))
        px, py = _display(ax, x, y)
        user_x, _ = _display(ax, 0.0, y)
        assert abs(px - user_x) > 8.0
        text = hit_test(fig, px, py)
        assert text is not None
        assert _URL in text
        assert "Sum" in text
    finally:
        plt.close(fig)


def test_hover_empty_space_selects_nothing() -> None:
    fig = build_figure(_fit())
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        px, py = _display(ax, 1.4, 0.05)
        user_x, _ = _display(ax, 0.0, 0.05)
        assert abs(px - user_x) > 8.0
        assert hit_test(fig, px, py) is None
    finally:
        plt.close(fig)


def test_no_show_writes_png(tmp_path: Path) -> None:
    source = _write(tmp_path, _payload())
    output = tmp_path / "plot.png"
    code = run(
        ["--input", str(source), "--output", str(output), "--no-show"]
    )
    assert code == 0
    data = output.read_bytes()
    assert len(data) > 0
    assert data.startswith(b"\x89PNG\r\n\x1a\n")


def test_version_two_returns_1(tmp_path: Path) -> None:
    payload = _payload()
    payload["version"] = 2
    source = _write(tmp_path, payload)
    assert run(["--input", str(source), "--no-show"]) == 1


@pytest.mark.parametrize("scale", [0, -1])
def test_nonpositive_scale_returns_1(tmp_path: Path, scale: int) -> None:
    payload = _payload()
    problems = payload["problems"]
    assert isinstance(problems, list)
    problems[0]["scale"] = scale
    source = _write(tmp_path, payload)
    assert run(["--input", str(source), "--no-show"]) == 1


def test_main_raises_system_exit(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as caught:
        main(["--input", str(missing), "--no-show"])
    assert caught.value.code == 1


def test_missing_file_returns_1(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert run(["--input", str(missing), "--no-show"]) == 1


def test_bad_json_returns_1(tmp_path: Path) -> None:
    source = tmp_path / "fit.json"
    source.write_text("{", encoding="utf-8")
    assert run(["--input", str(source), "--no-show"]) == 1


def test_wrong_format_returns_1(tmp_path: Path) -> None:
    payload = _payload()
    payload["format"] = "other"
    source = _write(tmp_path, payload)
    assert run(["--input", str(source), "--no-show"]) == 1


def test_wrong_model_returns_1(tmp_path: Path) -> None:
    payload = _payload()
    model = payload["model"]
    assert isinstance(model, dict)
    model["name"] = "linear"
    source = _write(tmp_path, payload)
    assert run(["--input", str(source), "--no-show"]) == 1


@pytest.mark.parametrize("key", ["users", "problems"])
def test_empty_collection_returns_1(tmp_path: Path, key: str) -> None:
    payload = _payload()
    payload[key] = []
    source = _write(tmp_path, payload)
    assert run(["--input", str(source), "--no-show"]) == 1


def test_nonfinite_skill_returns_1(tmp_path: Path) -> None:
    payload = _payload()
    users = payload["users"]
    assert isinstance(users, list)
    users[0]["skill"] = math.nan
    source = _write(tmp_path, payload)
    assert run(["--input", str(source), "--no-show"]) == 1


def test_nonfinite_location_returns_1(tmp_path: Path) -> None:
    payload = _payload()
    problems = payload["problems"]
    assert isinstance(problems, list)
    problems[0]["location"] = math.inf
    source = _write(tmp_path, payload)
    assert run(["--input", str(source), "--no-show"]) == 1
