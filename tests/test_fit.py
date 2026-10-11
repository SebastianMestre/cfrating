import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import expit
from scipy.stats import spearmanr

from cfrating.fit.cli import main, run
from cfrating.fit.labels import DEFAULT_OUTCOME_Y, apply_outcome_y, parse_outcome_y
from cfrating.fit.logistic import make_objective

_MODEL_KEYS = [
    "name",
    "formula",
    "y_solved",
    "y_failed",
    "y_none",
    "skill_mean",
    "skill_std",
    "skill_std_kind",
    "location_prior_sd",
    "log_scale_prior_sd",
]


def _url(contest_id: int, problem_index: str) -> str:
    return f"https://codeforces.com/contest/{contest_id}/problem/{problem_index}"


def write_observations(path: Path, rows: list[tuple[object, ...]], extra: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        header = [
            "handle",
            "contest_id",
            "problem_index",
            "problem_name",
            "problem_url",
            "outcome",
        ]
        if extra:
            header.append("source")
        writer.writerow(header)
        writer.writerows(rows)


def _reject(token: str) -> None:
    raise ValueError(token)


def load_fit(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    return json.loads(text, parse_constant=_reject)


def assert_model(doc: dict, y_solved: int, y_failed: int, y_none: int | None) -> None:
    assert doc["format"] == "cfrating.fit"
    assert doc["version"] == 1
    model = doc["model"]
    assert list(model) == _MODEL_KEYS
    assert model["name"] == "logistic"
    assert model["formula"] == "1 / (1 + exp(-(skill - location) / scale))"
    assert model["y_solved"] == y_solved
    assert model["y_failed"] == y_failed
    assert model["y_none"] is y_none if y_none is None else model["y_none"] == y_none
    assert model["skill_mean"] == 0.0
    assert model["skill_std"] == 1.0
    assert model["skill_std_kind"] == "population"
    assert model["location_prior_sd"] == 2.0
    assert model["log_scale_prior_sd"] == 0.5


def assert_constraint(doc: dict) -> None:
    skills = np.array([user["skill"] for user in doc["users"]], dtype=np.float64)
    mean = float(np.mean(skills))
    std = float(np.sqrt(np.mean(np.square(skills - mean))))
    assert abs(mean) < 1e-8
    assert abs(std - 1.0) < 1e-8
    assert skills.size >= 2
    assert doc["problems"]
    for problem in doc["problems"]:
        assert problem["scale"] > 0.0
        assert np.isfinite(problem["scale"])
        assert np.isfinite(problem["location"])
    assert np.all(np.isfinite(skills))


def _fit(tmp_path: Path, name: str, rows: list[tuple[object, ...]], y: str | None = None) -> dict:
    source = tmp_path / f"{name}.csv"
    dest = tmp_path / f"{name}.json"
    write_observations(source, rows)
    argv = ["--input", str(source), "--output", str(dest)]
    if y is not None:
        argv.extend(["--y", y])
    assert run(argv) == 0
    return load_fit(dest)


def test_outcome_table() -> None:
    assert DEFAULT_OUTCOME_Y == {"solved": 1.0, "failed": 0.0, "none": 0.0}
    assert parse_outcome_y("solved=1,failed=0,none=0") == DEFAULT_OUTCOME_Y
    table = parse_outcome_y("none=ignore,failed=0,solved=1")
    assert table == {"solved": 1.0, "failed": 0.0, "none": None}
    y, keep = apply_outcome_y(np.array(["none", "solved", "failed"], dtype=object), table)
    assert keep.tolist() == [False, True, True]
    assert y[1] == 1.0
    assert y[2] == 0.0
    with pytest.raises(ValueError):
        parse_outcome_y("solved=1,failed=0")
    with pytest.raises(ValueError):
        apply_outcome_y(np.array(["skipped"], dtype=object), DEFAULT_OUTCOME_Y)


def test_check_grad() -> None:
    outcomes = np.array(
        ["solved", "failed", "none", "solved", "failed", "solved"],
        dtype=object,
    )
    user_index = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
    problem_index = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)
    objective = make_objective(outcomes, user_index, problem_index, 3, 2, DEFAULT_OUTCOME_Y)
    theta = np.array([0.2, -1.4, 0.7, 0.3, -0.2, 0.1, -0.15], dtype=np.float64)

    def value(point: np.ndarray) -> float:
        return objective(point)[0]

    def gradient(point: np.ndarray) -> np.ndarray:
        return objective(point)[1]

    error = float(check_grad(value, gradient, theta))
    assert error < 1e-4


def test_default_fit(tmp_path: Path) -> None:
    source = tmp_path / "observations.csv"
    dest = tmp_path / "fit.json"
    rows = [
        ("bob", 1, "B", "Beta", _url(1, "B"), "none", "extra"),
        ("alice", 1, "A", "Alpha", _url(1, "A"), "solved", "extra"),
        ("alice", 1, "B", "Beta", _url(1, "B"), "solved", "extra"),
        ("bob", 1, "A", "Alpha", _url(1, "A"), "failed", "extra"),
    ]
    write_observations(source, rows, extra=True)
    with pytest.raises(SystemExit) as caught:
        main(["--input", str(source), "--output", str(dest)])
    assert caught.value.code == 0
    text = dest.read_text(encoding="utf-8")
    assert '"y_solved": 1' in text
    assert '"y_failed": 0' in text
    assert '"y_none": 0' in text
    doc = load_fit(dest)
    assert_model(doc, 1, 0, 0)
    assert_constraint(doc)
    assert [user["handle"] for user in doc["users"]] == ["alice", "bob"]
    assert [
        (problem["contest_id"], problem["problem_index"], problem["problem_name"], problem["problem_url"])
        for problem in doc["problems"]
    ] == [
        (1, "A", "Alpha", _url(1, "A")),
        (1, "B", "Beta", _url(1, "B")),
    ]


def test_failed_matches_none_under_default_table(tmp_path: Path) -> None:
    failed_rows = [
        ("ann", 3, "B", "Bee", _url(3, "B"), "solved"),
        ("ann", 3, "A", "Aye", _url(3, "A"), "failed"),
        ("bob", 3, "A", "Aye", _url(3, "A"), "solved"),
        ("bob", 3, "B", "Bee", _url(3, "B"), "failed"),
        ("ann", 4, "A", "Cee", _url(4, "A"), "none"),
        ("bob", 4, "A", "Cee", _url(4, "A"), "solved"),
    ]
    none_rows = [
        tuple("none" if value == "failed" else value for value in row) for row in failed_rows
    ]
    failed_doc = _fit(tmp_path, "failed", failed_rows)
    none_doc = _fit(tmp_path, "none", none_rows)
    assert_model(failed_doc, 1, 0, 0)
    assert_model(none_doc, 1, 0, 0)
    failed_skills = [user["skill"] for user in failed_doc["users"]]
    none_skills = [user["skill"] for user in none_doc["users"]]
    assert np.max(np.abs(np.array(failed_skills) - np.array(none_skills))) <= 1e-6
    for left, right in zip(failed_doc["problems"], none_doc["problems"], strict=True):
        assert left["contest_id"] == right["contest_id"]
        assert left["problem_index"] == right["problem_index"]
        assert abs(left["location"] - right["location"]) <= 1e-6
        assert abs(left["scale"] - right["scale"]) <= 1e-6


def test_ignore_drops_rows(tmp_path: Path) -> None:
    rows = [
        ("alpha", 10, "A", "Alpha", _url(10, "A"), "solved"),
        ("alpha", 10, "B", "Beta", _url(10, "B"), "solved"),
        ("alpha", 11, "C", "OnlyNone", _url(11, "C"), "none"),
        ("beta", 10, "A", "Alpha", _url(10, "A"), "failed"),
        ("beta", 10, "B", "Beta", _url(10, "B"), "solved"),
        ("beta", 11, "C", "OnlyNone", _url(11, "C"), "none"),
        ("gamma", 10, "A", "Alpha", _url(10, "A"), "none"),
        ("gamma", 10, "B", "Beta", _url(10, "B"), "none"),
        ("gamma", 11, "C", "OnlyNone", _url(11, "C"), "none"),
    ]
    default_doc = _fit(tmp_path, "default", rows)
    ignore_doc = _fit(tmp_path, "ignore", rows, y="solved=1,failed=0,none=ignore")
    assert_model(ignore_doc, 1, 0, None)
    assert '"y_none": null' in (tmp_path / "ignore.json").read_text(encoding="utf-8")
    assert_constraint(ignore_doc)
    assert [user["handle"] for user in ignore_doc["users"]] == ["alpha", "beta"]
    assert [(problem["contest_id"], problem["problem_index"]) for problem in ignore_doc["problems"]] == [
        (10, "A"),
        (10, "B"),
    ]
    assert "gamma" in {user["handle"] for user in default_doc["users"]}
    assert (11, "C") in {
        (problem["contest_id"], problem["problem_index"]) for problem in default_doc["problems"]
    }
    default_skills = {user["handle"]: user["skill"] for user in default_doc["users"]}
    ignore_skills = {user["handle"]: user["skill"] for user in ignore_doc["users"]}
    gap = max(abs(default_skills[handle] - ignore_skills[handle]) for handle in ignore_skills)
    assert gap > 1e-6


def test_one_handle_exits_2(tmp_path: Path) -> None:
    source = tmp_path / "one.csv"
    dest = tmp_path / "one.json"
    write_observations(
        source,
        [
            ("only", 1, "A", "Alpha", _url(1, "A"), "solved"),
            ("only", 1, "B", "Beta", _url(1, "B"), "failed"),
        ],
    )
    assert run(["--input", str(source), "--output", str(dest)]) == 2
    assert not dest.exists()


def test_duplicate_key_exits_1(tmp_path: Path) -> None:
    source = tmp_path / "dup.csv"
    dest = tmp_path / "dup.json"
    write_observations(
        source,
        [
            ("ann", 1, "A", "Alpha", _url(1, "A"), "solved"),
            ("bob", 1, "A", "Alpha", _url(1, "A"), "failed"),
            ("ann", 1, "A", "Alpha", _url(1, "A"), "none"),
        ],
    )
    assert run(["--input", str(source), "--output", str(dest)]) == 1


def test_missing_column_exits_1(tmp_path: Path) -> None:
    source = tmp_path / "missing.csv"
    dest = tmp_path / "missing.json"
    source.write_text(
        "handle,contest_id,problem_index,problem_name,problem_url\n"
        "alice,1,A,Alpha,https://codeforces.com/contest/1/problem/A\n"
        "bob,1,A,Alpha,https://codeforces.com/contest/1/problem/A\n",
        encoding="utf-8",
    )
    assert run(["--input", str(source), "--output", str(dest)]) == 1


def test_recovery(tmp_path: Path) -> None:
    rng = np.random.Generator(np.random.PCG64(0))
    true_skills = np.linspace(-3.0, 3.0, 12)
    assert float(np.max(true_skills) - np.min(true_skills)) > 2.0
    true_locations = np.linspace(-2.0, 2.0, 8)
    true_scales = np.array([0.9, 1.0, 1.1, 0.95, 1.05, 1.0, 0.85, 1.15])
    assert float(np.max(np.abs(true_scales - 1.0))) <= 0.2
    probabilities = expit(
        (true_skills[:, None] - true_locations[None, :]) / true_scales[None, :]
    )
    draws = rng.binomial(1, probabilities)
    handles = [f"u{index:02d}" for index in range(12)]
    indexes = [chr(ord("A") + index) for index in range(8)]
    rows: list[tuple[object, ...]] = []
    for user in range(12):
        for problem in range(8):
            outcome = "solved" if int(draws[user, problem]) == 1 else "failed"
            rows.append(
                (
                    handles[user],
                    2000,
                    indexes[problem],
                    f"Problem {indexes[problem]}",
                    _url(2000, indexes[problem]),
                    outcome,
                )
            )
    doc = _fit(tmp_path, "recovery", rows)
    assert_constraint(doc)
    assert [user["handle"] for user in doc["users"]] == handles
    estimated = np.array([user["skill"] for user in doc["users"]], dtype=np.float64)
    center = true_skills - np.mean(true_skills)
    truth = center / np.sqrt(np.mean(np.square(center)))
    correlation = float(spearmanr(estimated, truth).statistic)
    assert correlation > 0.8


def test_usage_exits_1(tmp_path: Path) -> None:
    assert run([]) == 1
    assert run(["--input", str(tmp_path / "in.csv")]) == 1
    assert run(["--output", str(tmp_path / "out.json")]) == 1


def test_bad_y_exits_1(tmp_path: Path) -> None:
    source = tmp_path / "obs.csv"
    dest = tmp_path / "fit.json"
    write_observations(
        source,
        [
            ("ann", 1, "A", "Alpha", _url(1, "A"), "solved"),
            ("bob", 1, "A", "Alpha", _url(1, "A"), "failed"),
        ],
    )
    argv = ["--input", str(source), "--output", str(dest)]
    assert run([*argv, "--y", "solved=1,failed=0"]) == 1
    assert run([*argv, "--y", "solved=1,failed=0,none=drop"]) == 1
    assert run([*argv, "--y", "solved=1,failed=0,none=0,solved=1"]) == 1


def test_bad_outcome_exits_1(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    dest = tmp_path / "bad.json"
    write_observations(
        source,
        [
            ("ann", 1, "A", "Alpha", _url(1, "A"), "solved"),
            ("bob", 1, "A", "Alpha", _url(1, "A"), "skipped"),
        ],
    )
    assert run(["--input", str(source), "--output", str(dest)]) == 1


def test_conflicting_problem_exits_1(tmp_path: Path) -> None:
    source = tmp_path / "conflict.csv"
    dest = tmp_path / "conflict.json"
    write_observations(
        source,
        [
            ("ann", 1, "A", "Alpha", _url(1, "A"), "solved"),
            ("bob", 1, "A", "Beta", _url(1, "A"), "failed"),
        ],
    )
    assert run(["--input", str(source), "--output", str(dest)]) == 1


def test_header_only_exits_2(tmp_path: Path) -> None:
    source = tmp_path / "empty.csv"
    dest = tmp_path / "empty.json"
    source.write_text(
        "handle,contest_id,problem_index,problem_name,problem_url,outcome\n",
        encoding="utf-8",
    )
    assert run(["--input", str(source), "--output", str(dest)]) == 2


def test_all_ignored_exits_2(tmp_path: Path) -> None:
    source = tmp_path / "ignored.csv"
    dest = tmp_path / "ignored.json"
    write_observations(
        source,
        [
            ("ann", 1, "A", "Alpha", _url(1, "A"), "none"),
            ("bob", 1, "A", "Alpha", _url(1, "A"), "none"),
        ],
    )
    assert (
        run(
            [
                "--input",
                str(source),
                "--output",
                str(dest),
                "--y",
                "solved=1,failed=0,none=ignore",
            ]
        )
        == 2
    )


def test_ignore_leaves_one_user_exits_2(tmp_path: Path) -> None:
    source = tmp_path / "one-left.csv"
    dest = tmp_path / "one-left.json"
    write_observations(
        source,
        [
            ("ann", 1, "A", "Alpha", _url(1, "A"), "solved"),
            ("ann", 1, "B", "Beta", _url(1, "B"), "failed"),
            ("bob", 1, "A", "Alpha", _url(1, "A"), "none"),
            ("bob", 1, "B", "Beta", _url(1, "B"), "none"),
        ],
    )
    assert (
        run(
            [
                "--input",
                str(source),
                "--output",
                str(dest),
                "--y",
                "solved=1,failed=0,none=ignore",
            ]
        )
        == 2
    )


def test_missing_file_exits_1(tmp_path: Path) -> None:
    assert (
        run(
            [
                "--input",
                str(tmp_path / "missing.csv"),
                "--output",
                str(tmp_path / "fit.json"),
            ]
        )
        == 1
    )


def test_bad_contest_id_exits_1(tmp_path: Path) -> None:
    source = tmp_path / "contest.csv"
    dest = tmp_path / "contest.json"
    write_observations(
        source,
        [
            ("ann", "x", "A", "Alpha", _url(1, "A"), "solved"),
            ("bob", "x", "A", "Alpha", _url(1, "A"), "failed"),
        ],
    )
    assert run(["--input", str(source), "--output", str(dest)]) == 1


def test_likelihood_source_has_no_outcome_strings() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "cfrating" / "fit" / "logistic.py"
    ).read_text(encoding="utf-8")
    for word in ("solved", "failed", "none"):
        assert word not in source


def test_package_does_not_import_siblings() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "cfrating" / "fit"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "cfrating.gather" not in text
        assert "cfrating.plot" not in text
