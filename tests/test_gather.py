import csv
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from cfrating.gather.cli import main, run
from cfrating.gather.codeforces import (
    CONTEST_LIST_URL,
    RATED_LIST_URL,
    TIMEOUT_SECONDS,
    USER_AGENT,
    CodeforcesClient,
    CodeforcesError,
    TransportError,
    standings_url,
    urllib_fetch,
)
from cfrating.gather.cohort import finished_cf_contests, handles_in_country
from cfrating.gather.outcomes import classify_outcome, observation_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HEADER = b"handle,contest_id,problem_index,problem_name,problem_url,outcome\n"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def ok(result: object) -> tuple[int, bytes]:
    return 200, json.dumps({"status": "OK", "result": result}).encode()


def test_classify_outcome_examples() -> None:
    cases = [
        ({"points": 500}, {"points": 500, "rejectedAttemptCount": 1, "bestSubmissionTimeSeconds": 100}, "solved"),
        ({"points": 500}, {"points": 0, "rejectedAttemptCount": 2}, "failed"),
        ({"points": 500}, {"points": 200, "bestSubmissionTimeSeconds": 80}, "failed"),
        ({"points": 500}, {"points": 0, "rejectedAttemptCount": 0}, "none"),
        ({}, {"points": 1}, "solved"),
        ({}, {"points": 0, "rejectedAttemptCount": 1}, "failed"),
        ({}, {"points": 0, "rejectedAttemptCount": 0}, "none"),
    ]
    for problem, result, expected in cases:
        assert classify_outcome(problem, result) == expected


def test_missing_fields_and_non_positive_problem_points() -> None:
    assert classify_outcome({"points": 500}, {}) == "none"
    assert classify_outcome({"points": 0}, {"points": 1}) == "solved"
    assert classify_outcome({"points": 500}, {"points": 500.0, "rejectedAttemptCount": 1}) == "solved"
    present_null = {"points": 0, "rejectedAttemptCount": 0, "bestSubmissionTimeSeconds": None}
    assert classify_outcome({"points": 500}, present_null) == "failed"


def test_outcome_rows_match_design_examples() -> None:
    standings = load_fixture("outcome_standings.json")
    assert "points" not in standings["problems"][1]
    rows = observation_rows(["rejected", "full", "quiet", "partial"], [(1904, standings)])
    assert [(row["handle"], row["problem_index"], row["outcome"]) for row in rows] == [
        ("full", "A", "solved"),
        ("partial", "A", "failed"),
        ("quiet", "A", "none"),
        ("rejected", "A", "failed"),
        ("full", "B", "solved"),
        ("partial", "B", "none"),
        ("quiet", "B", "none"),
        ("rejected", "B", "failed"),
    ]
    assert rows[0]["problem_name"] == "Points Problem"
    assert rows[0]["problem_url"] == "https://codeforces.com/contest/1904/problem/A"
    assert rows[4]["problem_name"] == "Unscored Problem"
    assert rows[4]["problem_url"] == "https://codeforces.com/contest/1904/problem/B"


def test_case_insensitive_handle_uses_api_spelling() -> None:
    standings = load_fixture("case_team_standings.json")
    rows = observation_rows(["benq", "BENQ"], [(7, standings)])
    assert [(row["handle"], row["problem_index"], row["outcome"]) for row in rows] == [
        ("Benq", "A", "solved"),
        ("Benq", "B", "solved"),
    ]


def test_team_member_gets_team_results() -> None:
    standings = load_fixture("case_team_standings.json")
    rows = observation_rows(["BOB"], [(7, standings)])
    assert [(row["handle"], row["problem_index"], row["outcome"]) for row in rows] == [
        ("bob", "A", "failed"),
        ("bob", "B", "failed"),
    ]


def test_participant_types() -> None:
    standings = load_fixture("participant_standings.json")
    rows = observation_rows(
        ["ooc", "virtual", "mix", "manager", "both", "practice"],
        [(5, standings)],
    )
    assert [(row["handle"], row["problem_index"], row["outcome"], row["problem_url"]) for row in rows] == [
        ("both", "A", "failed", "https://codeforces.com/contest/5/problem/A"),
        ("mix", "A", "failed", "https://codeforces.com/contest/5/problem/A"),
        ("ooc", "A", "solved", "https://codeforces.com/contest/5/problem/A"),
    ]


@pytest.mark.parametrize("participant_type", ["CONTESTANT", "OUT_OF_COMPETITION"])
def test_duplicate_counted_party_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str], participant_type: str) -> None:
    party_row = {
        "party": {"members": [{"handle": "dup"}], "participantType": participant_type},
        "problemResults": [{"points": 1}],
    }
    standings = {
        "problems": [{"index": "A", "name": "A"}],
        "rows": [party_row, party_row],
    }
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        return ok(standings)

    output = tmp_path / "observations.csv"
    code = run(
        ["--output", str(output), "--handles", "dup", "--contests", "15", "--delay", "0"],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 1
    assert calls == [standings_url(15)]
    assert not output.exists()
    assert participant_type in capsys.readouterr().err


def test_row_sort(tmp_path: Path) -> None:
    standings = {
        3: {
            "problems": [{"index": "A", "name": "First", "points": 100}],
            "rows": [
                {
                    "party": {"members": [{"handle": "a"}], "participantType": "CONTESTANT"},
                    "problemResults": [{"points": 100}],
                },
                {
                    "party": {"members": [{"handle": "b"}], "participantType": "CONTESTANT"},
                    "problemResults": [{"points": 100}],
                },
            ],
        },
        20: {
            "problems": [
                {"index": "B", "name": "Bee, two", "points": 100},
                {"index": "A", "name": "Aye", "points": 100},
            ],
            "rows": [
                {
                    "party": {"members": [{"handle": "b"}], "participantType": "CONTESTANT"},
                    "problemResults": [
                        {"points": 0, "rejectedAttemptCount": 1},
                        {"points": 0, "rejectedAttemptCount": 0},
                    ],
                },
                {
                    "party": {"members": [{"handle": "a"}], "participantType": "CONTESTANT"},
                    "problemResults": [
                        {"points": 100},
                        {"points": 0, "rejectedAttemptCount": 1},
                    ],
                },
            ],
        },
    }
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        for contest_id, payload in standings.items():
            if url == standings_url(contest_id):
                return ok(payload)
        raise AssertionError(url)

    output = tmp_path / "observations.csv"
    code = run(
        ["--output", str(output), "--handles", "b,a", "--contests", "20,3", "--delay", "0"],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 0
    assert calls == [standings_url(20), standings_url(3)]
    raw = output.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert b'"Bee, two"' in raw
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["handle", "contest_id", "problem_index", "problem_name", "problem_url", "outcome"],
        ["a", "3", "A", "First", "https://codeforces.com/contest/3/problem/A", "solved"],
        ["b", "3", "A", "First", "https://codeforces.com/contest/3/problem/A", "solved"],
        ["a", "20", "B", "Bee, two", "https://codeforces.com/contest/20/problem/B", "solved"],
        ["b", "20", "B", "Bee, two", "https://codeforces.com/contest/20/problem/B", "failed"],
        ["a", "20", "A", "Aye", "https://codeforces.com/contest/20/problem/A", "failed"],
        ["b", "20", "A", "Aye", "https://codeforces.com/contest/20/problem/A", "none"],
    ]
    assert not (tmp_path / "selection.json").exists()


def test_absent_handle_writes_header_only(tmp_path: Path) -> None:
    standings = {
        "problems": [{"index": "A", "name": "Alone", "points": 500}],
        "rows": [
            {
                "party": {"members": [{"handle": "someone"}], "participantType": "CONTESTANT"},
                "problemResults": [{"points": 500, "rejectedAttemptCount": 0, "bestSubmissionTimeSeconds": 1}],
            }
        ],
    }

    def fetch(url: str) -> tuple[int, bytes]:
        assert url == standings_url(1)
        return ok(standings)

    output = tmp_path / "out" / "observations.csv"
    code = run(
        ["--output", str(output), "--handles", "nobody", "--contests", "1", "--delay", "0"],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 0
    assert output.read_bytes() == HEADER
    assert output.parent.is_dir()
    assert not (output.parent / "selection.json").exists()


def test_country_filter_keeps_exact_argentina() -> None:
    users = load_fixture("users_countries.json")
    assert handles_in_country(users, "Argentina") == ["Ana", "Bob", "ana", "Ñandú"]


def test_contest_filter_orders_finished_cf_and_ignores_rated() -> None:
    contests = load_fixture("contests_mixed.json")
    assert finished_cf_contests(contests, 10) == [60, 50, 100, 10]
    assert finished_cf_contests(contests, 1) == [60]
    assert finished_cf_contests(contests, 2) == [60, 50]
    assert finished_cf_contests(contests, 3) == [60, 50, 100]
    assert 100 in finished_cf_contests(contests, 10)
    assert 200 not in finished_cf_contests(contests, 10)
    assert 300 not in finished_cf_contests(contests, 10)


def test_discovery_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    users = load_fixture("discovery_users.json")
    contests = load_fixture("discovery_contests.json")
    standings = load_fixture("discovery_standings.json")
    events: list[tuple[str, object]] = []

    def fetch(url: str) -> tuple[int, bytes]:
        events.append(("fetch", url))
        if url == RATED_LIST_URL:
            return ok(users)
        if url == CONTEST_LIST_URL:
            return ok(contests)
        for contest_id, payload in standings.items():
            if url == standings_url(int(contest_id)):
                return ok(payload)
        raise AssertionError(url)

    def sleep(seconds: float) -> None:
        events.append(("sleep", seconds))

    output = tmp_path / "data" / "observations.csv"
    selection = tmp_path / "data" / "selection.json"
    code = run(
        [
            "--output",
            str(output),
            "--active-country",
            "Argentina",
            "--last-rounds",
            "50",
        ],
        fetch=fetch,
        sleep=sleep,
    )
    assert code == 0
    assert events == [
        ("fetch", RATED_LIST_URL),
        ("sleep", 2.0),
        ("fetch", CONTEST_LIST_URL),
        ("sleep", 2.0),
        ("fetch", standings_url(900)),
        ("sleep", 2.0),
        ("fetch", standings_url(800)),
    ]
    expected_selection = {
        "format": "cfrating.selection",
        "version": 1,
        "handles": ["Alpha", "zeta", "Ñandú"],
        "contest_ids": [900, 800],
        "criteria": {
            "active_country": "Argentina",
            "active_only": True,
            "contest_type": "CF",
            "phase": "FINISHED",
            "last_rounds": 50,
        },
    }
    text = selection.read_text(encoding="utf-8")
    assert text == json.dumps(expected_selection, indent=2, ensure_ascii=False) + "\n"
    assert not selection.read_bytes().startswith(b"\xef\xbb\xbf")
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["handle", "contest_id", "problem_index", "problem_name", "problem_url", "outcome"],
        ["Alpha", "800", "A", "Older", "https://codeforces.com/contest/800/problem/A", "failed"],
        ["Alpha", "900", "A", "Newer", "https://codeforces.com/contest/900/problem/A", "solved"],
        ["zeta", "900", "A", "Newer", "https://codeforces.com/contest/900/problem/A", "none"],
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Fetching" in captured.err
    assert f"Writing {selection}" in captured.err
    assert f"Writing {output}" in captured.err


def test_mixing_config_and_handles_returns_1(tmp_path: Path) -> None:
    config = tmp_path / "gather.json"
    config.write_text(json.dumps({"handles": ["tourist"], "contest_ids": [2268]}), encoding="utf-8")
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        return ok({"problems": [], "rows": []})

    output = tmp_path / "observations.csv"
    code = run(
        ["--output", str(output), "--config", str(config), "--handles", "h1,h2", "--delay", "0"],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 1
    assert calls == []
    assert not output.exists()


def test_zero_country_matches_returns_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        return ok([{"handle": "x", "country": "Brazil"}])

    output = tmp_path / "data" / "observations.csv"
    code = run(
        [
            "--output",
            str(output),
            "--active-country",
            "Argentina",
            "--last-rounds",
            "50",
            "--delay",
            "0",
        ],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 1
    assert calls == [RATED_LIST_URL]
    assert not output.exists()
    assert not (tmp_path / "data" / "selection.json").exists()
    assert "Argentina" in capsys.readouterr().err


def test_no_finished_contests_returns_1(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        if url == RATED_LIST_URL:
            return ok([{"handle": "Ana", "country": "Argentina"}])
        if url == CONTEST_LIST_URL:
            return ok([{"id": 1, "type": "ICPC", "phase": "FINISHED", "startTimeSeconds": 10}])
        raise AssertionError(url)

    output = tmp_path / "data" / "observations.csv"
    code = run(
        [
            "--output",
            str(output),
            "--active-country",
            "Argentina",
            "--last-rounds",
            "50",
            "--delay",
            "0",
        ],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 1
    assert calls == [RATED_LIST_URL, CONTEST_LIST_URL]
    assert not output.exists()
    assert not (tmp_path / "data" / "selection.json").exists()


def test_config_mode_reads_snapshot_and_skips_selection_file(tmp_path: Path) -> None:
    config = tmp_path / "selection.json"
    config.write_text(
        json.dumps(
            {
                "format": "cfrating.selection",
                "version": 1,
                "handles": ["tourist", "Benq"],
                "contest_ids": [2268, 2269],
                "criteria": {"active_country": "Argentina", "last_rounds": 50},
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        return ok(
            {
                "problems": [{"index": "A", "name": "A", "points": 500}],
                "rows": [
                    {
                        "party": {"members": [{"handle": "tourist"}], "participantType": "CONTESTANT"},
                        "problemResults": [{"points": 500, "rejectedAttemptCount": 0, "bestSubmissionTimeSeconds": 1}],
                    }
                ],
            }
        )

    output = tmp_path / "nested" / "observations.csv"
    code = run(
        ["--output", str(output), "--config", str(config), "--delay", "0"],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 0
    assert calls == [standings_url(2268), standings_url(2269)]
    assert not (output.parent / "selection.json").exists()
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["handle", "contest_id", "problem_index", "problem_name", "problem_url", "outcome"],
        ["tourist", "2268", "A", "A", "https://codeforces.com/contest/2268/problem/A", "solved"],
        ["tourist", "2269", "A", "A", "https://codeforces.com/contest/2269/problem/A", "solved"],
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"handles": [], "contest_ids": [1]},
        {"handles": ["a"], "contest_ids": []},
        {"handles": ["a"], "contest_ids": [0]},
        {"handles": ["a"], "contest_ids": [-3]},
        {"handles": [""], "contest_ids": [1]},
        {"handles": ["a"], "contest_ids": [1.5]},
        {"handles": ["a"], "contest_ids": [True]},
        [],
        {"handles": ["a"]},
    ],
)
def test_invalid_config_returns_1(tmp_path: Path, payload: object) -> None:
    config = tmp_path / "gather.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        return ok({"problems": [], "rows": []})

    code = run(
        ["--output", str(tmp_path / "o.csv"), "--config", str(config), "--delay", "0"],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 1
    assert calls == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--handles", "h1"],
        ["--contests", "1"],
        ["--active-country", "Argentina"],
        ["--last-rounds", "50"],
    ],
)
def test_partial_selection_returns_1(tmp_path: Path, extra: list[str]) -> None:
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        return ok([])

    code = run(
        ["--output", str(tmp_path / "o.csv"), *extra, "--delay", "0"],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 1
    assert calls == []


def test_http_failure_exits_1(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        return 500, b"no"

    code = run(
        ["--output", str(tmp_path / "o.csv"), "--handles", "a", "--contests", "1", "--delay", "0"],
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert code == 1
    assert len(calls) == 6


def test_retry_waits_double_and_not_before_first_attempt() -> None:
    events: list[object] = []

    def fetch(url: str) -> tuple[int, bytes]:
        events.append("fetch")
        if events.count("fetch") < 3:
            return 503, b"no"
        return ok([])

    def sleep(seconds: float) -> None:
        events.append(("sleep", seconds))

    client = CodeforcesClient(fetch, 2, sleep)
    assert client.rated_list() == []
    assert events == ["fetch", ("sleep", 2), "fetch", ("sleep", 4), "fetch"]


def test_call_limit_is_retried() -> None:
    payloads = [
        {"status": "OK", "comment": "Call limit exceeded", "result": [{"handle": "wrong"}]},
        {"status": "OK", "result": [{"handle": "right"}]},
    ]

    def fetch(url: str) -> tuple[int, bytes]:
        return 200, json.dumps(payloads.pop(0)).encode()

    client = CodeforcesClient(fetch, 0, lambda _seconds: None)
    assert client.rated_list() == [{"handle": "right"}]


def test_failed_status_is_not_retried() -> None:
    calls = 0

    def fetch(url: str) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        return 400, json.dumps({"status": "FAILED", "comment": "nope"}).encode()

    client = CodeforcesClient(fetch, 0, lambda _seconds: None)
    with pytest.raises(CodeforcesError, match="nope"):
        client.rated_list()
    assert calls == 1


def test_invalid_json_is_retried() -> None:
    bodies = [b"not-json", json.dumps({"status": "OK", "result": []}).encode()]

    def fetch(url: str) -> tuple[int, bytes]:
        return 200, bodies.pop(0)

    client = CodeforcesClient(fetch, 0, lambda _seconds: None)
    assert client.rated_list() == []


def test_delay_is_between_requests() -> None:
    events: list[tuple[str, object]] = []

    def fetch(url: str) -> tuple[int, bytes]:
        events.append(("fetch", url))
        return ok([])

    def sleep(seconds: float) -> None:
        events.append(("sleep", seconds))

    client = CodeforcesClient(fetch, 2, sleep)
    assert client.rated_list() == []
    assert client.contest_list() == []
    assert events == [
        ("fetch", RATED_LIST_URL),
        ("sleep", 2),
        ("fetch", CONTEST_LIST_URL),
    ]


def test_retries_exhausted() -> None:
    calls = 0

    def fetch(url: str) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        raise OSError("reset")

    client = CodeforcesClient(fetch, 0, lambda _seconds: None)
    with pytest.raises(CodeforcesError):
        client.rated_list()
    assert calls == 6


def test_urllib_fetch_sets_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class Response:
        status = 200

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> Response:
        headers = {name.lower(): value for name, value in request.header_items()}
        seen["user_agent"] = headers.get("user-agent")
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, body = urllib_fetch(RATED_LIST_URL)
    assert status == 200
    assert body == b"{}"
    assert seen["user_agent"] == USER_AGENT
    assert seen["timeout"] == TIMEOUT_SECONDS


def test_urllib_fetch_http_error_and_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_http(request: urllib.request.Request, timeout: float | None = None) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "unavailable",
            hdrs=None,
            fp=io.BytesIO(b"busy"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_http)
    assert urllib_fetch(CONTEST_LIST_URL) == (503, b"busy")

    def raise_url(request: urllib.request.Request, timeout: float | None = None) -> None:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url)
    with pytest.raises(TransportError, match="down"):
        urllib_fetch(RATED_LIST_URL)


def test_main_raises_systemexit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--output",
                str(tmp_path / "o.csv"),
                "--config",
                str(tmp_path / "c.json"),
                "--handles",
                "h1,h2",
            ]
        )
    assert caught.value.code == 1
