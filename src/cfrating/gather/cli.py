import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cfrating.gather.codeforces import (
    CodeforcesClient,
    CodeforcesError,
    Fetch,
    Sleep,
    urllib_fetch,
)
from cfrating.gather.cohort import CohortError, finished_cf_contests, handles_in_country
from cfrating.gather.outcomes import COLUMNS, GatherError, Observation, observation_rows


class UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


@dataclass(frozen=True)
class _Args:
    mode: Literal["discovery", "handles", "config"]
    output: Path
    delay: float
    selection: Path
    country: str = ""
    last_rounds: int = 0
    handles: tuple[str, ...] = ()
    contest_ids: tuple[int, ...] = ()
    config: Path | None = None


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))


def run(
    argv: Sequence[str] | None = None,
    *,
    fetch: Fetch | None = None,
    sleep: Sleep | None = None,
) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        args = _parse_args(argv)
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return 1
    except SystemExit as exc:
        code = exc.code
        if code in (0, None):
            return 0
        if isinstance(code, int):
            return code
        return 1
    try:
        _execute(args, fetch or urllib_fetch, sleep or time.sleep)
    except (GatherError, CodeforcesError, CohortError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def _execute(args: _Args, fetch: Fetch, sleep: Sleep) -> None:
    client = CodeforcesClient(fetch, args.delay, sleep)
    if args.mode == "discovery":
        handles, contest_ids = _discover(client, args)
    elif args.mode == "config":
        if args.config is None:
            raise GatherError("config path is missing")
        handles, contest_ids = load_config(args.config)
    else:
        handles = list(args.handles)
        contest_ids = list(args.contest_ids)
    collected: list[tuple[int, Mapping[str, Any]]] = []
    for contest_id in contest_ids:
        collected.append((contest_id, client.standings(contest_id)))
    rows = observation_rows(handles, collected)
    print(f"Writing {args.output}", file=sys.stderr)
    _write_observations(args.output, rows)


def _discover(client: CodeforcesClient, args: _Args) -> tuple[list[str], list[int]]:
    handles = handles_in_country(client.rated_list(), args.country)
    if not handles:
        raise GatherError(f"no users with country {args.country}")
    contest_ids = finished_cf_contests(client.contest_list(), args.last_rounds)
    if not contest_ids:
        raise GatherError("no finished Codeforces rounds")
    print(f"Writing {args.selection}", file=sys.stderr)
    _write_selection(args.selection, handles, contest_ids, args.country, args.last_rounds)
    return handles, contest_ids


def load_config(path: Path) -> tuple[list[str], list[int]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise GatherError(f"cannot read config {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GatherError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GatherError(f"config {path} must be a JSON object")
    raw_handles = data.get("handles")
    raw_ids = data.get("contest_ids")
    if not isinstance(raw_handles, list) or not raw_handles:
        raise GatherError(f"config {path} needs a non-empty handles array")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise GatherError(f"config {path} needs a non-empty contest_ids array")
    handles = _unique_strings(raw_handles, path)
    contest_ids = _unique_contest_ids(raw_ids, path)
    return handles, contest_ids


def _unique_strings(raw_handles: list[Any], path: Path) -> list[str]:
    handles: list[str] = []
    seen: set[str] = set()
    for handle in raw_handles:
        if not isinstance(handle, str) or handle == "":
            raise GatherError(f"config {path} has an invalid handle")
        if handle in seen:
            continue
        seen.add(handle)
        handles.append(handle)
    return handles


def _unique_contest_ids(raw_ids: list[Any], path: Path) -> list[int]:
    contest_ids: list[int] = []
    seen: set[int] = set()
    for contest_id in raw_ids:
        if isinstance(contest_id, bool) or not isinstance(contest_id, int) or contest_id <= 0:
            raise GatherError(f"config {path} has an invalid contest id")
        if contest_id in seen:
            continue
        seen.add(contest_id)
        contest_ids.append(contest_id)
    return contest_ids


def _parse_args(argv: Sequence[str]) -> _Args:
    parser = _Parser(prog="cfrating-gather", description="Write observations.csv from Codeforces standings.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--active-country", default=None)
    parser.add_argument("--last-rounds", type=int, default=None)
    parser.add_argument("--handles", default=None)
    parser.add_argument("--contests", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--selection", default=None)
    namespace = parser.parse_args(list(argv))
    discovery = namespace.active_country is not None or namespace.last_rounds is not None
    explicit = namespace.handles is not None or namespace.contests is not None
    configured = namespace.config is not None
    if int(discovery) + int(explicit) + int(configured) != 1:
        raise UsageError(
            "pass exactly one of --active-country with --last-rounds, "
            "--handles with --contests, or --config"
        )
    delay = float(namespace.delay)
    if not math.isfinite(delay) or delay < 0:
        raise UsageError("--delay must be a finite number that is zero or positive")
    output = Path(namespace.output)
    selection = Path(namespace.selection) if namespace.selection is not None else output.parent / "selection.json"
    if discovery:
        if namespace.active_country is None or namespace.last_rounds is None:
            raise UsageError("--active-country and --last-rounds must be used together")
        if namespace.last_rounds <= 0:
            raise UsageError("--last-rounds must be a positive integer")
        return _Args(
            mode="discovery",
            output=output,
            delay=delay,
            selection=selection,
            country=namespace.active_country,
            last_rounds=namespace.last_rounds,
        )
    if explicit:
        if namespace.handles is None or namespace.contests is None:
            raise UsageError("--handles and --contests must be used together")
        return _Args(
            mode="handles",
            output=output,
            delay=delay,
            selection=selection,
            handles=tuple(_parse_handles(namespace.handles)),
            contest_ids=tuple(_parse_contests(namespace.contests)),
        )
    if namespace.config is None:
        raise UsageError("--config is required")
    return _Args(
        mode="config",
        output=output,
        delay=delay,
        selection=selection,
        config=Path(namespace.config),
    )


def _parse_handles(text: str) -> list[str]:
    handles: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        handle = part.strip()
        if handle == "":
            raise UsageError("--handles must be a comma-separated list of handles")
        if handle in seen:
            continue
        seen.add(handle)
        handles.append(handle)
    return handles


def _parse_contests(text: str) -> list[int]:
    contest_ids: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token.isdigit():
            raise UsageError(f"invalid contest id: {part}")
        contest_id = int(token)
        if contest_id <= 0:
            raise UsageError(f"invalid contest id: {part}")
        if contest_id in seen:
            continue
        seen.add(contest_id)
        contest_ids.append(contest_id)
    if not contest_ids:
        raise UsageError("--contests must list at least one contest")
    return contest_ids


def _write_selection(
    path: Path,
    handles: Sequence[str],
    contest_ids: Sequence[int],
    country: str,
    last_rounds: int,
) -> None:
    payload = {
        "format": "cfrating.selection",
        "version": 1,
        "handles": sorted(handles),
        "contest_ids": list(contest_ids),
        "criteria": {
            "active_country": country,
            "active_only": True,
            "contest_type": "CF",
            "phase": "FINISHED",
            "last_rounds": last_rounds,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_observations(path: Path, rows: Sequence[Observation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row[column] for column in COLUMNS])


if __name__ == "__main__":
    main()
