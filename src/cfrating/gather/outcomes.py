from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

Outcome = Literal["solved", "failed", "none"]

COLUMNS: tuple[str, ...] = (
    "handle",
    "contest_id",
    "problem_index",
    "problem_name",
    "problem_url",
    "outcome",
)

_COUNTED_TYPES = ("CONTESTANT", "OUT_OF_COMPETITION")


class Observation(TypedDict):
    handle: str
    contest_id: int
    problem_index: str
    problem_name: str
    problem_url: str
    outcome: str


class GatherError(Exception):
    pass


def problem_url(contest_id: int, problem_index: str) -> str:
    return f"https://codeforces.com/contest/{contest_id}/problem/{problem_index}"


def classify_outcome(problem: Mapping[str, Any], result: Mapping[str, Any]) -> Outcome:
    earned = _number(result.get("points"))
    full = _positive_problem_points(problem)
    solved = earned > 0 if full is None else earned >= full
    if solved:
        return "solved"
    if _submitted(result):
        return "failed"
    return "none"


def observation_rows(
    handles: Sequence[str],
    contests: Sequence[tuple[int, Mapping[str, Any]]],
) -> list[Observation]:
    pending: list[tuple[int, int, str, Observation]] = []
    for contest_id, standings in contests:
        problems = standings.get("problems")
        rows = standings.get("rows")
        if not isinstance(problems, list) or not isinstance(rows, list):
            raise GatherError(f"contest {contest_id} standings are missing problems or rows")
        seen_spellings: set[str] = set()
        for handle in handles:
            selected = _select_party(rows, handle, contest_id)
            if selected is None:
                continue
            row, spelling = selected
            if spelling in seen_spellings:
                continue
            seen_spellings.add(spelling)
            results = row.get("problemResults")
            if not isinstance(results, list) or len(results) != len(problems):
                raise GatherError(
                    f"contest {contest_id} problem results do not match the problem list for {spelling}"
                )
            for position, problem in enumerate(problems):
                result = results[position]
                if not isinstance(problem, dict) or not isinstance(result, dict):
                    raise GatherError(f"contest {contest_id} has a malformed problem result")
                index = problem.get("index")
                name = problem.get("name")
                if not isinstance(index, str) or not isinstance(name, str):
                    raise GatherError(f"contest {contest_id} problem is missing an index or name")
                pending.append(
                    (
                        contest_id,
                        position,
                        spelling,
                        {
                            "handle": spelling,
                            "contest_id": contest_id,
                            "problem_index": index,
                            "problem_name": name,
                            "problem_url": problem_url(contest_id, index),
                            "outcome": classify_outcome(problem, result),
                        },
                    )
                )
    pending.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in pending]


def _select_party(
    rows: Sequence[Any],
    handle: str,
    contest_id: int,
) -> tuple[Mapping[str, Any], str] | None:
    folded = handle.casefold()
    matches: dict[str, list[tuple[Mapping[str, Any], str]]] = {
        participant_type: [] for participant_type in _COUNTED_TYPES
    }
    for row in rows:
        if not isinstance(row, dict):
            raise GatherError(f"contest {contest_id} has a malformed standings row")
        party = row.get("party")
        if not isinstance(party, dict):
            raise GatherError(f"contest {contest_id} has a standings row without a party")
        participant_type = party.get("participantType")
        if participant_type not in matches:
            continue
        members = party.get("members")
        if not isinstance(members, list):
            raise GatherError(f"contest {contest_id} party is missing members")
        spelling: str | None = None
        for member in members:
            if not isinstance(member, dict):
                continue
            member_handle = member.get("handle")
            if isinstance(member_handle, str) and member_handle.casefold() == folded:
                spelling = member_handle
                break
        if spelling is None:
            continue
        matches[str(participant_type)].append((row, spelling))
    for participant_type in _COUNTED_TYPES:
        group = matches[participant_type]
        if len(group) > 1:
            raise GatherError(
                f"{handle} has {len(group)} {participant_type} parties in contest {contest_id}"
            )
    for participant_type in _COUNTED_TYPES:
        if matches[participant_type]:
            return matches[participant_type][0]
    return None


def _positive_problem_points(problem: Mapping[str, Any]) -> float | None:
    if "points" not in problem:
        return None
    value = problem["points"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _submitted(result: Mapping[str, Any]) -> bool:
    if _number(result.get("rejectedAttemptCount")) > 0:
        return True
    if "bestSubmissionTimeSeconds" in result:
        return True
    return _number(result.get("points")) > 0


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)
