from collections.abc import Mapping, Sequence
from typing import Any


class CohortError(Exception):
    pass


def handles_in_country(users: Sequence[object], country: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for user in users:
        if not isinstance(user, Mapping):
            continue
        if user.get("country") != country:
            continue
        handle = user.get("handle")
        if not isinstance(handle, str) or handle == "" or handle in seen:
            continue
        seen.add(handle)
        found.append(handle)
    found.sort()
    return found


def finished_cf_contests(contests: Sequence[object], last_rounds: int) -> list[int]:
    if last_rounds <= 0:
        return []
    ranked: list[tuple[int, int]] = []
    for contest in contests:
        if not isinstance(contest, Mapping):
            continue
        if contest.get("type") != "CF" or contest.get("phase") != "FINISHED":
            continue
        contest_id = _positive_int(contest.get("id"))
        if contest_id is None:
            raise CohortError("finished Codeforces contest is missing a positive integer id")
        start = contest.get("startTimeSeconds")
        if isinstance(start, bool) or not isinstance(start, int):
            raise CohortError(f"contest {contest_id} is missing an integer startTimeSeconds")
        ranked.append((start, contest_id))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[int] = []
    seen: set[int] = set()
    for _start, contest_id in ranked:
        if contest_id in seen:
            continue
        seen.add(contest_id)
        selected.append(contest_id)
        if len(selected) == last_rounds:
            break
    return selected


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value
