import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

USER_AGENT = "cfrating/0.1 (+https://github.com/SebastianMestre/cfrating)"
RATED_LIST_URL = "https://codeforces.com/api/user.ratedList?activeOnly=true"
CONTEST_LIST_URL = "https://codeforces.com/api/contest.list?gym=false"
TIMEOUT_SECONDS = 120
MAX_RETRIES = 5

Fetch = Callable[[str], tuple[int, bytes]]
Sleep = Callable[[float], None]


class CodeforcesError(Exception):
    pass


class TransportError(OSError):
    pass


class _Retry(Exception):
    pass


def standings_url(contest_id: int) -> str:
    return f"https://codeforces.com/api/contest.standings?contestId={contest_id}"


def urllib_fetch(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read()
            return int(response.status), body
    except urllib.error.HTTPError as exc:
        body = exc.read()
        if isinstance(body, str):
            body = body.encode("utf-8")
        return int(exc.code), body
    except urllib.error.URLError as exc:
        raise TransportError(str(exc.reason)) from exc
    except TimeoutError as exc:
        raise TransportError(str(exc)) from exc


class CodeforcesClient:
    def __init__(self, fetch: Fetch, delay: float, sleep: Sleep) -> None:
        self._fetch = fetch
        self._delay = delay
        self._sleep = sleep
        self._started = False

    def rated_list(self) -> list[Any]:
        result = self._get(RATED_LIST_URL)
        if not isinstance(result, list):
            raise CodeforcesError("rated list result is not a list")
        return result

    def contest_list(self) -> list[Any]:
        result = self._get(CONTEST_LIST_URL)
        if not isinstance(result, list):
            raise CodeforcesError("contest list result is not a list")
        return result

    def standings(self, contest_id: int) -> dict[str, Any]:
        result = self._get(standings_url(contest_id))
        if not isinstance(result, dict):
            raise CodeforcesError(f"standings for contest {contest_id} are not an object")
        return result

    def _get(self, url: str) -> Any:
        if self._started:
            self._sleep(self._delay)
        self._started = True
        print(f"Fetching {url}", file=sys.stderr)
        wait = self._delay
        last_error = f"request failed for {url}"
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                print(f"Retrying {url}", file=sys.stderr)
                self._sleep(wait)
                wait *= 2
            try:
                status, body = self._fetch(url)
                return _interpret(url, status, body)
            except _Retry as exc:
                last_error = str(exc)
            except OSError as exc:
                last_error = f"transport error for {url}: {exc}"
        raise CodeforcesError(last_error)


def _interpret(url: str, status: int, body: bytes) -> Any:
    payload = _decode_payload(body)
    if isinstance(payload, dict):
        if _is_call_limit(payload):
            raise _Retry(f"API {payload.get('status')} for {url}: {payload.get('comment', '')}")
        api_status = payload.get("status")
        if api_status not in (None, "OK"):
            raise CodeforcesError(f"API {api_status} for {url}: {payload.get('comment', '')}")
        if api_status == "OK":
            if status != 200:
                raise _Retry(f"HTTP {status} for {url}")
            if "result" not in payload:
                raise CodeforcesError(f"missing result for {url}")
            return payload["result"]
    if status != 200:
        raise _Retry(f"HTTP {status} for {url}")
    if payload is None:
        raise _Retry(f"invalid JSON for {url}")
    raise _Retry(f"JSON object expected for {url}")


def _decode_payload(body: bytes) -> Any | None:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _is_call_limit(payload: dict[str, Any]) -> bool:
    comment = payload.get("comment")
    if not isinstance(comment, str):
        return False
    folded = comment.casefold()
    return "limit" in folded and "exceed" in folded
