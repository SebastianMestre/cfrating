# Interfaces

`cfrating-gather`, `cfrating-fit`, and `cfrating-plot` exchange data through two files. A rewrite of one program can leave the others in place if it reads and writes these files.

```
cfrating-gather  -- observations.csv -->  cfrating-fit  -- fit.json -->  cfrating-plot
```

Code is split into `src/cfrating/gather/`, `src/cfrating/fit/`, and `src/cfrating/plot/`. Each package reads and writes the files on its own. The packages do not import one another.

## observations.csv

UTF-8, no byte-order mark, LF line endings, RFC 4180 quoting. One header row. Columns, in this order:

| column | contents |
| --- | --- |
| handle | Codeforces handle, spelled as the API spells it |
| contest_id | integer contest id |
| problem_index | statement index, such as `A` or `D2` |
| problem_name | problem title |
| problem_url | absolute `https` URL |
| outcome | `solved`, `failed`, or `none` |

Those six columns are required. Further columns are ignored. A duplicate `(handle, contest_id, problem_index)` key is an error in `cfrating-fit`.

For an ordinary round, `problem_url` is `https://codeforces.com/contest/{contest_id}/problem/{problem_index}`. Downstream programs copy the string as written.

Sort rows by `contest_id` ascending, then by the problem order in the standings `problems` array, then by `handle` ascending in Unicode code-point order.

There is one row for each problem in a contest when the handle took part in that contest. Taking part means the handle is a member of a party whose `participantType` is `CONTESTANT` or `OUT_OF_COMPETITION`. `VIRTUAL`, `PRACTICE`, and `MANAGER` do not count. Match handles case-insensitively, and write the API spelling. A handle on a team gets that team's problem results. If the same handle has two parties of the same counted type in one contest, gathering fails. If both a `CONTESTANT` party and an `OUT_OF_COMPETITION` party exist, keep the `CONTESTANT` party.

`outcome` describes the official contest standing, so submissions after the contest do not appear.

- `solved` means the party has full points on the problem.
- `failed` means the party submitted and does not have full points.
- `none` means the party did not submit.

Full points: when the problem object has a positive `points` field, the result's `points` must be greater than or equal to it. When the problem object has no positive `points` field, any result with `points` greater than 0 is full points. A submission exists when `rejectedAttemptCount` is greater than 0, or `bestSubmissionTimeSeconds` is present, or the result's `points` are greater than 0. Missing `points` and `rejectedAttemptCount` are treated as 0. `problemResults[i]` belongs to `problems[i]`.

Examples for a problem whose `points` field is 500:

| result | outcome |
| --- | --- |
| points 500, rejectedAttemptCount 1, bestSubmissionTimeSeconds 100 | solved |
| points 0, rejectedAttemptCount 2 | failed |
| points 200, bestSubmissionTimeSeconds 80 | failed |
| points 0, rejectedAttemptCount 0, no bestSubmissionTimeSeconds | none |

Examples when the problem object has no `points` field:

| result | outcome |
| --- | --- |
| points 1 | solved |
| points 0, rejectedAttemptCount 1 | failed |
| points 0, rejectedAttemptCount 0 | none |

## Gather config

UTF-8 JSON object used by `--config`:

```json
{
  "handles": ["tourist", "Benq"],
  "contest_ids": [1904, 1905]
}
```

`handles` is a non-empty array of strings. `contest_ids` is a non-empty array of positive integers. Other keys are ignored.

## fit.json

UTF-8 JSON. Version 1. Two-space indent, a trailing newline, and no `NaN` or `Infinity`.

```json
{
  "format": "cfrating.fit",
  "version": 1,
  "model": {
    "name": "logistic",
    "formula": "1 / (1 + exp(-(skill - location) / scale))",
    "y_solved": 1,
    "y_failed": 0,
    "y_none": 0,
    "skill_mean": 0.0,
    "skill_std": 1.0,
    "skill_std_kind": "population",
    "location_prior_sd": 2.0,
    "log_scale_prior_sd": 0.5
  },
  "users": [
    {"handle": "Benq", "skill": -0.2}
  ],
  "problems": [
    {
      "contest_id": 1904,
      "problem_index": "A",
      "problem_name": "Example",
      "problem_url": "https://codeforces.com/contest/1904/problem/A",
      "location": 0.1,
      "scale": 1.0
    }
  ]
}
```

The `model` keys are fixed. `y_solved`, `y_failed`, and `y_none` record the mapping used for that file, so a run with `--y` may differ from the default `1`, `0`, `0` shown above. `null` means that outcome was dropped. `users` is sorted by `handle`. `problems` is sorted by `contest_id`, then `problem_index`, in Unicode code-point order. The same `(contest_id, problem_index)` must carry one `problem_name` and one `problem_url` across the input rows. Users or problems with no rows left after `ignore` are omitted. If that leaves fewer than 2 users or zero problems, fit exits 2.

`skill`, `location`, and `scale` are finite. `scale` is strictly positive. With `n` users, `n >= 2`, and at least one problem:

```
mean = sum(skill) / n
std = sqrt(sum((skill - mean)^2) / n)
```

`mean` is 0 and `std` is 1, apart from floating-point error under `1e-8` in absolute value.

`format` must be `cfrating.fit` and `version` must be `1`. `cfrating-plot` also requires `model.name` to be `logistic`.

## Model

For skill `s`, location `b`, and scale `a`:

```
P(solved) = 1 / (1 + exp(-(s - b) / a))
```

The CSV keeps `solved`, `failed`, and `none` as three strings. The likelihood does not branch on those strings. It calls one mapping table. The default table treats both `failed` and `none` as did not solve:

| outcome | default label |
| --- | --- |
| solved | 1 |
| failed | 0 |
| none | 0 |

That table is `DEFAULT_OUTCOME_Y` in `src/cfrating/fit/labels.py`. `--y solved=1,failed=0,none=0` replaces it for one run. Each value is `0`, `1`, or `ignore`. `ignore` drops those rows from the likelihood. `fit.json` records the table that was used: `y_solved`, `y_failed`, and `y_none` are `0`, `1`, or `null` when that outcome was ignored.

`cfrating-fit` minimizes the Bernoulli negative log likelihood of the rows plus a normal prior. Write `u = log(scale)`. Independently for each problem, `location ~ Normal(0, 2^2)` and `u ~ Normal(0, 0.5^2)`. Each prior contributes `x^2 / (2 σ^2)` to the objective.

On every evaluation, standardize the free skill vector to mean 0 and population standard deviation 1 before computing probabilities. The objective is undefined when that vector has population standard deviation 0. `scale = exp(u)`.

Starting skills are `logit(r)` where `r` is the mean of `y` over that user's kept rows, clipped to `[0.05, 0.95]`. Users are taken in handle-sorted order for this step. If those starting skills are identical, add `i * 1e-3` to user `i` (0-based in that order) before standardization. Starting `location` is `-logit(q)` with `q` the mean of `y` over that problem's kept rows, clipped the same way. Starting `u` is 0.

The reference optimizer is `scipy.optimize.minimize` with `L-BFGS-B`. A rewrite should match this objective. It does not have to reproduce the same floating-point parameter vector.

`cfrating-plot` draws `P(solved)` against a skill coordinate `x` for every problem, and a vertical dashed line at each user's skill.

## Default cohort

The starting user set is every handle in `user.ratedList?activeOnly=true` whose `country` field is exactly `Argentina`. Codeforces defines `activeOnly` as having taken part in a rated contest during the last month. On 11 Oct 2026 that list had 32 handles.

The starting contest set is the 50 most recent contests from `contest.list?gym=false` with `type` `CF` and `phase` `FINISHED`, ordered by `startTimeSeconds` descending and then `id` descending. `CF` is a Codeforces round. `ICPC` and `IOI` contests are left out. Whether the round was rated is not a filter. Div. 1 and Div. 2 are separate contests, so both can appear in the 50.

Discovery writes a selection snapshot, then uses it as the handle list and contest list:

```json
{
  "format": "cfrating.selection",
  "version": 1,
  "handles": ["CodigoL", "Edu175"],
  "contest_ids": [2271, 2268],
  "criteria": {
    "active_country": "Argentina",
    "active_only": true,
    "contest_type": "CF",
    "phase": "FINISHED",
    "last_rounds": 50
  }
}
```

`handles` are sorted by Unicode code point. `contest_ids` stay in the recency order above. `--config` reads `handles` and `contest_ids` from either this snapshot or the shorter config object. Other keys are ignored, so a snapshot can be passed back as `--config`.

## Commands

```
cfrating-gather --output data/observations.csv --active-country Argentina --last-rounds 50
cfrating-gather --output data/observations.csv --handles tourist,Benq --contests 2268,2269
cfrating-gather --output data/observations.csv --config examples/gather.json
cfrating-fit --input data/observations.csv --output data/fit.json
cfrating-fit --input data/observations.csv --output data/fit.json --y solved=1,failed=0,none=ignore
cfrating-plot --input data/fit.json
cfrating-plot --input data/fit.json --output plot.png --no-show
```

`--output` is required for gather and fit. `--input` is required for fit and plot.

Gather takes one of three mutually exclusive selections: `--active-country` together with `--last-rounds`, or `--handles` together with `--contests`, or `--config`. `--handles` is a comma-separated list. `--contests` is a comma-separated list of positive integers. `--last-rounds` is a positive integer. `--active-country` is matched exactly against the API `country` field.

Discovery calls `https://codeforces.com/api/user.ratedList?activeOnly=true` and `https://codeforces.com/api/contest.list?gym=false`, then the standings calls. It writes the snapshot to `--selection`, which defaults to `selection.json` in the same directory as `--output`. If the country matches nobody, or no contest matches, gather exits 1.

`--y` is optional. When set, it must assign all three outcomes. The default is `solved=1,failed=0,none=0`.

`--delay` is the wait in seconds between Codeforces requests. The default is 2.

On the plot command, `--output` writes a PNG. The window opens unless `--no-show` is set.

Exit status is 0 on success. Exit status is 1 for usage errors, unreadable or invalid files, HTTP failures, and API failures. Exit status is 2 when fit is given fewer than 2 distinct handles, zero rows, or zero problems.

Gather calls `https://codeforces.com/api/contest.standings?contestId={id}` with no other query parameters. Codeforces rejects extra parameters on that method for anonymous requests. The response is the official standings. The participant-type filter still applies to the rows that come back.

It waits at least `--delay` seconds between requests. A transport failure, invalid JSON, a non-OK HTTP status whose body is not a Codeforces `FAILED` object, or a comment that the call limit was exceeded is retried up to 5 times. A `FAILED` response that is not a call limit stops the run. The first retry waits `--delay` seconds, and each later retry doubles that wait.

## Plot interaction

The x-axis is skill. The y-axis is `P(solve)`, limited to 0 and 1. The x limits cover every user skill and every problem location, with 1.5 units of padding on each side.

Hovering within 8 pixels of a user line shows the handle and the skill. Hovering within 8 pixels of a curve shows the problem name, the problem URL, the location, and the scale. When a line and a curve are both within 8 pixels, the user line is shown.
