# cfrating

Estimate a skill for each Codeforces user and a location and scale for each problem, using official contest standings.

`cfrating-gather`, `cfrating-fit`, and `cfrating-plot` pass data through `observations.csv` and `fit.json`. The columns, the JSON fields, and the likelihood are specified in [docs/interfaces.md](docs/interfaces.md).

Python 3.14.

```bash
uv sync
```

## Gather

The starting cohort is active Argentina users and the 50 most recent finished Codeforces rounds. Active means the handle is in `user.ratedList?activeOnly=true` (a rated contest in the last month) and `country` is `Argentina`. A round is a contest with `type` `CF` and `phase` `FINISHED`. Div. 1 and Div. 2 are separate contests. Rated and unrated rounds are both included.

```bash
uv run cfrating-gather --output data/observations.csv --active-country Argentina --last-rounds 50
```

That writes `data/selection.json` (the handles and contest ids) and `data/observations.csv`. One row is a user in a contest they entered, and one problem from that contest. `outcome` is `solved`, `failed`, or `none`. `none` means no submission during the contest. The standings do not include practice after the contest ends.

Requests wait 2 seconds apart. Expect a few minutes for 50 rounds.

A fixed handle list and contest list:

```bash
uv run cfrating-gather --output data/observations.csv --config examples/gather.json
```

## Fit

```bash
uv run cfrating-fit --input data/observations.csv --output data/fit.json
```

By default, `solved` is a success and both `failed` and `none` count as did not solve. The mapping is the table `DEFAULT_OUTCOME_Y` in `src/cfrating/fit/labels.py`. The fitter does not branch on the outcome strings anywhere else. To drop non-submissions from the likelihood:

```bash
uv run cfrating-fit --input data/observations.csv --output data/fit.json --y solved=1,failed=0,none=ignore
```

`0`, `1`, and `ignore` are the three values. The mapping used for a run is stored on the `model` object in `fit.json`.

Each user gets a skill. The skills of the fitted users have mean 0 and population standard deviation 1. Each problem gets a location `b` and a scale `a`:

```
P(solve) = 1 / (1 + exp(-(skill - b) / a))
```

## Plot

```bash
uv run cfrating-plot --input data/fit.json
```

The x-axis is skill. Each curve is one problem's `P(solve)`. Each dashed vertical line is one user. Hover a line for the handle, or a curve for the problem name and URL.

```bash
uv run cfrating-plot --input data/fit.json --output plot.png --no-show
```

## Tests

```bash
uv run pytest
```
