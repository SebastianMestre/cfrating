import math
from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

from cfrating.fit.labels import apply_outcome_y

LOCATION_PRIOR_SD = 2.0
LOG_SCALE_PRIOR_SD = 0.5

_CLIP_LOW = 0.05
_CLIP_HIGH = 0.95
_TIE_STEP = 1e-3

Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]


def _mean_std(values: NDArray[np.float64]) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.sqrt(np.mean(np.square(values - mean))))
    return mean, std


def _standardize(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    mean = float(np.mean(values))
    centered = np.asarray(values - mean, dtype=np.float64)
    std = float(np.sqrt(np.mean(np.square(centered))))
    if std == 0.0 or not math.isfinite(std):
        return centered, std
    return centered / std, std


def _group_mean(
    values: NDArray[np.float64],
    index: NDArray[np.intp],
    groups: int,
) -> NDArray[np.float64]:
    totals = np.zeros(groups, dtype=np.float64)
    counts = np.zeros(groups, dtype=np.float64)
    np.add.at(totals, index, values)
    np.add.at(counts, index, 1.0)
    if np.any(counts == 0.0):
        raise ValueError("empty group in design")
    return totals / counts


def _select(
    y: NDArray[np.float64],
    keep: NDArray[np.bool_],
    user_index: np.ndarray,
    problem_index: np.ndarray,
    n_users: int,
    n_problems: int,
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp]]:
    y = np.array(y, dtype=np.float64, copy=True)
    users = np.array(user_index, dtype=np.intp, copy=True)
    problems = np.array(problem_index, dtype=np.intp, copy=True)
    if y.ndim != 1 or users.shape != y.shape or problems.shape != y.shape:
        raise ValueError("outcome and index lengths differ")
    if y.size == 0 or not bool(np.all(keep)):
        raise ValueError("every row passed to the fitter must be kept")
    if np.unique(users).size != n_users or np.unique(problems).size != n_problems:
        raise ValueError("index does not match group counts")
    if int(users.min()) < 0 or int(problems.min()) < 0:
        raise ValueError("negative index")
    if int(users.max()) >= n_users or int(problems.max()) >= n_problems:
        raise ValueError("index out of range")
    return y, users, problems


def starting_parameters(
    outcomes: np.ndarray,
    user_index: np.ndarray,
    problem_index: np.ndarray,
    n_users: int,
    n_problems: int,
    table: Mapping[str, float | None],
) -> NDArray[np.float64]:
    y, keep = apply_outcome_y(outcomes, table)
    y, users, problems = _select(y, keep, user_index, problem_index, n_users, n_problems)
    skills = np.asarray(
        logit(np.clip(_group_mean(y, users, n_users), _CLIP_LOW, _CLIP_HIGH)),
        dtype=np.float64,
    )
    if np.all(skills == skills[0]):
        skills = skills + np.arange(n_users, dtype=np.float64) * _TIE_STEP
    locations = -np.asarray(
        logit(np.clip(_group_mean(y, problems, n_problems), _CLIP_LOW, _CLIP_HIGH)),
        dtype=np.float64,
    )
    log_scales = np.zeros(n_problems, dtype=np.float64)
    return np.concatenate([skills, locations, log_scales])


def loss_and_gradient(
    theta: NDArray[np.floating],
    n_users: int,
    n_problems: int,
    user_index: NDArray[np.intp],
    problem_index: NDArray[np.intp],
    y: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    theta = np.asarray(theta, dtype=np.float64)
    expected = n_users + 2 * n_problems
    if theta.shape != (expected,):
        raise ValueError("unexpected parameter vector length")
    skills_free = theta[:n_users]
    locations = theta[n_users : n_users + n_problems]
    log_scales = theta[n_users + n_problems :]
    skills, std = _standardize(skills_free)
    undefined = math.inf, np.full(theta.shape, np.nan, dtype=np.float64)
    if std == 0.0 or not math.isfinite(std):
        return undefined
    scales = np.exp(log_scales)
    if not np.all(np.isfinite(scales)) or not np.all(scales > 0.0):
        return undefined
    t = (skills[user_index] - locations[problem_index]) / scales[problem_index]
    residual = expit(t) - y
    location_var = LOCATION_PRIOR_SD**2
    log_scale_var = LOG_SCALE_PRIOR_SD**2
    nll = float(np.sum(np.logaddexp(0.0, t) - y * t))
    nll += float(np.sum(np.square(locations) / (2.0 * location_var)))
    nll += float(np.sum(np.square(log_scales) / (2.0 * log_scale_var)))
    if not math.isfinite(nll):
        return nll, np.full(theta.shape, np.nan, dtype=np.float64)
    row_scale = scales[problem_index]
    skill_grad = np.zeros(n_users, dtype=np.float64)
    location_grad = np.zeros(n_problems, dtype=np.float64)
    log_scale_grad = np.zeros(n_problems, dtype=np.float64)
    np.add.at(skill_grad, user_index, residual / row_scale)
    np.add.at(location_grad, problem_index, -residual / row_scale)
    np.add.at(log_scale_grad, problem_index, -residual * t)
    location_grad += locations / location_var
    log_scale_grad += log_scales / log_scale_var
    count = float(n_users)
    skill_jacobian = (
        np.eye(n_users) - (1.0 / count) - (np.outer(skills, skills) / count)
    ) / std
    free_skill_grad = skill_jacobian.T @ skill_grad
    gradient = np.concatenate([free_skill_grad, location_grad, log_scale_grad])
    return nll, gradient


def make_objective(
    outcomes: np.ndarray,
    user_index: np.ndarray,
    problem_index: np.ndarray,
    n_users: int,
    n_problems: int,
    table: Mapping[str, float | None],
) -> Objective:
    y, keep = apply_outcome_y(outcomes, table)
    y, users, problems = _select(y, keep, user_index, problem_index, n_users, n_problems)

    def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        return loss_and_gradient(theta, n_users, n_problems, users, problems, y)

    return objective


def _finalize(
    theta: NDArray[np.float64],
    n_users: int,
    n_problems: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    skills_free = np.asarray(theta[:n_users], dtype=np.float64)
    locations = np.array(theta[n_users : n_users + n_problems], dtype=np.float64, copy=True)
    log_scales = np.asarray(theta[n_users + n_problems :], dtype=np.float64)
    skills, std = _standardize(skills_free)
    if std == 0.0 or not math.isfinite(std):
        raise RuntimeError("skill standard deviation is 0")
    scales = np.exp(log_scales)
    mean, skill_std = _mean_std(skills)
    if abs(mean) >= 1e-8 or abs(skill_std - 1.0) >= 1e-8:
        if skill_std == 0.0 or not math.isfinite(skill_std):
            raise RuntimeError("skill standard deviation is 0")
        skills = (skills - mean) / skill_std
        locations = (locations - mean) / skill_std
        scales = scales / skill_std
    if (
        not np.all(np.isfinite(skills))
        or not np.all(np.isfinite(locations))
        or not np.all(np.isfinite(scales))
        or not np.all(scales > 0.0)
    ):
        raise RuntimeError("optimizer returned a non-finite result")
    return skills, locations, scales


def fit_outcomes(
    outcomes: np.ndarray,
    user_index: np.ndarray,
    problem_index: np.ndarray,
    n_users: int,
    n_problems: int,
    table: Mapping[str, float | None],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    theta0 = starting_parameters(
        outcomes, user_index, problem_index, n_users, n_problems, table
    )
    objective = make_objective(
        outcomes, user_index, problem_index, n_users, n_problems, table
    )
    result = minimize(objective, theta0, method="L-BFGS-B", jac=True)
    if (
        result.x.shape != theta0.shape
        or not np.all(np.isfinite(result.x))
        or not np.isfinite(result.fun)
    ):
        raise RuntimeError("optimizer returned a non-finite result")
    return _finalize(np.asarray(result.x, dtype=np.float64), n_users, n_problems)
