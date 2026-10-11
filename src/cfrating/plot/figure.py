import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.text import Annotation

from cfrating.plot.load import Fit, Problem, User

_HIT_RADIUS_PX = 8.0
_SAMPLE_COUNT = 400
_X_PAD = 1.5
_HOVER_STATE = "_cfrating_hover"


@dataclass(frozen=True)
class _HoverState:
    ax: Axes
    xmin: float
    xmax: float
    users: tuple[User, ...]
    problems: tuple[Problem, ...]


def build_figure(fit: Fit) -> Figure:
    fig, ax = plt.subplots()
    xmin, xmax = _x_limits(fit)
    xs = np.linspace(xmin, xmax, _SAMPLE_COUNT)
    for problem in fit.problems:
        ax.plot(xs, _logistic(xs, problem.location, problem.scale))
    for user in fit.users:
        ax.plot(
            [user.skill, user.skill],
            [0.0, 1.0],
            linestyle="--",
            color="0.25",
        )
    ax.set_xlabel("skill")
    ax.set_ylabel("P(solve)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.0)
    setattr(
        fig,
        _HOVER_STATE,
        _HoverState(
            ax=ax,
            xmin=xmin,
            xmax=xmax,
            users=fit.users,
            problems=fit.problems,
        ),
    )
    _connect_hover(fig, ax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_autoscale_on(False)
    return fig


def hit_test(fig: Figure, pixel_x: float, pixel_y: float) -> str | None:
    state = getattr(fig, _HOVER_STATE, None)
    if not isinstance(state, _HoverState):
        return None
    x = float(pixel_x)
    y = float(pixel_y)
    user = _closest_user(state, x, y)
    if user is not None:
        return _user_text(user)
    problem = _closest_problem(state, x, y)
    if problem is not None:
        return _problem_text(problem)
    return None


def _x_limits(fit: Fit) -> tuple[float, float]:
    values = [user.skill for user in fit.users]
    values.extend(problem.location for problem in fit.problems)
    return min(values) - _X_PAD, max(values) + _X_PAD


def _logistic(x: np.ndarray, location: float, scale: float) -> np.ndarray:
    z = np.clip((x - location) / scale, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _closest_user(state: _HoverState, pixel_x: float, pixel_y: float) -> User | None:
    best: tuple[float, User] | None = None
    for user in state.users:
        start = state.ax.transData.transform((user.skill, 0.0))
        end = state.ax.transData.transform((user.skill, 1.0))
        distance = _segment_distance(
            pixel_x,
            pixel_y,
            float(start[0]),
            float(start[1]),
            float(end[0]),
            float(end[1]),
        )
        if distance <= _HIT_RADIUS_PX and (best is None or distance < best[0]):
            best = (distance, user)
    if best is None:
        return None
    return best[1]


def _closest_problem(
    state: _HoverState, pixel_x: float, pixel_y: float
) -> Problem | None:
    xs = np.linspace(state.xmin, state.xmax, _SAMPLE_COUNT)
    best: tuple[float, Problem] | None = None
    for problem in state.problems:
        points = state.ax.transData.transform(
            np.column_stack((xs, _logistic(xs, problem.location, problem.scale)))
        )
        distance = _polyline_distance(pixel_x, pixel_y, np.asarray(points, dtype=float))
        if distance <= _HIT_RADIUS_PX and (best is None or distance < best[0]):
            best = (distance, problem)
    if best is None:
        return None
    return best[1]


def _segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    length_sq = abx * abx + aby * aby
    if length_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _polyline_distance(px: float, py: float, points: np.ndarray) -> float:
    if len(points) == 0:
        return math.inf
    if len(points) == 1:
        return math.hypot(float(points[0, 0]) - px, float(points[0, 1]) - py)
    start = points[:-1]
    delta = points[1:] - start
    length_sq = np.einsum("ij,ij->i", delta, delta)
    to_point = np.array([px, py], dtype=float) - start
    dots = np.einsum("ij,ij->i", to_point, delta)
    t = np.zeros(len(start), dtype=float)
    positive = length_sq > 0.0
    t[positive] = np.clip(dots[positive] / length_sq[positive], 0.0, 1.0)
    closest = start + t[:, np.newaxis] * delta
    offset = closest - np.array([px, py], dtype=float)
    return float(np.min(np.hypot(offset[:, 0], offset[:, 1])))


def _user_text(user: User) -> str:
    return f"{user.handle}\nskill {_format_number(user.skill)}"


def _problem_text(problem: Problem) -> str:
    return (
        f"{problem.name}\n"
        f"{problem.url}\n"
        f"location {_format_number(problem.location)}\n"
        f"scale {_format_number(problem.scale)}"
    )


def _format_number(value: float) -> str:
    return format(value, ".6g")


def _connect_hover(fig: Figure, ax: Axes) -> None:
    annotation = ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.92},
        arrowprops={"arrowstyle": "-"},
        zorder=5,
    )
    annotation.set_visible(False)
    annotation.set_clip_on(False)

    def on_move(event: MouseEvent) -> None:
        if event.x is None or event.y is None:
            _hide(fig, annotation)
            return
        text = hit_test(fig, float(event.x), float(event.y))
        if text is None:
            _hide(fig, annotation)
            return
        if event.xdata is not None and event.ydata is not None:
            annotation.xy = (float(event.xdata), float(event.ydata))
        else:
            data = ax.transData.inverted().transform((event.x, event.y))
            annotation.xy = (float(data[0]), float(data[1]))
        annotation.set_text(text)
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def _hide(fig: Figure, annotation: Annotation) -> None:
    if annotation.get_visible():
        annotation.set_visible(False)
        fig.canvas.draw_idle()
