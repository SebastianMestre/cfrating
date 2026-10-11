import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from cfrating.plot.figure import build_figure
from cfrating.plot.load import FitFileError, load_fit


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))


def run(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0):
            return 0
        return 1
    try:
        fit = load_fit(args.input)
    except FitFileError as exc:
        print(f"cfrating-plot: {exc}", file=sys.stderr)
        return 1
    fig = build_figure(fit)
    if args.output is not None:
        try:
            fig.savefig(args.output, dpi=120, format="png")
        except OSError as exc:
            print(f"cfrating-plot: {exc}", file=sys.stderr)
            plt.close(fig)
            return 1
    if args.no_show:
        plt.close(fig)
        return 0
    plt.show()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfrating-plot")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser


if __name__ == "__main__":
    main()
