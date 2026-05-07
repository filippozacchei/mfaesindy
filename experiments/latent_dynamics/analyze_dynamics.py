"""Analyze SINDy recovery and encoder-pushforward latent derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from utils import analyze_dynamics_results, create_dynamics_figure, run_full_study


def analyze_dynamics() -> dict[str, object]:
    return analyze_dynamics_results(run_full_study())


def plot_dynamics_analysis(
    sample_index: int = 0,
    save_path: str | Path | None = None,
) -> dict[str, object]:
    analysis = analyze_dynamics()
    fig, _ = create_dynamics_figure(analysis, sample_index=sample_index)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    plot_dynamics_analysis(sample_index=args.sample_index, save_path=args.save_path)
    plt.show()


if __name__ == "__main__":
    main()
