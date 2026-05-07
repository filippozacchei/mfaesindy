"""Run and plot the latent-dynamics sharing study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from utils import create_sharing_figure, run_full_study


def plot_sharing_study(
    sample_index: int = 0,
    time_index: int = -1,
    save_path: str | Path | None = None,
) -> dict[str, object]:
    results = run_full_study()
    fig, _ = create_sharing_figure(results, sample_index=sample_index, time_index=time_index)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0, help="Held-out sample to visualize.")
    parser.add_argument("--time-index", type=int, default=-1, help="Time index for field-space comparison.")
    parser.add_argument("--save-path", type=str, default=None, help="Optional path to save the figure.")
    args = parser.parse_args()

    plot_sharing_study(
        sample_index=args.sample_index,
        time_index=args.time_index,
        save_path=args.save_path,
    )
    plt.show()


if __name__ == "__main__":
    main()
