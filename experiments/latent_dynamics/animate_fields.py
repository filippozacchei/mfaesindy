"""Animate LF and HF field evolution for the latent-dynamics study dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from utils import TransientFieldDatasetConfig, create_field_animation, generate_multi_fidelity_dataset


def animate_fields(
    sample_index: int = 0,
    save_path: str | Path | None = None,
) -> animation.FuncAnimation:
    dataset = generate_multi_fidelity_dataset(TransientFieldDatasetConfig())
    ani, _ = create_field_animation(dataset, sample_index=sample_index)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            ani.save(output_path, writer=animation.PillowWriter(fps=15))
        else:
            ani.save(output_path, writer=animation.FFMpegWriter(fps=15))

    return ani


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0, help="Held-out sample to animate.")
    parser.add_argument("--save-path", type=str, default=None, help="Optional .gif or .mp4 output path.")
    args = parser.parse_args()

    animate_fields(sample_index=args.sample_index, save_path=args.save_path)
    plt.show()


if __name__ == "__main__":
    main()
