"""Plotting utilities for training diagnostics and rollout inspection."""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

cache_root = Path(tempfile.gettempdir()) / "mf_lnode_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from torch import Tensor

from mf_lnode.data.datasets import MultiFidelityTrajectorySample
from mf_lnode.data.scalers import MultiFidelityScaler


FIDELITY_COLORS = {
    "low": "#4C78A8",
    "high": "#E45756",
}
SPLIT_COLORS = {
    "train": "#54A24B",
    "val": "#F58518",
    "test": "#B279A2",
}
AVAILABILITY_MARKERS = {
    "paired": "o",
    "low_only": "^",
    "high_only": "s",
    "other": "D",
}
MAX_CHANNELS_PER_PANEL = 4

matplotlib.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def _to_numpy(tensor: Tensor) -> Any:
    return tensor.detach().cpu().numpy()


def _ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _fidelity_color(fidelity: str) -> str:
    return FIDELITY_COLORS.get(fidelity, "#72B7B2")


def _availability_category(group: MultiFidelityTrajectorySample) -> str:
    fidelities = set(group.trajectories)
    if {"low", "high"}.issubset(fidelities):
        return "paired"
    if fidelities == {"low"}:
        return "low_only"
    if fidelities == {"high"}:
        return "high_only"
    return "other"


def _parameter_label(parameter: Tensor) -> str:
    values = [f"{float(value):.3f}" for value in parameter]
    return f"mu=({', '.join(values)})"


def _channel_label(channel_index: int) -> str:
    if channel_index == 0:
        return "q"
    if channel_index == 1:
        return "v"
    return f"sensor {channel_index}"


def _measurement_formula(channel_index: int) -> str:
    formulas = {
        0: "q(t)",
        1: "v(t)",
        2: "q(t) v(t)",
        3: "q(t)^2",
        4: "v(t)^2",
        5: "sin(1.3 q(t))",
        6: "cos(1.1 v(t))",
        7: "tanh(1.5 q(t) - 0.45 v(t))",
        8: "exp(-mu0 t) q(t)",
        9: "exp(-mu0 t) v(t)",
        10: "0.5 (v(t)^2 + (mu1 q(t))^2)",
        11: "q(t) sin(mu1 t)",
        12: "v(t) cos(mu1 t)",
        13: "E(t) sin(0.5 t + mu0)",
        14: "q(t)^2 v(t)",
        15: "v(t)^2 q(t)",
    }
    if channel_index in formulas:
        return formulas[channel_index]
    offset = channel_index - 16
    return (
        f"sin(a{offset} q + b{offset}) cos(c{offset} v - d{offset}) + "
        f"0.08 E(t) + 0.04 sin(w{offset} t)"
    )


def _select_channel_indices(
    observation_dim: int,
    max_channels: int = MAX_CHANNELS_PER_PANEL,
) -> list[int]:
    if observation_dim <= max_channels:
        return list(range(observation_dim))

    selected: list[int] = [0]
    if observation_dim > 1:
        selected.append(1)

    remaining_slots = max_channels - len(selected)
    if remaining_slots <= 0:
        return selected[:max_channels]

    start = len(selected)
    last = observation_dim - 1
    if remaining_slots == 1:
        candidates = [last]
    else:
        candidates = [
            round(start + (step + 1) * (last - start) / remaining_slots)
            for step in range(remaining_slots - 1)
        ]
        candidates.append(last)
    for candidate in candidates:
        candidate = int(candidate)
        if 0 <= candidate < observation_dim and candidate not in selected:
            selected.append(candidate)

    cursor = observation_dim - 1
    while len(selected) < max_channels and cursor >= 0:
        if cursor not in selected:
            selected.append(cursor)
        cursor -= 1
    return selected[:max_channels]


def _heatmap_ticks(observation_dim: int) -> list[int]:
    if observation_dim <= 12:
        return list(range(observation_dim))
    ticks = sorted(set([0, 1, 2, 5, 7, 10, observation_dim - 1]))
    return [tick for tick in ticks if 0 <= tick < observation_dim]


def _paired_counts(groups: Sequence[MultiFidelityTrajectorySample]) -> dict[str, int]:
    counts = {"paired": 0, "low_only": 0, "high_only": 0, "other": 0}
    for group in groups:
        counts[_availability_category(group)] += 1
    return counts


def _grid_shape_from_sample(sample: Any) -> tuple[int, int]:
    height = int(sample.metadata.get("grid_height", 0))
    width = int(sample.metadata.get("grid_width", 0))
    if height <= 0 or width <= 0:
        raise ValueError("Grid metadata is missing or invalid.")
    return height, width


def _grid_shape_from_entry(entry: Mapping[str, Any]) -> tuple[int, int]:
    height = int(entry.get("grid_height", 0))
    width = int(entry.get("grid_width", 0))
    if height <= 0 or width <= 0:
        raise ValueError("Grid entry metadata is missing or invalid.")
    return height, width


def _reshape_field_snapshot(flattened_snapshot: Tensor, grid_shape: tuple[int, int]) -> Tensor:
    height, width = grid_shape
    return flattened_snapshot.reshape(height, width)


def _reshape_field_series(flattened_series: Tensor, grid_shape: tuple[int, int]) -> Tensor:
    height, width = grid_shape
    return flattened_series.reshape(flattened_series.shape[0], height, width)


def _resize_field_series(field_series: Tensor, target_shape: tuple[int, int]) -> Tensor:
    return F.interpolate(
        field_series.unsqueeze(1),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _selected_snapshot_indices(length: int, count: int = 3) -> list[int]:
    if length <= count:
        return list(range(length))
    return sorted(
        {
            0,
            max(0, length // 2),
            length - 1,
        }
    )


def _select_story_group(groups: Sequence[MultiFidelityTrajectorySample]) -> MultiFidelityTrajectorySample:
    paired = [group for group in groups if {"low", "high"}.issubset(group.trajectories)]
    if paired:
        return paired[len(paired) // 2]
    if groups:
        return groups[len(groups) // 2]
    raise ValueError("groups must contain at least one sample.")


def _draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str = "#1F2937",
) -> None:
    x, y = xy
    width, height = wh
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.73, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + width / 2, y + height * 0.38, body, ha="center", va="center", fontsize=9)


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.6,
        color="#374151",
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.04,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="#374151",
        )


def _extract_latent_entries(data: Mapping[str, Any]) -> dict[str, dict[str, Tensor]]:
    if "per_fidelity" in data:
        extracted: dict[str, dict[str, Tensor]] = {}
        for fidelity, entry in data["per_fidelity"].items():
            extracted[fidelity] = {
                "times": entry["target_times"][0],
                "latent_trajectory": entry["latent_trajectory"][0],
                "z0": entry["z0"][0],
            }
        return extracted

    extracted = {}
    for fidelity, entry in data.items():
        extracted[fidelity] = {
            "times": entry["rollout_times"],
            "latent_trajectory": entry["latent_trajectory"],
            "z0": entry["z0"],
        }
    return extracted


def plot_test_problem_summary(
    groups: Sequence[MultiFidelityTrajectorySample],
    output_path: str | Path,
) -> Path:
    """Save a narrative figure describing the synthetic test problem."""

    output_path = _ensure_parent(output_path)
    if not groups:
        raise ValueError("groups must contain at least one sample.")

    counts = _paired_counts(groups)
    group = _select_story_group(groups)
    observation_dim = max(sample.observation_dim for sample in group.trajectories.values())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0][0].axis("off")
    axes[0][0].text(0.02, 0.96, "Synthetic Test Problem", fontsize=16, fontweight="bold", va="top")
    axes[0][0].text(
        0.02,
        0.79,
        "Parameterized damped oscillator\nq'' + mu0 q' + mu1^2 q = 0",
        fontsize=13,
        va="top",
    )
    axes[0][0].text(
        0.02,
        0.52,
        "Observation model\n"
        f"- high fidelity: lifted 2D oscillator state into {observation_dim} observation channels\n"
        "- low fidelity: nonlinear, lagged, channel-mixed surrogate\n"
        "- paired and unpaired grouped samples",
        fontsize=10,
        va="top",
    )
    axes[0][0].text(
        0.02,
        0.20,
        "Dataset composition\n"
        f"paired = {counts['paired']}\n"
        f"low only = {counts['low_only']}\n"
        f"high only = {counts['high_only']}",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F1D3", edgecolor="#D6C49A"),
    )

    if {"low", "high"}.issubset(group.trajectories):
        low = group.trajectories["low"]
        high = group.trajectories["high"]
        selected_dims = _select_channel_indices(min(low.observation_dim, high.observation_dim))

        axes[0][1].plot(_to_numpy(high.times), _to_numpy(high.observations[:, 0]), color=_fidelity_color("high"), linewidth=2.3, label="high q")
        axes[0][1].plot(_to_numpy(low.times), _to_numpy(low.observations[:, 0]), color=_fidelity_color("low"), linewidth=2.0, label="low q")
        if high.observation_dim > 1:
            axes[0][1].plot(_to_numpy(high.times), _to_numpy(high.observations[:, 1]), color=_fidelity_color("high"), linestyle="--", linewidth=1.8, alpha=0.8, label="high v")
        if low.observation_dim > 1:
            axes[0][1].plot(_to_numpy(low.times), _to_numpy(low.observations[:, 1]), color=_fidelity_color("low"), linestyle="--", linewidth=1.8, alpha=0.8, label="low v")
        axes[0][1].set_title(
            f"Representative paired trajectory | {_parameter_label(group.parameter)} | "
            f"{high.observation_dim} channels"
        )
        axes[0][1].set_xlabel("time")
        axes[0][1].set_ylabel("channel value")
        axes[0][1].grid(alpha=0.25)
        axes[0][1].legend(frameon=False, ncol=2)

        axes[1][0].plot(_to_numpy(high.observations[:, 0]), _to_numpy(high.observations[:, 1]), color=_fidelity_color("high"), linewidth=2.1, label="high")
        axes[1][0].plot(_to_numpy(low.observations[:, 0]), _to_numpy(low.observations[:, 1]), color=_fidelity_color("low"), linewidth=2.0, label="low")
        axes[1][0].set_title("Phase portrait")
        axes[1][0].set_xlabel("q")
        axes[1][0].set_ylabel("v")
        axes[1][0].grid(alpha=0.25)
        axes[1][0].legend(frameon=False)

        shared_dim = min(low.observation_dim, high.observation_dim)
        overall_gap = (high.observations[:, :shared_dim] - low.observations[:, :shared_dim]).pow(2).mean(dim=-1).sqrt()
        axes[1][1].plot(
            _to_numpy(high.times),
            _to_numpy(overall_gap),
            color="#111111",
            linewidth=2.4,
            label="all-channel RMSE gap",
        )
        for dim in selected_dims:
            axes[1][1].plot(
                _to_numpy(high.times),
                _to_numpy(high.observations[:, dim] - low.observations[:, dim]),
                linewidth=1.8,
                label=_channel_label(dim),
            )
        axes[1][1].set_title("Cross-fidelity observation gap")
        axes[1][1].set_xlabel("time")
        axes[1][1].set_ylabel("gap / RMSE")
        axes[1][1].grid(alpha=0.25)
        axes[1][1].legend(frameon=False)
    else:
        for ax in (axes[0][1], axes[1][0], axes[1][1]):
            ax.axis("off")

    fig.suptitle("Test Problem and Multi-Fidelity Observation Model", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_measurement_overview(
    groups: Sequence[MultiFidelityTrajectorySample],
    output_path: str | Path,
) -> Path:
    """Save a dedicated figure explaining the synthetic measurements."""

    output_path = _ensure_parent(output_path)
    if not groups:
        raise ValueError("groups must contain at least one sample.")

    group = _select_story_group(groups)
    if "high" not in group.trajectories:
        raise ValueError("A high-fidelity trajectory is required for the measurement overview.")

    high = group.trajectories["high"]
    low = group.trajectories.get("low")
    selected_dims = _select_channel_indices(high.observation_dim)
    tick_positions = _heatmap_ticks(high.observation_dim)
    formulas = [f"m{idx}: {_measurement_formula(idx)}" for idx in range(min(high.observation_dim, 12))]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    axes[0][0].axis("off")
    axes[0][0].text(0.02, 0.96, "What Are The Measurements?", fontsize=16, fontweight="bold", va="top")
    axes[0][0].text(
        0.02,
        0.84,
        f"Representative sample | {_parameter_label(group.parameter)}",
        fontsize=11,
        va="top",
    )
    axes[0][0].text(
        0.02,
        0.72,
        "Measurement model\n"
        "- latent physics: damped oscillator with state [q(t), v(t)]\n"
        "- high fidelity: x_high(t) = h(q(t), v(t), t, mu)\n"
        "- low fidelity: x_low(t) = B(x_high(t), t, mu) + noise\n"
        "- all channels observe the same trajectory through different nonlinear sensors",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEF5FB", edgecolor="#B8D2E6"),
    )
    split_index = (len(formulas) + 1) // 2
    axes[0][0].text(
        0.02,
        0.38,
        "\n".join(formulas[:split_index]),
        fontsize=9,
        va="top",
        family="monospace",
    )
    axes[0][0].text(
        0.54,
        0.38,
        "\n".join(formulas[split_index:]) + (
            "\n..." if high.observation_dim > len(formulas) else ""
        ),
        fontsize=9,
        va="top",
        family="monospace",
    )

    for dim in selected_dims:
        axes[0][1].plot(
            _to_numpy(high.times),
            _to_numpy(high.observations[:, dim]),
            linewidth=2.1,
            label=f"HF {_channel_label(dim)}",
        )
        if low is not None and dim < low.observation_dim:
            axes[0][1].plot(
                _to_numpy(low.times),
                _to_numpy(low.observations[:, dim]),
                linewidth=1.8,
                linestyle="--",
                label=f"LF {_channel_label(dim)}",
            )
    axes[0][1].set_title("Selected Measurement Channels")
    axes[0][1].set_xlabel("time")
    axes[0][1].set_ylabel("measurement value")
    axes[0][1].grid(alpha=0.25)
    axes[0][1].legend(frameon=False, ncol=2)

    if low is not None:
        channel_rmse = (high.observations - low.observations).pow(2).mean(dim=0).sqrt()
        axes[0][2].bar(
            list(range(high.observation_dim)),
            _to_numpy(channel_rmse),
            color="#C44E52",
            alpha=0.85,
        )
        axes[0][2].set_title("Per-Channel LF Bias Magnitude")
        axes[0][2].set_xlabel("measurement channel")
        axes[0][2].set_ylabel("RMSE(high, low)")
        axes[0][2].grid(alpha=0.25, axis="y")
    else:
        axes[0][2].axis("off")

    hf_image = axes[1][0].imshow(
        _to_numpy(high.observations.T),
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[float(high.times[0]), float(high.times[-1]), 0, high.observation_dim - 1],
    )
    axes[1][0].set_title("High-Fidelity Measurement Field")
    axes[1][0].set_xlabel("time")
    axes[1][0].set_ylabel("measurement channel")
    axes[1][0].set_yticks(tick_positions)
    axes[1][0].set_yticklabels([_channel_label(tick) for tick in tick_positions])
    fig.colorbar(hf_image, ax=axes[1][0], fraction=0.046, pad=0.02)

    if low is not None:
        lf_image = axes[1][1].imshow(
            _to_numpy(low.observations.T),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[float(low.times[0]), float(low.times[-1]), 0, low.observation_dim - 1],
        )
        axes[1][1].set_title("Low-Fidelity Measurement Field")
        axes[1][1].set_xlabel("time")
        axes[1][1].set_ylabel("measurement channel")
        axes[1][1].set_yticks(tick_positions)
        axes[1][1].set_yticklabels([_channel_label(tick) for tick in tick_positions])
        fig.colorbar(lf_image, ax=axes[1][1], fraction=0.046, pad=0.02)

        gap = low.observations - high.observations
        gap_scale = float(gap.abs().max().clamp_min(1e-6))
        gap_image = axes[1][2].imshow(
            _to_numpy(gap.T),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-gap_scale,
            vmax=gap_scale,
            extent=[float(low.times[0]), float(low.times[-1]), 0, low.observation_dim - 1],
        )
        axes[1][2].set_title("Structured LF Bias: low - high")
        axes[1][2].set_xlabel("time")
        axes[1][2].set_ylabel("measurement channel")
        axes[1][2].set_yticks(tick_positions)
        axes[1][2].set_yticklabels([_channel_label(tick) for tick in tick_positions])
        fig.colorbar(gap_image, ax=axes[1][2], fraction=0.046, pad=0.02)
    else:
        axes[1][1].axis("off")
        axes[1][2].axis("off")

    fig.suptitle("Measurement Definition and Fidelity Bias Structure", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cfd_test_problem_summary(
    groups: Sequence[MultiFidelityTrajectorySample],
    output_path: str | Path,
) -> Path:
    """Save a narrative figure for the CFD-like different-grid example."""

    output_path = _ensure_parent(output_path)
    if not groups:
        raise ValueError("groups must contain at least one sample.")

    counts = _paired_counts(groups)
    group = _select_story_group(groups)
    if "high" not in group.trajectories:
        raise ValueError("A high-fidelity trajectory is required for the CFD test problem plot.")

    high = group.trajectories["high"]
    low = group.trajectories.get("low")
    high_shape = _grid_shape_from_sample(high)
    low_shape = _grid_shape_from_sample(low) if low is not None else None
    snapshot_index = high.length // 2
    high_snapshot = _reshape_field_snapshot(high.observations[snapshot_index], high_shape)
    low_snapshot = _reshape_field_snapshot(low.observations[snapshot_index], low_shape) if low is not None else None

    amplitude_scale = float(high_snapshot.abs().max().clamp_min(1e-6))
    if low_snapshot is not None:
        amplitude_scale = max(amplitude_scale, float(low_snapshot.abs().max().clamp_min(1e-6)))

    observation_lines = ["Observation model"]
    if low_shape is not None:
        observation_lines.append(f"- low fidelity: coarse grid {low_shape[0]}x{low_shape[1]}")
    observation_lines.extend(
        [
            f"- high fidelity: fine grid {high_shape[0]}x{high_shape[1]}",
            "- same scalar quantity on different grids",
            "- low fidelity includes stronger diffusion and phase bias",
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0][0].axis("off")
    axes[0][0].text(0.02, 0.96, "CFD-Like Test Problem", fontsize=16, fontweight="bold", va="top")
    axes[0][0].text(
        0.02,
        0.82,
        "Parameterized unsteady scalar field\n"
        "advected vortices + wake oscillations + shear layer",
        fontsize=13,
        va="top",
    )
    axes[0][0].text(
        0.02,
        0.58,
        "\n".join(observation_lines),
        fontsize=10,
        va="top",
    )
    axes[0][0].text(
        0.02,
        0.20,
        "Dataset composition\n"
        f"paired = {counts['paired']}\n"
        f"low only = {counts['low_only']}\n"
        f"high only = {counts['high_only']}\n"
        f"{_parameter_label(group.parameter)}",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8F1D3", edgecolor="#D6C49A"),
    )

    high_image = axes[0][1].imshow(
        _to_numpy(high_snapshot),
        origin="lower",
        cmap="RdBu_r",
        vmin=-amplitude_scale,
        vmax=amplitude_scale,
    )
    axes[0][1].set_title(f"High-fidelity snapshot | t = {float(high.times[snapshot_index]):.2f}")
    axes[0][1].set_xlabel("x-index")
    axes[0][1].set_ylabel("y-index")

    if low_snapshot is not None:
        low_image = axes[1][0].imshow(
            _to_numpy(low_snapshot),
            origin="lower",
            cmap="RdBu_r",
            vmin=-amplitude_scale,
            vmax=amplitude_scale,
        )
        axes[1][0].set_title(f"Low-fidelity snapshot | t = {float(low.times[snapshot_index]):.2f}")
        axes[1][0].set_xlabel("x-index")
        axes[1][0].set_ylabel("y-index")
    else:
        axes[1][0].axis("off")

    for fidelity, sample in sorted(group.trajectories.items()):
        field_norm = sample.observations.pow(2).mean(dim=-1).sqrt()
        axes[1][1].plot(
            _to_numpy(sample.times),
            _to_numpy(field_norm),
            linewidth=2.2,
            color=_fidelity_color(fidelity),
            label=f"{fidelity} field RMS",
        )
    axes[1][1].set_title("Global Field Amplitude Over Time")
    axes[1][1].set_xlabel("time")
    axes[1][1].set_ylabel("RMS field magnitude")
    axes[1][1].grid(alpha=0.25)
    axes[1][1].legend(frameon=False)

    fig.colorbar(high_image, ax=[axes[0][1], axes[1][0]], fraction=0.035, pad=0.02, label="field value")

    fig.suptitle("Test Problem and Different-Grid Observation Model", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cfd_measurement_overview(
    groups: Sequence[MultiFidelityTrajectorySample],
    output_path: str | Path,
) -> Path:
    """Save a figure explaining how different-grid CFD-like measurements are defined."""

    output_path = _ensure_parent(output_path)
    if not groups:
        raise ValueError("groups must contain at least one sample.")

    group = _select_story_group(groups)
    if "low" not in group.trajectories or "high" not in group.trajectories:
        raise ValueError("A paired low/high sample is required for the CFD measurement overview.")

    low = group.trajectories["low"]
    high = group.trajectories["high"]
    low_shape = _grid_shape_from_sample(low)
    high_shape = _grid_shape_from_sample(high)
    snapshot_index = high.length // 2

    high_snapshot = _reshape_field_snapshot(high.observations[snapshot_index], high_shape)
    low_snapshot = _reshape_field_snapshot(low.observations[snapshot_index], low_shape)
    upsampled_low = _resize_field_series(low_snapshot.unsqueeze(0), high_shape)[0]
    amplitude_scale = max(
        float(high_snapshot.abs().max().clamp_min(1e-6)),
        float(upsampled_low.abs().max().clamp_min(1e-6)),
    )
    center_row = high_shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    axes[0][0].axis("off")
    axes[0][0].text(0.02, 0.96, "Measurement Definition", fontsize=16, fontweight="bold", va="top")
    axes[0][0].text(
        0.02,
        0.82,
        f"At each time t, the observation is the flattened field on a grid.\n"
        f"low snapshot length = {low_shape[0] * low_shape[1]}\n"
        f"high snapshot length = {high_shape[0] * high_shape[1]}",
        fontsize=11,
        va="top",
    )
    axes[0][0].text(
        0.02,
        0.58,
        "Interpretation\n"
        "- each measurement dimension is one grid-cell value\n"
        "- vectorization uses row-major flattening of the 2D field\n"
        "- low and high observe the same scalar field on different grids\n"
        "- gap comes from both coarse resolution and low-fidelity model bias",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEF5FB", edgecolor="#B8D2E6"),
    )
    axes[0][0].text(
        0.02,
        0.24,
        f"Representative snapshot time: t = {float(high.times[snapshot_index]):.2f}\n"
        f"{_parameter_label(group.parameter)}",
        fontsize=10,
        va="top",
    )

    high_image = axes[0][1].imshow(
        _to_numpy(high_snapshot),
        origin="lower",
        cmap="RdBu_r",
        vmin=-amplitude_scale,
        vmax=amplitude_scale,
    )
    axes[0][1].set_title(f"High grid {high_shape[0]}x{high_shape[1]}")
    axes[0][1].set_xlabel("x-index")
    axes[0][1].set_ylabel("y-index")

    low_image = axes[0][2].imshow(
        _to_numpy(low_snapshot),
        origin="lower",
        cmap="RdBu_r",
        vmin=-amplitude_scale,
        vmax=amplitude_scale,
    )
    axes[0][2].set_title(f"Low grid {low_shape[0]}x{low_shape[1]}")
    axes[0][2].set_xlabel("x-index")
    axes[0][2].set_ylabel("y-index")

    axes[1][0].plot(
        torch.linspace(0.0, 1.0, high.observation_dim),
        _to_numpy(high.observations[snapshot_index]),
        linewidth=1.8,
        color=_fidelity_color("high"),
        label="high vectorized snapshot",
    )
    axes[1][0].plot(
        torch.linspace(0.0, 1.0, low.observation_dim),
        _to_numpy(low.observations[snapshot_index]),
        linewidth=1.6,
        color=_fidelity_color("low"),
        label="low vectorized snapshot",
    )
    axes[1][0].set_title("Vectorized Observation at One Time")
    axes[1][0].set_xlabel("normalized flattened index")
    axes[1][0].set_ylabel("cell value")
    axes[1][0].grid(alpha=0.25)
    axes[1][0].legend(frameon=False)

    axes[1][1].plot(
        _to_numpy(torch.linspace(-1.0, 1.0, high_shape[1])),
        _to_numpy(high_snapshot[center_row]),
        linewidth=2.2,
        color=_fidelity_color("high"),
        label="high centerline",
    )
    axes[1][1].plot(
        _to_numpy(torch.linspace(-1.0, 1.0, high_shape[1])),
        _to_numpy(upsampled_low[center_row]),
        linewidth=2.0,
        linestyle="--",
        color=_fidelity_color("low"),
        label="low centerline (upsampled)",
    )
    axes[1][1].set_title("Centerline Comparison")
    axes[1][1].set_xlabel("x")
    axes[1][1].set_ylabel("field value")
    axes[1][1].grid(alpha=0.25)
    axes[1][1].legend(frameon=False)

    bias_map = upsampled_low - high_snapshot
    bias_scale = float(bias_map.abs().max().clamp_min(1e-6))
    bias_image = axes[1][2].imshow(
        _to_numpy(bias_map),
        origin="lower",
        cmap="coolwarm",
        vmin=-bias_scale,
        vmax=bias_scale,
    )
    axes[1][2].set_title("Structured LF Bias on HF Grid")
    axes[1][2].set_xlabel("x-index")
    axes[1][2].set_ylabel("y-index")
    fig.colorbar(high_image, ax=[axes[0][1], axes[0][2]], fraction=0.035, pad=0.02, label="field value")
    fig.colorbar(bias_image, ax=axes[1][2], fraction=0.046, pad=0.02, label="low - high")

    fig.suptitle("Measurement Definition and Cross-Grid Observation Structure", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cfd_dataset_overview(
    groups: Sequence[MultiFidelityTrajectorySample],
    output_path: str | Path,
    max_groups: int = 3,
) -> Path:
    """Save representative CFD-like field snapshots across held-out samples."""

    output_path = _ensure_parent(output_path)
    if not groups:
        raise ValueError("groups must contain at least one sample.")

    paired = [group for group in groups if _availability_category(group) == "paired"]
    unpaired = [group for group in groups if _availability_category(group) != "paired"]
    selected = (paired + unpaired)[:max_groups]
    fig, axes = plt.subplots(len(selected), 4, figsize=(15, 4.3 * len(selected)), squeeze=False, constrained_layout=True)
    shared_scale = max(
        float(sample.observations.abs().max().clamp_min(1e-6))
        for group in selected
        for sample in group.trajectories.values()
    )
    last_image = None

    for row, group in enumerate(selected):
        label = f"{group.pairing_id or f'group-{row}'} | {_parameter_label(group.parameter)}"
        for fidelity_index, fidelity in enumerate(("low", "high")):
            sample = group.trajectories.get(fidelity)
            if sample is None:
                axes[row][fidelity_index * 2].axis("off")
                axes[row][fidelity_index * 2 + 1].axis("off")
                continue

            grid_shape = _grid_shape_from_sample(sample)
            for col_offset, snapshot_index in enumerate((sample.length // 2, sample.length - 1)):
                ax = axes[row][fidelity_index * 2 + col_offset]
                field = _reshape_field_snapshot(sample.observations[snapshot_index], grid_shape)
                image = ax.imshow(
                    _to_numpy(field),
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-shared_scale,
                    vmax=shared_scale,
                )
                last_image = image
                ax.set_title(
                    f"{fidelity} {grid_shape[0]}x{grid_shape[1]} | t = {float(sample.times[snapshot_index]):.2f}"
                )
                ax.set_xlabel("x-index")
                if fidelity_index == 0 and col_offset == 0:
                    ax.set_ylabel(label)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, fraction=0.018, pad=0.01, label="field value")

    fig.suptitle("Different-Grid CFD-Like Dataset Overview", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cfd_window_rollout(
    window_rollout: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save short-window rollout field diagnostics for the different-grid example."""

    output_path = _ensure_parent(output_path)
    fidelities = sorted(window_rollout)
    fig, axes = plt.subplots(len(fidelities), 5, figsize=(18, 4.2 * len(fidelities)), squeeze=False, constrained_layout=True)

    for row, fidelity in enumerate(fidelities):
        entry = window_rollout[fidelity]
        grid_shape = _grid_shape_from_entry(entry)
        context_field = _reshape_field_snapshot(entry["context_targets"][-1], grid_shape)
        target_field = _reshape_field_snapshot(entry["targets"][-1], grid_shape)
        prediction_field = _reshape_field_snapshot(entry["predictions"][-1], grid_shape)
        error_field = prediction_field - target_field
        scale = max(
            float(context_field.abs().max().clamp_min(1e-6)),
            float(target_field.abs().max().clamp_min(1e-6)),
            float(prediction_field.abs().max().clamp_min(1e-6)),
        )
        error_scale = float(error_field.abs().max().clamp_min(1e-6))
        rmse_time = (entry["predictions"] - entry["targets"]).pow(2).mean(dim=-1).sqrt()

        images = [
            (context_field, f"{fidelity} context anchor", -scale, scale, "RdBu_r"),
            (target_field, f"{fidelity} target final", -scale, scale, "RdBu_r"),
            (prediction_field, f"{fidelity} prediction final", -scale, scale, "RdBu_r"),
            (error_field, f"{fidelity} prediction error", -error_scale, error_scale, "coolwarm"),
        ]
        for column, (field, title, vmin, vmax, cmap) in enumerate(images):
            image = axes[row][column].imshow(
                _to_numpy(field),
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[row][column].set_title(title)
            axes[row][column].set_xlabel("x-index")
            if column == 0:
                axes[row][column].set_ylabel("y-index")
            fig.colorbar(image, ax=axes[row][column], fraction=0.046, pad=0.02)

        axes[row][4].plot(
            _to_numpy(entry["target_times"]),
            _to_numpy(rmse_time),
            color=_fidelity_color(fidelity),
            linewidth=2.2,
        )
        axes[row][4].fill_between(
            _to_numpy(entry["target_times"]),
            0.0,
            _to_numpy(rmse_time),
            color=_fidelity_color(fidelity),
            alpha=0.25,
        )
        axes[row][4].set_title(f"{fidelity} short-window RMSE")
        axes[row][4].set_xlabel("time")
        axes[row][4].set_ylabel("field RMSE")
        axes[row][4].grid(alpha=0.25)

    fig.suptitle("Representative Windowed Rollout", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cfd_full_trajectory_rollout(
    full_rollout: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save long-horizon field snapshots from full rollouts on different grids."""

    output_path = _ensure_parent(output_path)
    fidelities = sorted(full_rollout)
    snapshot_indices = _selected_snapshot_indices(
        min(full_rollout[fidelity]["prediction"].shape[0] for fidelity in fidelities),
        count=3,
    )
    fig, axes = plt.subplots(
        len(fidelities),
        2 * len(snapshot_indices),
        figsize=(5.0 * len(snapshot_indices) * 2, 4.0 * len(fidelities)),
        squeeze=False,
        constrained_layout=True,
    )
    last_image = None

    for row, fidelity in enumerate(fidelities):
        entry = full_rollout[fidelity]
        grid_shape = _grid_shape_from_entry(entry)
        row_scale = max(
            float(entry["rollout_targets"].abs().max().clamp_min(1e-6)),
            float(entry["prediction"].abs().max().clamp_min(1e-6)),
        )
        for snapshot_rank, snapshot_index in enumerate(snapshot_indices):
            target_field = _reshape_field_snapshot(entry["rollout_targets"][snapshot_index], grid_shape)
            prediction_field = _reshape_field_snapshot(entry["prediction"][snapshot_index], grid_shape)
            time_value = float(entry["rollout_times"][snapshot_index])

            for col_offset, (field, name) in enumerate(((target_field, "target"), (prediction_field, "prediction"))):
                column = 2 * snapshot_rank + col_offset
                image = axes[row][column].imshow(
                    _to_numpy(field),
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-row_scale,
                    vmax=row_scale,
                )
                last_image = image
                axes[row][column].set_title(f"{fidelity} {name} | t = {time_value:.2f}")
                axes[row][column].set_xlabel("x-index")
                if column == 0:
                    axes[row][column].set_ylabel("y-index")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, fraction=0.018, pad=0.01, label="field value")

    fig.suptitle("Full-Trajectory Rollout Snapshots", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cfd_rollout_error_profile(
    full_rollout: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save time-resolved field error diagnostics for the different-grid example."""

    output_path = _ensure_parent(output_path)
    fidelities = sorted(full_rollout)
    fig, axes = plt.subplots(len(fidelities), 2, figsize=(13, 4.2 * len(fidelities)), squeeze=False, constrained_layout=True)

    for row, fidelity in enumerate(fidelities):
        entry = full_rollout[fidelity]
        errors = entry["prediction"] - entry["rollout_targets"]
        times = entry["rollout_times"]
        rmse_time = errors.pow(2).mean(dim=-1).sqrt()
        relative_l2 = errors.norm(dim=-1) / entry["rollout_targets"].norm(dim=-1).clamp_min(1e-6)
        target_energy = entry["rollout_targets"].pow(2).mean(dim=-1).sqrt()
        prediction_energy = entry["prediction"].pow(2).mean(dim=-1).sqrt()

        axes[row][0].plot(_to_numpy(times), _to_numpy(rmse_time), color=_fidelity_color(fidelity), linewidth=2.2, label="RMSE")
        axes[row][0].plot(_to_numpy(times), _to_numpy(relative_l2), color="#111111", linewidth=1.9, linestyle="--", label="relative L2")
        axes[row][0].set_title(f"{fidelity} rollout error over time")
        axes[row][0].set_xlabel("time")
        axes[row][0].set_ylabel("error")
        axes[row][0].grid(alpha=0.25)
        axes[row][0].legend(frameon=False)

        axes[row][1].plot(_to_numpy(times), _to_numpy(target_energy), color="#111111", linewidth=2.2, label="target field RMS")
        axes[row][1].plot(_to_numpy(times), _to_numpy(prediction_energy), color=_fidelity_color(fidelity), linewidth=2.0, label="prediction field RMS")
        axes[row][1].set_title(f"{fidelity} field amplitude tracking")
        axes[row][1].set_xlabel("time")
        axes[row][1].set_ylabel("field RMS")
        axes[row][1].grid(alpha=0.25)
        axes[row][1].legend(frameon=False)

    fig.suptitle("Rollout Error Profile", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cfd_fidelity_gap(
    full_rollout: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save cross-grid fidelity gap diagnostics after upsampling LF to the HF grid."""

    output_path = _ensure_parent(output_path)
    if "low" not in full_rollout or "high" not in full_rollout:
        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        ax.text(0.5, 0.5, "Fidelity gap plot requires both low and high fidelities.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    low_entry = full_rollout["low"]
    high_entry = full_rollout["high"]
    low_shape = _grid_shape_from_entry(low_entry)
    high_shape = _grid_shape_from_entry(high_entry)
    snapshot_index = min(low_entry["prediction"].shape[0], high_entry["prediction"].shape[0]) // 2

    low_target = _reshape_field_series(low_entry["rollout_targets"], low_shape)
    low_prediction = _reshape_field_series(low_entry["prediction"], low_shape)
    high_target = _reshape_field_series(high_entry["rollout_targets"], high_shape)
    high_prediction = _reshape_field_series(high_entry["prediction"], high_shape)

    low_target_up = _resize_field_series(low_target, high_shape)
    low_prediction_up = _resize_field_series(low_prediction, high_shape)

    target_gap = high_target[snapshot_index] - low_target_up[snapshot_index]
    prediction_gap = high_prediction[snapshot_index] - low_prediction_up[snapshot_index]
    amplitude_scale = max(
        float(high_target[snapshot_index].abs().max().clamp_min(1e-6)),
        float(low_target_up[snapshot_index].abs().max().clamp_min(1e-6)),
        float(high_prediction[snapshot_index].abs().max().clamp_min(1e-6)),
        float(low_prediction_up[snapshot_index].abs().max().clamp_min(1e-6)),
    )
    gap_scale = max(
        float(target_gap.abs().max().clamp_min(1e-6)),
        float(prediction_gap.abs().max().clamp_min(1e-6)),
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        (low_target_up[snapshot_index], "LF target (upsampled)", -amplitude_scale, amplitude_scale, "RdBu_r"),
        (high_target[snapshot_index], "HF target", -amplitude_scale, amplitude_scale, "RdBu_r"),
        (target_gap, "Target gap: HF - LF_up", -gap_scale, gap_scale, "coolwarm"),
        (low_prediction_up[snapshot_index], "LF prediction (upsampled)", -amplitude_scale, amplitude_scale, "RdBu_r"),
        (high_prediction[snapshot_index], "HF prediction", -amplitude_scale, amplitude_scale, "RdBu_r"),
        (prediction_gap, "Predicted gap: HF - LF_up", -gap_scale, gap_scale, "coolwarm"),
    ]
    amplitude_image = None
    gap_image = None
    for axis, (field, title, vmin, vmax, cmap) in zip(axes.flatten(), panels):
        image = axis.imshow(_to_numpy(field), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        if cmap == "RdBu_r":
            amplitude_image = image
        else:
            gap_image = image
        axis.set_title(title)
        axis.set_xlabel("x-index")
        axis.set_ylabel("y-index")
    if amplitude_image is not None:
        fig.colorbar(amplitude_image, ax=[axes[0][0], axes[0][1], axes[1][0], axes[1][1]], fraction=0.025, pad=0.02, label="field value")
    if gap_image is not None:
        fig.colorbar(gap_image, ax=[axes[0][2], axes[1][2]], fraction=0.035, pad=0.02, label="gap value")

    fig.suptitle("Cross-Fidelity Gap Analysis (LF Upsampled to HF Grid)", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_method_overview(
    model_config: Any,
    output_path: str | Path,
) -> Path:
    """Save a conceptual end-to-end method overview diagram."""

    output_path = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    high_head_label = (
        "High head\nD_low + Delta"
        if getattr(model_config, "hierarchical_decoder", False)
        else "High head\nD_high(z,mu)"
    )
    training_signals = [
        "- rollout / reconstruction losses in observation space",
        "- temporal latent consistency from shifted future windows",
        "- paired latent alignment across fidelities",
    ]
    if getattr(model_config, "hierarchical_decoder", False):
        training_signals.append("- hierarchical discrepancy regularization for high fidelity")

    _draw_box(ax, (0.03, 0.38), (0.13, 0.24), "Multi-fidelity windows", "x_f[t-k:t]\n+ parameters mu", "#E8F1FB")
    _draw_box(ax, (0.21, 0.38), (0.12, 0.24), "Adapters", f"fidelity-specific\nobs -> {model_config.adapter_dim}", "#DAEEF3")
    _draw_box(ax, (0.38, 0.38), (0.14, 0.24), "Temporal encoder", f"shared backbone\nhidden = {model_config.encoder_hidden_dim}", "#D8F0E0")
    _draw_box(ax, (0.57, 0.38), (0.12, 0.24), "Shared latent z0", f"projector\nlatent = {model_config.latent_dim}", "#FCE8D5")
    _draw_box(ax, (0.74, 0.38), (0.14, 0.24), "Latent Neural ODE", "z'(t) = g(z, mu)\nshared dynamics", "#F7DDE2")
    _draw_box(ax, (0.74, 0.08), (0.14, 0.18), "Shared decoder", f"hidden = {model_config.decoder_hidden_dim}", "#D8F0E0")
    _draw_box(ax, (0.90, 0.53), (0.08, 0.16), "Low head", "D_low(z,mu)", "#DAEEF3")
    _draw_box(ax, (0.90, 0.27), (0.08, 0.16), "High head", high_head_label, "#FBE4E6")

    _draw_arrow(ax, (0.16, 0.50), (0.21, 0.50), "per fidelity")
    _draw_arrow(ax, (0.33, 0.50), (0.38, 0.50), "shared")
    _draw_arrow(ax, (0.52, 0.50), (0.57, 0.50), "shared latent")
    _draw_arrow(ax, (0.69, 0.50), (0.74, 0.50), "rollout")
    _draw_arrow(ax, (0.81, 0.38), (0.81, 0.26))
    _draw_arrow(ax, (0.88, 0.18), (0.90, 0.35), "decode")
    _draw_arrow(ax, (0.88, 0.18), (0.90, 0.61))
    _draw_arrow(ax, (0.88, 0.50), (0.90, 0.61))

    ax.text(0.05, 0.88, "Method summary", fontsize=17, fontweight="bold")
    ax.text(
        0.05,
        0.80,
        "Encode short multi-fidelity observation windows into a shared latent initial state,\n"
        "evolve that state with a parameter-conditioned latent Neural ODE,\n"
        "then decode back to each fidelity with shared-latent consistency and fidelity-specific outputs.",
        fontsize=11,
        va="top",
    )
    ax.text(
        0.05,
        0.18,
        "Training signals\n" + "\n".join(training_signals),
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F1D3", edgecolor="#D6C49A"),
    )
    ax.text(
        0.54,
        0.77,
        "Shared components are highlighted in green/orange/red.\n"
        "Fidelity-specific pieces are shown in blue/pink.",
        fontsize=10,
        va="top",
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_architecture_breakdown(
    model_config: Any,
    output_path: str | Path,
) -> Path:
    """Save an architecture-oriented figure with module roles and dimensions."""

    output_path = _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)

    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

    hierarchical = getattr(model_config, "hierarchical_decoder", False)

    axes[0].text(0.02, 0.96, "Encoder and shared latent space", fontsize=15, fontweight="bold", va="top")
    _draw_box(axes[0], (0.05, 0.70), (0.26, 0.16), "Per-fidelity observations", f"window = {model_config.window_size}\nobs dims = {model_config.fidelity_dims}", "#E8F1FB")
    _draw_box(axes[0], (0.38, 0.70), (0.20, 0.16), "Adapters", f"shared input width\nadapter dim = {model_config.adapter_dim}", "#DAEEF3")
    _draw_box(axes[0], (0.65, 0.70), (0.24, 0.16), "Temporal encoder", f"hidden = {model_config.encoder_hidden_dim}\nlayers = {model_config.num_hidden_layers}", "#D8F0E0")
    _draw_box(axes[0], (0.30, 0.40), (0.34, 0.18), "Shared latent projector", f"latent dim = {model_config.latent_dim}\nshared_latent_projection = {model_config.shared_latent_projection}", "#FCE8D5")
    _draw_box(axes[0], (0.69, 0.38), (0.20, 0.22), "Latent dynamics", f"hidden = {model_config.dynamics_hidden_dim}\nsolver = {model_config.solver_backend}", "#F7DDE2")
    _draw_arrow(axes[0], (0.31, 0.78), (0.38, 0.78))
    _draw_arrow(axes[0], (0.58, 0.78), (0.65, 0.78))
    _draw_arrow(axes[0], (0.52, 0.70), (0.47, 0.58), "shared z0")
    _draw_arrow(axes[0], (0.64, 0.49), (0.69, 0.49), "z(t)")

    axes[1].text(0.02, 0.96, "Decoder and multi-fidelity outputs", fontsize=15, fontweight="bold", va="top")
    _draw_box(axes[1], (0.06, 0.70), (0.22, 0.16), "Latent rollout", "z(t), mu", "#FCE8D5")
    _draw_box(axes[1], (0.35, 0.70), (0.24, 0.16), "Shared decoder", f"hidden = {model_config.decoder_hidden_dim}\nshared_decoder_backbone = {model_config.shared_decoder_backbone}", "#D8F0E0")
    _draw_box(axes[1], (0.68, 0.76), (0.22, 0.12), "Low-fidelity head", "predict x_low(t)", "#DAEEF3")
    if hierarchical:
        _draw_box(axes[1], (0.68, 0.54), (0.22, 0.14), "High-fidelity head", "predict x_high base", "#FBE4E6")
        _draw_box(axes[1], (0.68, 0.30), (0.22, 0.14), "Discrepancy branch", "hierarchical = True\nDelta_h<-l(z,mu)", "#F7DDE2")
        _draw_box(axes[1], (0.30, 0.22), (0.34, 0.18), "Final outputs", "low: D_low(z,mu)\nhigh: D_low + Delta", "#F8F1D3")
    else:
        _draw_box(axes[1], (0.68, 0.54), (0.22, 0.14), "High-fidelity head", "predict x_high(t)", "#FBE4E6")
        _draw_box(axes[1], (0.68, 0.30), (0.22, 0.14), "Independent grids", "hierarchical = False\nseparate output dims", "#F7DDE2")
        _draw_box(axes[1], (0.30, 0.22), (0.34, 0.18), "Final outputs", "low: D_low(z,mu)\nhigh: D_high(z,mu)", "#F8F1D3")
    _draw_arrow(axes[1], (0.28, 0.78), (0.35, 0.78))
    _draw_arrow(axes[1], (0.59, 0.78), (0.68, 0.82))
    _draw_arrow(axes[1], (0.59, 0.78), (0.68, 0.61))
    _draw_arrow(axes[1], (0.59, 0.74), (0.68, 0.37))
    _draw_arrow(axes[1], (0.79, 0.76), (0.47, 0.40))
    _draw_arrow(
        axes[1],
        (0.79, 0.30),
        (0.47, 0.32),
        "additive correction" if hierarchical else "separate reconstruction path",
    )

    axes[1].text(
        0.04,
        0.08,
        "Design intent\n"
        "- shared latent coordinates for all fidelities\n"
        "- shared latent dynamics g(z,mu)\n"
        "- fidelity-specific observation maps\n"
        + ("- optional hierarchical high-fidelity correction" if hierarchical else "- support for different fidelity output grids"),
        fontsize=10,
        va="bottom",
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_training_strategy_overview(
    experiment_config: Any,
    history: Sequence[Mapping[str, float]],
    fidelity_weight_history: Sequence[Mapping[str, float]],
    output_path: str | Path,
) -> Path:
    """Save a figure summarizing the objective and training strategy."""

    output_path = _ensure_parent(output_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    axes[0][0].axis("off")
    loss_cfg = experiment_config.loss
    train_cfg = experiment_config.training
    data_cfg = experiment_config.data
    model_cfg = experiment_config.model
    strategy_lines = [
        "1. encode short windows",
        "2. roll out shared latent ODE",
        "3. decode per fidelity",
        "4. align paired latents",
    ]
    if getattr(model_cfg, "hierarchical_decoder", False):
        strategy_lines.append("5. regularize hierarchical discrepancy")
        strategy_lines.append("6. shift emphasis toward high fidelity later in training")
    else:
        strategy_lines.append("5. shift emphasis toward high fidelity later in training")
    axes[0][0].text(0.02, 0.96, "Training objective", fontsize=15, fontweight="bold", va="top")
    axes[0][0].text(
        0.02,
        0.78,
        "L = lambda_roll L_roll + lambda_latent L_latent\n"
        "  + lambda_align L_align + lambda_mf L_mf\n"
        "  + lambda_disc L_disc + lambda_reg L_reg",
        fontsize=13,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F1D3", edgecolor="#D6C49A"),
    )
    axes[0][0].text(
        0.02,
        0.40,
        f"weights\n"
        f"rollout = {loss_cfg.rollout_weight}\n"
        f"latent consistency = {loss_cfg.latent_consistency_weight}\n"
        f"latent alignment = {loss_cfg.latent_alignment_weight}\n"
        f"multifidelity = {loss_cfg.multifidelity_weight}\n"
        f"discrepancy = {loss_cfg.discrepancy_weight}\n"
        f"regularization = {loss_cfg.regularization_weight}",
        fontsize=10,
        va="top",
    )

    epochs = [entry["epoch"] for entry in history] if history else list(range(1, len(fidelity_weight_history) + 1))
    for fidelity in sorted({key for weights in fidelity_weight_history for key in weights}):
        axes[0][1].plot(
            epochs[: len(fidelity_weight_history)],
            [weights.get(fidelity, 1.0) for weights in fidelity_weight_history],
            linewidth=2.2,
            color=_fidelity_color(fidelity),
            label=fidelity,
        )
    axes[0][1].set_title("Curriculum weighting over epochs")
    axes[0][1].set_xlabel("Epoch")
    axes[0][1].set_ylabel("fidelity loss weight")
    axes[0][1].grid(alpha=0.25)
    axes[0][1].legend(frameon=False)

    axes[1][0].set_title("Windowed offline training setup")
    categories = ["context", "rollout", "future window"]
    values = [data_cfg.window_size, data_cfg.rollout_horizon + 1, data_cfg.window_size]
    colors = ["#D8F0E0", "#FBE4E6", "#E8F1FB"]
    axes[1][0].bar(categories, values, color=colors)
    axes[1][0].set_ylabel("time samples")
    axes[1][0].grid(alpha=0.25, axis="y")
    axes[1][0].text(
        0.02,
        0.95,
        f"future_context_shift = {data_cfg.future_context_shift}",
        transform=axes[1][0].transAxes,
        va="top",
        fontsize=10,
    )

    axes[1][1].axis("off")
    axes[1][1].text(0.02, 0.96, "Optimization protocol", fontsize=15, fontweight="bold", va="top")
    axes[1][1].text(
        0.02,
        0.76,
        f"offline training only\n"
        f"epochs = {train_cfg.epochs}\n"
        f"batch size = {train_cfg.batch_size}\n"
        f"learning rate = {train_cfg.learning_rate}\n"
        f"weight decay = {train_cfg.weight_decay}\n"
        f"grad clip = {train_cfg.max_grad_norm}\n"
        f"AMP = {train_cfg.use_amp}",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EAF4E2", edgecolor="#B7D3A8"),
    )
    axes[1][1].text(
        0.55,
        0.76,
        "Batch logic\n"
        "- paired low/high groups\n"
        "- low-only groups\n"
        "- high-only groups\n"
        "- masked losses over available fidelities",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7E7EA", edgecolor="#D7B2BC"),
    )
    axes[1][1].text(
        0.02,
        0.28,
        "Strategy summary\n" + "\n".join(strategy_lines),
        fontsize=10,
        va="top",
    )

    fig.suptitle("Training Strategy", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_dataset_overview(
    groups: Sequence[MultiFidelityTrajectorySample],
    output_path: str | Path,
    max_groups: int = 4,
) -> Path:
    """Save representative raw trajectories across fidelities."""

    output_path = _ensure_parent(output_path)
    if not groups:
        raise ValueError("groups must contain at least one sample.")

    paired = [group for group in groups if _availability_category(group) == "paired"]
    unpaired = [group for group in groups if _availability_category(group) != "paired"]
    selected = (paired + unpaired)[:max_groups]
    max_dim = max(
        sample.observation_dim
        for group in selected
        for sample in group.trajectories.values()
    )
    selected_dims = _select_channel_indices(max_dim)

    fig, axes = plt.subplots(
        len(selected),
        len(selected_dims),
        figsize=(5.4 * len(selected_dims), 3.1 * len(selected)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, group in enumerate(selected):
        label = f"{group.pairing_id or f'group-{row}'} | {_parameter_label(group.parameter)}"
        for col, dim in enumerate(selected_dims):
            ax = axes[row][col]
            plotted = False
            for fidelity in sorted(group.trajectories):
                sample = group.trajectories[fidelity]
                if dim >= sample.observation_dim:
                    continue
                ax.plot(
                    _to_numpy(sample.times),
                    _to_numpy(sample.observations[:, dim]),
                    linewidth=2.0,
                    color=_fidelity_color(fidelity),
                    label=fidelity if row == 0 else None,
                )
                plotted = True
            if not plotted:
                ax.axis("off")
                continue
            if row == 0:
                ax.set_title(f"Channel {_channel_label(dim)}")
            if col == 0:
                ax.set_ylabel(label)
            ax.set_xlabel("time")
            ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    subtitle = ""
    if max_dim > len(selected_dims):
        subtitle = f" (showing representative channels out of {max_dim})"
    fig.suptitle(f"Synthetic Multi-Fidelity Trajectory Overview{subtitle}", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_parameter_coverage(
    split_groups: Mapping[str, Sequence[MultiFidelityTrajectorySample]],
    output_path: str | Path,
) -> Path:
    """Save parameter-space coverage and split composition summaries."""

    output_path = _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    counts: dict[str, dict[str, int]] = {
        split: defaultdict(int)
        for split in split_groups
    }
    for split, groups in split_groups.items():
        for group in groups:
            parameter = group.parameter
            x = float(parameter[0])
            y = float(parameter[1]) if parameter.shape[0] > 1 else 0.0
            category = _availability_category(group)
            counts[split][category] += 1
            axes[0].scatter(
                x,
                y,
                s=70,
                alpha=0.85,
                color=SPLIT_COLORS.get(split, "#333333"),
                marker=AVAILABILITY_MARKERS[category],
                edgecolor="white",
                linewidth=0.6,
            )

    axes[0].set_title("Parameter Coverage by Split")
    axes[0].set_xlabel(r"$\mu_0$")
    axes[0].set_ylabel(r"$\mu_1$")
    axes[0].grid(alpha=0.25)

    split_handles = [
        Line2D([0], [0], color=color, marker="o", linestyle="", markersize=8, label=split)
        for split, color in SPLIT_COLORS.items()
        if split in split_groups
    ]
    availability_handles = [
        Line2D(
            [0],
            [0],
            color="#666666",
            marker=marker,
            linestyle="",
            markersize=8,
            label=label.replace("_", " "),
        )
        for label, marker in AVAILABILITY_MARKERS.items()
        if any(counts[split].get(label, 0) > 0 for split in split_groups)
    ]
    axes[0].legend(handles=split_handles + availability_handles, frameon=False, loc="best")

    split_names = list(split_groups)
    bottoms = [0] * len(split_names)
    category_order = ("paired", "low_only", "high_only", "other")
    category_colors = {
        "paired": "#4C78A8",
        "low_only": "#72B7B2",
        "high_only": "#F58518",
        "other": "#B279A2",
    }
    for category in category_order:
        values = [counts[split].get(category, 0) for split in split_names]
        axes[1].bar(
            split_names,
            values,
            bottom=bottoms,
            color=category_colors[category],
            label=category.replace("_", " "),
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axes[1].set_title("Split Composition")
    axes[1].set_ylabel("Number of grouped trajectories")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(frameon=False)

    fig.suptitle("Dataset Coverage and Split Summary", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_training_history(
    history: Sequence[Mapping[str, float]],
    output_path: str | Path,
    fidelity_weight_history: Sequence[Mapping[str, float]] | None = None,
) -> Path:
    """Save a presentation-style training dashboard."""

    output_path = _ensure_parent(output_path)
    if not history:
        raise ValueError("history must contain at least one epoch.")

    epochs = [entry["epoch"] for entry in history]
    single_epoch = len(epochs) == 1
    marker_kwargs = {"marker": "o", "markersize": 6} if single_epoch else {}
    available_keys = {key for entry in history for key in entry}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    for split, style in (("train", "-"), ("val", "--")):
        total_key = f"{split}/total_loss"
        if total_key in available_keys:
            axes[0][0].plot(
                epochs,
                [entry.get(total_key, float("nan")) for entry in history],
                style,
                linewidth=2.2,
                label=f"{split} total",
                **marker_kwargs,
            )
        rollout_key = f"{split}/rollout_loss"
        if rollout_key in available_keys:
            axes[0][0].plot(
                epochs,
                [entry.get(rollout_key, float("nan")) for entry in history],
                style,
                linewidth=1.8,
                alpha=0.75,
                label=f"{split} rollout",
                **marker_kwargs,
            )
    axes[0][0].set_title("Optimization Progress")
    axes[0][0].set_xlabel("Epoch")
    axes[0][0].set_ylabel("Loss")
    axes[0][0].grid(alpha=0.25)
    axes[0][0].legend(frameon=False)

    component_keys = [
        ("train/multifidelity_loss", "multifidelity", "#4C78A8"),
        ("train/latent_consistency_loss", "latent consistency", "#F58518"),
        ("train/latent_alignment_loss", "latent alignment", "#E45756"),
        ("train/discrepancy_loss", "discrepancy", "#54A24B"),
        ("train/regularization_loss", "regularization", "#B279A2"),
    ]
    for key, label, color in component_keys:
        if key in available_keys:
            axes[0][1].plot(
                epochs,
                [entry.get(key, float("nan")) for entry in history],
                linewidth=2.0,
                color=color,
                label=label,
                **marker_kwargs,
            )
    axes[0][1].set_title("Training Objective Breakdown")
    axes[0][1].set_xlabel("Epoch")
    axes[0][1].set_ylabel("Component value")
    axes[0][1].grid(alpha=0.25)
    axes[0][1].legend(frameon=False)

    per_fidelity_keys = sorted(key for key in available_keys if key.endswith("/mse"))
    for key in per_fidelity_keys:
        split, fidelity, _ = key.split("/")
        axes[1][0].plot(
            epochs,
            [entry.get(key, float("nan")) for entry in history],
            "--" if split == "val" else "-",
            linewidth=2.0 if split == "train" else 1.6,
            color=_fidelity_color(fidelity),
            alpha=1.0 if split == "train" else 0.7,
            label=f"{split} {fidelity}",
            **marker_kwargs,
        )
    axes[1][0].set_title("Per-Fidelity Rollout MSE")
    axes[1][0].set_xlabel("Epoch")
    axes[1][0].set_ylabel("MSE")
    axes[1][0].grid(alpha=0.25)
    if per_fidelity_keys:
        axes[1][0].legend(frameon=False, ncol=2)

    if fidelity_weight_history:
        for fidelity in sorted({key for weights in fidelity_weight_history for key in weights}):
            axes[1][1].plot(
                epochs[: len(fidelity_weight_history)],
                [weights.get(fidelity, 1.0) for weights in fidelity_weight_history],
                linewidth=2.2,
                color=_fidelity_color(fidelity),
                label=fidelity,
                **marker_kwargs,
            )
        axes[1][1].set_title("Curriculum Fidelity Weights")
        axes[1][1].set_xlabel("Epoch")
        axes[1][1].set_ylabel("Loss weight")
        axes[1][1].grid(alpha=0.25)
        axes[1][1].legend(frameon=False)
    else:
        axes[1][1].axis("off")

    if single_epoch:
        fig.text(
            0.5,
            0.015,
            "Single-epoch smoke run: training traces are shown as markers rather than curves.",
            ha="center",
            va="center",
            fontsize=9,
            color="#4B5563",
        )

    fig.suptitle("Training Dashboard", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_representative_rollout(
    batch: dict[str, Any],
    outputs: dict[str, Any],
    scaler: MultiFidelityScaler | None,
    output_path: str | Path,
) -> Path:
    """Save a representative windowed rollout with residual panels."""

    output_path = _ensure_parent(output_path)
    fidelities = sorted(outputs["per_fidelity"])
    observation_dim = max(
        int(outputs["per_fidelity"][fidelity]["predictions"].shape[-1])
        for fidelity in fidelities
    )
    selected_dims = _select_channel_indices(observation_dim)
    fig, axes = plt.subplots(
        len(fidelities) * 2,
        len(selected_dims),
        figsize=(5.2 * len(selected_dims), 2.7 * len(fidelities) * 2),
        squeeze=False,
        sharex="col",
        constrained_layout=True,
    )

    for fidelity_index, fidelity in enumerate(fidelities):
        tensors = batch["windows"][fidelity]
        fidelity_output = outputs["per_fidelity"][fidelity]
        context_times = tensors["context_times"][0]
        target_times = tensors["target_times"][0]
        context = tensors["context_observations"][0]
        targets = fidelity_output["targets"][0]
        predictions = fidelity_output["predictions"][0]

        if scaler is not None:
            context = scaler.inverse_transform(fidelity, context)
            targets = scaler.inverse_transform(fidelity, targets)
            predictions = scaler.inverse_transform(fidelity, predictions)

        for col, dim in enumerate(selected_dims):
            signal_ax = axes[fidelity_index * 2][col]
            residual_ax = axes[fidelity_index * 2 + 1][col]
            if dim >= targets.shape[-1]:
                signal_ax.axis("off")
                residual_ax.axis("off")
                continue

            residual = predictions[:, dim] - targets[:, dim]
            rollout_start = float(context_times[-1])
            rollout_end = float(target_times[-1])
            signal_ax.axvspan(rollout_start, rollout_end, color="#D9EAF7", alpha=0.35)
            residual_ax.axvspan(rollout_start, rollout_end, color="#D9EAF7", alpha=0.35)
            signal_ax.plot(
                _to_numpy(context_times),
                _to_numpy(context[:, dim]),
                color="#7F7F7F",
                linestyle=":",
                linewidth=2.1,
                label="context" if col == 0 else None,
            )
            signal_ax.plot(
                _to_numpy(target_times),
                _to_numpy(targets[:, dim]),
                color="#111111",
                linewidth=2.4,
                label="target" if col == 0 else None,
            )
            signal_ax.plot(
                _to_numpy(target_times),
                _to_numpy(predictions[:, dim]),
                color=_fidelity_color(fidelity),
                linewidth=2.1,
                label="prediction" if col == 0 else None,
            )
            residual_ax.axhline(0.0, color="#333333", linewidth=1.0)
            residual_ax.plot(
                _to_numpy(target_times),
                _to_numpy(residual),
                color=_fidelity_color(fidelity),
                linewidth=1.8,
            )
            residual_ax.fill_between(
                _to_numpy(target_times),
                0.0,
                _to_numpy(residual),
                color=_fidelity_color(fidelity),
                alpha=0.25,
            )

            signal_ax.set_title(f"{fidelity} fidelity | {_channel_label(dim)}")
            signal_ax.grid(alpha=0.25)
            residual_ax.grid(alpha=0.25)
            residual_ax.set_xlabel("time")
            if col == 0:
                signal_ax.set_ylabel("channel")
                residual_ax.set_ylabel("error")
                signal_ax.legend(frameon=False)

    fig.suptitle("Representative Windowed Rollout", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_full_trajectory_rollout(
    full_rollout: Mapping[str, Mapping[str, Tensor | str]],
    output_path: str | Path,
) -> Path:
    """Save long-horizon trajectory rollouts from one representative paired sample."""

    output_path = _ensure_parent(output_path)
    fidelities = sorted(full_rollout)
    observation_dim = max(int(full_rollout[fidelity]["rollout_targets"].shape[-1]) for fidelity in fidelities)
    selected_dims = _select_channel_indices(observation_dim)
    fig, axes = plt.subplots(
        len(fidelities),
        len(selected_dims),
        figsize=(5.4 * len(selected_dims), 3.2 * len(fidelities)),
        squeeze=False,
        constrained_layout=True,
    )

    first_entry = full_rollout[fidelities[0]]
    title = (
        f"Full-Trajectory Rollout | {first_entry.get('pairing_id', 'sample')} | "
        f"{_parameter_label(first_entry['parameter'])}"
    )

    for row, fidelity in enumerate(fidelities):
        entry = full_rollout[fidelity]
        full_times = entry["full_times"]
        full_targets = entry["full_targets"]
        rollout_times = entry["rollout_times"]
        rollout_targets = entry["rollout_targets"]
        predictions = entry["prediction"]
        context_end = float(entry["context_times"][-1])
        mse = float((predictions - rollout_targets).pow(2).mean())

        for col, dim in enumerate(selected_dims):
            ax = axes[row][col]
            if dim >= full_targets.shape[-1]:
                ax.axis("off")
                continue
            ax.axvspan(float(full_times[0]), context_end, color="#F3E8C8", alpha=0.45)
            ax.plot(
                _to_numpy(full_times),
                _to_numpy(full_targets[:, dim]),
                color="#111111",
                linewidth=2.1,
                label="target" if col == 0 else None,
            )
            ax.plot(
                _to_numpy(rollout_times),
                _to_numpy(predictions[:, dim]),
                color=_fidelity_color(fidelity),
                linewidth=2.0,
                label="rollout" if col == 0 else None,
            )
            ax.axvline(context_end, color="#666666", linestyle="--", linewidth=1.2)
            ax.set_title(f"{fidelity} fidelity | {_channel_label(dim)} | MSE={mse:.3e}")
            ax.set_xlabel("time")
            if col == 0:
                ax.set_ylabel("value")
                ax.legend(frameon=False)
            ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_rollout_error_profile(
    full_rollout: Mapping[str, Mapping[str, Tensor | str]],
    output_path: str | Path,
) -> Path:
    """Save time-resolved error diagnostics for full-trajectory rollouts."""

    output_path = _ensure_parent(output_path)
    fidelities = sorted(full_rollout)
    fig, axes = plt.subplots(len(fidelities), 2, figsize=(12, 4.0 * len(fidelities)), squeeze=False, constrained_layout=True)

    for row, fidelity in enumerate(fidelities):
        entry = full_rollout[fidelity]
        errors = entry["prediction"] - entry["rollout_targets"]
        times = entry["rollout_times"]
        rmse_time = errors.pow(2).mean(dim=-1).sqrt()
        cumulative = torch.cumsum(errors.pow(2).mean(dim=-1), dim=0) / torch.arange(1, errors.shape[0] + 1, dtype=errors.dtype)
        cumulative = cumulative.sqrt()
        selected_dims = _select_channel_indices(errors.shape[-1])

        for dim in selected_dims:
            axes[row][0].plot(
                _to_numpy(times),
                _to_numpy(errors[:, dim].abs()),
                linewidth=1.8,
                label=_channel_label(dim),
            )
        axes[row][0].plot(
            _to_numpy(times),
            _to_numpy(rmse_time),
            color="#111111",
            linewidth=2.4,
            label="RMSE",
        )
        axes[row][0].set_title(f"{fidelity} | absolute error by selected channel")
        axes[row][0].set_xlabel("time")
        axes[row][0].set_ylabel("absolute error")
        axes[row][0].grid(alpha=0.25)
        axes[row][0].legend(frameon=False)

        axes[row][1].plot(
            _to_numpy(times),
            _to_numpy(cumulative),
            color=_fidelity_color(fidelity),
            linewidth=2.2,
        )
        axes[row][1].fill_between(_to_numpy(times), 0.0, _to_numpy(cumulative), color=_fidelity_color(fidelity), alpha=0.25)
        axes[row][1].set_title(f"{fidelity} | cumulative rollout RMSE")
        axes[row][1].set_xlabel("time")
        axes[row][1].set_ylabel("RMSE")
        axes[row][1].grid(alpha=0.25)

    fig.suptitle("Rollout Error Profile", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_fidelity_gap(
    full_rollout: Mapping[str, Mapping[str, Tensor | str]],
    output_path: str | Path,
) -> Path:
    """Save high-vs-low fidelity gap diagnostics for one representative paired rollout."""

    output_path = _ensure_parent(output_path)
    if "low" not in full_rollout or "high" not in full_rollout:
        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        ax.text(0.5, 0.5, "Fidelity gap plot requires both low and high fidelities.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    low_entry = full_rollout["low"]
    high_entry = full_rollout["high"]
    steps = min(
        low_entry["rollout_targets"].shape[0],
        high_entry["rollout_targets"].shape[0],
    )
    dims = min(
        low_entry["rollout_targets"].shape[-1],
        high_entry["rollout_targets"].shape[-1],
    )
    times = high_entry["rollout_times"][:steps]
    selected_dims = _select_channel_indices(dims)

    fig, axes = plt.subplots(
        2,
        len(selected_dims),
        figsize=(5.4 * len(selected_dims), 6.0),
        squeeze=False,
        constrained_layout=True,
    )
    for column, dim in enumerate(selected_dims):
        target_gap = high_entry["rollout_targets"][:steps, dim] - low_entry["rollout_targets"][:steps, dim]
        predicted_gap = high_entry["prediction"][:steps, dim] - low_entry["prediction"][:steps, dim]
        gap_error = predicted_gap - target_gap

        axes[0][column].plot(_to_numpy(times), _to_numpy(target_gap), color="#111111", linewidth=2.2, label="target gap")
        axes[0][column].plot(_to_numpy(times), _to_numpy(predicted_gap), color="#C44E52", linewidth=2.0, label="predicted gap")
        axes[0][column].set_title(f"Fidelity gap | {_channel_label(dim)}")
        axes[0][column].set_xlabel("time")
        axes[0][column].set_ylabel("high - low")
        axes[0][column].grid(alpha=0.25)
        axes[0][column].legend(frameon=False)

        axes[1][column].axhline(0.0, color="#333333", linewidth=1.0)
        axes[1][column].plot(_to_numpy(times), _to_numpy(gap_error), color="#4C78A8", linewidth=1.8)
        axes[1][column].fill_between(_to_numpy(times), 0.0, _to_numpy(gap_error), color="#4C78A8", alpha=0.25)
        axes[1][column].set_title(f"Gap residual | {_channel_label(dim)}")
        axes[1][column].set_xlabel("time")
        axes[1][column].set_ylabel("prediction error")
        axes[1][column].grid(alpha=0.25)

    fig.suptitle("Cross-Fidelity Gap Analysis", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_latent_diagnostics(
    data: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Save latent trajectory, norm, coordinates, and cross-fidelity alignment diagnostics."""

    output_path = _ensure_parent(output_path)
    entries = _extract_latent_entries(data)
    fidelities = sorted(entries)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for fidelity in fidelities:
        times = entries[fidelity]["times"]
        latent = entries[fidelity]["latent_trajectory"]
        axes[0][0].plot(
            _to_numpy(latent[:, 0]),
            _to_numpy(latent[:, 1] if latent.shape[-1] > 1 else latent[:, 0]),
            linewidth=2.0,
            color=_fidelity_color(fidelity),
            label=fidelity,
        )
        axes[0][1].plot(
            _to_numpy(times),
            _to_numpy(latent.norm(dim=-1)),
            linewidth=2.0,
            color=_fidelity_color(fidelity),
            label=fidelity,
        )

    primary_fidelity = "high" if "high" in entries else fidelities[0]
    primary_times = entries[primary_fidelity]["times"]
    primary_latent = entries[primary_fidelity]["latent_trajectory"]
    dims_to_plot = min(3, primary_latent.shape[-1])
    dim_colors = ["#4C78A8", "#F58518", "#54A24B"]
    for dim in range(dims_to_plot):
        axes[1][0].plot(
            _to_numpy(primary_times),
            _to_numpy(primary_latent[:, dim]),
            linewidth=2.0,
            color=dim_colors[dim],
            label=f"latent dim {dim}",
        )

    axes[0][0].set_title("Latent Phase Portrait")
    axes[0][0].set_xlabel("latent dim 0")
    axes[0][0].set_ylabel("latent dim 1" if primary_latent.shape[-1] > 1 else "latent dim 0")
    axes[0][0].grid(alpha=0.25)
    axes[0][0].legend(frameon=False)

    axes[0][1].set_title("Latent Norm Over Time")
    axes[0][1].set_xlabel("time")
    axes[0][1].set_ylabel(r"$||z(t)||_2$")
    axes[0][1].grid(alpha=0.25)
    axes[0][1].legend(frameon=False)

    axes[1][0].set_title(f"Latent Coordinates Over Time ({primary_fidelity})")
    axes[1][0].set_xlabel("time")
    axes[1][0].set_ylabel("value")
    axes[1][0].grid(alpha=0.25)
    axes[1][0].legend(frameon=False)

    if len(fidelities) >= 2:
        reference = entries["high"] if "high" in entries else entries[fidelities[0]]
        other_key = "low" if "low" in entries else fidelities[1]
        other = entries[other_key]
        steps = min(reference["latent_trajectory"].shape[0], other["latent_trajectory"].shape[0])
        latent_distance = (
            reference["latent_trajectory"][:steps] - other["latent_trajectory"][:steps]
        ).norm(dim=-1)
        axes[1][1].plot(
            _to_numpy(reference["times"][:steps]),
            _to_numpy(latent_distance),
            color="#C44E52",
            linewidth=2.0,
        )
        axes[1][1].set_title(f"Cross-Fidelity Latent Distance ({other_key} vs reference)")
        axes[1][1].set_xlabel("time")
        axes[1][1].set_ylabel("distance")
        axes[1][1].grid(alpha=0.25)
    else:
        axes[1][1].text(0.5, 0.5, "Only one fidelity available.", ha="center", va="center")
        axes[1][1].axis("off")

    fig.suptitle("Latent Dynamics Diagnostics", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_test_metrics(
    metrics: Mapping[str, float],
    output_path: str | Path,
) -> Path:
    """Save per-fidelity and global metric summaries."""

    output_path = _ensure_parent(output_path)
    fidelities = sorted({key.split("/")[0] for key in metrics if "/" in key})
    metric_names = ("mse", "mae", "relative_l2")
    global_keys = [
        "rollout_loss",
        "multifidelity_loss",
        "latent_consistency_loss",
        "latent_alignment_loss",
        "discrepancy_loss",
        "regularization_loss",
        "total_loss",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    bar_width = 0.22
    positions = list(range(len(fidelities)))
    for metric_index, metric_name in enumerate(metric_names):
        offsets = [position + (metric_index - 1) * bar_width for position in positions]
        values = [metrics.get(f"{fidelity}/{metric_name}", 0.0) for fidelity in fidelities]
        axes[0].bar(offsets, values, width=bar_width, label=metric_name)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(fidelities)
    axes[0].set_title("Per-Fidelity Test Metrics")
    axes[0].set_ylabel("metric value")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend(frameon=False)

    global_values = [metrics.get(key, 0.0) for key in global_keys if key in metrics]
    global_labels = [key.replace("_", " ") for key in global_keys if key in metrics]
    axes[1].barh(global_labels, global_values, color="#4C78A8")
    axes[1].set_title("Global Objective Summary")
    axes[1].set_xlabel("value")
    axes[1].grid(alpha=0.25, axis="x")

    fig.suptitle("Test-Set Metric Summary", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
