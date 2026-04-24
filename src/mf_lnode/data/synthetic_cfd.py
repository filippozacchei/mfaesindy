"""CFD-like multi-fidelity field generators on different spatial grids."""

from __future__ import annotations

import torch
from torch import Tensor

from mf_lnode.configs.schema import CFDGridDataConfig
from mf_lnode.data.datasets import MultiFidelityTrajectorySample, TrajectorySample


def _make_grid(grid_shape: tuple[int, int], dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Create a structured Cartesian grid over [-1, 1] x [-1, 1]."""

    height, width = grid_shape
    y = torch.linspace(-1.0, 1.0, height, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def simulate_cfd_like_field(
    parameter: Tensor,
    times: Tensor,
    grid_shape: tuple[int, int],
    fidelity: str,
) -> Tensor:
    """Simulate a smooth unsteady scalar field that resembles simple CFD data.

    The field is not the solution of a CFD solver, but it combines advected vortex
    cores, wake-like oscillations, and a shear layer to produce spatially coherent
    snapshots on structured grids.
    """

    if parameter.shape[0] != 2:
        raise ValueError("The CFD-like synthetic field expects a 2D parameter vector.")
    if fidelity not in {"low", "high"}:
        raise ValueError("fidelity must be either 'low' or 'high'.")

    xx, yy = _make_grid(grid_shape, dtype=times.dtype)
    advection = parameter[0]
    swirl = parameter[1]

    is_low = fidelity == "low"
    phase_lag = 0.16 if is_low else 0.0
    diffusion = 0.08 if is_low else 0.02
    amplitude = 0.82 if is_low else 1.0
    wake_shift = 0.05 if is_low else 0.0

    snapshots: list[Tensor] = []
    for time in times:
        tau = time + phase_lag

        center_x = -0.55 + (0.18 + 0.02 * advection) * tau + 0.12 * torch.sin(0.45 * tau + 0.35 * advection)
        center_y = 0.24 * torch.sin(0.26 * tau + 0.55 * swirl)
        sigma_vortex = 0.18 + 0.03 * advection + diffusion
        radius_vortex = ((xx - center_x) ** 2 + (yy - center_y) ** 2) / sigma_vortex**2
        leading_vortex = amplitude * (1.0 + 0.25 * swirl) * torch.exp(-radius_vortex) * torch.sin(
            3.2 * (yy - center_y) - 1.9 * (xx - center_x)
        )

        recirc_x = -0.18 + (0.10 + 0.03 * advection) * tau
        recirc_y = -0.32 + 0.16 * torch.cos(0.31 * tau + 0.25)
        sigma_recirc = 0.26 + 1.2 * diffusion
        radius_recirc = ((xx - recirc_x) ** 2 + (yy - recirc_y) ** 2) / sigma_recirc**2
        recirculation = -0.78 * torch.exp(-radius_recirc) * torch.cos(
            2.3 * (yy - recirc_y) + 1.4 * (xx - recirc_x)
        )

        wake_center = -0.82 + (0.30 + 0.05 * advection) * tau + wake_shift
        wake = 0.72 * torch.exp(-((xx - wake_center) ** 2) / (0.20 + diffusion)) * torch.sin(
            (4.2 + 0.4 * swirl) * yy + 2.6 * tau
        )

        shear_layer = 0.34 * torch.tanh(
            (yy - 0.16 * torch.sin(2.1 * (xx + 0.1) - 0.82 * tau)) / (0.10 + diffusion)
        )
        boundary_mode = 0.14 * torch.cos(torch.pi * xx) * torch.sin(1.5 * torch.pi * yy) * torch.exp(-0.08 * tau)

        field = (
            leading_vortex
            + recirculation
            + wake * torch.exp(-0.09 * tau)
            + shear_layer * torch.exp(-0.05 * tau)
            + boundary_mode
        )

        if is_low:
            field = (
                0.84 * field
                + 0.18 * torch.tanh(1.35 * field)
                + 0.08 * torch.roll(field, shifts=1, dims=1)
                - 0.05 * torch.roll(field, shifts=1, dims=0)
            )

        snapshots.append(field)

    return torch.stack(snapshots, dim=0)


def build_cfd_grid_multifidelity_dataset(
    config: CFDGridDataConfig,
) -> list[MultiFidelityTrajectorySample]:
    """Generate paired and partially paired CFD-like trajectories on different grids."""

    generator = torch.Generator().manual_seed(config.seed)
    base_times = torch.linspace(0.0, config.time_end, config.trajectory_length)
    groups: list[MultiFidelityTrajectorySample] = []

    low_height, low_width = config.low_grid_shape
    high_height, high_width = config.high_grid_shape

    def sample_parameter() -> Tensor:
        advection = torch.empty(1).uniform_(0.8, 1.6, generator=generator)
        swirl = torch.empty(1).uniform_(0.7, 1.4, generator=generator)
        return torch.cat((advection, swirl), dim=0)

    def build_observations(parameter: Tensor) -> tuple[Tensor, Tensor]:
        low_field = simulate_cfd_like_field(
            parameter=parameter,
            times=base_times,
            grid_shape=config.low_grid_shape,
            fidelity="low",
        )
        high_field = simulate_cfd_like_field(
            parameter=parameter,
            times=base_times,
            grid_shape=config.high_grid_shape,
            fidelity="high",
        )
        low_noise = torch.randn(low_field.shape, generator=generator, dtype=low_field.dtype)
        high_noise = torch.randn(high_field.shape, generator=generator, dtype=high_field.dtype)
        low_field = low_field + config.low_noise_std * low_noise
        high_field = high_field + config.high_noise_std * high_noise
        return low_field.reshape(base_times.shape[0], -1), high_field.reshape(base_times.shape[0], -1)

    def make_group(include_low: bool, include_high: bool, group_index: int) -> MultiFidelityTrajectorySample:
        parameter = sample_parameter()
        low_obs, high_obs = build_observations(parameter)
        pairing_id = f"cfd-group-{group_index:04d}"
        trajectories: dict[str, TrajectorySample] = {}
        if include_low:
            trajectories["low"] = TrajectorySample(
                times=base_times,
                observations=low_obs,
                parameter=parameter,
                fidelity="low",
                trajectory_id=f"{pairing_id}-low",
                pairing_id=pairing_id,
                metadata={
                    "synthetic": True,
                    "family": "cfd_like_field",
                    "grid_height": low_height,
                    "grid_width": low_width,
                },
            )
        if include_high:
            trajectories["high"] = TrajectorySample(
                times=base_times,
                observations=high_obs,
                parameter=parameter,
                fidelity="high",
                trajectory_id=f"{pairing_id}-high",
                pairing_id=pairing_id,
                metadata={
                    "synthetic": True,
                    "family": "cfd_like_field",
                    "grid_height": high_height,
                    "grid_width": high_width,
                },
            )
        return MultiFidelityTrajectorySample(
            trajectories=trajectories,
            pairing_id=pairing_id,
            metadata={"paired": include_low and include_high},
        )

    group_index = 0
    for _ in range(config.num_paired):
        groups.append(make_group(include_low=True, include_high=True, group_index=group_index))
        group_index += 1
    for _ in range(config.num_low_only):
        groups.append(make_group(include_low=True, include_high=False, group_index=group_index))
        group_index += 1
    for _ in range(config.num_high_only):
        groups.append(make_group(include_low=False, include_high=True, group_index=group_index))
        group_index += 1

    return groups
