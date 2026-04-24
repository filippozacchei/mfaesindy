"""Synthetic two-fidelity trajectory generation for examples and tests."""

from __future__ import annotations

import torch
from torch import Tensor

from mf_lnode.configs.schema import SyntheticDataConfig
from mf_lnode.data.datasets import MultiFidelityTrajectorySample, TrajectorySample


def simulate_damped_oscillator(
    parameter: Tensor,
    times: Tensor,
    initial_state: Tensor | None = None,
) -> Tensor:
    """Simulate a simple parameterized damped oscillator with RK4."""

    if parameter.shape[0] != 2:
        raise ValueError("The synthetic oscillator expects a 2D parameter vector.")
    state = initial_state if initial_state is not None else torch.tensor([1.0, 0.0], dtype=times.dtype)
    state = state.to(device=times.device, dtype=times.dtype)
    damping = parameter[0]
    frequency = parameter[1]
    trajectory = [state]

    def rhs(_: Tensor, current_state: Tensor) -> Tensor:
        displacement, velocity = current_state.unbind(-1)
        acceleration = -(frequency**2) * displacement - damping * velocity
        return torch.stack((velocity, acceleration), dim=-1)

    current = state
    for t0, t1 in zip(times[:-1], times[1:]):
        dt = t1 - t0
        k1 = rhs(t0, current)
        k2 = rhs(t0 + 0.5 * dt, current + 0.5 * dt * k1)
        k3 = rhs(t0 + 0.5 * dt, current + 0.5 * dt * k2)
        k4 = rhs(t1, current + dt * k3)
        current = current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory.append(current)
    return torch.stack(trajectory, dim=0)


def lift_oscillator_observations(
    latent_state: Tensor,
    times: Tensor,
    parameter: Tensor,
    observation_dim: int,
) -> Tensor:
    """Lift the 2D oscillator state into a richer sensor-like observation space.

    The first two channels retain the physical displacement and velocity so that
    downstream plots can still expose an interpretable phase portrait. Additional
    channels are deterministic nonlinear sensor lifts of the same latent physics.
    """

    if observation_dim < 2:
        raise ValueError("observation_dim must be at least 2 for the synthetic example.")

    displacement = latent_state[:, 0]
    velocity = latent_state[:, 1]
    damping = parameter[0]
    frequency = parameter[1]
    energy = 0.5 * (velocity.square() + (frequency * displacement).square())
    phase = frequency * times
    envelope = torch.exp(-damping * times)

    features = [
        displacement,
        velocity,
        displacement * velocity,
        displacement.square(),
        velocity.square(),
        torch.sin(1.3 * displacement),
        torch.cos(1.1 * velocity),
        torch.tanh(1.5 * displacement - 0.45 * velocity),
        envelope * displacement,
        envelope * velocity,
        energy,
        displacement * torch.sin(phase),
        velocity * torch.cos(phase),
        energy * torch.sin(0.5 * times + damping),
        displacement.square() * velocity,
        velocity.square() * displacement,
    ]

    sensor_index = 0
    while len(features) < observation_dim:
        scale = 0.75 + 0.12 * sensor_index
        shift = 0.17 * sensor_index
        features.append(
            torch.sin(scale * displacement + shift) * torch.cos(0.7 * velocity - shift)
            + 0.08 * (1.0 + 0.03 * sensor_index) * energy
            + 0.04 * torch.sin(times * (0.8 + 0.05 * sensor_index))
        )
        sensor_index += 1

    return torch.stack(features[:observation_dim], dim=-1)


def apply_low_fidelity_bias(
    reference_observations: Tensor,
    times: Tensor,
    parameter: Tensor,
) -> Tensor:
    """Apply a structured low-fidelity bias to the high-fidelity observations.

    The distortion intentionally mixes several effects:
    - temporal lag/smoothing
    - cross-channel mixing
    - nonlinear compression and interaction terms
    - parameter-dependent time-varying sensor drift
    """

    sensor_ids = torch.arange(
        reference_observations.shape[-1],
        device=reference_observations.device,
        dtype=reference_observations.dtype,
    )
    lagged = torch.roll(reference_observations, shifts=3, dims=0)
    lagged[:3] = reference_observations[:3]
    rolled_left = torch.roll(reference_observations, shifts=1, dims=-1)
    rolled_right = torch.roll(reference_observations, shifts=-1, dims=-1)
    channel_scale = 0.60 + 0.14 * torch.sin(0.55 * sensor_ids + parameter[0])

    phase = times.unsqueeze(-1) * (parameter[1] + 0.10 * (sensor_ids + 1.0))
    drift = 0.09 * torch.sin(phase + 0.42 * sensor_ids) + 0.05 * torch.cos(0.72 * phase - parameter[0])
    compression = torch.tanh(1.35 * reference_observations + 0.28 * rolled_left)
    interaction = reference_observations * rolled_right
    cross_channel = rolled_left - 0.65 * rolled_right
    nonlinear_warp = torch.sin(phase + 0.55 * reference_observations) * (
        0.05 + 0.01 * sensor_ids
    )

    return (
        channel_scale.unsqueeze(0) * reference_observations
        + 0.22 * lagged
        + 0.16 * compression
        + 0.08 * interaction
        + 0.06 * cross_channel
        + nonlinear_warp
        + drift
    )


def build_synthetic_multifidelity_dataset(
    config: SyntheticDataConfig,
) -> list[MultiFidelityTrajectorySample]:
    """Generate paired and partially paired two-fidelity trajectories."""

    generator = torch.Generator().manual_seed(config.seed)
    base_times = torch.linspace(0.0, config.time_end, config.trajectory_length)
    groups: list[MultiFidelityTrajectorySample] = []

    def sample_parameter() -> Tensor:
        damping = torch.empty(1).uniform_(0.05, 0.25, generator=generator)
        frequency = torch.empty(1).uniform_(0.8, 1.6, generator=generator)
        return torch.cat((damping, frequency), dim=0)

    def build_observations(parameter: Tensor) -> tuple[Tensor, Tensor]:
        truth = simulate_damped_oscillator(parameter=parameter, times=base_times)
        high = lift_oscillator_observations(
            latent_state=truth,
            times=base_times,
            parameter=parameter,
            observation_dim=config.observation_dim,
        )
        low = apply_low_fidelity_bias(
            reference_observations=high,
            times=base_times,
            parameter=parameter,
        )
        low_noise = torch.randn(low.shape, generator=generator, dtype=low.dtype)
        high_noise = torch.randn(high.shape, generator=generator, dtype=high.dtype)
        low = low + config.low_noise_std * low_noise
        high = high + config.high_noise_std * high_noise
        return low, high

    def make_group(include_low: bool, include_high: bool, group_index: int) -> MultiFidelityTrajectorySample:
        parameter = sample_parameter()
        low_obs, high_obs = build_observations(parameter)
        pairing_id = f"group-{group_index:04d}"
        trajectories: dict[str, TrajectorySample] = {}
        if include_low:
            trajectories["low"] = TrajectorySample(
                times=base_times,
                observations=low_obs,
                parameter=parameter,
                fidelity="low",
                trajectory_id=f"{pairing_id}-low",
                pairing_id=pairing_id,
                metadata={"synthetic": True, "family": "damped_oscillator"},
            )
        if include_high:
            trajectories["high"] = TrajectorySample(
                times=base_times,
                observations=high_obs,
                parameter=parameter,
                fidelity="high",
                trajectory_id=f"{pairing_id}-high",
                pairing_id=pairing_id,
                metadata={"synthetic": True, "family": "damped_oscillator"},
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
