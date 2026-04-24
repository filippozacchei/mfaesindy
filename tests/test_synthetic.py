from __future__ import annotations

import torch

from mf_lnode.data.synthetic import (
    apply_low_fidelity_bias,
    build_synthetic_multifidelity_dataset,
    lift_oscillator_observations,
    simulate_damped_oscillator,
)


def test_simulate_damped_oscillator_returns_expected_shape():
    trajectory = simulate_damped_oscillator(
        parameter=torch.tensor([0.1, 1.2]),
        times=torch.linspace(0.0, 1.0, 10),
    )
    assert trajectory.shape == (10, 2)


def test_build_synthetic_multifidelity_dataset_counts_groups(tiny_config):
    groups = build_synthetic_multifidelity_dataset(tiny_config.synthetic)
    expected = (
        tiny_config.synthetic.num_paired
        + tiny_config.synthetic.num_low_only
        + tiny_config.synthetic.num_high_only
    )
    assert len(groups) == expected


def test_lifted_observations_are_high_dimensional_and_preserve_physical_channels():
    times = torch.linspace(0.0, 2.0, 16)
    parameter = torch.tensor([0.12, 1.1])
    latent_state = simulate_damped_oscillator(parameter=parameter, times=times)
    observations = lift_oscillator_observations(
        latent_state=latent_state,
        times=times,
        parameter=parameter,
        observation_dim=12,
    )

    assert observations.shape == (16, 12)
    assert torch.allclose(observations[:, 0], latent_state[:, 0])
    assert torch.allclose(observations[:, 1], latent_state[:, 1])


def test_low_fidelity_bias_is_nonlinear():
    times = torch.linspace(0.0, 2.0, 16)
    parameter = torch.tensor([0.12, 1.1])
    latent_state = simulate_damped_oscillator(parameter=parameter, times=times)
    high = lift_oscillator_observations(
        latent_state=latent_state,
        times=times,
        parameter=parameter,
        observation_dim=10,
    )

    biased = apply_low_fidelity_bias(high, times=times, parameter=parameter)
    scaled_biased = apply_low_fidelity_bias(2.0 * high, times=times, parameter=parameter)

    assert biased.shape == high.shape
    assert not torch.allclose(biased, high)
    assert not torch.allclose(scaled_biased, 2.0 * biased)
