"""Synthetic transient PDE-style dataset with configurable LF distortion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TransientFieldDatasetConfig:
    n_train: int = 64
    n_test: int = 16
    n_time: int = 100
    n_space_hf: int = 128
    n_space_lf: int = 48
    t_end: float = 5.0
    damping: float = 0.15
    frequency_range: tuple[float, float] = (1.5, 2.5)
    amplitude_range: tuple[float, float] = (0.8, 1.2)
    phase_range: tuple[float, float] = (0.0, np.pi)
    lf_blur_radius: int = 3
    lf_amplitude_scale: float = 0.9
    lf_nonlinear_bias: float = 0.05
    lf_distortion_strength: float = 1.0
    lf_noise_std: float = 0.0
    lf_dynamics_mismatch: float = 0.0
    seed: int = 7


@dataclass(slots=True)
class MultiFidelityDataset:
    x_grid: np.ndarray
    t_grid: np.ndarray
    train_hf: np.ndarray
    train_lf: np.ndarray
    test_hf: np.ndarray
    test_lf: np.ndarray
    train_latent: np.ndarray
    train_latent_dt: np.ndarray
    train_lf_latent: np.ndarray
    train_lf_latent_dt: np.ndarray
    test_latent: np.ndarray
    test_latent_dt: np.ndarray
    test_lf_latent: np.ndarray
    test_lf_latent_dt: np.ndarray
    train_params: np.ndarray
    test_params: np.ndarray


def _latent_trajectory(
    t: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float,
    damping: float,
) -> np.ndarray:
    decay = np.exp(-damping * t)
    z1 = amplitude * decay * np.cos(frequency * t + phase)
    z2 = amplitude * decay * np.sin(frequency * t + phase)
    return np.stack([z1, z2], axis=-1)


def _latent_derivative(
    t: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float,
    damping: float,
) -> np.ndarray:
    decay = np.exp(-damping * t)
    angle = frequency * t + phase
    z1_dt = amplitude * decay * (-damping * np.cos(angle) - frequency * np.sin(angle))
    z2_dt = amplitude * decay * (-damping * np.sin(angle) + frequency * np.cos(angle))
    return np.stack([z1_dt, z2_dt], axis=-1)


def _decode_to_field(latent: np.ndarray, x: np.ndarray) -> np.ndarray:
    phi1 = np.sin(np.pi * x)
    phi2 = np.sin(2.0 * np.pi * x)
    phi3 = np.cos(3.0 * np.pi * x)

    z1 = latent[:, 0][:, None]
    z2 = latent[:, 1][:, None]
    return z1 * phi1[None, :] + z2 * phi2[None, :] + 0.35 * (z1 * z2) * phi3[None, :]


def _moving_average_blur(field: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return field
    kernel = np.ones(2 * radius + 1, dtype=np.float64)
    kernel /= kernel.sum()
    padded = np.pad(field, ((0, 0), (radius, radius)), mode="wrap")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)


def _make_low_fidelity_latent(
    t: np.ndarray,
    latent_hf: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float,
    damping: float,
    config: TransientFieldDatasetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mismatch = float(config.lf_dynamics_mismatch)
    if mismatch <= 0.0:
        latent_lf_dt = _latent_derivative(
            t=t,
            amplitude=amplitude,
            frequency=frequency,
            phase=phase,
            damping=damping,
        )
        return latent_hf, latent_lf_dt

    lf_frequency = frequency * (1.0 + 0.10 * mismatch)
    lf_damping = damping * (1.0 + 0.20 * mismatch)
    latent_lf = _latent_trajectory(
        t=t,
        amplitude=amplitude,
        frequency=lf_frequency,
        phase=phase,
        damping=lf_damping,
    )
    latent_lf_dt = _latent_derivative(
        t=t,
        amplitude=amplitude,
        frequency=lf_frequency,
        phase=phase,
        damping=lf_damping,
    )
    return latent_lf, latent_lf_dt


def _make_low_fidelity(
    field_hf: np.ndarray,
    n_space_lf: int,
    x_hf: np.ndarray,
    x_lf: np.ndarray,
    config: TransientFieldDatasetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    field_lf = np.vstack([np.interp(x_lf, x_hf, snapshot) for snapshot in field_hf])

    strength = float(config.lf_distortion_strength)
    blur_radius = max(0, int(round(config.lf_blur_radius * strength)))
    amplitude_scale = max(0.25, 1.0 - strength * (1.0 - config.lf_amplitude_scale))
    nonlinear_bias = strength * config.lf_nonlinear_bias

    field_lf = _moving_average_blur(field_lf, radius=blur_radius)
    field_lf = amplitude_scale * field_lf
    field_lf = field_lf + nonlinear_bias * np.tanh(field_lf)

    field_lf_hf_grid = np.vstack([np.interp(x_hf, x_lf, snapshot) for snapshot in field_lf])
    if config.lf_noise_std > 0.0:
        noise_scale = float(config.lf_noise_std) * np.std(field_lf_hf_grid)
        field_lf_hf_grid = field_lf_hf_grid + noise_scale * rng.standard_normal(field_lf_hf_grid.shape)
    return field_lf_hf_grid


def _sample_parameters(rng: np.random.Generator, count: int, config: TransientFieldDatasetConfig) -> np.ndarray:
    amplitudes = rng.uniform(*config.amplitude_range, size=count)
    frequencies = rng.uniform(*config.frequency_range, size=count)
    phases = rng.uniform(*config.phase_range, size=count)
    return np.stack([amplitudes, frequencies, phases], axis=-1)


def _build_split(
    params: np.ndarray,
    x_hf: np.ndarray,
    x_lf: np.ndarray,
    t_grid: np.ndarray,
    config: TransientFieldDatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hf_fields = []
    lf_fields = []
    latents = []
    latent_derivatives = []
    lf_latents = []
    lf_latent_derivatives = []

    for amplitude, frequency, phase in params:
        latent = _latent_trajectory(
            t=t_grid,
            amplitude=float(amplitude),
            frequency=float(frequency),
            phase=float(phase),
            damping=config.damping,
        )
        latent_dt = _latent_derivative(
            t=t_grid,
            amplitude=float(amplitude),
            frequency=float(frequency),
            phase=float(phase),
            damping=config.damping,
        )
        lf_latent, lf_latent_dt = _make_low_fidelity_latent(
            t=t_grid,
            latent_hf=latent,
            amplitude=float(amplitude),
            frequency=float(frequency),
            phase=float(phase),
            damping=config.damping,
            config=config,
        )
        hf_field = _decode_to_field(latent, x_hf)
        lf_source_field = _decode_to_field(lf_latent, x_hf)
        lf_field = _make_low_fidelity(lf_source_field, config.n_space_lf, x_hf, x_lf, config, rng=rng)

        latents.append(latent.astype(np.float32))
        latent_derivatives.append(latent_dt.astype(np.float32))
        lf_latents.append(lf_latent.astype(np.float32))
        lf_latent_derivatives.append(lf_latent_dt.astype(np.float32))
        hf_fields.append(hf_field.astype(np.float32))
        lf_fields.append(lf_field.astype(np.float32))

    return (
        np.stack(hf_fields),
        np.stack(lf_fields),
        np.stack(latents),
        np.stack(latent_derivatives),
        np.stack(lf_latents),
        np.stack(lf_latent_derivatives),
    )


def generate_multi_fidelity_dataset(config: TransientFieldDatasetConfig) -> MultiFidelityDataset:
    rng = np.random.default_rng(config.seed)
    x_hf = np.linspace(0.0, 1.0, config.n_space_hf, endpoint=False, dtype=np.float64)
    x_lf = np.linspace(0.0, 1.0, config.n_space_lf, endpoint=False, dtype=np.float64)
    t_grid = np.linspace(0.0, config.t_end, config.n_time, dtype=np.float64)

    train_params = _sample_parameters(rng, config.n_train, config)
    test_params = _sample_parameters(rng, config.n_test, config)

    train_hf, train_lf, train_latent, train_latent_dt, train_lf_latent, train_lf_latent_dt = _build_split(
        train_params,
        x_hf,
        x_lf,
        t_grid,
        config,
        rng=rng,
    )
    test_hf, test_lf, test_latent, test_latent_dt, test_lf_latent, test_lf_latent_dt = _build_split(
        test_params,
        x_hf,
        x_lf,
        t_grid,
        config,
        rng=rng,
    )

    return MultiFidelityDataset(
        x_grid=x_hf.astype(np.float32),
        t_grid=t_grid.astype(np.float32),
        train_hf=train_hf,
        train_lf=train_lf,
        test_hf=test_hf,
        test_lf=test_lf,
        train_latent=train_latent,
        train_latent_dt=train_latent_dt,
        train_lf_latent=train_lf_latent,
        train_lf_latent_dt=train_lf_latent_dt,
        test_latent=test_latent,
        test_latent_dt=test_latent_dt,
        test_lf_latent=test_lf_latent,
        test_lf_latent_dt=test_lf_latent_dt,
        train_params=train_params.astype(np.float32),
        test_params=test_params.astype(np.float32),
    )
