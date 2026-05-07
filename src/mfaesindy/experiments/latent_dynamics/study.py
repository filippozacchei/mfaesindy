"""Training helpers for comparing alignment and sharing regimes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .autoencoder import AutoencoderArchitecture, MultiFidelityAutoencoder
from .dataset import MultiFidelityDataset, TransientFieldDatasetConfig, generate_multi_fidelity_dataset


@dataclass(slots=True)
class SharingStudyConfig:
    dataset: TransientFieldDatasetConfig = field(default_factory=TransientFieldDatasetConfig)
    latent_dim: int = 2
    epochs: int = 300
    learning_rate: float = 1e-3
    device: str = "cpu"


@dataclass(slots=True)
class RegimeConfig:
    name: str
    share_encoder_tail: bool
    share_decoder_head: bool
    lambda_align: float
    lambda_derivative: float = 0.0
    lambda_sindy: float = 0.0


DEFAULT_REGIMES: tuple[RegimeConfig, ...] = (
    RegimeConfig(
        name="separate_no_alignment",
        share_encoder_tail=False,
        share_decoder_head=False,
        lambda_align=0.0,
    ),
    RegimeConfig(
        name="separate_with_alignment",
        share_encoder_tail=False,
        share_decoder_head=False,
        lambda_align=1.0,
    ),
    RegimeConfig(
        name="shared_no_alignment",
        share_encoder_tail=True,
        share_decoder_head=True,
        lambda_align=0.0,
    ),
    RegimeConfig(
        name="shared_with_alignment",
        share_encoder_tail=True,
        share_decoder_head=True,
        lambda_align=1.0,
    ),
)


DYNAMICS_ALIGNMENT_REGIMES: tuple[RegimeConfig, ...] = (
    RegimeConfig(
        name="shared_with_derivative_alignment",
        share_encoder_tail=True,
        share_decoder_head=True,
        lambda_align=1.0,
        lambda_derivative=1.0,
    ),
    RegimeConfig(
        name="shared_with_sindy_alignment",
        share_encoder_tail=True,
        share_decoder_head=True,
        lambda_align=1.0,
        lambda_sindy=0.0,
    ),
)


def _flatten_snapshots(fields: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(fields.reshape(-1, fields.shape[-1]), dtype=torch.float32)


def _finite_difference_torch(trajectories: torch.Tensor, dt: float) -> torch.Tensor:
    derivatives = torch.empty_like(trajectories)
    if trajectories.shape[1] < 3:
        derivatives[:, 0] = 0.0
        if trajectories.shape[1] == 2:
            derivatives[:, 1] = (trajectories[:, 1] - trajectories[:, 0]) / dt
        return derivatives

    derivatives[:, 1:-1] = (trajectories[:, 2:] - trajectories[:, :-2]) / (2.0 * dt)
    derivatives[:, 0] = (-3.0 * trajectories[:, 0] + 4.0 * trajectories[:, 1] - trajectories[:, 2]) / (2.0 * dt)
    derivatives[:, -1] = (3.0 * trajectories[:, -1] - 4.0 * trajectories[:, -2] + trajectories[:, -3]) / (2.0 * dt)
    return derivatives


def _build_polynomial_library_torch(trajectories: torch.Tensor) -> torch.Tensor:
    flat = trajectories.reshape(-1, trajectories.shape[-1]).to(torch.float64)
    z1 = flat[:, 0:1]
    z2 = flat[:, 1:2]
    return torch.cat(
        [
            torch.ones((flat.shape[0], 1), dtype=flat.dtype, device=flat.device),
            z1,
            z2,
            z1**2,
            z1 * z2,
            z2**2,
        ],
        dim=1,
    )


def _fit_sindy_coefficients_torch(
    trajectories: torch.Tensor,
    derivatives: torch.Tensor,
    ridge: float = 1.0e-8,
) -> torch.Tensor:
    theta = _build_polynomial_library_torch(trajectories)
    dz = derivatives.reshape(-1, derivatives.shape[-1]).to(torch.float64)
    eye = torch.eye(theta.shape[1], dtype=theta.dtype, device=theta.device)
    coefficients = torch.linalg.solve(theta.T @ theta + ridge * eye, theta.T @ dz)
    return coefficients.to(trajectories.dtype)


def _relative_mse_torch(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    scale = 0.5 * (torch.mean(reference.detach() ** 2) + torch.mean(estimate.detach() ** 2))
    return torch.mean((reference - estimate) ** 2) / (scale + eps)


def encode_split(
    model: MultiFidelityAutoencoder,
    lf_fields: np.ndarray,
    hf_fields: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    x_lf = _flatten_snapshots(lf_fields).to(device)
    x_hf = _flatten_snapshots(hf_fields).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x_lf, x_hf)

    return {
        "z_lf": outputs["z_lf"].cpu().numpy().reshape(lf_fields.shape[0], lf_fields.shape[1], -1),
        "z_hf": outputs["z_hf"].cpu().numpy().reshape(hf_fields.shape[0], hf_fields.shape[1], -1),
        "xhat_lf": outputs["xhat_lf"].cpu().numpy().reshape(lf_fields.shape),
        "xhat_hf": outputs["xhat_hf"].cpu().numpy().reshape(hf_fields.shape),
    }


def _aligned_latent_error(learned_latent: np.ndarray, true_latent: np.ndarray) -> float:
    learned_flat = learned_latent.reshape(-1, learned_latent.shape[-1]).astype(np.float64)
    true_flat = true_latent.reshape(-1, true_latent.shape[-1]).astype(np.float64)

    learned_mean = learned_flat.mean(axis=0, keepdims=True)
    true_mean = true_flat.mean(axis=0, keepdims=True)
    learned_centered = learned_flat - learned_mean
    true_centered = true_flat - true_mean

    transform, *_ = np.linalg.lstsq(learned_centered, true_centered, rcond=None)
    aligned = learned_centered @ transform + true_mean
    return float(np.mean((aligned - true_flat) ** 2))


def _evaluate_alignment(
    model: MultiFidelityAutoencoder,
    dataset: MultiFidelityDataset,
    device: torch.device,
) -> dict[str, float]:
    x_lf = _flatten_snapshots(dataset.test_lf).to(device)
    x_hf = _flatten_snapshots(dataset.test_hf).to(device)
    encoded = encode_split(model, dataset.test_lf, dataset.test_hf, device=device)

    z_lf = torch.as_tensor(
        encoded["z_lf"].reshape(-1, encoded["z_lf"].shape[-1]),
        dtype=torch.float32,
        device=device,
    )
    z_hf = torch.as_tensor(
        encoded["z_hf"].reshape(-1, encoded["z_hf"].shape[-1]),
        dtype=torch.float32,
        device=device,
    )
    xhat_lf = torch.as_tensor(
        encoded["xhat_lf"].reshape(-1, encoded["xhat_lf"].shape[-1]),
        dtype=torch.float32,
        device=device,
    )
    xhat_hf = torch.as_tensor(
        encoded["xhat_hf"].reshape(-1, encoded["xhat_hf"].shape[-1]),
        dtype=torch.float32,
        device=device,
    )

    return {
        "lf_reconstruction_mse": float(torch.mean((x_lf - xhat_lf) ** 2).cpu()),
        "hf_reconstruction_mse": float(torch.mean((x_hf - xhat_hf) ** 2).cpu()),
        "cross_fidelity_latent_mse": float(torch.mean((z_lf - z_hf) ** 2).cpu()),
        "lf_aligned_true_latent_mse": _aligned_latent_error(encoded["z_lf"], dataset.test_latent),
        "hf_aligned_true_latent_mse": _aligned_latent_error(encoded["z_hf"], dataset.test_latent),
    }


def train_autoencoder_for_study(
    dataset: MultiFidelityDataset,
    regime: RegimeConfig,
    config: SharingStudyConfig,
) -> tuple[MultiFidelityAutoencoder, dict[str, float]]:
    device = torch.device(config.device)
    model = MultiFidelityAutoencoder(
        architecture=AutoencoderArchitecture(
            input_dim=dataset.train_hf.shape[-1],
            latent_dim=config.latent_dim,
        ),
        share_encoder_tail=regime.share_encoder_tail,
        share_decoder_head=regime.share_decoder_head,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    x_lf = _flatten_snapshots(dataset.train_lf).to(device)
    x_hf = _flatten_snapshots(dataset.train_hf).to(device)
    n_train, n_time = dataset.train_lf.shape[:2]
    dt = float(dataset.t_grid[1] - dataset.t_grid[0])

    for _ in range(config.epochs):
        model.train()
        outputs = model(x_lf, x_hf)
        loss_rec = criterion(outputs["xhat_lf"], x_lf) + criterion(outputs["xhat_hf"], x_hf)
        loss_align = criterion(outputs["z_lf"], outputs["z_hf"])

        z_lf = outputs["z_lf"].reshape(n_train, n_time, -1)
        z_hf = outputs["z_hf"].reshape(n_train, n_time, -1)
        dz_lf = _finite_difference_torch(z_lf, dt)
        dz_hf = _finite_difference_torch(z_hf, dt)

        if regime.lambda_derivative > 0.0:
            loss_derivative = criterion(dz_lf, dz_hf)
        else:
            loss_derivative = torch.zeros((), dtype=x_lf.dtype, device=device)

        if regime.lambda_sindy > 0.0:
            coef_lf = _fit_sindy_coefficients_torch(z_lf, dz_lf)
            coef_hf = _fit_sindy_coefficients_torch(z_hf, dz_hf)
            loss_sindy = _relative_mse_torch(coef_lf, coef_hf)
        else:
            loss_sindy = torch.zeros((), dtype=x_lf.dtype, device=device)

        loss = (
            loss_rec
            + regime.lambda_align * loss_align
            + regime.lambda_derivative * loss_derivative
            + regime.lambda_sindy * loss_sindy
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metrics = _evaluate_alignment(model, dataset, device)
    model.eval()
    with torch.no_grad():
        final_outputs = model(x_lf, x_hf)
        final_z_lf = final_outputs["z_lf"].reshape(n_train, n_time, -1)
        final_z_hf = final_outputs["z_hf"].reshape(n_train, n_time, -1)
        final_dz_lf = _finite_difference_torch(final_z_lf, dt)
        final_dz_hf = _finite_difference_torch(final_z_hf, dt)
        final_coef_lf = _fit_sindy_coefficients_torch(final_z_lf, final_dz_lf)
        final_coef_hf = _fit_sindy_coefficients_torch(final_z_hf, final_dz_hf)

    metrics["train_derivative_alignment_mse"] = float(torch.mean((final_dz_lf - final_dz_hf) ** 2).cpu())
    metrics["train_sindy_alignment_mse"] = float(torch.mean((final_coef_lf - final_coef_hf) ** 2).cpu())
    return model, metrics


def compare_sharing_strategies(
    config: SharingStudyConfig,
    regimes: tuple[RegimeConfig, ...] | None = None,
) -> dict[str, dict[str, float]]:
    dataset = generate_multi_fidelity_dataset(config.dataset)
    results: dict[str, dict[str, float]] = {}

    for regime in regimes or DEFAULT_REGIMES:
        _, metrics = train_autoencoder_for_study(dataset=dataset, regime=regime, config=config)
        results[regime.name] = metrics

    return results


def run_sharing_study(
    config: SharingStudyConfig,
    regimes: tuple[RegimeConfig, ...] | None = None,
) -> dict[str, object]:
    dataset = generate_multi_fidelity_dataset(config.dataset)
    device = torch.device(config.device)
    regime_results: dict[str, dict[str, object]] = {}

    for regime in regimes or DEFAULT_REGIMES:
        model, metrics = train_autoencoder_for_study(dataset=dataset, regime=regime, config=config)
        latents = encode_split(model, dataset.test_lf, dataset.test_hf, device=device)
        regime_results[regime.name] = {
            "regime": regime,
            "model": model,
            "metrics": metrics,
            "latents": latents,
        }

    return {
        "dataset": dataset,
        "regimes": regime_results,
    }
