"""Shared utilities for the latent-dynamics experiment scripts and notebook."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mfaesindy.experiments.latent_dynamics.dataset import (
    TransientFieldDatasetConfig,
    generate_multi_fidelity_dataset,
)
from mfaesindy.experiments.latent_dynamics.study import (
    DYNAMICS_ALIGNMENT_REGIMES,
    SharingStudyConfig,
    run_sharing_study,
)


REGIME_SPECS: list[tuple[str, str]] = [
    ("separate_no_alignment", "Separate Encoders\nNo Alignment Loss"),
    ("separate_with_alignment", "Separate Encoders\nWith Alignment Loss"),
    ("shared_no_alignment", "Shared Tail/Head\nNo Alignment Loss"),
    ("shared_with_alignment", "Shared Tail/Head\nWith Alignment Loss"),
]

DYNAMICS_ALIGNMENT_SPECS: list[tuple[str, str]] = [
    ("shared_with_derivative_alignment", "Shared Tail/Head\nState + Derivative Alignment"),
    ("shared_with_sindy_alignment", "Shared Tail/Head\nState + SINDy Alignment (lambda=0)"),
]

REGIME_TITLE_MAP: dict[str, str] = dict(REGIME_SPECS + DYNAMICS_ALIGNMENT_SPECS)


def finite_difference(trajectories: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(trajectories, dt, axis=1, edge_order=2).astype(np.float64)


def flatten_time_series(array: np.ndarray) -> np.ndarray:
    return array.reshape(-1, array.shape[-1]).astype(np.float64)


def build_polynomial_library(trajectories: np.ndarray) -> np.ndarray:
    flat = flatten_time_series(trajectories)
    z1 = flat[:, 0:1]
    z2 = flat[:, 1:2]
    return np.hstack(
        [
            np.ones((flat.shape[0], 1), dtype=np.float64),
            z1,
            z2,
            z1**2,
            z1 * z2,
            z2**2,
        ]
    )


def stlsq(
    theta: np.ndarray,
    dz: np.ndarray,
    threshold: float = 0.02,
    alpha: float = 1.0e-8,
    max_iter: int = 8,
) -> np.ndarray:
    n_features = theta.shape[1]
    coefficients = np.linalg.solve(theta.T @ theta + alpha * np.eye(n_features), theta.T @ dz)

    for _ in range(max_iter):
        small = np.abs(coefficients) < threshold
        coefficients[small] = 0.0
        for column in range(coefficients.shape[1]):
            active = ~small[:, column]
            if not np.any(active):
                continue
            theta_active = theta[:, active]
            coefficients[active, column] = np.linalg.solve(
                theta_active.T @ theta_active + alpha * np.eye(active.sum()),
                theta_active.T @ dz[:, column],
            )
    return coefficients


def fit_sindy_coefficients(
    trajectories: np.ndarray,
    derivatives: np.ndarray,
    threshold: float = 0.02,
) -> np.ndarray:
    theta = build_polynomial_library(trajectories)
    dz = flatten_time_series(derivatives)
    return stlsq(theta, dz, threshold=threshold)


def pushforward_latent_derivative(
    encoder: torch.nn.Module,
    fields: np.ndarray,
    field_derivatives: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    encoder = encoder.to(device)
    encoder.eval()
    latent_dim = int(encoder(torch.as_tensor(fields[0, 0], dtype=torch.float32, device=device).unsqueeze(0)).shape[-1])
    pushed = np.zeros((fields.shape[0], fields.shape[1], latent_dim), dtype=np.float32)

    for traj_idx in range(fields.shape[0]):
        for time_idx in range(fields.shape[1]):
            x = torch.as_tensor(fields[traj_idx, time_idx], dtype=torch.float32, device=device)
            x = x.requires_grad_(True)
            dxdt = torch.as_tensor(field_derivatives[traj_idx, time_idx], dtype=torch.float32, device=device)

            z = encoder(x.unsqueeze(0)).squeeze(0)
            dzdt_components = []
            for dim in range(z.shape[0]):
                grad = torch.autograd.grad(z[dim], x, retain_graph=True)[0]
                dzdt_components.append(torch.dot(grad, dxdt))
            pushed[traj_idx, time_idx] = torch.stack(dzdt_components).detach().cpu().numpy()

    return pushed


def coefficient_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - estimate) / max(np.linalg.norm(reference), 1.0e-12))


def regime_specs_for(regimes: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (key, REGIME_TITLE_MAP.get(key, key.replace("_", " ").title()))
        for key in regimes.keys()
    ]


def run_full_study(
    config: SharingStudyConfig | None = None,
    regimes: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    return run_sharing_study(config or SharingStudyConfig(), regimes=regimes)


def run_dynamics_alignment_study(config: SharingStudyConfig | None = None) -> dict[str, Any]:
    return run_sharing_study(config or SharingStudyConfig(), regimes=DYNAMICS_ALIGNMENT_REGIMES)


def analyze_dynamics_results(results: dict[str, Any]) -> dict[str, Any]:
    dataset = results["dataset"]
    dt = float(dataset.t_grid[1] - dataset.t_grid[0])
    device = torch.device("cpu")
    true_coef = fit_sindy_coefficients(dataset.train_latent, dataset.train_latent_dt)

    analysis: dict[str, Any] = {
        "dataset": dataset,
        "true_coefficients": true_coef,
        "regimes": {},
    }

    hf_field_dt = finite_difference(dataset.test_hf, dt)
    lf_field_dt = finite_difference(dataset.test_lf, dt)

    for regime_name, regime_data in results["regimes"].items():
        model = regime_data["model"]
        train_encoded = {
            "z_lf": model.forward(
                torch.as_tensor(dataset.train_lf.reshape(-1, dataset.train_lf.shape[-1]), dtype=torch.float32),
                torch.as_tensor(dataset.train_hf.reshape(-1, dataset.train_hf.shape[-1]), dtype=torch.float32),
            )["z_lf"].detach().cpu().numpy().reshape(dataset.train_lf.shape[0], dataset.train_lf.shape[1], -1),
            "z_hf": model.forward(
                torch.as_tensor(dataset.train_lf.reshape(-1, dataset.train_lf.shape[-1]), dtype=torch.float32),
                torch.as_tensor(dataset.train_hf.reshape(-1, dataset.train_hf.shape[-1]), dtype=torch.float32),
            )["z_hf"].detach().cpu().numpy().reshape(dataset.train_hf.shape[0], dataset.train_hf.shape[1], -1),
        }
        test_encoded = regime_data["latents"]

        hf_latent_dt_fd = finite_difference(test_encoded["z_hf"], dt)
        lf_latent_dt_fd = finite_difference(test_encoded["z_lf"], dt)
        hf_latent_dt_push = pushforward_latent_derivative(model.encoder_hf, dataset.test_hf, hf_field_dt, device=device)
        lf_latent_dt_push = pushforward_latent_derivative(model.encoder_lf, dataset.test_lf, lf_field_dt, device=device)

        hf_coef = fit_sindy_coefficients(train_encoded["z_hf"], finite_difference(train_encoded["z_hf"], dt))
        lf_coef = fit_sindy_coefficients(train_encoded["z_lf"], finite_difference(train_encoded["z_lf"], dt))

        analysis["regimes"][regime_name] = {
            "hf_coefficients": hf_coef,
            "lf_coefficients": lf_coef,
            "hf_lf_coefficient_disagreement": coefficient_error(hf_coef, lf_coef),
            "hf_fd_vs_pushforward_mse": float(np.mean((hf_latent_dt_fd - hf_latent_dt_push) ** 2)),
            "lf_fd_vs_pushforward_mse": float(np.mean((lf_latent_dt_fd - lf_latent_dt_push) ** 2)),
            "test_encoded": test_encoded,
            "hf_latent_dt_fd": hf_latent_dt_fd,
            "lf_latent_dt_fd": lf_latent_dt_fd,
            "hf_latent_dt_push": hf_latent_dt_push,
            "lf_latent_dt_push": lf_latent_dt_push,
        }

    return analysis


def format_reconstruction_metrics(metrics: dict[str, float]) -> str:
    return "\n".join(
        [
            f"LF rec MSE: {metrics['lf_reconstruction_mse']:.3e}",
            f"HF rec MSE: {metrics['hf_reconstruction_mse']:.3e}",
            f"LF/HF latent MSE: {metrics['cross_fidelity_latent_mse']:.3e}",
            f"LF aligned-to-true MSE: {metrics['lf_aligned_true_latent_mse']:.3e}",
            f"HF aligned-to-true MSE: {metrics['hf_aligned_true_latent_mse']:.3e}",
        ]
    )


def plot_latent_panel(
    ax: plt.Axes,
    t: np.ndarray,
    true_latent: np.ndarray,
    lf_latent: np.ndarray,
    hf_latent: np.ndarray,
    title: str,
) -> None:
    ax.plot(t, true_latent[:, 0], color="black", linewidth=2.2, label="true z1")
    ax.plot(t, true_latent[:, 1], color="dimgray", linewidth=2.2, linestyle="--", label="true z2")
    ax.plot(t, lf_latent[:, 0], color="tab:orange", linewidth=1.8, label="LF enc z1")
    ax.plot(t, lf_latent[:, 1], color="tab:orange", linewidth=1.8, linestyle="--", label="LF enc z2")
    ax.plot(t, hf_latent[:, 0], color="tab:blue", linewidth=1.8, label="HF enc z1")
    ax.plot(t, hf_latent[:, 1], color="tab:blue", linewidth=1.8, linestyle="--", label="HF enc z2")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("latent coordinate")
    ax.grid(alpha=0.2)


def plot_phase_panel(
    ax: plt.Axes,
    true_latent: np.ndarray,
    lf_latent: np.ndarray,
    hf_latent: np.ndarray,
    title: str,
) -> None:
    ax.plot(true_latent[:, 0], true_latent[:, 1], color="black", linewidth=2.2, label="true latent")
    ax.plot(lf_latent[:, 0], lf_latent[:, 1], color="tab:orange", linewidth=1.8, label="LF encoding")
    ax.plot(hf_latent[:, 0], hf_latent[:, 1], color="tab:blue", linewidth=1.8, linestyle="--", label="HF encoding")
    ax.scatter(true_latent[0, 0], true_latent[0, 1], color="black", s=35, zorder=4)
    ax.scatter(lf_latent[0, 0], lf_latent[0, 1], color="tab:orange", s=35, zorder=4)
    ax.scatter(hf_latent[0, 0], hf_latent[0, 1], color="tab:blue", s=35, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")


def plot_field_panel(
    ax: plt.Axes,
    x: np.ndarray,
    reference_field: np.ndarray,
    lf_field: np.ndarray,
    hf_field: np.ndarray,
    title: str,
    time_index: int,
    reference_label: str = "reference",
    lf_label: str = "LF field",
    hf_label: str = "HF field",
) -> None:
    ax.plot(x, reference_field[time_index], color="black", linewidth=2.2, label=reference_label)
    ax.plot(x, lf_field[time_index], color="tab:orange", linewidth=1.8, label=lf_label)
    ax.plot(x, hf_field[time_index], color="tab:blue", linewidth=1.8, linestyle="--", label=hf_label)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("state")
    ax.grid(alpha=0.2)


def create_sharing_figure(
    results: dict[str, Any],
    sample_index: int = 0,
    time_index: int = -1,
) -> tuple[plt.Figure, np.ndarray]:
    dataset = results["dataset"]
    regimes = results["regimes"]
    t = dataset.t_grid
    x = dataset.x_grid
    true_latent = dataset.test_latent[sample_index]
    true_field = dataset.test_hf[sample_index]
    regime_specs = regime_specs_for(regimes)
    n_regimes = len(regime_specs)
    fig, axes = plt.subplots(3, n_regimes, figsize=(5.5 * n_regimes, 13), squeeze=False)

    for col, (key, title) in enumerate(regime_specs):
        regime = regimes[key]
        plot_latent_panel(axes[0, col], t, true_latent, regime["latents"]["z_lf"][sample_index], regime["latents"]["z_hf"][sample_index], title)
        plot_phase_panel(axes[1, col], true_latent, regime["latents"]["z_lf"][sample_index], regime["latents"]["z_hf"][sample_index], f"Phase Space\n{title}")
        plot_field_panel(
            axes[2, col],
            x,
            true_field,
            regime["latents"]["xhat_lf"][sample_index],
            regime["latents"]["xhat_hf"][sample_index],
            f"Reconstructions\n{title}",
            time_index=time_index,
            reference_label="true HF field",
            lf_label="LF reconstruction",
            hf_label="HF reconstruction",
        )
        axes[0, col].text(
            0.02,
            0.02,
            format_reconstruction_metrics(regime["metrics"]),
            transform=axes[0, col].transAxes,
            fontsize=8,
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
        )
        axes[0, col].legend(loc="upper right", fontsize=8)
        axes[1, col].legend(loc="upper right", fontsize=8)
        axes[2, col].legend(loc="upper right", fontsize=8)

    fig.suptitle("Effect Of Alignment Loss And Weight Sharing On Latent Dynamics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes


def create_dynamics_figure(
    analysis: dict[str, Any],
    sample_index: int = 0,
) -> tuple[plt.Figure, np.ndarray]:
    dataset = analysis["dataset"]
    regime_specs = regime_specs_for(analysis["regimes"])
    n_regimes = len(regime_specs)
    fig, axes = plt.subplots(2, n_regimes, figsize=(5.2 * n_regimes, 8), squeeze=False)
    for col, (regime_name, regime_title) in enumerate(regime_specs):
        regime = analysis["regimes"][regime_name]
        encoded = regime["test_encoded"]
        true_latent = dataset.test_latent[sample_index]
        axes[0, col].plot(true_latent[:, 0], true_latent[:, 1], label="true latent", color="black", linewidth=2.2)
        axes[0, col].plot(encoded["z_hf"][sample_index][:, 0], encoded["z_hf"][sample_index][:, 1], label="HF latent", color="tab:blue")
        axes[0, col].plot(encoded["z_lf"][sample_index][:, 0], encoded["z_lf"][sample_index][:, 1], label="LF latent", color="tab:orange")
        axes[0, col].set_title(regime_title)
        axes[0, col].set_xlabel("z1")
        axes[0, col].set_ylabel("z2")
        axes[0, col].grid(alpha=0.2)
        axes[0, col].legend(fontsize=8)

        axes[1, col].plot(regime["hf_latent_dt_fd"][sample_index][:, 0], label="HF fd", color="tab:blue")
        axes[1, col].plot(regime["hf_latent_dt_push"][sample_index][:, 0], label="HF push", color="tab:blue", linestyle="--")
        axes[1, col].plot(regime["lf_latent_dt_fd"][sample_index][:, 0], label="LF fd", color="tab:orange")
        axes[1, col].plot(regime["lf_latent_dt_push"][sample_index][:, 0], label="LF push", color="tab:orange", linestyle="--")
        axes[1, col].set_xlabel("time index")
        axes[1, col].set_ylabel("dz1/dt")
        axes[1, col].grid(alpha=0.2)
        axes[1, col].legend(fontsize=8)
        axes[1, col].text(
            0.02,
            0.02,
            "\n".join(
                [
                    f"HF/LF coef disagreement: {regime['hf_lf_coefficient_disagreement']:.3e}",
                    f"HF fd/push MSE: {regime['hf_fd_vs_pushforward_mse']:.3e}",
                    f"LF fd/push MSE: {regime['lf_fd_vs_pushforward_mse']:.3e}",
                ]
            ),
            transform=axes[1, col].transAxes,
            fontsize=8,
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
        )
    fig.suptitle("SINDy Recovery And Encoder-Pushforward Derivative Consistency", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes


def create_field_animation(
    dataset: Any,
    sample_index: int = 0,
) -> tuple[animation.FuncAnimation, plt.Figure]:
    x = dataset.x_grid
    t = dataset.t_grid
    lf = dataset.test_lf[sample_index]
    hf = dataset.test_hf[sample_index]
    y_min = float(min(lf.min(), hf.min()))
    y_max = float(max(lf.max(), hf.max()))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    (hf_line,) = ax.plot(x, hf[0], color="tab:blue", linewidth=2.2, label="HF field")
    (lf_line,) = ax.plot(x, lf[0], color="tab:orange", linewidth=2.0, linestyle="--", label="LF field")
    title = ax.set_title(f"LF/HF field evolution, t = {t[0]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("state")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(y_min - 0.05, y_max + 0.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    def update(frame_index: int):
        hf_line.set_ydata(hf[frame_index])
        lf_line.set_ydata(lf[frame_index])
        title.set_text(f"LF/HF field evolution, t = {t[frame_index]:.2f}")
        return hf_line, lf_line, title

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=80, blit=True, repeat=True)
    return ani, fig
