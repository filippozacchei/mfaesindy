"""Dataclass-based configuration schema for `mf_lnode`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SyntheticDataConfig:
    """Settings for the built-in synthetic two-fidelity example."""

    num_paired: int = 24
    num_low_only: int = 8
    num_high_only: int = 4
    trajectory_length: int = 48
    time_end: float = 6.0
    parameter_dim: int = 2
    observation_dim: int = 12
    low_noise_std: float = 0.08
    high_noise_std: float = 0.02
    seed: int = 7


@dataclass(slots=True)
class CFDGridDataConfig:
    """Settings for the CFD-like different-grid synthetic example."""

    num_paired: int = 20
    num_low_only: int = 6
    num_high_only: int = 4
    trajectory_length: int = 48
    time_end: float = 8.0
    parameter_dim: int = 2
    low_grid_shape: tuple[int, int] = (16, 16)
    high_grid_shape: tuple[int, int] = (32, 32)
    low_noise_std: float = 0.015
    high_noise_std: float = 0.004
    seed: int = 11


@dataclass(slots=True)
class DataConfig:
    """Data-windowing and split parameters."""

    window_size: int = 6
    rollout_horizon: int = 8
    future_context_shift: int = 1
    train_fraction: float = 0.7
    val_fraction: float = 0.15


@dataclass(slots=True)
class ModelConfig:
    """Neural network and latent ODE model configuration."""

    window_size: int = 6
    parameter_dim: int = 2
    fidelity_dims: dict[str, int] = field(default_factory=lambda: {"low": 12, "high": 12})
    latent_dim: int = 8
    adapter_dim: int = 16
    encoder_hidden_dim: int = 64
    decoder_hidden_dim: int = 64
    dynamics_hidden_dim: int = 64
    num_hidden_layers: int = 2
    shared_encoder_backbone: bool = True
    shared_latent_projection: bool = True
    shared_decoder_backbone: bool = True
    hierarchical_decoder: bool = True
    hierarchy_low_fidelity: str = "low"
    hierarchy_high_fidelity: str = "high"
    solver_backend: str = "rk4"
    solver_steps_per_interval: int = 1
    conditioning: str = "concat"
    activation: str = "silu"

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if self.parameter_dim < 0:
            raise ValueError("parameter_dim must be non-negative.")
        if self.hierarchical_decoder:
            low = self.hierarchy_low_fidelity
            high = self.hierarchy_high_fidelity
            if low not in self.fidelity_dims or high not in self.fidelity_dims:
                raise ValueError("Hierarchical decoder fidelities must exist in fidelity_dims.")
            if self.fidelity_dims[low] != self.fidelity_dims[high]:
                raise ValueError(
                    "Hierarchical decoding requires matching observation dimensions "
                    "for the low and high fidelities."
                )


@dataclass(slots=True)
class LossConfig:
    """Weights for the default composite training objective."""

    rollout_weight: float = 1.0
    latent_consistency_weight: float = 0.2
    latent_alignment_weight: float = 0.5
    multifidelity_weight: float = 1.0
    discrepancy_weight: float = 1e-3
    regularization_weight: float = 1e-4
    align_latent_trajectories: bool = True
    fidelity_weights_start: dict[str, float] = field(
        default_factory=lambda: {"low": 1.0, "high": 0.5}
    )
    fidelity_weights_end: dict[str, float] = field(
        default_factory=lambda: {"low": 0.5, "high": 2.0}
    )


@dataclass(slots=True)
class TrainingConfig:
    """Optimizer, runtime, and checkpointing settings."""

    epochs: int = 12
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float | None = 1.0
    device: str = "cpu"
    use_amp: bool = True
    output_dir: str = "artifacts"
    checkpoint_name: str = "best.pt"
    transition_epochs: int = 8
    warmup_epochs: int = 0
    log_every: int = 1
    tensorboard: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment configuration for the synthetic example."""

    seed: int = 7
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    cfd_grid: CFDGridDataConfig = field(default_factory=CFDGridDataConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
