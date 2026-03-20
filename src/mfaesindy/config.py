"""Configuration models for experiments and training."""

from pydantic import BaseModel, Field


class AutoencoderConfig(BaseModel):
    input_dim: int
    latent_dim: int
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64])
    share_encoder_tail: bool = True
    share_decoder_head: bool = True


class SINDyConfig(BaseModel):
    threshold: float = 1e-2
    tikhonov: float = 1e-6
    control_variate_mode: str = "scalar"


class LossConfig(BaseModel):
    lambda_align: float = 1.0
    lambda_dyn: float = 1.0
    lambda_mf: float = 0.0
    lambda_smooth: float = 0.0


class TrainingConfig(BaseModel):
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    device: str = "cpu"


class ExperimentConfig(BaseModel):
    autoencoder: AutoencoderConfig
    sindy: SINDyConfig = Field(default_factory=SINDyConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
