"""Experiment slice for studying latent dynamics agreement across fidelities."""

from .autoencoder import MultiFidelityAutoencoder
from .dataset import TransientFieldDatasetConfig, generate_multi_fidelity_dataset
from .study import SharingStudyConfig, compare_sharing_strategies

__all__ = [
    "MultiFidelityAutoencoder",
    "TransientFieldDatasetConfig",
    "generate_multi_fidelity_dataset",
    "SharingStudyConfig",
    "compare_sharing_strategies",
]
