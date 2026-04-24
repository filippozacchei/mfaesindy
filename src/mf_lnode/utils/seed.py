"""Reproducibility helpers."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Global seed value.
    deterministic:
        If `True`, ask PyTorch for deterministic kernels when possible.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
