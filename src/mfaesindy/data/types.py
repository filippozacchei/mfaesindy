"""Typed containers for low- and high-fidelity trajectory data."""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TrajectoryBatch:
    x_lf: torch.Tensor
    x_hf: torch.Tensor | None = None
    dx_lf: torch.Tensor | None = None
    dx_hf: torch.Tensor | None = None
    time: torch.Tensor | None = None
    paired_mask: torch.Tensor | None = None
