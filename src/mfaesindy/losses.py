"""Loss functions for representation learning and dynamics regularization."""

from __future__ import annotations

import torch
from torch import nn


class MultiFidelityLoss(nn.Module):
    def __init__(
        self,
        lambda_align: float = 1.0,
        lambda_dyn: float = 1.0,
        lambda_mf: float = 0.0,
        lambda_smooth: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_dyn = lambda_dyn
        self.lambda_mf = lambda_mf
        self.lambda_smooth = lambda_smooth

    def reconstruction(
        self,
        x_lf: torch.Tensor,
        xhat_lf: torch.Tensor,
        x_hf: torch.Tensor | None = None,
        xhat_hf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = torch.mean((x_lf - xhat_lf) ** 2)
        if x_hf is not None and xhat_hf is not None:
            loss = loss + torch.mean((x_hf - xhat_hf) ** 2)
        return loss

    def alignment(self, z_lf: torch.Tensor, z_hf: torch.Tensor | None = None) -> torch.Tensor:
        if z_hf is None:
            return z_lf.new_tensor(0.0)
        return torch.mean((z_lf - z_hf) ** 2)

    def smoothness(self, z: torch.Tensor | None) -> torch.Tensor:
        if z is None or z.shape[0] < 3:
            tensor = z if z is not None else torch.zeros(1)
            return tensor.new_tensor(0.0)
        accel = z[2:] - 2 * z[1:-1] + z[:-2]
        return torch.mean(accel**2)
