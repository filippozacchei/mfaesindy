"""PySINDy integration points for latent-space regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pysindy as ps
import torch


@dataclass(slots=True)
class SparseRegressionResult:
    coefficients: torch.Tensor
    model: ps.SINDy | None = None


class LatentSINDyRegressor:
    def __init__(self, threshold: float = 1e-2) -> None:
        optimizer = ps.STLSQ(threshold=threshold)
        self.model = ps.SINDy(optimizer=optimizer)

    def fit(self, z: np.ndarray, t: np.ndarray | float, x_dot: np.ndarray | None = None) -> ps.SINDy:
        self.model.fit(z, t=t, x_dot=x_dot)
        return self.model

    def coefficients(self) -> torch.Tensor:
        coef = self.model.coefficients()
        return torch.as_tensor(coef, dtype=torch.float32)
