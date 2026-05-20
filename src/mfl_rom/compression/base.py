from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseAutoencoder(nn.Module, ABC):
    def __init__(self, fidelities: tuple[str, ...]) -> None:
        super().__init__()
        self.fidelities = fidelities

    def _check_fidelity(self, fidelity: str) -> None:
        if fidelity not in self.fidelities:
            raise ValueError(
                f"Unknown fidelity '{fidelity}'. Expected one of "
                f"{self.fidelities}."
            )

    @abstractmethod
    def encode(self, x: torch.Tensor, fidelity: str) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor, fidelity: str) -> torch.Tensor:
        pass

    def reconstruct(self, x: torch.Tensor, fidelity: str) -> torch.Tensor:
        z = self.encode(x, fidelity)
        return self.decode(z, fidelity)
