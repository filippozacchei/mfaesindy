"""Normalization utilities for multi-fidelity trajectory data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import Tensor

from mf_lnode.data.datasets import MultiFidelityTrajectorySample, TrajectorySample


@dataclass(slots=True)
class TensorStandardScaler:
    """Feature-wise standardization for tensors with feature dimension last."""

    eps: float = 1e-6
    mean: Tensor | None = None
    std: Tensor | None = None

    def fit(self, tensors: Iterable[Tensor]) -> "TensorStandardScaler":
        flattened = [tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors]
        if not flattened:
            raise ValueError("At least one tensor is required to fit the scaler.")
        concatenated = torch.cat(flattened, dim=0)
        self.mean = concatenated.mean(dim=0, keepdim=True)
        self.std = concatenated.std(dim=0, keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, tensor: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("The scaler must be fitted before calling transform().")
        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)

    def inverse_transform(self, tensor: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("The scaler must be fitted before calling inverse_transform().")
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


@dataclass(slots=True)
class MultiFidelityScaler:
    """A collection of standard scalers indexed by fidelity name."""

    scalers: dict[str, TensorStandardScaler] = field(default_factory=dict)

    @classmethod
    def fit_from_groups(
        cls,
        groups: Sequence[MultiFidelityTrajectorySample],
    ) -> "MultiFidelityScaler":
        per_fidelity: dict[str, list[Tensor]] = {}
        for group in groups:
            for fidelity, sample in group.trajectories.items():
                per_fidelity.setdefault(fidelity, []).append(sample.observations)
        scalers = {
            fidelity: TensorStandardScaler().fit(tensors)
            for fidelity, tensors in per_fidelity.items()
        }
        return cls(scalers=scalers)

    def transform_trajectory(self, sample: TrajectorySample) -> TrajectorySample:
        """Return a new trajectory with standardized observations."""

        scaler = self.scalers[sample.fidelity]
        return TrajectorySample(
            times=sample.times,
            observations=scaler.transform(sample.observations),
            parameter=sample.parameter,
            fidelity=sample.fidelity,
            trajectory_id=sample.trajectory_id,
            pairing_id=sample.pairing_id,
            metadata=dict(sample.metadata),
        )

    def transform_group(
        self,
        group: MultiFidelityTrajectorySample,
    ) -> MultiFidelityTrajectorySample:
        """Return a scaled copy of a grouped multi-fidelity sample."""

        return MultiFidelityTrajectorySample(
            trajectories={
                fidelity: self.transform_trajectory(sample)
                for fidelity, sample in group.trajectories.items()
            },
            pairing_id=group.pairing_id,
            metadata=dict(group.metadata),
        )

    def transform_groups(
        self,
        groups: Sequence[MultiFidelityTrajectorySample],
    ) -> list[MultiFidelityTrajectorySample]:
        """Transform a sequence of grouped samples."""

        return [self.transform_group(group) for group in groups]

    def inverse_transform(self, fidelity: str, tensor: Tensor) -> Tensor:
        """Map standardized predictions back to the original observation space."""

        return self.scalers[fidelity].inverse_transform(tensor)

