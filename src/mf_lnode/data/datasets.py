"""Dataset abstractions for offline multi-fidelity trajectory learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


MetadataValue = str | int | float | bool


@dataclass(slots=True)
class TrajectorySample:
    """One trajectory observed at a single fidelity."""

    times: Tensor
    observations: Tensor
    parameter: Tensor
    fidelity: str
    trajectory_id: str
    pairing_id: str | None = None
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.times.ndim != 1:
            raise ValueError("times must be a 1D tensor.")
        if self.observations.ndim != 2:
            raise ValueError("observations must be a 2D tensor of shape [time, features].")
        if self.times.shape[0] != self.observations.shape[0]:
            raise ValueError("times and observations must share the same trajectory length.")
        if self.parameter.ndim != 1:
            raise ValueError("parameter must be a 1D tensor.")

    @property
    def length(self) -> int:
        """Number of observation times in the trajectory."""

        return int(self.times.shape[0])

    @property
    def observation_dim(self) -> int:
        """Observation feature dimension."""

        return int(self.observations.shape[1])


@dataclass(slots=True)
class MultiFidelityTrajectorySample:
    """A grouped sample containing one or more fidelities of the same system instance."""

    trajectories: dict[str, TrajectorySample]
    pairing_id: str | None = None
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("At least one fidelity trajectory is required.")
        inferred_pair = self.pairing_id
        for fidelity, sample in self.trajectories.items():
            if sample.fidelity != fidelity:
                raise ValueError("Trajectory fidelity keys must match the sample fidelity field.")
            inferred_pair = inferred_pair or sample.pairing_id
        self.pairing_id = inferred_pair

    @property
    def fidelities(self) -> tuple[str, ...]:
        """Available fidelities for the grouped sample."""

        return tuple(sorted(self.trajectories))

    @property
    def parameter(self) -> Tensor:
        """Representative parameter vector for the grouped sample."""

        first_key = next(iter(self.trajectories))
        return self.trajectories[first_key].parameter


@dataclass(slots=True)
class WindowedTrajectorySlice:
    """A temporal window with an associated rollout target for a single fidelity."""

    context_times: Tensor
    context_observations: Tensor
    target_times: Tensor
    target_observations: Tensor
    future_context_times: Tensor
    future_context_observations: Tensor
    parameter: Tensor
    fidelity: str
    trajectory_id: str
    pairing_id: str | None
    start_index: int


@dataclass(slots=True)
class WindowedMultiFidelitySample:
    """A grouped windowed sample retaining every available fidelity."""

    windows: dict[str, WindowedTrajectorySlice]
    group_index: int
    pairing_id: str | None = None


class MultiFidelityTrajectoryDataset(Dataset[MultiFidelityTrajectorySample]):
    """Dataset over grouped multi-fidelity trajectories."""

    def __init__(self, groups: Sequence[MultiFidelityTrajectorySample]) -> None:
        self.groups = list(groups)
        self._fidelity_dims = self._infer_fidelity_dims(self.groups)

    @staticmethod
    def _infer_fidelity_dims(
        groups: Sequence[MultiFidelityTrajectorySample],
    ) -> dict[str, int]:
        dims: dict[str, int] = {}
        for group in groups:
            for fidelity, sample in group.trajectories.items():
                dim = sample.observation_dim
                if fidelity in dims and dims[fidelity] != dim:
                    raise ValueError(f"Inconsistent observation dimension for fidelity '{fidelity}'.")
                dims[fidelity] = dim
        return dims

    @property
    def fidelities(self) -> tuple[str, ...]:
        """All fidelity names present in the dataset."""

        return tuple(sorted(self._fidelity_dims))

    @property
    def fidelity_dims(self) -> dict[str, int]:
        """Observation dimension for each fidelity."""

        return dict(self._fidelity_dims)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> MultiFidelityTrajectorySample:
        return self.groups[index]


class WindowedTrajectoryDataset(Dataset[WindowedMultiFidelitySample]):
    """Temporal window view over grouped trajectories.

    Each dataset item preserves all fidelities available in the original grouped sample,
    while providing context windows, rollout targets, and a shifted future window for
    latent consistency.
    """

    def __init__(
        self,
        group_dataset: MultiFidelityTrajectoryDataset,
        window_size: int,
        rollout_horizon: int,
        future_context_shift: int = 1,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if rollout_horizon < 1:
            raise ValueError("rollout_horizon must be positive.")
        if future_context_shift < 1:
            raise ValueError("future_context_shift must be positive.")

        self.group_dataset = group_dataset
        self.window_size = window_size
        self.rollout_horizon = rollout_horizon
        self.future_context_shift = future_context_shift
        self.index: list[tuple[int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        required_tail = max(self.rollout_horizon, self.future_context_shift)
        for group_index, group in enumerate(self.group_dataset.groups):
            counts: list[int] = []
            for sample in group.trajectories.values():
                count = sample.length - self.window_size - required_tail + 1
                counts.append(max(count, 0))
            for start_index in range(min(counts, default=0)):
                self.index.append((group_index, start_index))

    def _slice_trajectory(
        self,
        sample: TrajectorySample,
        start_index: int,
    ) -> WindowedTrajectorySlice:
        window_end = start_index + self.window_size
        future_start = start_index + self.future_context_shift
        future_end = future_start + self.window_size
        rollout_anchor = window_end - 1
        target_end = rollout_anchor + self.rollout_horizon + 1

        return WindowedTrajectorySlice(
            context_times=sample.times[start_index:window_end],
            context_observations=sample.observations[start_index:window_end],
            target_times=sample.times[rollout_anchor:target_end],
            target_observations=sample.observations[rollout_anchor:target_end],
            future_context_times=sample.times[future_start:future_end],
            future_context_observations=sample.observations[future_start:future_end],
            parameter=sample.parameter,
            fidelity=sample.fidelity,
            trajectory_id=sample.trajectory_id,
            pairing_id=sample.pairing_id,
            start_index=start_index,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> WindowedMultiFidelitySample:
        group_index, start_index = self.index[index]
        group = self.group_dataset[group_index]
        return WindowedMultiFidelitySample(
            windows={
                fidelity: self._slice_trajectory(sample, start_index)
                for fidelity, sample in group.trajectories.items()
            },
            group_index=group_index,
            pairing_id=group.pairing_id,
        )


def collate_windowed_groups(
    samples: Sequence[WindowedMultiFidelitySample],
) -> dict[str, Any]:
    """Collate grouped windows into per-fidelity mini-batches."""

    fidelities = sorted({fidelity for sample in samples for fidelity in sample.windows})
    batch: dict[str, Any] = {
        "group_count": len(samples),
        "pairing_ids": [sample.pairing_id for sample in samples],
        "windows": {},
    }
    for fidelity in fidelities:
        selected = [(batch_index, sample.windows[fidelity]) for batch_index, sample in enumerate(samples) if fidelity in sample.windows]
        batch["windows"][fidelity] = {
            "group_ids": torch.tensor([item[0] for item in selected], dtype=torch.long),
            "context_times": torch.stack([item[1].context_times for item in selected], dim=0),
            "context_observations": torch.stack(
                [item[1].context_observations for item in selected],
                dim=0,
            ),
            "target_times": torch.stack([item[1].target_times for item in selected], dim=0),
            "target_observations": torch.stack(
                [item[1].target_observations for item in selected],
                dim=0,
            ),
            "future_context_times": torch.stack(
                [item[1].future_context_times for item in selected],
                dim=0,
            ),
            "future_context_observations": torch.stack(
                [item[1].future_context_observations for item in selected],
                dim=0,
            ),
            "parameters": torch.stack([item[1].parameter for item in selected], dim=0),
            "trajectory_ids": [item[1].trajectory_id for item in selected],
            "pairing_ids": [item[1].pairing_id for item in selected],
        }
    return batch


def split_group_dataset(
    groups_or_dataset: Sequence[MultiFidelityTrajectorySample] | MultiFidelityTrajectoryDataset,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[MultiFidelityTrajectoryDataset, MultiFidelityTrajectoryDataset, MultiFidelityTrajectoryDataset]:
    """Randomly split grouped trajectories by sample identity."""

    groups = (
        groups_or_dataset.groups
        if isinstance(groups_or_dataset, MultiFidelityTrajectoryDataset)
        else list(groups_or_dataset)
    )
    total = len(groups)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()

    train_end = int(total * train_fraction)
    val_end = train_end + int(total * val_fraction)

    train = [groups[index] for index in permutation[:train_end]]
    val = [groups[index] for index in permutation[train_end:val_end]]
    test = [groups[index] for index in permutation[val_end:]]
    return (
        MultiFidelityTrajectoryDataset(train),
        MultiFidelityTrajectoryDataset(val),
        MultiFidelityTrajectoryDataset(test),
    )


def split_group_dataset_by_parameter(
    groups_or_dataset: Sequence[MultiFidelityTrajectorySample] | MultiFidelityTrajectoryDataset,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[MultiFidelityTrajectoryDataset, MultiFidelityTrajectoryDataset, MultiFidelityTrajectoryDataset]:
    """Split grouped trajectories so whole parameter settings stay in the same partition."""

    groups = (
        groups_or_dataset.groups
        if isinstance(groups_or_dataset, MultiFidelityTrajectoryDataset)
        else list(groups_or_dataset)
    )
    grouped_by_parameter: dict[tuple[float, ...], list[MultiFidelityTrajectorySample]] = {}
    for group in groups:
        key = tuple(float(value) for value in group.parameter.tolist())
        grouped_by_parameter.setdefault(key, []).append(group)

    parameter_keys = list(grouped_by_parameter)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(parameter_keys), generator=generator).tolist()
    shuffled_keys = [parameter_keys[index] for index in permutation]

    train_end = int(len(shuffled_keys) * train_fraction)
    val_end = train_end + int(len(shuffled_keys) * val_fraction)

    def collect(keys: Sequence[tuple[float, ...]]) -> list[MultiFidelityTrajectorySample]:
        collected: list[MultiFidelityTrajectorySample] = []
        for key in keys:
            collected.extend(grouped_by_parameter[key])
        return collected

    return (
        MultiFidelityTrajectoryDataset(collect(shuffled_keys[:train_end])),
        MultiFidelityTrajectoryDataset(collect(shuffled_keys[train_end:val_end])),
        MultiFidelityTrajectoryDataset(collect(shuffled_keys[val_end:])),
    )

