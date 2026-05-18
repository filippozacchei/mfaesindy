from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MFTrajectory",
    "State",
    "TrainingTrajectoryDataset",
    "Trajectory",
]


def _validate_non_empty_array(
    values: np.ndarray,
    *,
    name: str,
    min_ndim: int,
) -> None:
    if values.ndim < min_ndim or any(dim == 0 for dim in values.shape):
        raise ValueError(
            f"{name} must be a non-empty NumPy array with at least "
            f"{min_ndim} dimensions."
        )


def _validate_channel_names(
    channel_names: tuple[str, ...] | None,
    values: np.ndarray,
    *,
    name: str,
) -> None:
    if channel_names is not None and len(channel_names) != values.shape[-1]:
        raise ValueError(
            f"If provided, channel_names must describe the last axis of "
            f"{name} and have length equal to {name}.shape[-1]."
        )


@dataclass(frozen=True)
class State:
    """
    One instantaneous state snapshot stored as a NumPy array.

    Spatial dimensions are preserved in their natural tensor form. If
    ``channel_names`` is provided, it is assumed to describe the last axis of
    ``values`` and must have length equal to ``values.shape[-1]``.
    """
    values: np.ndarray
    channel_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)

        _validate_non_empty_array(values, name="values", min_ndim=1)
        _validate_channel_names(self.channel_names, values, name="values")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


@dataclass(frozen=True)
class Trajectory:
    """
    One sampled trajectory at one fidelity.

    The first axis of ``states`` is the time axis. The remaining axes preserve
    the natural tensor layout of a single state snapshot.
    """

    time: np.ndarray
    states: np.ndarray
    parameters: np.ndarray
    fidelity: str
    channel_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time)
        states = np.asarray(self.states)
        parameters = np.asarray(self.parameters)

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "parameters", parameters)

        if time.ndim != 1 or time.size == 0:
            raise ValueError("time must be a non-empty one-dimensional array.")
        _validate_non_empty_array(states, name="states", min_ndim=2)
        if parameters.ndim != 1:
            raise ValueError("parameters must be a one-dimensional array.")
        if not self.fidelity:
            raise ValueError("fidelity must be a non-empty string.")
        if time.shape[0] != states.shape[0]:
            raise ValueError(
                "time.shape[0] must match the number of trajectory states."
            )
        _validate_channel_names(self.channel_names, states, name="states")

    @property
    def num_steps(self) -> int:
        return self.time.shape[0]

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.states.shape[1:]

    def snapshot(self, index: int) -> State:
        return State(self.states[index], channel_names=self.channel_names)


@dataclass(frozen=True)
class MFTrajectory:
    """
    One paired low-/high-fidelity trajectory sample.

    This container is intentionally strict about operating conditions: the
    paired LF and HF trajectories must correspond to the same parameter vector.
    Their time grids may differ.
    """

    lf_trajectory: Trajectory
    hf_trajectory: Trajectory

    def __post_init__(self) -> None:
        if self.lf_trajectory.fidelity != "LF":
            raise ValueError("lf_trajectory must have fidelity equal to 'LF'.")
        if self.hf_trajectory.fidelity != "HF":
            raise ValueError("hf_trajectory must have fidelity equal to 'HF'.")
        if (
            self.lf_trajectory.parameters.shape
            != self.hf_trajectory.parameters.shape
        ):
            raise ValueError(
                "LF and HF trajectories must have parameter vectors with "
                "the same shape."
            )
        if not np.allclose(
            self.lf_trajectory.parameters,
            self.hf_trajectory.parameters,
        ):
            raise ValueError(
                "LF and HF trajectories must share the same parameter values."
            )

    @property
    def parameters(self) -> np.ndarray:
        return self.hf_trajectory.parameters

@dataclass(frozen=True)
class TrainingTrajectoryDataset:
    """
    Minimal training-data container for paired MF samples and extra single-
    fidelity trajectories.

    ``mf_samples`` contains paired LF/HF trajectories.
    ``lf_samples`` contains additional low-fidelity-only trajectories.
    ``hf_samples`` contains additional high-fidelity-only trajectories.
    """

    mf_samples: list[MFTrajectory]
    lf_samples: list[Trajectory]
    hf_samples: list[Trajectory]

    def __post_init__(self) -> None:
        mf_samples = list(self.mf_samples)
        lf_samples = list(self.lf_samples)
        hf_samples = list(self.hf_samples)

        object.__setattr__(self, "mf_samples", mf_samples)
        object.__setattr__(self, "lf_samples", lf_samples)
        object.__setattr__(self, "hf_samples", hf_samples)

        for sample in mf_samples:
            if not isinstance(sample, MFTrajectory):
                raise TypeError(
                    "mf_samples must contain only MFTrajectory objects."
                )
        for trajectory in lf_samples:
            if not isinstance(trajectory, Trajectory):
                raise TypeError(
                    "lf_samples must contain only Trajectory objects."
                )
            if trajectory.fidelity != "LF":
                raise ValueError(
                    "lf_samples must contain only trajectories with fidelity "
                    "equal to 'LF'."
                )
        for trajectory in hf_samples:
            if not isinstance(trajectory, Trajectory):
                raise TypeError(
                    "hf_samples must contain only Trajectory objects."
                )
            if trajectory.fidelity != "HF":
                raise ValueError(
                    "hf_samples must contain only trajectories with fidelity "
                    "equal to 'HF'."
                )

    @property
    def num_mf_samples(self) -> int:
        return len(self.mf_samples)

    @property
    def num_lf_samples(self) -> int:
        return len(self.lf_samples)

    @property
    def num_hf_samples(self) -> int:
        return len(self.hf_samples)
