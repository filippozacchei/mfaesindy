from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    def snapshot(self, index: int) -> State:
        return State(self.states[index], channel_names=self.channel_names)
