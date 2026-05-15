from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class State:
    """
    One instantaneous state snapshot stored as a NumPy array.

    Spatial dimensions are preserved in their natural tensor form. If
    ``channel_names`` is provided, it is assumed to describe the last axis of
    ``values`` and must have length equal to ``values.shape[-1]``.
    """
    values: np.ndarray
    channel_names: tuple[str,...] | None = None
    
    def __post_init__(self):
        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)

        if values.ndim < 1 or any(dim == 0 for dim in values.shape):
            raise ValueError("values must be a non-empty NumPy array representing a single state snapshot.")

        if self.channel_names is not None and len(self.channel_names) != values.shape[-1]:
            raise ValueError(
                "If provided, channel_names must describe the last axis of values "
                "and have length equal to values.shape[-1]."
            )
            
