"""Schedules for fidelity-specific loss weighting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FidelityWeightScheduler:
    """Linearly interpolate per-fidelity loss weights over training epochs."""

    initial_weights: dict[str, float]
    final_weights: dict[str, float]
    transition_epochs: int
    warmup_epochs: int = 0

    def weights_at(self, epoch: int) -> dict[str, float]:
        """Return the fidelity weights to use at a given epoch index."""

        if self.transition_epochs <= 0:
            alpha = 1.0
        elif epoch < self.warmup_epochs:
            alpha = 0.0
        else:
            alpha = min((epoch - self.warmup_epochs) / self.transition_epochs, 1.0)
        keys = set(self.initial_weights) | set(self.final_weights)
        return {
            key: (1.0 - alpha) * self.initial_weights.get(key, 1.0)
            + alpha * self.final_weights.get(key, self.initial_weights.get(key, 1.0))
            for key in keys
        }

