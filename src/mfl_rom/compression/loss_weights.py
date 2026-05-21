from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import ItemsView, Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import torch
from torch import nn

__all__ = [
    "LossWeightStrategy",
    "NTKLossWeights",
    "StaticLossWeights",
    "ensure_loss_weight_strategy",
    "shared_trainable_parameters",
]


class LossWeightStrategy(ABC):
    """
    Base interface for loss-weight selection in multifidelity training.

    The current training pipeline keeps the total objective outside the core
    training loop: the user defines a `loss_fn`, and that function combines the
    primitive losses with scalar weights. This interface gives that pattern a
    small extension point so future adaptive schemes can be added cleanly.

    Today the simplest use case is:

    ```python
    loss_weights = StaticLossWeights(
        {
            "lf_reconstruction": 0.5,
            "hf_reconstruction": 1.0,
            "state_alignment": 0.1,
            "sindy_residual": 1.0e-3,
        }
    )

    loss = loss_weights["hf_reconstruction"] * reconstruction
    ```

    A future dynamic strategy can subclass `LossWeightStrategy`, implement
    `resolve()`, and optionally update its internal state through `update()`
    without changing the training loop API.
    """

    @abstractmethod
    def resolve(self) -> dict[str, float]:
        """Return the current scalar weights as a plain dictionary."""

    def update(self, **context: object) -> None:
        """
        Optional hook for adaptive strategies.

        Static strategies ignore this hook. A dynamic strategy may use it to
        update internal weights from metrics, gradients, or epoch counters.
        """

    def as_dict(self) -> dict[str, float]:
        """Return the current weights as a detached plain dictionary."""
        return self.resolve()

    def __getitem__(self, name: str) -> float:
        return self.resolve()[name]

    def get(self, name: str, default: float | None = None) -> float | None:
        return self.resolve().get(name, default)

    def items(self) -> ItemsView[str, float]:
        return self.resolve().items()


@dataclass(frozen=True, slots=True)
class StaticLossWeights(LossWeightStrategy):
    """
    Immutable fixed loss weights.

    Parameters
    ----------
    weights
        Mapping from loss names to non-negative finite scalar weights.

    Notes
    -----
    This class is intentionally small. It is meant to replace ad hoc weight
    dictionaries in notebooks and scripts while preserving the same usage style.

    Example
    -------
    ```python
    base_weights = StaticLossWeights(
        {
            "lf_reconstruction": 0.5,
            "hf_reconstruction": 1.0,
            "state_alignment": 0.1,
            "sindy_residual": 1.0e-3,
        }
    )

    no_sindy = base_weights.with_overrides({"sindy_residual": 0.0})
    ```
    """

    _weights: Mapping[str, float]

    def __init__(self, weights: Mapping[str, float]) -> None:
        normalized: dict[str, float] = {}

        for name, value in weights.items():
            scalar = float(value)
            if not isfinite(scalar):
                raise ValueError(f"Loss weight '{name}' must be finite.")
            if scalar < 0.0:
                raise ValueError(f"Loss weight '{name}' must be non-negative.")
            normalized[name] = scalar

        object.__setattr__(self, "_weights", MappingProxyType(normalized))

    @property
    def weights(self) -> Mapping[str, float]:
        """Read-only view of the stored fixed weights."""
        return self._weights

    def resolve(self) -> dict[str, float]:
        return dict(self._weights)

    def with_overrides(
        self,
        overrides: Mapping[str, float] | None = None,
    ) -> StaticLossWeights:
        """
        Return a new `StaticLossWeights` object with selected overrides.

        This is useful for compact ablations, for example turning the SINDy
        term on or off while keeping the rest of the setup unchanged.
        """
        if overrides is None:
            return StaticLossWeights(self._weights)

        merged = self.resolve()
        merged.update(overrides)
        return StaticLossWeights(merged)


def shared_trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """
    Return the trainable parameters associated with the shared latent pathway.

    The current autoencoder implementations expose their shared components
    through a small set of conventional attribute names. This helper extracts
    those parameters so adaptive loss balancing can focus on the part of the
    model that is actually shared across fidelities.

    If no shared-path attributes are found, the function falls back to all
    trainable model parameters.
    """
    module_names = (
        "shared_encoder",
        "to_latent",
        "to_shared",
        "from_latent",
        "shared_decoder",
    )

    parameters: list[nn.Parameter] = []
    for name in module_names:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            parameters.extend(
                parameter
                for parameter in module.parameters()
                if parameter.requires_grad
            )

    if parameters:
        return tuple(parameters)

    return tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )


class NTKLossWeights(LossWeightStrategy):
    """
    NTK-inspired adaptive loss weights based on shared-parameter gradient norms.

    This class does not construct the full neural tangent kernel explicitly.
    Instead, it uses a lightweight surrogate: for each primitive loss term, it
    measures the gradient norm with respect to the shared autoencoder
    parameters and adjusts the scalar weight inversely to that scale.

    The intent is similar to NTK-based balancing in PINNs:
    losses that already induce large shared-parameter updates are downweighted,
    while losses with weak shared-parameter influence are upweighted.

    Parameters
    ----------
    weights
        Base scalar weights used both as initialization and as the reference
        scale for normalization.
    ema_decay
        Exponential moving average factor for the observed gradient norms.
    power
        Exponent used in the inverse scaling rule. `power=1.0` corresponds to
        an inverse gradient-norm weighting.
    epsilon
        Small positive number for numerical stability.
    update_interval
        Recompute the adaptive weights every `update_interval` update calls.
    min_weight, max_weight
        Optional clipping bounds applied after normalization.

    Notes
    -----
    The strategy is intentionally stateful: `update(...)` should be called
    inside the user-defined `loss_fn` before the final weighted loss is formed.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        ema_decay: float = 0.9,
        power: float = 1.0,
        epsilon: float = 1.0e-12,
        update_interval: int = 1,
        min_weight: float = 0.0,
        max_weight: float | None = None,
    ) -> None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1.")
        if power <= 0.0:
            raise ValueError("power must be positive.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if update_interval < 1:
            raise ValueError("update_interval must be at least 1.")
        if min_weight < 0.0:
            raise ValueError("min_weight must be non-negative.")
        if max_weight is not None and max_weight < min_weight:
            raise ValueError("max_weight must be greater than min_weight.")

        self._base_weights = StaticLossWeights(weights)
        self._current_weights = self._base_weights.resolve()
        self._grad_norm_ema = {
            name: None for name in self._current_weights
        }
        self.ema_decay = float(ema_decay)
        self.power = float(power)
        self.epsilon = float(epsilon)
        self.update_interval = int(update_interval)
        self.min_weight = float(min_weight)
        self.max_weight = max_weight
        self._num_updates = 0

    @property
    def base_weights(self) -> Mapping[str, float]:
        """Read-only view of the initial scalar weights."""
        return self._base_weights.weights

    @property
    def grad_norm_ema(self) -> dict[str, float | None]:
        """Current exponential moving averages of shared-parameter norms."""
        return dict(self._grad_norm_ema)

    def with_overrides(
        self,
        overrides: Mapping[str, float] | None = None,
    ) -> NTKLossWeights:
        """
        Return a fresh adaptive strategy with overridden base weights.

        The adaptive state is intentionally reset. This is convenient for
        ablations where each run should start from the same initial condition.
        """
        merged = self._base_weights.resolve()
        if overrides is not None:
            merged.update(overrides)

        return NTKLossWeights(
            merged,
            ema_decay=self.ema_decay,
            power=self.power,
            epsilon=self.epsilon,
            update_interval=self.update_interval,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
        )

    def resolve(self) -> dict[str, float]:
        return dict(self._current_weights)

    def update(
        self,
        *,
        name: str,
        loss: torch.Tensor,
        model: nn.Module | None = None,
        parameters: tuple[nn.Parameter, ...] | None = None,
        **context: object,
    ) -> None:
        """
        Update the adaptive weights from one primitive loss term.

        Parameters
        ----------
        name
            Name of the primitive term being observed, e.g.
            `"hf_reconstruction"` or `"state_alignment"`.
        loss
            Scalar torch loss before weighting.
        model
            Autoencoder model. Used to infer the shared trainable parameters
            when `parameters` is not provided.
        parameters
            Optional explicit parameter tuple. If omitted, the shared-path
            parameters are selected automatically from `model`.
        """
        if name not in self._current_weights:
            raise KeyError(f"Unknown loss name '{name}'.")
        if not torch.is_grad_enabled() or not loss.requires_grad:
            return
        if parameters is None:
            if model is None:
                raise ValueError("Either model or parameters must be provided.")
            parameters = shared_trainable_parameters(model)
        if len(parameters) == 0:
            return

        self._num_updates += 1
        if self._num_updates % self.update_interval != 0:
            return

        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

        grad_energy = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for gradient in gradients:
            if gradient is not None:
                grad_energy = grad_energy + gradient.pow(2).sum()

        grad_norm = float(torch.sqrt(grad_energy + self.epsilon).detach().cpu())

        previous = self._grad_norm_ema[name]
        if previous is None:
            updated = grad_norm
        else:
            updated = (
                self.ema_decay * previous
                + (1.0 - self.ema_decay) * grad_norm
            )
        self._grad_norm_ema[name] = updated
        self._recompute_weights()

    def _recompute_weights(self) -> None:
        base = self._base_weights.resolve()
        proposal: dict[str, float] = {}

        for name, base_weight in base.items():
            ema_norm = self._grad_norm_ema[name]
            if ema_norm is None or base_weight == 0.0:
                proposal[name] = base_weight
            else:
                proposal[name] = base_weight / (
                    ema_norm**self.power + self.epsilon
                )

        total_base = sum(base.values())
        total_proposal = sum(proposal.values())
        if total_proposal > 0.0:
            scale = total_base / total_proposal
            proposal = {
                name: value * scale for name, value in proposal.items()
            }

        clipped: dict[str, float] = {}
        for name, value in proposal.items():
            lower_bounded = max(self.min_weight, value)
            if self.max_weight is None:
                clipped[name] = lower_bounded
            else:
                clipped[name] = min(self.max_weight, lower_bounded)

        self._current_weights = clipped


def ensure_loss_weight_strategy(
    weights: Mapping[str, float] | LossWeightStrategy,
) -> LossWeightStrategy:
    """
    Normalize either a plain mapping or an existing strategy object.

    This helper is useful when user-facing code wants to accept either:
    - a simple dictionary of scalar weights, or
    - a future adaptive weighting strategy
    """
    if isinstance(weights, LossWeightStrategy):
        return weights
    return StaticLossWeights(weights)
