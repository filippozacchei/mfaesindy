"""Parameterized latent dynamics and differentiable Neural ODE integration."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def _activation_from_name(name: str) -> nn.Module:
    lookup: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return lookup[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation '{name}'.") from error


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(_activation_from_name(activation))
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class ParameterizedVectorField(nn.Module):
    """MLP vector field for latent dynamics conditioned on system parameters."""

    def __init__(
        self,
        latent_dim: int,
        parameter_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
        activation: str = "silu",
        conditioning: str = "concat",
    ) -> None:
        super().__init__()
        if conditioning != "concat":
            raise ValueError("Only concat conditioning is implemented in v1.")
        self.latent_dim = latent_dim
        self.parameter_dim = parameter_dim
        self.conditioning = conditioning
        self.network = _build_mlp(
            input_dim=latent_dim + parameter_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
        )

    def forward(self, t: Tensor, z: Tensor, parameters: Tensor) -> Tensor:
        """Evaluate the latent vector field.

        The time `t` is accepted for Neural ODE API compatibility, but the default
        v1 vector field is autonomous apart from parameter conditioning.
        """

        _ = t
        features = torch.cat((z, parameters), dim=-1)
        return self.network(features)


class LatentDynamics(nn.Module):
    """Thin wrapper exposing the parameterized vector field as a dynamics module."""

    def __init__(self, vector_field: ParameterizedVectorField) -> None:
        super().__init__()
        self.vector_field = vector_field

    def forward(self, t: Tensor, z: Tensor, parameters: Tensor) -> Tensor:
        return self.vector_field(t, z, parameters)


class NeuralODESolver:
    """Differentiable Neural ODE integrator.

    The built-in solver supports fixed-step Euler and RK4. If `torchdiffeq` is
    installed, the optional `torchdiffeq` backend can also be selected.
    """

    def __init__(self, backend: str = "rk4", steps_per_interval: int = 1) -> None:
        if backend not in {"euler", "rk4", "torchdiffeq"}:
            raise ValueError(f"Unsupported solver backend '{backend}'.")
        if steps_per_interval < 1:
            raise ValueError("steps_per_interval must be at least 1.")
        self.backend = backend
        self.steps_per_interval = steps_per_interval

    def integrate(
        self,
        dynamics: LatentDynamics,
        z0: Tensor,
        times: Tensor,
        parameters: Tensor,
    ) -> Tensor:
        """Integrate the latent state over a batch of time grids."""

        if times.ndim == 1:
            return self._integrate_shared_grid(dynamics, z0, times, parameters)
        if times.ndim != 2:
            raise ValueError("times must be either [T] or [B, T].")
        if torch.allclose(times, times[:1].expand_as(times)):
            return self._integrate_shared_grid(dynamics, z0, times[0], parameters)
        trajectories = [
            self._integrate_shared_grid(
                dynamics=dynamics,
                z0=z0[index : index + 1],
                times=times[index],
                parameters=parameters[index : index + 1],
            )[0]
            for index in range(times.shape[0])
        ]
        return torch.stack(trajectories, dim=0)

    def _integrate_shared_grid(
        self,
        dynamics: LatentDynamics,
        z0: Tensor,
        times: Tensor,
        parameters: Tensor,
    ) -> Tensor:
        if times.ndim != 1:
            raise ValueError("The shared-grid integrator expects a 1D time tensor.")
        if self.backend == "torchdiffeq":
            try:
                from torchdiffeq import odeint
            except ImportError as error:
                raise RuntimeError(
                    "torchdiffeq is not installed. Use backend='rk4' or add the optional dependency."
                ) from error

            class _WrappedDynamics(nn.Module):
                def __init__(self, base: LatentDynamics, params: Tensor) -> None:
                    super().__init__()
                    self.base = base
                    self.params = params

                def forward(self, t: Tensor, z: Tensor) -> Tensor:
                    return self.base(t, z, self.params)

            trajectory = odeint(_WrappedDynamics(dynamics, parameters), z0, times, method="rk4")
            return trajectory.transpose(0, 1)

        states = [z0]
        current = z0
        current_time = times[0]
        for next_time in times[1:]:
            if next_time <= current_time:
                raise ValueError("times must be strictly increasing.")
            dt = (next_time - current_time) / self.steps_per_interval
            sub_time = current_time
            for _ in range(self.steps_per_interval):
                current = self._step(dynamics, current, sub_time, dt, parameters)
                sub_time = sub_time + dt
            states.append(current)
            current_time = next_time
        return torch.stack(states, dim=1)

    def _step(
        self,
        dynamics: LatentDynamics,
        z: Tensor,
        time: Tensor,
        dt: Tensor,
        parameters: Tensor,
    ) -> Tensor:
        if self.backend == "euler":
            return z + dt * dynamics(time, z, parameters)
        k1 = dynamics(time, z, parameters)
        k2 = dynamics(time + 0.5 * dt, z + 0.5 * dt * k1, parameters)
        k3 = dynamics(time + 0.5 * dt, z + 0.5 * dt * k2, parameters)
        k4 = dynamics(time + dt, z + dt * k3, parameters)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class RolloutEngine:
    """Utility class for latent-space rollouts."""

    def __init__(self, dynamics: LatentDynamics, solver: NeuralODESolver) -> None:
        self.dynamics = dynamics
        self.solver = solver

    def rollout(self, z0: Tensor, times: Tensor, parameters: Tensor) -> Tensor:
        """Propagate latent states over the provided time grid."""

        return self.solver.integrate(self.dynamics, z0, times, parameters)

