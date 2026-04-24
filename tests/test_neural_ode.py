from __future__ import annotations

import torch

from mf_lnode.dynamics.neural_ode import LatentDynamics, NeuralODESolver, ParameterizedVectorField


class ZeroDynamics(torch.nn.Module):
    def forward(self, t, z, parameters):
        _ = (t, parameters)
        return torch.zeros_like(z)


def test_parameterized_vector_field_shape():
    field = ParameterizedVectorField(latent_dim=4, parameter_dim=2, hidden_dim=8)
    z = torch.randn(5, 4)
    parameters = torch.randn(5, 2)
    derivative = field(torch.tensor(0.0), z, parameters)
    assert derivative.shape == z.shape


def test_neural_ode_solver_preserves_constant_state():
    solver = NeuralODESolver(backend="rk4")
    z0 = torch.randn(2, 3)
    parameters = torch.randn(2, 2)
    times = torch.linspace(0.0, 1.0, 5)
    trajectory = solver.integrate(ZeroDynamics(), z0, times, parameters)
    assert trajectory.shape == (2, 5, 3)
    assert torch.allclose(trajectory, z0.unsqueeze(1).expand_as(trajectory))


def test_latent_dynamics_wraps_vector_field():
    field = ParameterizedVectorField(latent_dim=3, parameter_dim=2, hidden_dim=6)
    dynamics = LatentDynamics(field)
    output = dynamics(torch.tensor(0.0), torch.randn(2, 3), torch.randn(2, 2))
    assert output.shape == (2, 3)

