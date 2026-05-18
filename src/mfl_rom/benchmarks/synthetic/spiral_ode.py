from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def latent_rhs(
    t: float,
    z: np.ndarray,
) -> np.ndarray:
    """Evaluate the fixed two-dimensional latent benchmark dynamics."""
    alpha, omega, beta = 0.1, 1.0, 1.0

    z1, z2 = z
    r2 = z1**2 + z2**2

    return np.array([
        alpha * z1 - omega * z2 - beta * r2 * z1,
        omega * z1 + alpha * z2 - beta * r2 * z2,
    ])


def generate_latent_trajectory(
    initial_condition: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """
    Integrate the latent benchmark dynamics over a prescribed time grid.
    """
    initial_condition = np.asarray(initial_condition, dtype=float)
    time = np.asarray(time, dtype=float)

    if initial_condition.shape != (2,):
        raise ValueError(
            "initial_condition must be a one-dimensional array with shape "
            "(2,)."
        )
    if time.ndim != 1 or time.size < 2:
        raise ValueError(
            "time must be a one-dimensional array with at least two entries."
        )
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be strictly increasing.")

    t_span = (float(time[0]), float(time[-1]))
    sol = solve_ivp(
        fun=latent_rhs,
        t_span=t_span,
        y0=initial_condition,
        t_eval=time,
    )
    if not sol.success:
        raise RuntimeError(
            f"latent trajectory integration failed: {sol.message}"
        )

    return sol.y.T
