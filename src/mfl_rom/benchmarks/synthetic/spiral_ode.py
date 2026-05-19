from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from mfl_rom.data import MFTrajectory, Trajectory

HF_GRID_SHAPE = (64, 64)
LF_GRID_SHAPE = (16, 16)
LF_BIAS = 0.01
LF_NOISE_STD = 0.01

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


def sample_ic(
    rng: np.random.Generator,
    radius_range: tuple[float, float] = (0.05, 1.0),
) -> np.ndarray:
    """Sample one latent initial condition in polar coordinates."""
    radius_min, radius_max = radius_range
    if radius_min <= 0.0 or radius_min >= radius_max:
        raise ValueError("radius_range must satisfy 0 < min < max.")

    radius = rng.uniform(radius_min, radius_max)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return radius * np.array([np.cos(angle), np.sin(angle)])


def generate_latent_trajectory(
    ic: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """
    Integrate the latent benchmark dynamics over a prescribed time grid.
    """
    ic = np.asarray(ic, dtype=float)
    time = np.asarray(time, dtype=float)

    if ic.shape != (2,):
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
        y0=ic,
        t_eval=time,
    )
    if not sol.success:
        raise RuntimeError(
            f"latent trajectory integration failed: {sol.message}"
        )

    return sol.y.T


def _make_grid(grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, grid_shape[0])
    y = np.linspace(-1.0, 1.0, grid_shape[1])
    return np.meshgrid(x, y, indexing="ij")


def _lift_latent(
    latent_state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    z1, z2 = latent_state
    return (
        z1 * np.sin(np.pi * x) * np.sin(np.pi * y)
        + z2 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    )


def _generate_observed_states(
    latent_states: np.ndarray,
    fidelity: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    
    if fidelity not in ["HF","LF"]:
        raise ValueError("fidelity must be either 'HF' or 'LF'.")

    grid_shape = HF_GRID_SHAPE if fidelity =="HF" else LF_GRID_SHAPE
    
    x, y = _make_grid(grid_shape)
    states = np.empty(
        (latent_states.shape[0], *grid_shape),
        dtype=float,
    )
    rng = np.random.default_rng() if rng is None else rng

    for index, latent_state in enumerate(latent_states):
        snapshot = _lift_latent(latent_state, x, y)
        if fidelity == "LF":
            snapshot = (
                snapshot
                + LF_BIAS * np.cos(np.pi * x) * np.cos(np.pi * y)
                + LF_NOISE_STD * rng.standard_normal(
                    size=snapshot.shape
                )
            )
        states[index, ...] = snapshot

    return states


def generate_trajectory(
    ic: np.ndarray,
    time: np.ndarray,
    fidelity: str,
    rng: np.random.Generator | None = None,
) -> Trajectory:
    """
    Generate one observed trajectory at a prescribed fidelity.

    Since this benchmark uses fixed latent dynamics, the conditioning vector
    stored in ``Trajectory.parameters`` is the latent initial condition.
    """
    ic = np.asarray(ic, dtype=float)
    time = np.asarray(time, dtype=float)
    latent_states = generate_latent_trajectory(
        ic=ic,
        time=time,
    )
    states = _generate_observed_states(
        latent_states,
        fidelity=fidelity,
        rng=rng,
    )
    return Trajectory(
        time=time,
        states=states,
        parameters=ic,
        fidelity=fidelity
    )


def generate_mf_trajectory(
    ic: np.ndarray,
    lf_time: np.ndarray,
    hf_time: np.ndarray,
    rng: np.random.Generator | None = None,
) -> MFTrajectory:
    """Generate one paired LF/HF benchmark trajectory sample."""
    lf_trajectory = generate_trajectory(
        ic=ic,
        time=lf_time,
        fidelity="LF",
        rng=rng,
    )
    hf_trajectory = generate_trajectory(
        ic=ic,
        time=hf_time,
        fidelity="HF",
    )
    return MFTrajectory(
        lf_trajectory=lf_trajectory,
        hf_trajectory=hf_trajectory,
    )
