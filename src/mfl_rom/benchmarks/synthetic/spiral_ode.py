import numpy as np

from scipy.integrate import solve_ivp

def latent_rhs(
    t: float, 
    z: np.ndarray
) -> np.ndarray:
    alpha, omega, beta = 0.1, 1.0, 1.0

    z1, z2 = z
    r2 = z1**2 + z2**2

    return np.array([
        alpha * z1 - omega * z2 - beta * r2 * z1,
        omega * z1 + alpha * z2 - beta * r2 * z2,
    ])
    
def generate_latent_trajectory(
    initial_condition: np.ndarray,
    time: np.ndarray
) -> np.ndarray:
    t_span = time[0], time[1]
    sol = solve_ivp(fun=latent_rhs, 
                    t_span=t_span, 
                    y0=initial_condition, 
                    t_eval=time,
                    method='RK45')
    return sol.y.T

    
