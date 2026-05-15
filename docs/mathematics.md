# Mathematical Formulation

Consider a parametric dynamical system with state field
$$
\mathbf{u}(\mathbf{x}, t; \boldsymbol{\mu}) : \Omega \times [0, T) \to \mathbb{R}^d,
$$
where $\mathbf{x} \in \Omega \subset \mathbb{R}^n$ denotes the spatial coordinate, $t$ is time, and $\boldsymbol{\mu} \in \mathbb{R}^p$ is a vector of system parameters.

We assume that $\mathbf{u}$ is governed by a known or unknown partial differential equation of the form
$$
\partial_t \mathbf{u} = \mathcal{N}(\mathbf{u}; \boldsymbol{\mu}).
$$

High-fidelity data are assumed to be obtained from expensive simulations or experiments, while additional lower-fidelity information is available for the same class of parametric dynamical systems.

## Data

We observe time-dependent trajectories from two fidelities: low fidelity (LF) and high fidelity (HF).
For each operating condition $\boldsymbol{\mu} \in \mathbb{R}^{d_\mu}$ and fidelity $f \in \{\mathrm{LF}, \mathrm{HF}\},$ we denote the continuous-time state trajectory by
$$
\mathbf{u}^{(f)}(t; \boldsymbol{\mu}) \in \mathbb{R}^{n_f}, \qquad t \in [0, T].
$$
The state dimension $n_f$ may depend on the fidelity.

In practice, the observed data consist of sampled trajectories of the form
$$
\mathcal{T}^{(f)}(\boldsymbol{\mu})
=
\left(
\mathbf{t}^{(f)},
\mathbf{U}^{(f)},
\boldsymbol{\mu},
f
\right),
$$
where
$$
\mathbf{t}^{(f)} = (t_0, \dots, t_{m_f-1}) \in \mathbb{R}^{m_f},
$$
and
$$
\mathbf{U}^{(f)}
=
\left[
\mathbf{u}^{(f)}(t_0; \boldsymbol{\mu}),
\dots,
\mathbf{u}^{(f)}(t_{m_f-1}; \boldsymbol{\mu})
\right]^\top
\in \mathbb{R}^{m_f \times n_f}.
$$

Low fidelity is generally an approximation of the high-fidelity dynamics, differing in resolution, observation quality, or missing physics.

The goal is to construct a reduced-order surrogate for the high-fidelity dynamics by leveraging the greater availability of low-fidelity trajectories. Unlike idealized multifidelity settings, low- and high-fidelity data are not assumed to differ only in spatial or temporal resolution, nor merely through additive noise. In practical settings such as CFD and experiments, the discrepancy between fidelities may also arise from systematic observation bias, imperfect parameter matching, and mild model-form mismatch in the underlying time evolution.

The reduced-order model is composed of two main components:
- a multifidelity compression model;
- a multifidelity dynamical forecasting model.


## Compression
Because both HF and LF trajectories are high-dimensional, the first layer of the method is an autoencoder-style reduced-order model. Each fidelity is encoded into a latent representation, possibly with a shared core and fidelity-specific private components:
$$
E_f\!\left(\mathbf{u}^{(f)}_{0:k}, \mu, \boldsymbol{\mu}\right)
\mapsto
\left(z_s, z_f^{p}\right),
$$
  
where:

- $z_s \in \mathbb{R}^{r_s}$ is the shared latent core;
- $z_f^p \in \mathbb{R}^{r_p}$ is an optional fidelity-specific private latent block.

The decoder reconstructs observations as
$$
\hat \mathbf{u}^{(f)} = D_f(z_s, z_f^p; \boldsymbol{\mu}).
$$  
This decomposition is useful because the latent space does not need to be fully shared. Forcing the entire latent state to coincide is often too strong. The shared coordinates should represent the common physical evolution, while private coordinates may absorb fidelity-specific sensing effects, missing physics, or experimental artifacts.

## Forecasting

Once a latent representation has been learned, the surrogate advances the latent state in time. Several dynamics models are possible.

### MF SINDy

The shared latent core satisfies

$$
\dot z_s \approx \Theta(z_s; \boldsymbol{\mu})\,\Xi_{\mathrm{shared}},
$$
where $\Theta$ is a SINDy library and $\Xi_{\mathrm{shared}}$ is a sparse coefficient matrix.

In the simplest formulation, only the shared latent core $z_s$ is propagated in time. The private latent block $z_f^p$ is treated as a fidelity-specific representation variable inferred from the encoded initial condition and used by the decoder to reconstruct fidelity-dependent effects.

This formulation is attractive because, although $\Theta$ is nonlinear in the latent state, the regression is linear in the unknown coefficients. Therefore, the identification of $\Xi_{\mathrm{shared}}$ can be posed as a multifidelity sparse linear regression problem. The corresponding regression viewpoint, together with its connection to the approximate control-variate framework of Qian et al., is summarized in [multifidelity_linear_regression.md](./multifidelity_linear_regression.md).

This option is especially appealing when:
- interpretability matters;
- HF data are scarce;
- the dominant shared dynamics are expected to be relatively low-order.

### Neural ODE Refinement

## Training
