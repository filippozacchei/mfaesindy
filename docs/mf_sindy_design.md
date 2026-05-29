# MF-SINDy Design

This note defines the implementation design for multifidelity SINDy class.

## Scope

- fit MF-SINDy offline on top of a frozen autoencoder;
- use only the shared latent coordinates;
- use one common feature library for LF and HF;
- build the multifidelity-corrected normal equations once;
- run STRidge on those corrected statistics;
- start with fixed scalar multifidelity weights `beta_A = beta_B = 1`.

This keeps the multifidelity estimator exact at the regression-statistics level
while avoiding support-dependent re-estimation in the first implementation.

## Notation

- `r_s`: dimension of the shared latent space.
- `q`: number of candidate library features.
- `N_HF`: number of paired HF regression rows.
- `N_LF`: number of LF regression rows in the full LF pool.
- `Theta(z)`: feature library evaluated at latent state `z`.
- `Xi in R^{q x r_s}`: coefficient matrix of the shared latent dynamics.

Each regression row is built from a latent state and its time derivative:

`z_i in R^{r_s}`, `z_dot_i in R^{r_s}`.

Stacking rows gives

- `X in R^{N x q}` with `X[i, :] = Theta(z_i)`;
- `Y in R^{N x r_s}` with `Y[i, :] = z_dot_i`.

## Latent Regression Rows

For one trajectory sampled at times `t_0 < ... < t_{m-1}`, encode all states to
shared latent coordinates:

`z_k = E_shared(x_k)`.

Version 1 uses centered finite differences through `np.gradient` and discards
the boundary rows:

`z_dot_k = d z / d t (t_k)` for `k = 1, ..., m - 2`.  (MF-1)

The retained regression rows are therefore

`(z_k, z_dot_k)` for `k = 1, ..., m - 2`.  (MF-2)

For paired LF/HF samples, both trajectories are first resampled to a common
overlap time grid before encoding, so LF and HF regression rows are aligned in
physical time.

## High-Fidelity Ridge Regression

For any regression view `(X, Y)`, define the empirical normal equations

`A = (1 / N) X^T X`,  (MF-3)

`B = (1 / N) X^T Y`.  (MF-4)

The ridge solution on the full support is

`(A + lambda_2 I) Xi = B`.  (MF-5)

Here `A in R^{q x q}` and `B in R^{q x r_s}`.

## Multifidelity Statistics

Version 1 uses three regression views:

- paired HF rows: `(X_HF, Y_HF)`;
- paired LF rows: `(X_LF^pair, Y_LF^pair)`;
- full LF rows: `(X_LF^all, Y_LF^all)`.

Their normal equations are

- `(A_HF, B_HF)`;
- `(A_LF^pair, B_LF^pair)`;
- `(A_LF^all, B_LF^all)`.

The multifidelity-corrected statistics are

`A_MF = A_HF + beta_A (A_LF^all - A_LF^pair)`,  (MF-6)

`B_MF = B_HF + beta_B (B_LF^all - B_LF^pair)`.  (MF-7)

Version 1 fixes `beta_A = beta_B = 1`, so the default estimator is

`A_MF = A_HF + A_LF^all - A_LF^pair`,  (MF-8)

`B_MF = B_HF + B_LF^all - B_LF^pair`.  (MF-9)

The multifidelity ridge problem is then

`(A_MF + lambda_2 I) Xi = B_MF`.  (MF-10)

## STRidge on Corrected Statistics

Sparsity is enforced after the multifidelity correction, not before it.

For one target column `j`, let `S^(k)` be the active support at iteration `k`.
At each iteration:

1. solve the restricted ridge problem

   `(A_MF[S, S] + lambda_2 I) xi_j[S] = B_MF[S, j]`;  (MF-11)

2. threshold the coefficients

   `S^(k+1) = {i : |xi_j^(k)[i]| >= lambda_1}`;  (MF-12)

3. repeat until the support stabilizes or a maximum iteration count is reached.

An optional final debias step solves the unregularized restricted system on the
final support:

`A_MF[S, S] xi_j[S] = B_MF[S, j]`.  (MF-13)

The sparse loop is therefore support-restricted, but the multifidelity
statistics `A_MF` and `B_MF` stay fixed in version 1.

## Feature Library

PySINDy is used only for the standard library machinery:

- `fit` on latent states;
- `transform` to build `Theta(z)`;
- `get_feature_names` for reporting.

The multifidelity regression estimator remains custom.

For the synthetic spiral benchmark, the first library should be a polynomial
library of degree 3, because the latent ground-truth dynamics contain cubic
terms.

## Module Mapping

- `src/mfl_rom/dynamics/latent_data.py`
  - build regression views from encoded trajectories;
  - implement (MF-1) and (MF-2).
- `src/mfl_rom/dynamics/libraries.py`
  - fit and evaluate the PySINDy feature library.
- `src/mfl_rom/dynamics/statistics.py`
  - implement (MF-3) through (MF-10).
- `src/mfl_rom/dynamics/ridge.py`
  - dense ridge solves on full or restricted support.
- `src/mfl_rom/dynamics/stridge.py`
  - implement (MF-11) through (MF-13).
- `src/mfl_rom/dynamics/model.py`
  - expose a fitted `MFSINDyModel` with `fit`, `predict_rhs`, and `simulate`.

## Version 1 Limitations

- HF-only extra trajectories are not used yet.
- Control-variate weights are fixed, not estimated from pilot covariances.
- The fitter assumes shared-latent dynamics only.
- Joint end-to-end training with the autoencoder remains separate from the
  offline MF-SINDy fit.
