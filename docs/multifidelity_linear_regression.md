# Multifidelity Linear Regression

This note formalizes the multifidelity linear regression viewpoint that underlies the MF-SINDy component in [mathematics.md](./mathematics.md). The intended hierarchy is:

1. define the high-fidelity linear regression problem;
2. replace its empirical regression statistics with multifidelity estimators;
3. embed those estimators inside a STRidge loop to obtain a sparse latent dynamics model.

The key point is that multifidelity regression is the base estimation mechanism, while MF-SINDy is the sparse model-discovery procedure built on top of it.

## High-Fidelity Ridge Regression

For one component of the shared latent dynamics, let

$$
y \in \mathbb{R}^{N},
\qquad
X \in \mathbb{R}^{N \times q},
$$

where \(y\) contains sampled latent time derivatives and \(X\) is the evaluated SINDy feature library. The high-fidelity ridge problem is

$$
\hat \xi_{\mathrm{HF}}(\lambda_2)
=
\arg\min_{\xi \in \mathbb{R}^{q}}
\frac{1}{2N}\|y - X\xi\|_2^2
+
\frac{\lambda_2}{2}\|\xi\|_2^2.
$$

Equivalently, if

$$
A_{\mathrm{HF}} = \frac{1}{N}X^\top X,
\qquad
b_{\mathrm{HF}} = \frac{1}{N}X^\top y,
$$

then

$$
\left(A_{\mathrm{HF}} + \lambda_2 I\right)\hat \xi_{\mathrm{HF}} = b_{\mathrm{HF}}.
$$

For latent dynamics with \(r_s\) shared coordinates, this regression is applied columnwise, one target \(y_j\) for each component of \(\dot z_s\).

## Multifidelity Regression Statistics

Assume now that we have:

- a paired high-/low-fidelity set of size \(N_{\mathrm{HF}}\);
- an additional low-fidelity-only set, so that the total number of low-fidelity samples is \(N_{\mathrm{LF}} \geq N_{\mathrm{HF}}\).

For a fixed latent coordinate, let

$$
\left(X_{\mathrm{HF}}, y_{\mathrm{HF}}\right),
\qquad
\left(X_{\mathrm{LF}}^{\mathrm{pair}}, y_{\mathrm{LF}}^{\mathrm{pair}}\right),
\qquad
\left(X_{\mathrm{LF}}^{\mathrm{all}}, y_{\mathrm{LF}}^{\mathrm{all}}\right)
$$

denote the corresponding regression data.

The high-fidelity target statistics are

$$
A_{\mathrm{HF}} = \frac{1}{N_{\mathrm{HF}}}X_{\mathrm{HF}}^\top X_{\mathrm{HF}},
\qquad
b_{\mathrm{HF}} = \frac{1}{N_{\mathrm{HF}}}X_{\mathrm{HF}}^\top y_{\mathrm{HF}}.
$$

The low-fidelity analogues are

$$
A_{\mathrm{LF}}^{\mathrm{pair}}
=
\frac{1}{N_{\mathrm{HF}}}
\left(X_{\mathrm{LF}}^{\mathrm{pair}}\right)^\top X_{\mathrm{LF}}^{\mathrm{pair}},
\qquad
A_{\mathrm{LF}}^{\mathrm{all}}
=
\frac{1}{N_{\mathrm{LF}}}
\left(X_{\mathrm{LF}}^{\mathrm{all}}\right)^\top X_{\mathrm{LF}}^{\mathrm{all}},
$$

and

$$
b_{\mathrm{LF}}^{\mathrm{pair}}
=
\frac{1}{N_{\mathrm{HF}}}
\left(X_{\mathrm{LF}}^{\mathrm{pair}}\right)^\top y_{\mathrm{LF}}^{\mathrm{pair}},
\qquad
b_{\mathrm{LF}}^{\mathrm{all}}
=
\frac{1}{N_{\mathrm{LF}}}
\left(X_{\mathrm{LF}}^{\mathrm{all}}\right)^\top y_{\mathrm{LF}}^{\mathrm{all}}.
$$

Following the approximate control-variate viewpoint of Qian et al., the low-fidelity data are not used to replace the high-fidelity regression problem. Instead, they are used to build lower-variance estimators of the high-fidelity statistics. A generic multifidelity form is

$$
\hat A_{\mathrm{MF}}
=
A_{\mathrm{HF}}
+
B_A
\left(
A_{\mathrm{LF}}^{\mathrm{all}} - A_{\mathrm{LF}}^{\mathrm{pair}}
\right),
$$

$$
\hat b_{\mathrm{MF}}
=
b_{\mathrm{HF}}
+
B_b
\left(
b_{\mathrm{LF}}^{\mathrm{all}} - b_{\mathrm{LF}}^{\mathrm{pair}}
\right),
$$

where \(B_A\) and \(B_b\) denote control-variate weights.

To make the definition precise for the matrix-valued statistic \(A\), let

$$
\Delta a_{\mathrm{LF}}
=
\mathrm{vec}\!\left(
A_{\mathrm{LF}}^{\mathrm{all}} - A_{\mathrm{LF}}^{\mathrm{pair}}
\right),
\qquad
a_{\mathrm{HF}} = \mathrm{vec}(A_{\mathrm{HF}}),
$$

and

$$
\Delta b_{\mathrm{LF}}
=
b_{\mathrm{LF}}^{\mathrm{all}} - b_{\mathrm{LF}}^{\mathrm{pair}}.
$$

Then the optimal linear control-variate weights are

$$
B_A^\star
=
-\,\mathrm{Cov}(a_{\mathrm{HF}}, \Delta a_{\mathrm{LF}})
\left[\mathrm{Cov}(\Delta a_{\mathrm{LF}}, \Delta a_{\mathrm{LF}})\right]^{-1},
$$

$$
B_b^\star
=
-\,\mathrm{Cov}(b_{\mathrm{HF}}, \Delta b_{\mathrm{LF}})
\left[\mathrm{Cov}(\Delta b_{\mathrm{LF}}, \Delta b_{\mathrm{LF}})\right]^{-1}.
$$

Equivalently, if one writes the multifidelity estimator in the more classical control-variate form

$$
\hat A_{\mathrm{MF}}
=
A_{\mathrm{HF}}
- 
\widetilde B_A
\left(
A_{\mathrm{LF}}^{\mathrm{pair}} - A_{\mathrm{LF}}^{\mathrm{all}}
\right),
$$

$$
\hat b_{\mathrm{MF}}
=
b_{\mathrm{HF}}
- 
\widetilde B_b
\left(
b_{\mathrm{LF}}^{\mathrm{pair}} - b_{\mathrm{LF}}^{\mathrm{all}}
\right),
$$

then \(\widetilde B_A = -B_A\) and \(\widetilde B_b = -B_b\), and the optimal formulas take the familiar positive covariance form.

In practice, these covariances are not known and must be estimated from pilot data or from the paired multifidelity sample set. Simpler restrictions, such as scalar, diagonal, or blockwise weights, may also be imposed for robustness.

The resulting multifidelity ridge problem is then

$$
\left(\hat A_{\mathrm{MF}} + \lambda_2 I\right)\hat \xi_{\mathrm{MF}} = \hat b_{\mathrm{MF}}.
$$

This is the basic multifidelity linear regression object that the library should support.

## MF-SINDy via STRidge

SINDy adds sparsity on top of the multifidelity ridge solve. For one shared latent coordinate, start from the library regression

$$
\dot z_{s,j} \approx \Theta(z_s; \boldsymbol{\mu})\,\xi_j,
$$

where \(\xi_j \in \mathbb{R}^q\) is sparse.

MF-SINDy then applies a sequential threshold ridge procedure:

1. Build and normalize the candidate feature library \(X = \Theta(z_s; \boldsymbol{\mu})\).
2. Estimate the regression statistics \(\hat A_{\mathrm{MF}}\) and \(\hat b_{\mathrm{MF}}\) from paired HF/LF samples and additional LF samples.
3. Solve the multifidelity ridge problem on the current active support.
4. Threshold small coefficients:

$$
S^{(k+1)} = \left\{ i : |\xi_i^{(k)}| \geq \lambda_1 \right\}.
$$

5. Restrict the regression to the active set and resolve:

$$
\left(
\hat A_{\mathrm{MF}}^{(k)}[S^{(k+1)}, S^{(k+1)}] + \lambda_2 I
\right)
\xi_{S^{(k+1)}}^{(k+1)}
=
\hat b_{\mathrm{MF}}^{(k)}[S^{(k+1)}].
$$

6. Repeat until the support stabilizes or a stopping rule is met.
7. Optionally perform a final debiasing refit on the selected support.

Thus, the multifidelity piece enters at every ridge subproblem inside STRidge. It is not a separate post-processing step.

## Interpretation for the Library Design

This decomposition has direct architectural consequences:

- `compression/` must expose shared latent trajectories and the derivative targets needed by regression-based dynamics models.
- `dynamics/` should separate:
  - library evaluation;
  - multifidelity estimation of regression statistics;
  - STRidge support selection and refitting.
- `data/` must support both paired HF/LF samples and additional LF-only samples.
- `training/` should be able to prepare regression-ready batches for both the multifidelity estimator and the STRidge loop.

## References

- Elizabeth Qian, Dayoung Kang, Vignesh Sella, and Anirban Chaudhuri, [*Multifidelity linear regression for scientific machine learning from scarce data*](https://arxiv.org/abs/2403.08627), arXiv:2403.08627, revised July 1, 2024.
- Samuel H. Rudy, Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz, [*Data-driven discovery of partial differential equations*](https://arxiv.org/abs/1609.06401), arXiv:1609.06401, 2016.
