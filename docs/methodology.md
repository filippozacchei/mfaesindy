# Methodology

We consider the problem of identifying sparse latent dynamics from multi-fidelity trajectory data, where low-fidelity (LF) simulations are abundant but biased and high-fidelity (HF) simulations are accurate but expensive. The proposed framework combines a multi-fidelity autoencoder, used to construct a shared latent representation across fidelities, with a multi-fidelity SINDy regression, used to identify sparse governing equations in that latent space.

We assume that, on the paired subset, LF and HF trajectories are generated from the same input parameter and initial condition. While the two fidelities may be defined on different time discretizations, we require that one fidelity can be evaluated or interpolated at the time instances of the other, enabling the construction of paired samples for latent alignment and control-variate estimation

## Shared Latent Representation

Let \(x_{LF}(t), x_{HF}(t) \in \mathbb{R}^{n_x}\) denote LF and HF observations of the same dynamical system. Two encoders,
\[
E_{LF}: \mathbb{R}^{n_x} \to \mathbb{R}^{n_z},
\qquad
E_{HF}: \mathbb{R}^{n_x} \to \mathbb{R}^{n_z},
\]
map the observations to latent coordinates
\[
z_{LF} = E_{LF}(x_{LF}),
\qquad
z_{HF} = E_{HF}(x_{HF}),
\]
while two decoders,
\[
D_{LF}: \mathbb{R}^{n_z} \to \mathbb{R}^{n_x},
\qquad
D_{HF}: \mathbb{R}^{n_z} \to \mathbb{R}^{n_x},
\]
reconstruct the corresponding fidelity level.

The autoencoder is trained by combining reconstruction and latent-alignment losses, \[ \mathcal{L}_{rec}= \|x_{LF} - D_{LF}(z_{LF})\|^2 + \|x_{HF} - D_{HF}(z_{HF})\|^2, \] \[\mathcal{L}_{align}= \|z_{LF} - z_{HF}\|^2,\]
which yield the representation-learning objective
\[ \mathcal{L}_{AE}
= \mathcal{L}_{rec} + \lambda_{align}\mathcal{L}_{align}.\]

When additional coupling is needed, one may also penalize discrepancies between paired latent derivatives,
\[ \mathcal{L}_{align}^{dyn}
= \|\dot z_{LF} - \dot z_{HF}\|^2, \]
and augment the representation loss accordingly. In practice, latent derivatives may be estimated by finite differences or, when the model structure permits, by automatic differentiation.

To balance expressiveness and consistency across fidelities, the two encoders may share their final layers and the two decoders may share their initial layers. The non-shared layers capture fidelity-specific distortions, whereas the shared layers enforce a common latent coordinate system suitable for dynamical identification. This construction is conceptually related to shared-latent-space models for cross-domain representation learning [1].

## Latent Dynamics Identification

Once the trajectories are embedded in the latent space, we identify their dynamics using SINDy [2], following the broader idea of learning coordinates and dynamics jointly in a latent representation [3]. Let \(z(t) \in \mathbb{R}^{n_z}\) denote a latent trajectory and let
\[
\Theta(z) \in \mathbb{R}^{N \times p}
\]
be the library matrix formed from \(N\) latent samples and \(p\) candidate features. The latent dynamics are approximated by
\[
\dot z \approx \Theta(z)\Xi,
\]
where \(\Xi \in \mathbb{R}^{p \times n_z}\) is a sparse coefficient matrix.

Our objective is to estimate the HF latent dynamical operator. In moment form, the ideal coefficients satisfy the normal equations
\[
C_{XX}^{HF}\Xi = C_{XY}^{HF},
\]
with
\[
C_{XX}^{HF} = \mathbb{E}\!\left[\Theta_{HF}^\top \Theta_{HF}\right],
\qquad
C_{XY}^{HF} = \mathbb{E}\!\left[\Theta_{HF}^\top \dot z_{HF}\right].
\]
Because HF trajectories are limited, direct empirical estimation of these moments is statistically inefficient. We therefore estimate them using a multi-fidelity control-variate construction inspired by multi-fidelity Monte Carlo (MFMC) [4, 5].

## MFMC Estimation of Regression Moments

Assume that \(m_{LF}\) LF trajectories and \(m_{HF}\) paired HF trajectories are available, with \(m_{LF} \ge m_{HF}\). Let \(\Theta_{LF}^{(HF)}\) and \(\dot z_{LF}^{(HF)}\) denote LF quantities evaluated on the paired HF inputs. We first define the sample estimators
\[ \widehat C_{XY}^{LF} = \frac{1}{m_{LF}} \Theta_{LF}^\top \dot z_{LF}, \qquad \widehat C_{XY}^{HF} = \frac{1}{m_{HF}} \Theta_{HF}^\top \dot z_{HF}, \] \[ \widehat C_{XY}^{LF,paired} = \frac{1}{m_{HF}} \Theta_{LF}^{(HF)\top}\dot z_{LF}^{(HF)}. \] The MFMC estimator of the cross-moment is then \[
\widehat C_{XY}^{MF}= \widehat C_{XY}^{LF}+ A_{XY}\!\left(\widehat C_{XY}^{HF}- \widehat C_{XY}^{LF,paired}\right),\] where \(A_{XY}\) is a control-variate coefficient estimated from the paired dataset. Depending on the desired level of complexity, \(A_{XY}\) may be taken as a scalar, a diagonal matrix, or a componentwise coefficient.

To preserve consistency with the HF regression target, the regressor moment is estimated using the same control-variate structure: \[\widehat C_{XX}^{LF}=\frac{1}{m_{LF}} \Theta_{LF}^\top \Theta_{LF},\qquad\widehat C_{XX}^{HF}=\frac{1}{m_{HF}} \Theta_{HF}^\top \Theta_{HF},\] \[\widehat C_{XX}^{LF,paired}=\frac{1}{m_{HF}} \Theta_{LF}^{(HF)\top}\Theta_{LF}^{(HF)},\] \[\widehat C_{XX}^{MF}=\widehat C_{XX}^{LF}+A_{XX}\!\left(\widehat C_{XX}^{HF}-\widehat C_{XX}^{LF,paired}\right).\] The coefficient matrix is then updated from \[\Xi =\left(\widehat C_{XX}^{MF} + \beta I\right)^{-1}\widehat C_{XY}^{MF},\] where \(\beta > 0\) is a Tikhonov regularization parameter used to improve numerical stability. Sparsity is enforced with sequential thresholded least squares (STLSQ), as in the original SINDy formulation [2].

This estimator makes the role of each fidelity explicit: LF data provide a low-variance baseline estimate, while the paired HF data supply a bias-correcting control-variate term.

## Control-Variate Coefficients and Sample Allocation

For a scalar random quantity \(Q\), the classical two-level MFMC coefficient is \[
A^\star=\frac{\mathrm{Cov}(Q_{HF}, Q_{LF})}{\mathrm{Var}(Q_{LF})}.\]
In the present setting, \(Q\) represents the moment contribution being estimated from paired samples. For example, one may define
\[
Q_{XY} = \Theta(z)^\top \dot z,
\qquad
Q_{XX} = \Theta(z)^\top \Theta(z),
\]
and estimate separate coefficients \(A_{XY}\) and \(A_{XX}\). When full matrix-valued coefficients are unnecessary, scalar or diagonal approximations are sufficient and considerably simpler to estimate.

The number of LF and HF samples is selected according to standard MFMC principles [4, 5]. If \(c_{LF}\) and \(c_{HF}\) denote the cost of a single LF and HF sample, respectively, then an efficient two-level allocation satisfies
\[
\frac{m_{LF}}{m_{HF}}
\propto
\sqrt{
\frac{c_{HF}}{c_{LF}}
\cdot
\frac{\mathrm{Var}(Q_{LF})}{\mathrm{Var}(Q_{HF} - Q_{LF})}
}.
\]
In practice, the variances and correlations entering this expression are estimated from an initial pilot dataset of paired trajectories.

## Training Procedure

The method is implemented as an alternating optimization procedure.

First, the multi-fidelity autoencoder is trained on a small paired dataset in order to produce an initial shared latent representation. Second, latent trajectories and latent derivatives are computed on the pilot set, and the empirical statistics required for MFMC sample allocation and for the control-variate coefficients are estimated. Third, additional LF and paired HF trajectories are generated according to the selected allocation rule.

Finally, the full model is trained by alternating between two steps:

1. update the encoder-decoder parameters using the current representation and dynamics losses;
2. recompute \(\widehat C_{XX}^{MF}\) and \(\widehat C_{XY}^{MF}\), then update \(\Xi\) by regularized least squares followed by thresholding.

This alternating strategy couples representation learning with sparse dynamical identification while preserving a clear separation between the neural-network updates and the sparse regression step.

## Dynamics-Aware Losses

To incorporate dynamical information directly into training, we augment the representation objective with a SINDy residual loss evaluated on HF latent trajectories,\[\mathcal{L}_{dyn}
=\|\dot z_{HF} - \Theta(z_{HF})\Xi\|^2.
\]Because the target of the regression is the HF latent dynamics, this term is defined on HF samples; if desired, an analogous term may also be evaluated on the paired LF subset.

To further encourage LF and HF trajectories to induce the same latent vector field, we optionally introduce the cross-fidelity consistency term

\[ \mathcal{L}_{mf\text{-}dyn}=\|\Theta(z_{HF})\Xi - \Theta(z_{LF})\Xi\|^2,\]

evaluated only on paired samples at matched times. The resulting training objective becomes

\[ \mathcal{L}=\mathcal{L}_{rec}+\lambda_{align}\mathcal{L}_{align}+\lambda_{dyn}\mathcal{L}_{dyn}+\lambda_{mf}\mathcal{L}_{mf\text{-}dyn}.\]

If the derivative estimates are noisy, we also include the latent smoothness regularizer

\[\mathcal{L}_{smooth}=\sum_t \|z_{t+1} - 2z_t + z_{t-1}\|^2,\]

which penalizes rapid temporal oscillations and stabilizes numerical differentiation. In that case, the overall objective is

\[\mathcal{L}_{total}=\mathcal{L} + \lambda_{smooth}\mathcal{L}_{smooth}.\]

## Optional Extension: High-Fidelity Correction Terms

The main formulation assumes that LF and HF trajectories are governed by the same sparse latent dynamics and differ primarily through state-space distortion and moment-estimation bias. When this assumption is too restrictive, the model may be extended by introducing an HF-specific correction:
\[
\dot z_{LF} = \Theta(z)\Xi,
\qquad
\dot z_{HF} = \Theta(z)\Xi + \Theta(z)\Xi_{\Delta}.
\]
Here \(\Xi\) represents the shared dynamical core, whereas \(\Xi_{\Delta}\) captures additional HF mechanisms that are not resolved at LF. This variant preserves interpretability while allowing the HF model to enrich the shared latent dynamics through a sparse corrective term.

## Optional Extension: Time Continuous Encoder

An important extension of the proposed framework consists in adopting a continuous-time latent representation. Instead of relying on discrete-time observations and finite-difference approximations of derivatives, one may define a time-aware encoder \(z(t) = E(x(t), t)\), allowing latent trajectories to be differentiated using automatic differentiation. This removes the need for aligned time grids across fidelities and provides more accurate and stable estimates of latent derivatives.

In this setting, low- and high-fidelity trajectories can be evaluated at arbitrary time instances, enabling more flexible construction of paired samples for both latent alignment and multi-fidelity estimation. Moreover, the resulting formulation is naturally compatible with continuous-time dynamical systems and neural ordinary differential equation frameworks. While this extension increases modeling flexibility and numerical robustness, it also introduces additional complexity, and is therefore left as future work

## References

[1] Liu, Ming-Yu, Thomas M. Breuel, and Jan Kautz. “Unsupervised Image-to-Image Translation Networks.” In *Advances in Neural Information Processing Systems 30*, 700-708, 2017.

[2] Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz. “Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems.” *Proceedings of the National Academy of Sciences* 113, no. 15 (2016): 3932-3937. https://doi.org/10.1073/pnas.1517384113.

[3] Champion, Kathleen, Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton. “Data-Driven Discovery of Coordinates and Governing Equations.” *Proceedings of the National Academy of Sciences* 116, no. 45 (2019): 22445-22451. https://doi.org/10.1073/pnas.1906995116.

[4] Peherstorfer, Benjamin, Karen Willcox, and Max Gunzburger. “Optimal Model Management for Multifidelity Monte Carlo Estimation.” *SIAM Journal on Scientific Computing* 38, no. 5 (2016): A3163-A3194. https://doi.org/10.1137/15M1046472.

[5] Peherstorfer, Benjamin, Max Gunzburger, and Karen Willcox. “Convergence Analysis of Multifidelity Monte Carlo Estimation.” *Numerische Mathematik* 139, no. 3 (2018): 683-707. https://doi.org/10.1007/s00211-018-0945-7.


- spatio-temporal pde 
- experimental scenario dario lopez pintor spray

- get an example where I know SINDy-AE where they work and make it degrade one of the two Fidelities. 
