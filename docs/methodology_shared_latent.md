# Methodology: Shared-Latent Multi-Fidelity Reduced-Order Modeling

This note summarizes the methodological design space discussed for the final application of interest: low-fidelity CFD aerosol trajectories and high-fidelity experimental trajectories, both high-dimensional, with systematic nonlinear mismatch between the two sources. It is intended as a candidate methodology section for the paper and as a source document for future repository documentation.

## Problem Setting

We observe time-dependent trajectories from two fidelities:

- low fidelity (LF): CFD simulations;
- high fidelity (HF): experiments.

For each operating condition, let

\[
x^{(f)}(t) \in \mathbb{R}^{n_f}, \qquad f \in \{\mathrm{LF}, \mathrm{HF}\},
\]

denote the observed state at fidelity \(f\). The state dimension may differ across fidelities. We also assume access to operating parameters and, when available, initial-condition descriptors,

\[
\mu \in \mathbb{R}^{d_\mu}, \qquad \eta \in \mathbb{R}^{d_\eta}.
\]

The target is to construct a reduced-order surrogate for the HF dynamics while exploiting the larger availability of LF trajectories. Unlike idealized multi-fidelity settings, CFD and experiments are not assumed to differ only by resolution or noise; they may also differ through systematic observation bias, imperfect parameter matching, and mild model-form mismatch in the time evolution itself.

This motivates a formulation in which:

- a dominant latent dynamical backbone is shared across fidelities;
- fidelity-specific discrepancy terms absorb what is not shared;
- HF remains the final predictive target.

## Latent Representation

Because both CFD and experimental trajectories are high-dimensional, the first layer of the method is an autoencoder-style reduced-order model. Each fidelity is encoded into a latent representation, possibly with a shared core and fidelity-specific private components:

\[
E_f\!\left(x^{(f)}_{0:k}, \mu, \eta\right)
\mapsto
\left(z_s, z_f^{p}\right),
\]

where:

- \(z_s \in \mathbb{R}^{r_s}\) is the shared latent core;
- \(z_f^p \in \mathbb{R}^{r_p}\) is an optional fidelity-specific private latent block.

The decoder reconstructs observations as

\[
\hat x^{(f)} = D_f(z_s, z_f^p, \mu, \eta).
\]

This decomposition is useful because the latent space does not need to be fully shared. In a CFD-to-experiment setting, forcing the entire latent state to coincide is often too strong. The shared coordinates should represent the common physical evolution, while private coordinates may absorb fidelity-specific sensing effects, missing physics, or experimental artifacts.

## Autoencoder Design Options

Several encoder-decoder organizations are possible.

### 1. Fully shared latent space

Both fidelities are mapped to a common latent state,

\[
E_{\mathrm{LF}}(x^{\mathrm{LF}}, \mu, \eta) \to z,
\qquad
E_{\mathrm{HF}}(x^{\mathrm{HF}}, \mu, \eta) \to z.
\]

This is the simplest and most restrictive option. It is appropriate only when CFD and experiments are believed to differ mainly through observation bias and not through materially different latent evolution.

### 2. Principally shared autoencoders

The encoders and decoders share most of their layers, but retain small fidelity-specific adapters or heads. This is often the most practical compromise when one wants to bias the model toward a common representation without imposing complete identity.

### 3. Shared core plus private latent components

The latent state is explicitly split into shared and private blocks:

\[
z^{(f)} = \left(z_s, z_f^p\right).
\]

This is the most natural formulation when CFD and experiments share dominant physics but exhibit real discrepancy. In that case:

- \(z_s\) carries the common dynamical content;
- \(z_f^p\) carries fidelity-specific residual structure.

### 4. Parameter- and initial-condition-informed latent space

The latent representation is conditioned on known operating settings and initial-condition descriptors:

\[
z = E_f\!\left(x^{(f)}_{0:k}, \mu, \eta\right).
\]

This is important when apparent CFD/experiment mismatch partly reflects unresolved conditioning mismatch rather than pure fidelity effects. In aerosol applications this may include flow rate, particle loading, spray conditions, inlet state, geometry descriptors, or experimentally inferred context variables.

## Latent Dynamics Models

Once a latent representation has been learned, the surrogate advances the latent state in time. Several dynamics models are possible.

### 1. Shared latent SINDy with MFMC estimation

The shared latent core satisfies

\[
\dot z_s \approx \Theta(z_s, \mu, \eta)\,\Xi_{\mathrm{shared}},
\]

where \(\Theta\) is a SINDy library and \(\Xi_{\mathrm{shared}}\) is a sparse coefficient matrix. This is attractive because, although \(\Theta\) is nonlinear in the latent state, the regression is linear in the unknown coefficients. Therefore, multifidelity Monte Carlo or control-variate estimators can be used to estimate the HF regression quantities from abundant LF data and scarce HF data.

This option is especially appealing when:

- interpretability matters;
- HF data are scarce;
- the dominant shared dynamics are expected to be relatively low-order.

### 2. Shared latent Neural ODE

The latent dynamics are modeled continuously in time as

\[
\dot z_s = f_\theta(z_s, \mu, \eta).
\]

This is more flexible than SINDy and may better capture strongly nonlinear transient behavior, but it is harder to train and more difficult to combine cleanly with multifidelity control-variate estimation.

### 3. Shared SINDy core with HF residual dynamics

The most promising hybrid for the present problem is

\[
\dot z_s^{\mathrm{LF}} \approx \Theta(z_s, \mu, \eta)\,\Xi_{\mathrm{shared}},
\]

\[
\dot z_s^{\mathrm{HF}} \approx \Theta(z_s, \mu, \eta)\,\Xi_{\mathrm{shared}} + \delta_{\mathrm{HF}}(z_s, \mu, \eta).
\]

The residual term \(\delta_{\mathrm{HF}}\) can be:

- a sparse SINDy-style correction \(\Theta(z_s,\mu,\eta)\Xi_\Delta\);
- a small neural residual model;
- or omitted if the shared dynamics already suffice.

This hybrid uses LF trajectories to learn the dominant latent backbone and spends scarce HF data only on the smaller correction.

### 4. Shared latent dynamics plus HF decoder correction

If the main mismatch is observational rather than dynamical, one may keep a common latent dynamics model and place the discrepancy in the decoder:

\[
\hat x^{\mathrm{HF}} = D_{\mathrm{HF}}(z_s, z_{\mathrm{HF}}^p, \mu, \eta) + r_{\mathrm{HF}}(z_s, \mu, \eta).
\]

This is useful when CFD and experiments share the same latent evolution but differ strongly in how that evolution is observed.

## Enforcing Cross-Fidelity Consistency

The shared latent structure must be enforced carefully.

### Case 1: Paired LF-HF trajectories are available

When LF and HF trajectories are paired under matched operating conditions, the cleanest mechanism is direct alignment on the shared latent core:

\[
\mathcal{L}_{\mathrm{align}} =
\|z_s^{\mathrm{LF}} - z_s^{\mathrm{HF}}\|^2.
\]

One may also align latent rollouts or latent derivatives on matched times:

\[
\mathcal{L}_{\mathrm{roll-align}} =
\sum_t \|z_s^{\mathrm{LF}}(t) - z_s^{\mathrm{HF}}(t)\|^2,
\]

\[
\mathcal{L}_{\mathrm{dyn-align}} =
\sum_t \|\dot z_s^{\mathrm{LF}}(t) - \dot z_s^{\mathrm{HF}}(t)\|^2.
\]

These losses should typically be soft rather than hard, since perfect equality is unrealistic in the presence of genuine discrepancy.

### Case 2: Paired data are not available

Without paired trajectories, one should not claim that the two fidelities share the exact same latent coordinates. Instead, one should aim for a shared latent dynamical structure. This can be encouraged through:

- shared encoder and decoder backbones;
- a shared latent projection block;
- common parameter/IC conditioning;
- a common latent dynamics model fitted on the shared coordinates;
- optional distributional or moment matching on the latent core.

In this unpaired case, shared dynamics is a weaker but still meaningful form of cross-fidelity coupling.

## Training Objective

A generic training objective can be written as

\[
\mathcal{L}
=
\lambda_{\mathrm{rec}} \mathcal{L}_{\mathrm{rec}}
+ \lambda_{\mathrm{align}} \mathcal{L}_{\mathrm{align}}
+ \lambda_{\mathrm{dyn}} \mathcal{L}_{\mathrm{dyn}}
+ \lambda_{\mathrm{mf}} \mathcal{L}_{\mathrm{mf}}
+ \lambda_{\mathrm{priv}} \mathcal{L}_{\mathrm{priv}}
+ \lambda_{\mathrm{reg}} \mathcal{L}_{\mathrm{reg}}.
\]

Typical terms are:

- reconstruction loss for each fidelity;
- paired latent alignment on the shared core when pairs exist;
- latent dynamics residual or rollout loss;
- multifidelity discrepancy or correction loss;
- regularization on private latent coordinates;
- standard smoothness or weight regularization.

If MFMC-SINDy is used, a practical training strategy is staged rather than fully end-to-end:

1. train the autoencoder and latent alignment model;
2. encode trajectories into latent space;
3. fit the shared latent SINDy operator using multifidelity regression;
4. fit the HF residual dynamics or decoder correction;
5. optionally alternate between representation updates and latent dynamics refits.

This is more stable than jointly training all components from scratch with a Neural ODE.

## Relation to Two-Step Multi-Fidelity Models

A classical two-step model typically proceeds as follows:

1. train a surrogate for the LF system;
2. train a second map that converts LF predictions into HF predictions.

Schematically,

\[
(\mu,\eta) \mapsto \hat x^{\mathrm{LF}}(t),
\qquad
\hat x^{\mathrm{HF}}(t) = G\!\left(\hat x^{\mathrm{LF}}(t), \mu, \eta\right).
\]

This family includes residual correction models and many practical multi-fidelity surrogates.

### Main difference in philosophy

The two-step model treats LF primarily as an intermediate predictor to be corrected. By contrast, the shared-latent methodology uses LF trajectories to learn the reduced dynamical backbone itself. The distinction is fundamental:

- in a two-step model, LF helps mainly as an input to a transfer map;
- in a shared-latent model, LF helps estimate the latent manifold and latent dynamics.

### Potential advantages of the shared-latent approach

- LF data can reduce the sample complexity of the dynamical model, not only of the correction map.
- Unpaired LF trajectories remain useful because they still inform the shared backbone.
- The method provides a decomposition into common dynamics and HF-specific discrepancy.
- SINDy-based latent modeling enables the use of MFMC-style regression where it is mathematically natural.
- The framework is more compatible with interpretation of what part of the dynamics is shared and what part is missing in CFD.

### Potential advantages of the two-step approach

- It is simpler to train.
- It does not require strong latent alignment assumptions.
- It may be more robust when CFD and experiments differ substantially.
- It can be a very strong predictive baseline when the main objective is accuracy rather than scientific decomposition.

### Key limitation of the two-step approach

LF forecast errors propagate directly into the HF correction stage. Moreover, the latent structure shared between CFD and experiment is not modeled explicitly; the method corrects predictions, but does not directly learn a common reduced dynamical core.

## Recommended Backbone for the Present Application

For aerosol CFD-to-experiment transfer, the most defensible first backbone is:

- principally shared autoencoders with parameter/IC conditioning;
- a shared latent core \(z_s\) and optional private fidelity-specific block \(z_f^p\);
- soft paired alignment on \(z_s\) when matched LF-HF trajectories are available;
- a shared latent SINDy backbone estimated with MFMC-style multifidelity regression;
- an optional HF residual dynamics term or HF decoder correction to model discrepancy.

This backbone reflects the scientific assumption that CFD and experiments share dominant physical evolution but not complete equality in either observation space or latent dynamics.

## Summary of the Design Space

The methodology space can therefore be summarized as follows:

- Representation:
  - fully shared latent space;
  - principally shared autoencoders;
  - shared core plus private latent coordinates;
  - parameter/IC-informed latent space.
- Dynamics:
  - shared latent SINDy with MFMC;
  - shared latent Neural ODE;
  - shared SINDy core with HF residual dynamics;
  - shared dynamics plus HF decoder correction.
- Coupling:
  - paired latent alignment when LF-HF pairs exist;
  - shared latent dynamics regularization when pairs do not exist.
- Baseline comparison:
  - always compare against a two-step LF-surrogate-plus-HF-correction model.

The central methodological claim is not that CFD and experiments are identical, but that they contain enough common dynamical structure for a shared latent reduced-order backbone to be statistically and scientifically useful.
