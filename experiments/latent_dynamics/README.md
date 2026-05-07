# Latent Dynamics Study

This folder now serves two purposes:

- it contains the code and notebook for the latent-dynamics experiment;
- it acts as a short planning note for the presentation built around that notebook.

## Brainstorming Report

### Long-Term Goal

The final objective is not just to compare autoencoders. The real target is a strong
multi-fidelity surrogate model:

- high-fidelity (HF) data are accurate but expensive;
- low-fidelity (LF) data are cheaper and more abundant, but biased;
- the surrogate should exploit both and produce HF-quality transient predictions.

At the highest level, the intended pipeline is:

`LF/HF observations -> encoder(s) -> shared latent space -> latent dynamics model -> decoder`

The decoder of greatest interest is the HF decoder, because the end goal is HF surrogate
prediction rather than LF reconstruction alone.

### Scope Of The Current Notebook

The notebook does not yet implement the full surrogate pipeline. Instead, it isolates a
necessary prerequisite:

Can LF and HF observations of the same underlying transient dynamics be embedded into a
shared latent space that induces consistent latent dynamics?

This is narrower than the full method, but it is the right first question. If the
latent spaces are not compatible, then any downstream latent surrogate will be unstable
or difficult to interpret.

### Main Research Question

It helps to keep two levels of the question separate.

Broad question:

How can we learn a unified latent representation across multiple fidelity levels for the
same parametric dynamical process?

Concrete notebook question:

Do latent alignment and partial weight sharing help LF and HF autoencoders recover
compatible latent dynamics?

### Why Use An Autoencoder Instead Of Only A Decoder

A decoder-only model is usually too weak for this setting unless the latent variables are
already known or are obtained from an external reduction method.

Reasons:

- the latent coordinates are not directly observed;
- LF and HF data need to be mapped into a common latent space;
- the encoder is what makes latent alignment across fidelities possible.

So the practical position is:

- during training, use an encoder-decoder formulation;
- at deployment, the full autoencoder may not be necessary.

For example, an eventual inference path could be:

`LF field -> LF encoder -> latent dynamics -> HF decoder`

or, in a parametric setting:

`parameters / initial conditions -> latent initializer -> latent dynamics -> HF decoder`

### Candidate Full Architectures

The architecture space is richer than just `AE + Neural ODE` versus `AE + SINDy`.
The literature suggests a spectrum from very simple multi-fidelity correction models to
more structured latent and operator-learning architectures.

#### 1. Observation-Space Correction Models

The simplest multi-fidelity baseline is often not a latent model at all. One can predict
HF outputs directly from LF outputs by learning either:

- an additive correction, `x_HF = x_LF + correction(x_LF, params, t)`;
- a nonlinear transfer map, `x_HF = g(x_LF, params, t)`.

This family is conceptually close to composite multi-fidelity networks such as Meng and
Karniadakis (2020). It is attractive when:

- LF and HF states already live on comparable grids;
- the main goal is prediction accuracy;
- interpretability is secondary.

Its weakness is that it does not enforce a clean shared latent representation, so it is
less useful if the real scientific objective is latent dynamics identification.

#### 2. Shared-Latent Autoencoder With Discrete-Time Dynamics

This is the most direct extension of the current notebook.

Architecture:

`LF/HF encoders -> shared latent state -> latent stepper -> HF decoder`

where the latent stepper can be a simple residual MLP, GRU, or discrete-time integrator.

This option is appealing because:

- it is simpler to train than a continuous-time model;
- it works well on regularly sampled trajectories;
- it preserves the key idea of shared latent coordinates across fidelities.

This is probably the most practical next model if the goal is to move from the current
representation study to a first end-to-end surrogate.

#### 3. Shared-Latent Autoencoder + Neural ODE / Latent ODE

This is the natural continuous-time version.

Architecture:

`LF/HF encoders -> shared latent initial state -> Neural ODE -> HF decoder`

This family is connected to Neural ODEs (Chen et al., 2018) and Latent ODEs (Rubanova
et al., 2019). It becomes attractive when:

- sampling in time is irregular;
- LF and HF trajectories live on different time grids;
- extrapolation in continuous time matters.

Its advantages are flexibility and a physically natural time-continuous formulation.
Its disadvantages are higher optimization cost, possible solver sensitivity, and less
transparent dynamics than a sparse model.

#### 4. Shared-Latent Autoencoder + SINDy

This is the most interpretable direction and is conceptually closest to the current
methodology note.

Architecture:

`LF/HF encoders -> shared latent state -> sparse latent vector field -> decoder`

This follows the spirit of SINDy-AE style methods such as Champion et al. (2019).
It is attractive when:

- we want explicit governing equations in latent coordinates;
- we care about sparsity and model structure;
- scientific interpretation matters as much as prediction.

The main risk is that predictive performance may be limited if the latent dynamics are
too nonlinear or if the learned coordinates do not admit a sufficiently sparse law.

#### 5. Hybrid Latent Dynamics Models

A strong compromise is to separate prediction and interpretation.

Examples:

- neural latent dynamics for rollout, plus SINDy as an analysis tool;
- neural latent dynamics regularized toward sparse or linear structure;
- shared latent dynamics plus an HF-specific sparse correction term;
- Koopman-inspired linear latent evolution with nonlinear encoders and decoders.

This family is attractive because the neural component can absorb modeling complexity,
while the structured component can still provide diagnostics and physical insight.

#### 6. Operator-Learning Architectures

If the final goal is a surrogate over families of trajectories or PDE solutions, then
operator-learning models may be more powerful than a classical autoencoder pipeline.

Relevant families include:

- DeepONet-style operator learners;
- Fourier Neural Operator (FNO) and related neural operators;
- multi-fidelity DeepONet / multi-fidelity neural operator variants.

These architectures are especially attractive when:

- the input is a function, field, boundary condition, or parameter field;
- the output is an entire trajectory or field evolution;
- geometry or discretization changes matter.

Their main strength is surrogate power at scale. Their main weakness for this project is
that they do not automatically provide the kind of explicit latent dynamical
interpretation that motivates the current notebook.

#### 7. Multi-Modal Multi-Fidelity Architectures

The literature also suggests a broader setting than just LF and HF fields. In many real
problems, one may have several data modalities at different fidelities:

- simulated fields;
- sparse sensors;
- scalar operating conditions;
- images or videos;
- geometry descriptors;
- experimental measurements.

In that setting, a natural architecture is:

`modality-specific encoders -> shared latent fusion block -> latent dynamics -> task-specific decoders`

The fusion block can be implemented in several ways:

- simple concatenation followed by an MLP;
- product-of-experts or variational fusion;
- cross-attention or transformer-style fusion;
- partially shared encoders with missing-modality robustness.

This is where multimodal VAE-style ideas become relevant. For the long term, this may be
the most realistic formulation if the project eventually combines simulations and
experiments.

#### What Seems Most Relevant Here

For this project, a sensible progression is:

1. Current notebook:
   shared-latent representation study with LF/HF encoders.
2. First end-to-end surrogate:
   shared-latent autoencoder with a discrete-time latent propagator and HF decoder.
3. More ambitious extension:
   replace the discrete propagator with a Neural ODE or structured latent dynamics.
4. Stronger benchmarking direction:
   compare against an observation-space multi-fidelity correction model and an
   operator-learning baseline.

So, yes, autoencoders are one important family, but they are not the only serious option.
The larger design space includes simpler residual multi-fidelity models and more powerful
operator or multimodal fusion architectures.

### What The Current Experiment Actually Tests

The experiment in this folder is deliberately controlled.

- the true latent dynamics are known and two-dimensional;
- the HF field is generated by a nonlinear decoding of those latent coordinates;
- the LF field is built from the HF field using blur, amplitude scaling, and nonlinear
  distortion;
- this lets us separate representation issues from full surrogate-design issues.

The notebook compares three regimes:

1. separate encoders, no latent alignment loss;
2. separate encoders, with latent alignment loss;
3. partially shared encoders/decoders, with latent alignment loss.

This comparison is important because it separates the effect of alignment from the effect
of weight sharing.

### Current Takeaways

The current default run supports a clear qualitative message.

- reconstruction is good in all three regimes;
- latent alignment is the main driver of LF/HF latent agreement;
- partial sharing acts more as a refinement or regularizer than as the only mechanism
  creating agreement;
- latent coordinates alone are not enough: we also need to inspect derivative and vector
  field consistency.

Representative numbers from a default run:

- without alignment, LF/HF latent mismatch is about `3.06`;
- with alignment, LF/HF latent mismatch drops to about `1e-6`;
- SINDy coefficient disagreement drops from about `1.93` to the `1e-3` to `1e-4` range.

So the main presentation claim should not be "sharing solves everything." A more defensible
claim is:

Alignment is what mainly makes LF and HF latent representations compatible, while sharing
can improve the quality and regularity of the learned representation.

### Suggested Presentation Outline

The cleanest flow for the talk is:

1. Motivation and research question
2. Synthetic test case and experimental setup
3. Methods compared in the notebook
4. Results on representation agreement and dynamics consistency
5. Broader architecture options and literature positioning

This order keeps the talk honest:

- first show the precise question;
- then show the controlled evidence;
- only afterwards discuss the full surrogate architecture.

### Literature Positioning

The broader methodology is documented in [../../docs/methodology.md](../../docs/methodology.md).
The main literature anchors currently considered are:

- multi-fidelity residual and correction networks;
- Neural ODE and Latent ODE continuous-time latent models;
- SINDy for sparse system identification;
- SINDy-AE / learned-coordinate dynamics identification;
- operator learning through DeepONet and neural operators;
- multi-fidelity operator learning;
- shared-latent-space learning across domains or fidelities;
- multimodal latent-variable models with missing-modality robustness;
- multi-fidelity Monte Carlo and control-variate estimation for efficient use of LF/HF data.

The references already collected or worth adding to the repository bibliography include:

- Brunton et al. (2016)
- Champion et al. (2019)
- Chen et al. (2018)
- Rubanova et al. (2019)
- Meng and Karniadakis (2020)
- Lu et al. (2021)
- Li et al. (2021)
- Lu et al. (2022)
- Peherstorfer et al. (2016, 2018)
- Liu et al. (2017)
- Wu and Goodman (2018)

These papers suggest that the long-term project can be framed in at least three
different communities:

- scientific machine learning for latent dynamical systems;
- multi-fidelity surrogate modeling;
- multimodal representation learning.

That is useful for the presentation because it means we can position the current notebook
as a controlled latent-consistency study inside a much larger design space.

## Folder Layout

- `latent_dynamics_study.ipynb`
  Main notebook and presentation reference.
- `utils.py`
  Shared plotting and analysis helpers used by the notebook and scripts.
- `plot_study.py`
  Trains the representation study and plots the main latent-sharing figure.
- `analyze_dynamics.py`
  Runs the dynamics analysis and plots SINDy and derivative-consistency diagnostics.
- `animate_fields.py`
  Animates the LF/HF field evolution.
- `../../src/mfaesindy/experiments/latent_dynamics/dataset.py`
  Synthetic transient field generator with known latent dynamics.
- `../../src/mfaesindy/experiments/latent_dynamics/autoencoder.py`
  Two-fidelity autoencoder with configurable sharing.
- `../../src/mfaesindy/experiments/latent_dynamics/study.py`
  Training and evaluation helpers for the three comparison regimes.

## Run

From the repository root:

```bash
python experiments/latent_dynamics/plot_study.py
```

To save the main representation figure:

```bash
python experiments/latent_dynamics/plot_study.py \
  --save-path experiments/latent_dynamics/artifacts/latent_sharing.png
```

To run the dynamics analysis:

```bash
python experiments/latent_dynamics/analyze_dynamics.py \
  --save-path experiments/latent_dynamics/artifacts/dynamics_analysis.png
```

## Output

The main outputs in this folder are:

- a representation figure comparing latent trajectories, phase portraits, and field
  reconstructions across the three regimes;
- a dynamics figure comparing LF/HF latent phase portraits and finite-difference versus
  encoder-pushforward latent derivatives;
- the notebook narrative, which should remain the main reference for the presentation.
