# mf_lnode

`mf_lnode` is a research-oriented Python library for offline multi-fidelity latent Neural ODE reduced-order modeling. The first milestone focuses on paired and partially paired multi-fidelity trajectories, shared-latent autoencoding with fidelity-specific observation adapters/heads, and a shared latent Neural ODE conditioned on system parameters.

## Package Structure

```text
src/mf_lnode/
  configs/      Dataclass-based experiment, model, data, loss, and training configs
  data/         Trajectory containers, windowed datasets, synthetic generators, scaling
  dynamics/     Parameterized latent vector fields and Neural ODE integration
  evaluation/   Rollout and per-fidelity metrics
  losses/       Composable rollout, latent consistency, multifidelity, and regularization losses
  models/       Shared encoder/decoder building blocks and the LatentNeuralODEModel
  training/     Fidelity schedules, trainer, callbacks, checkpointing
  utils/        Reproducibility helpers
examples/
  train_synthetic.py
tests/
```

## Key Abstractions

- `TrajectorySample` represents one offline trajectory with times, observations, parameters, fidelity metadata, and optional pairing identifiers.
- `MultiFidelityTrajectorySample` groups trajectories that belong to the same physical sample across fidelities.
- `WindowedTrajectoryDataset` converts grouped trajectories into short temporal context windows plus rollout targets.
- `LatentNeuralODEModel` combines fidelity-specific observation adapters with shared latent projection, shared dynamics, and fidelity-specific reconstruction heads.
- `LossComposer` combines rollout, temporal latent consistency, paired latent alignment, multifidelity weighting, discrepancy regularization, and latent regularization.
- `Trainer` owns optimization, validation, checkpointing, AMP, gradient clipping, and fidelity weighting schedules.
- The built-in synthetic problem lifts a 2D damped oscillator into a higher-dimensional observation space; the low-fidelity stream adds both noise and a structured nonlinear sensor bias.
- A second built-in example uses CFD-like scalar fields observed on different spatial grids, with low and high fidelity carrying different output dimensions.

## Current Limitations

- The default synthetic example uses the same observation dimension across fidelities.
- Hierarchical decoding assumes the low- and high-fidelity outputs have matching dimensions.
- The built-in Neural ODE solver defaults to differentiable fixed-step RK4/Euler. `torchdiffeq` can be added later without changing model APIs.

## Extension Points

- Add multimodal encoders and decoders by extending the `data` sample containers and the fidelity adapter interface.
- Add probabilistic latent states by swapping deterministic heads for distribution-valued heads in `models`.
- Add physics-informed losses or symbolic sparse latent dynamics in `losses` and `dynamics`.
- Add more advanced paired-sample objectives on top of the current paired latent alignment by reusing `group_ids` and `pairing_ids` already produced by the collate pipeline.

## Quick Start

Run the synthetic example:

```bash
python examples/train_synthetic.py
```

Run the different-grid CFD-like example:

```bash
python examples/train_cfd_like_grids.py
```

Open the guided project walkthrough notebook:

```bash
jupyter notebook notebooks/mf_lnode_project_walkthrough.ipynb
```

The example saves JSON metrics plus a presentation-style figure bundle covering the
test problem, measurement definition, method overview, architecture, training strategy, dataset coverage,
training dynamics, rollout behavior, latent diagnostics, and held-out metrics under
the configured output directory. It also writes an `artifact_index.md` file so the
output folder can be browsed like a lightweight presentation/report. By default, the
synthetic trajectories expose 12 observation channels per fidelity, with low fidelity
constructed as a strongly biased noisy nonlinear distortion of the high-fidelity sensor field.
The CFD-like example uses coarse and fine 2D grids and produces the same style of
figure bundle, but with spatial field visualizations and cross-grid fidelity-gap plots.
The walkthrough notebook reuses those generated artifacts and adds detailed commentary
about the problem setup, architecture, losses, training strategy, data interpretation,
results, and recommended next project directions.

Run the test suite:

```bash
pytest
```
