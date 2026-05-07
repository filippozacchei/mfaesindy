# Repo Handoff and Research Roadmap

This document is a detailed to-do list for turning the current repository into a clean handoff-quality project and for preparing the next research phase around multi-fidelity shared-latent training, mismatch modeling, and baseline comparison.

It is intentionally action-oriented. It does not implement the changes; it defines the work packages we should complete together.

## Goal

By the end of this roadmap, the repository should:

- be understandable by a new user without oral explanation;
- have one clear package identity and one clear scientific story;
- include examples and notebooks that teach the method cell by cell;
- separate implemented functionality from research ideas and future extensions;
- support a paper-quality experimental plan around shared latent spaces, mismatch-aware multi-fidelity training, and baseline comparisons.

## Immediate Diagnosis of the Current Repo

The repo is already substantial, but it currently mixes several layers of maturity:

- a usable library under `src/mf_lnode/`;
- a second package namespace under `src/mfaesindy/` that looks experimental/incomplete;
- a `README.md` that presents the project as a latent Neural ODE library;
- a `docs/methodology.md` that presents a more ambitious autoencoder + SINDy + MFMC methodology that is not the same as the currently implemented main package;
- long example scripts under `examples/` that are good research demos, but not ideal teaching entry points;
- a generated paper-style notebook under `notebooks/build_project_walkthrough.py`, which is useful, but not a substitute for tutorial notebooks;
- tracked generated outputs under `artifacts/`, which are useful for inspection but should not be part of the core handoff story.

This means the first problem is not lack of content. The first problem is that the repo currently tells more than one story at once.

## Technical Assessment of the Research Idea

Yes, the core idea is scientifically reasonable:

- fidelity-specific encoders and decoders;
- a shared latent space;
- explicit pressure for LF and HF trajectories to align in latent coordinates;
- a shared latent dynamics model, possibly with an HF correction;
- weighted multi-fidelity training that exploits abundant LF data without letting LF bias dominate the final surrogate.

This is a good direction if the scientific assumption is:

> LF and HF are different observations or discretizations of the same underlying dynamical process, not completely different physics.

Under that assumption, the shared latent space is exactly the right inductive bias.

## Main risks

- If LF and HF are too different, hard latent alignment may over-constrain the model and damage HF performance.
- If the latent space is only enforced through reconstruction, the model may learn fidelity-specific latent coordinates that look similar only superficially.
- If mismatch penalties are introduced without normalization, the training weights may become arbitrary and the paper will be hard to defend.
- If the repo keeps mixing "implemented now" and "planned later", the paper and the library documentation will both become confusing.

## My recommendation

The right next scientific step is not "add many losses". The right next step is:

1. define exactly what mismatch means;
2. measure it consistently;
3. normalize it;
4. ablate the effect of each mismatch term;
5. compare against simple and strong baselines.

That is the structure that will make the method convincing.

## Phase 0: Freeze the Project Story

- [ ] Decide the primary identity of the repo.
  - Option A: the repo is first and foremost the `mf_lnode` library, and the SINDy/MFMC work is the next research extension.
  - Option B: the repo is becoming `mfaesindy`, and the latent Neural ODE implementation is an intermediate stage.
- [ ] Decide whether `src/mfaesindy/` stays, is promoted, is merged into `src/mf_lnode/`, or is removed.
- [ ] Decide whether `docs/methodology.md` describes:
  - the method implemented today;
  - the target paper methodology;
  - or both, but split into separate sections called `Implemented` and `Planned Extension`.
- [ ] Write one 3-5 sentence project definition that we will reuse everywhere.
- [ ] Define the target audience explicitly.
  - external researcher using the library;
  - collaborator extending the method;
  - reviewer reading the repo for the paper;
  - or all three, but with distinct entry points.

Deliverable:

- one short "project identity" note we will treat as the source of truth for the rest of the cleanup.

## Phase 1: Clean the Repository Structure

- [ ] Unify naming across repository, package, docs, and paper.
  - `mf_lnode` and `mfaesindy` should not coexist ambiguously.
- [ ] Decide which directories are product code, which are experiments, and which are paper material.
- [ ] Adopt a stable top-level structure such as:
  - `src/`
  - `tests/`
  - `examples/`
  - `notebooks/`
  - `docs/`
  - `paper/`
  - `research/` or `experiments/`
- [ ] Move brainstorming notes out of user-facing paths if they are not polished.
  - `experiments/latent_dynamics/README.md` is useful, but it reads like an internal research note.
- [ ] Remove tracked generated artifacts from the core repo narrative.
  - keep them locally or publish selected figures elsewhere;
  - do not make new users wonder whether `artifacts/` is part of the package.
- [ ] Ensure ignored generated content is not versioned again.
- [ ] Remove committed cache/build leftovers from the conceptual structure.
  - `__pycache__`
  - `*.egg-info`
- [ ] Add a short top-level architecture map to the docs.

Deliverable:

- a repo tree that makes sense without explanation.

## Phase 2: Rewrite the Documentation Stack

- [ ] Rewrite `README.md` for first-contact clarity.
  - what the library does;
  - what problem it solves;
  - what is implemented now;
  - how to install;
  - the 5-minute quickstart;
  - where to go next.
- [ ] Split documentation by reader intent instead of one long narrative.
  - `docs/installation.md`
  - `docs/quickstart.md`
  - `docs/concepts.md`
  - `docs/data_model.md`
  - `docs/training.md`
  - `docs/api_overview.md`
  - `docs/research_methodology.md`
- [ ] Rewrite the methodology note so that assumptions are explicit.
  - what is shared across fidelities;
  - what is fidelity-specific;
  - what pairing means;
  - what the learning target is;
  - what is implemented vs planned.
- [ ] Create a "How to use this repo" page for a new collaborator.
  - where datasets are built;
  - how configs flow;
  - where losses are defined;
  - where training happens;
  - where figures are produced.
- [ ] Add a "What is not implemented yet" page.
  - this will avoid over-claiming in both docs and paper.
- [ ] Add docstrings or short usage notes for every public API object exported in `src/mf_lnode/__init__.py`.
- [ ] Add a minimal contributor guide.
  - environment setup;
  - running tests;
  - style expectations;
  - where to put experiments versus reusable code.

Deliverable:

- documentation that supports both adoption and review.

## Phase 3: Redesign the Examples and Notebooks

The current examples are rich, but they are too long and too presentation-oriented to be the best teaching entry points.

- [ ] Keep the current long scripts as full demo pipelines only if they are clearly labeled as such.
- [ ] Create one minimal example per use case.
  - `examples/minimal_synthetic.py`
  - `examples/minimal_cfd_like.py`
- [ ] Create one "full experiment" example per use case.
  - these can keep the plotting and artifact generation.
- [ ] Split tutorial notebooks by learning objective.
  - Notebook 1: installation and quickstart
  - Notebook 2: synthetic dataset generation
  - Notebook 3: training the shared latent model
  - Notebook 4: evaluating rollouts and fidelity gaps
  - Notebook 5: ablations on alignment and weighting
- [ ] Make each notebook readable cell by cell.
  - each code cell should have a markdown cell immediately before it explaining:
    - what the next cell does;
    - why it matters;
    - what the user should inspect in the output.
- [ ] Avoid giant cells that do many things at once.
- [ ] Add expected output descriptions after important cells.
  - example: "This plot should show LF and HF trajectories with visible bias but correlated dynamics."
- [ ] Add "sanity check" cells throughout.
  - inspect tensor shapes;
  - inspect grouped samples;
  - inspect paired IDs;
  - inspect scaled versus unscaled data;
  - inspect one batch before training.
- [ ] Add a final summary section in each notebook.
  - what was learned;
  - what can break;
  - what to try next.
- [ ] Make notebook outputs optional and reproducible.
  - a notebook should still be useful even if outputs are cleared.

Deliverable:

- tutorials that are genuinely pedagogical, not only demonstrative.

## Phase 4: Clarify the Methodology for the Paper

This phase is essential because the repo currently spans at least two methodological stories:

- latent Neural ODE shared-dynamics modeling;
- planned shared-latent autoencoder + sparse dynamics / SINDy-style identification.

- [ ] Decide the paper claim precisely.
  - Is the paper about multi-fidelity latent forecasting?
  - Is it about interpretable latent dynamics identification?
  - Is it about both, with one as the main claim and one as an extension?
- [ ] Define the mathematical objects once and reuse them consistently.
  - observations;
  - parameters;
  - fidelities;
  - paired groups;
  - latent states;
  - latent dynamics;
  - mismatch terms.
- [ ] Separate assumptions from design choices.
  - assumption: shared underlying dynamics;
  - design choice: partially shared encoder/decoder weights;
  - design choice: discrepancy decoder;
  - design choice: SINDy or Neural ODE latent dynamics.
- [ ] State clearly which losses are scientifically motivated and which are engineering stabilizers.
- [ ] Add a methodology figure that matches the exact implementation/paper scope.
- [ ] Add a table named `Implemented now vs planned extension`.

Deliverable:

- a methodology section that is rigorous and internally consistent.

## Phase 5: Define Mismatch Carefully

This is the most important technical next step for the research extension.

### 5.1 Decide what "mismatch" means

- [ ] Define at least three candidate mismatch notions.
  - latent state mismatch:
    `||z_LF(t) - z_HF(t)||^2`
  - latent rollout mismatch:
    `sum_t ||z_LF(t) - z_HF(t)||^2`
  - latent dynamics mismatch:
    `||f_LF(z, mu) - f_HF(z, mu)||^2` or `||f(z_LF, mu) - f(z_HF, mu)||^2`
- [ ] Decide whether mismatch is measured:
  - only on paired samples;
  - only at matched times;
  - after interpolation to a common time grid;
  - in latent space only;
  - or also after decoding to a common observation space.
- [ ] Decide whether mismatch is symmetric or HF-anchored.
  - symmetric if both fidelities are treated equally;
  - HF-anchored if LF is only regularized toward HF.

### 5.2 Normalize mismatch terms

- [ ] Normalize mismatch by latent dimension and time horizon.
- [ ] Consider variance normalization so one latent channel does not dominate.
- [ ] Decide whether losses should be averaged per sample, per time step, or per trajectory.
- [ ] Define units and scaling explicitly in the paper.

### 5.3 Choose the first implementation target

- [ ] Start with one simple mismatch term first.
  - recommendation: paired latent rollout mismatch on matched times.
- [ ] Add one optional stronger term later.
  - recommendation: dynamics mismatch or HF correction mismatch.
- [ ] Do not add more than two new terms before the first ablation study.

Deliverable:

- one precise mismatch definition that can be defended mathematically and implemented cleanly.

## Phase 6: Multi-Fidelity Weighted Training Strategy

- [ ] Write the total objective in one equation and define every weight.
- [ ] Decide which weights are:
  - fixed hyperparameters;
  - scheduled over epochs;
  - estimated from data;
  - learned automatically.
- [ ] Start with a simple weighted objective:

  `L = w_rec L_rec + w_align L_align + w_mismatch L_mismatch + w_dyn L_dyn + w_reg L_reg`

- [ ] Decide whether `w_mismatch` should depend on:
  - fidelity gap magnitude;
  - data confidence;
  - number of LF/HF samples;
  - training epoch.
- [ ] Compare at least three weighting strategies.
  - fixed manual weights;
  - curriculum schedule from LF-heavy to HF-heavy;
  - normalized or uncertainty-aware weights.
- [ ] Track each term separately during training.
- [ ] Plot all loss components, not only the total loss.
- [ ] Add failure diagnostics.
  - latent collapse;
  - HF degradation despite low total loss;
  - LF domination;
  - mismatch weight explosion.

Deliverable:

- a weighting scheme that is interpretable, monitorable, and ablatable.

## Phase 7: Decide the Dynamics Model Scope

Before comparing against baselines, we need to lock the dynamics model used in the paper.

- [ ] Decide whether the first paper version uses:
  - shared latent Neural ODE;
  - shared latent discrete-time propagator;
  - shared latent SINDy;
  - or a hybrid where Neural ODE is predictive and SINDy is interpretive.
- [ ] If SINDy is introduced, define the exact role.
  - primary dynamics model;
  - auxiliary regularizer;
  - post hoc analysis layer;
  - or sparse correction on top of shared dynamics.
- [ ] Decide whether HF-specific correction is needed.
  - shared dynamics only;
  - shared dynamics plus sparse HF residual;
  - shared latent coordinates plus fidelity-specific observation maps only.
- [ ] Keep the first research paper simpler than the maximal design space.

Recommendation:

- For the first defensible paper, keep one shared latent dynamics model and one clearly defined mismatch mechanism.
- Treat HF-specific correction as an ablation or extension unless results prove it is essential.

Deliverable:

- a frozen model scope for the paper experiments.

## Phase 8: Baseline Plan

The paper should not compare only against your own variants.

### 8.1 Minimum baseline suite

- [ ] HF-only model trained on HF data only.
- [ ] LF-only model evaluated as a surrogate for HF after a simple transfer map.
- [ ] Observation-space residual correction baseline.
- [ ] Shared latent model without alignment.
- [ ] Shared latent model with alignment but without mismatch weighting.

### 8.2 Strong external baseline

- [ ] Add the Conti et al. multi-fidelity ROM baseline to the comparison plan.
- [ ] Read and summarize the exact method before implementing anything.
- [ ] Extract the ingredients that matter for a fair comparison.
  - HF reduced basis construction;
  - LF-to-HF latent/reduced-state mapping;
  - sequence model choice;
  - decoding back to the full field.
- [ ] Decide whether the baseline can be reproduced exactly on your synthetic testbeds or whether an adapted analogue is needed.
- [ ] Make the comparison fair in data budget.
  - same number of HF trajectories;
  - same LF availability;
  - same train/validation/test split;
  - same evaluation metrics.
- [ ] State clearly in the paper where the baseline is stronger or weaker by construction.

Important note:

Conti et al. is a strong and relevant baseline because it uses dimensionality reduction plus a multi-fidelity neural surrogate for time-dependent PDE settings, but it does not make the same claim you want to make about shared latent dynamical structure. That distinction should be explicit in the paper.

Deliverable:

- a baseline matrix with methods, data budgets, and metrics.

## Phase 9: Experimental Protocol

- [ ] Define datasets and difficulty levels.
  - easy synthetic;
  - harder synthetic with larger LF bias;
  - different-grid field example;
  - eventually one physically more credible benchmark.
- [ ] Define the split protocol.
  - random group split;
  - parameter-holdout split;
  - both, if possible.
- [ ] Define evaluation metrics.
  - reconstruction MSE;
  - rollout MSE;
  - long-horizon stability;
  - fidelity-gap reconstruction;
  - latent mismatch;
  - data-efficiency under HF scarcity.
- [ ] Define HF-scarcity studies.
  - vary number of HF paired samples;
  - hold LF pool fixed;
  - show where multi-fidelity actually helps.
- [ ] Define ablations.
  - no alignment;
  - alignment only;
  - alignment plus mismatch;
  - different weight schedules;
  - shared versus partially shared encoders/decoders;
  - with and without HF correction.
- [ ] Define statistical reporting.
  - seeds;
  - means and standard deviations;
  - representative qualitative rollouts;
  - failure cases.

Deliverable:

- an experiment table that prevents ad hoc evaluation.

## Phase 10: Paper Preparation Checklist

- [ ] Fix the main paper claim in one sentence.
- [ ] Build the paper around one central hypothesis.
  - example: a shared latent space with explicit cross-fidelity mismatch control improves HF prediction under limited HF data.
- [ ] Write the contributions list conservatively.
- [ ] Add one method figure, one training-objective figure, and one benchmark figure.
- [ ] Prepare a notation table.
- [ ] Add an ablation table focused on what each loss term contributes.
- [ ] Add a baseline comparison table.
- [ ] Add a limitations paragraph.
  - when shared latent assumptions fail;
  - when LF is too biased;
  - when alignment hurts;
  - when the method is less attractive than direct residual-learning baselines.
- [ ] Make sure all claims in the paper correspond either to code already in the repo or to explicitly labeled future work.

Deliverable:

- a paper that is honest, reproducible, and methodologically tight.

## Phase 11: Release and Handoff Checklist

- [ ] One-command install works.
- [ ] One-command quickstart works.
- [ ] One notebook can be run from top to bottom by a new user.
- [ ] Public APIs have stable names.
- [ ] Tests pass.
- [ ] Example outputs are reproducible.
- [ ] Documentation has no broken references.
- [ ] The repo contains no ambiguity about what is current versus future work.
- [ ] The handoff recipient can answer:
  - what this repo is;
  - how to run it;
  - how to extend it;
  - what research question it addresses.

## Suggested Execution Order

We should not do the tasks in arbitrary order. The most efficient order is:

1. freeze project identity and naming;
2. clean the repo structure;
3. rewrite README and docs skeleton;
4. redesign minimal examples and tutorial notebooks;
5. formalize mismatch and weighted objectives;
6. lock the paper model scope;
7. define the baseline matrix;
8. run ablations and comparisons;
9. write the paper around the finalized evidence.

## Recommended First Working Session

In the next pass, I recommend we do only the following:

- choose the official repo name and scope;
- decide what happens to `mf_lnode` versus `mfaesindy`;
- decide whether `docs/methodology.md` is "implemented method" or "paper roadmap";
- outline the new `README.md`;
- define the first mismatch term and the first baseline suite.

That will remove most of the current ambiguity.

## References To Keep In Mind

- Current library entry point: `README.md`
- Current methodology note: `docs/methodology.md`
- Current walkthrough generator: `notebooks/build_project_walkthrough.py`
- Current synthetic demo: `examples/train_synthetic.py`
- Current different-grid demo: `examples/train_cfd_like_grids.py`
- Current public package: `src/mf_lnode/`
- Current experimental secondary package: `src/mfaesindy/`
- Conti et al. baseline:
  - journal version: https://doi.org/10.1098/rspa.2023.0655
  - preprint entry: https://doi.org/10.48550/arXiv.2309.00325
