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

## Methodology Decisions Already Fixed

The discussion has already narrowed the design space substantially. The current working assumptions are:

- [x] Final application: low-fidelity CFD aerosol trajectories and high-fidelity experimental trajectories.
- [x] High-dimensional observations on both fidelities require an autoencoder-style reduced-order model.
- [x] A subset of LF-HF runs can be treated as paired under matched operating conditions.
- [x] The latent representation should be principally shared, with room for optional private fidelity-specific components.
- [x] The latent state should be informed by operating parameters and initial-condition descriptors.
- [x] The main latent forecasting backbone should be shared-core SINDy with MFMC-style multifidelity regression.
- [x] HF-specific residual dynamics or decoder correction are allowed as controlled extensions if the shared backbone is insufficient.
- [x] A two-step LF-surrogate-plus-HF-correction model remains a required baseline.

What is still open is not the overall methodology, but the exact implementation choices and the experimental protocol.

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

The current examples are rich, but they are too long and too presentation-oriented to be the best teaching entry points. They also reflect the older latent-Neural-ODE-first story rather than the newly chosen shared-latent MFMC-SINDy backbone.

- [ ] Reclassify the existing examples.
  - `examples/train_synthetic.py` should become a legacy or baseline example unless it is rewritten around the new backbone.
  - `examples/train_cfd_like_grids.py` should become either a legacy example or the seed for the new paired multi-fidelity example.
- [ ] Create one minimal methodology-aligned example.
  - target: `examples/minimal_shared_latent_sindy.py`
  - purpose: show paired LF-HF synthetic data, shared latent encoding, latent SINDy fit, and basic rollout evaluation.
- [ ] Create one minimal baseline example.
  - target: `examples/minimal_two_step_baseline.py`
  - purpose: show the Conti-style logic or a simpler two-step LF-surrogate-plus-HF-correction analogue.
- [ ] Create one full experiment example for the chosen backbone.
  - target: `examples/paired_mfmc_sindy_experiment.py`
  - purpose: run the staged autoencoder -> latent SINDy -> HF correction pipeline with plots and metrics.
- [ ] Keep only one heavy presentation-style example, and make sure it is explicitly labeled as a report/demo pipeline rather than the main entry point.
- [ ] Split tutorial notebooks by learning objective.
  - Notebook 1: installation and quickstart
  - Notebook 2: paired synthetic dataset generation and inspection
  - Notebook 3: training the shared latent autoencoder
  - Notebook 4: latent derivative estimation and SINDy library construction
  - Notebook 5: MFMC-SINDy fitting on paired LF-HF data
  - Notebook 6: baseline comparison and ablations
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

## Phase 3A: Make Examples Validate the Methodology

The examples should not only demonstrate usage. They should verify that the methodology behaves as claimed.

- [ ] Define one canonical synthetic paired benchmark for the new method.
  - LF should have a controlled nonlinear bias relative to HF.
  - paired metadata, parameter conditioning, and matched times should all be explicit.
- [ ] Ensure the minimal shared-latent example reports:
  - reconstruction error by fidelity;
  - paired latent alignment error;
  - latent SINDy regression residual;
  - rollout error;
  - fidelity-gap error.
- [ ] Ensure the minimal two-step baseline reports the same metrics where applicable.
- [ ] Ensure the full experiment example includes ablations:
  - without alignment;
  - with alignment;
  - with MFMC-SINDy;
  - with optional HF correction.
- [ ] Make every example write a compact metrics JSON that can be consumed by tests and by the notebook/report layer.
- [ ] Add one example that acts as a smoke benchmark for CI-scale validation.
  - quick runtime;
  - tiny synthetic paired dataset;
  - deterministic seed;
  - no heavy plotting required.

Deliverable:

- examples that double as methodological validation artifacts.

## Phase 4: Clarify the Methodology for the Paper

This phase is essential because the repo currently spans at least two methodological stories:

- latent Neural ODE shared-dynamics modeling;
- planned shared-latent autoencoder + sparse dynamics / SINDy-style identification.

- [x] Fix the main scientific direction.
  - primary story: multi-fidelity latent reduced-order modeling for HF prediction;
  - core mechanism: shared latent backbone with MFMC-SINDy on paired LF-HF data;
  - secondary extension: HF residual dynamics or decoder correction;
  - Neural ODE: optional refinement or comparison, not the first backbone.
- [ ] Convert the chosen backbone into one concise formal statement for the paper and the README.
- [ ] Define the mathematical objects once and reuse them consistently.
  - observations;
  - parameters;
  - fidelities;
  - paired groups;
  - shared latent core and optional private latent block;
  - latent dynamics and HF correction terms;
  - mismatch terms.
- [ ] Separate assumptions from design choices.
  - assumption: CFD and experiment share a dominant latent dynamical backbone;
  - assumption: a paired subset exists and is statistically meaningful;
  - design choice: principally shared encoder/decoder weights;
  - design choice: optional private latent coordinates;
  - design choice: MFMC-SINDy shared core with optional HF residual correction.
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

- [x] Start with one simple mismatch term first.
  - chosen first target: paired latent rollout mismatch on matched times for the shared latent core.
- [ ] Add one optional stronger term later.
  - recommendation: HF latent residual dynamics mismatch or HF decoder correction mismatch.
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

## Phase 7: Lock the Dynamics Model Scope

Before implementation expands further, we need to freeze the first model we will actually build and test.

- [x] First backbone:
  - principally shared autoencoder;
  - parameter/IC-informed latent encoding;
  - shared latent core with optional private fidelity-specific block;
  - shared latent SINDy estimated with MFMC on paired LF-HF data.
- [x] Neural ODE is not the first main model.
  - it stays as a later refinement or comparison.
- [ ] Define whether the first release uses:
  - shared core only;
  - shared core plus HF decoder correction;
  - shared core plus HF latent residual dynamics.
- [ ] Decide whether the HF correction is sparse first or neural first.
- [ ] Keep the first research paper simpler than the maximal design space.

Recommendation:

- For the first defensible paper, build the shared MFMC-SINDy core first.
- Treat HF correction as the first extension and Neural ODE refinement as the second extension.

Deliverable:

- a frozen model scope for the paper experiments.

## Phase 7A: Immediate Technical Next Steps

These are now the most useful concrete tasks.

- [ ] Decide whether the default latent architecture is:
  - fully shared latent state;
  - or shared latent core plus private fidelity-specific coordinates.
- [ ] Define the parameter and initial-condition inputs that will condition the encoder and latent dynamics.
  - list the exact fields expected from CFD;
  - list the exact fields expected from experiments;
  - identify which fields are paired and which may be noisy or incomplete.
- [ ] Define the paired-data contract in code and docs.
  - what counts as one LF-HF pair;
  - what metadata identify matched operating conditions;
  - how matched times are represented or interpolated.
- [ ] Define the first latent SINDy library.
  - polynomial order;
  - parameter dependence;
  - whether cross terms with conditioning variables are included.
- [ ] Define the first latent derivative estimator.
  - finite differences;
  - spline smoothing;
  - or another robust estimator for noisy HF experimental data.
- [ ] Write the MFMC-SINDy estimator in mathematical detail.
  - target HF moments;
  - LF baseline estimator;
  - paired correction term;
  - control-variate coefficient estimation;
  - regularized sparse solve.
- [ ] Decide whether the first HF correction is:
  - no correction;
  - decoder correction;
  - or latent residual dynamics correction.
- [ ] Build the minimum baseline set.
  - HF-only latent surrogate;
  - two-step LF-surrogate-plus-HF-correction baseline;
  - shared latent model without alignment;
  - shared latent model with alignment but without MFMC correction.

Deliverable:

- one precise first implementation target that can be coded without further methodological branching.

## Phase 7B: Implementation Work Packages

The methodology should now be translated into bounded engineering slices.

### 7B.1 Data and configuration layer

- [ ] Extend the data schema so one grouped sample can represent:
  - LF observations;
  - HF observations;
  - pairing metadata;
  - operating parameters;
  - initial-condition descriptors;
  - optional time-alignment metadata.
- [ ] Extend config objects for:
  - shared latent core dimension;
  - optional private latent dimension;
  - conditioning inputs;
  - SINDy library settings;
  - MFMC estimator settings;
  - HF correction settings.

### 7B.2 Model layer

- [ ] Implement or refactor the encoder/decoder stack so the model can expose:
  - shared latent core only;
  - or shared latent core plus private fidelity-specific coordinates.
- [ ] Expose parameter/IC conditioning consistently across encoder, decoder, and latent dynamics.
- [ ] Add a clean interface boundary between:
  - representation learning;
  - latent dynamics fitting;
  - HF correction modules.

### 7B.3 Latent-dynamics layer

- [ ] Add a SINDy library builder for the shared latent core.
- [ ] Add latent derivative estimation utilities robust to noisy HF trajectories.
- [ ] Add MFMC-style estimators for latent regression moments.
- [ ] Add sparse solve / thresholding utilities for the shared operator.
- [ ] Add an optional HF residual correction module.

### 7B.4 Training pipeline

- [ ] Implement the staged training workflow:
  - stage 1: train shared latent autoencoder;
  - stage 2: encode trajectories and estimate derivatives;
  - stage 3: fit shared latent SINDy with MFMC;
  - stage 4: fit optional HF correction;
  - stage 5: optional alternating refinement.
- [ ] Decide what remains inside the current `Trainer` abstraction and what should become a higher-level experiment pipeline.

Deliverable:

- a development plan that maps directly to code modules rather than research slogans.

## Phase 7C: Test Strategy for the New Methodology

The test suite must evolve with the methodology. Current tests mostly cover the existing latent-Neural-ODE library surface. The new backbone needs unit, integration, and example-level validation.

### 7C.1 Update existing tests

- [ ] Extend `tests/test_datasets.py` to validate:
  - paired LF-HF grouping;
  - parameter/IC metadata propagation;
  - matched-time or interpolation behavior.
- [ ] Extend `tests/test_autoencoder.py` and `tests/test_model.py` to validate:
  - shared latent core shape;
  - optional private latent coordinates;
  - conditioning inputs;
  - paired latent outputs.
- [ ] Extend `tests/test_losses.py` to validate:
  - paired shared-core alignment loss;
  - latent rollout mismatch loss;
  - optional HF correction penalties.
- [ ] Extend `tests/test_configs.py` and `tests/test_public_api.py` to cover the new configuration and public interfaces.

### 7C.2 Add new targeted tests

- [ ] Add `tests/test_sindy_library.py`.
  - feature generation;
  - parameter-conditioned library terms;
  - shape and ordering guarantees.
- [ ] Add `tests/test_latent_derivatives.py`.
  - derivative estimation on clean and noisy synthetic trajectories;
  - basic numerical sanity checks.
- [ ] Add `tests/test_mfmc_regression.py`.
  - LF baseline moment estimator;
  - paired correction term;
  - control-variate coefficient estimation;
  - recovery on a toy linear latent system.
- [ ] Add `tests/test_hf_correction.py` if residual dynamics or decoder correction are introduced.
- [ ] Add `tests/test_pipeline_shared_latent_sindy.py`.
  - end-to-end staged pipeline smoke test on a tiny paired synthetic dataset.

### 7C.3 Example and regression tests

- [ ] Add an example smoke test that runs the minimal shared-latent MFMC-SINDy example with a quick flag.
- [ ] Add an example smoke test for the minimal two-step baseline.
- [ ] Define tolerances for:
  - metrics file creation;
  - non-NaN losses and coefficients;
  - qualitative improvement over trivial baselines on the toy dataset.

Deliverable:

- a test matrix that validates not only code correctness but also the intended methodological behavior.

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
- [ ] Create a methodology Beamer deck for internal discussions and paper-prep presentations.
  - target: `paper/methodology_beamer.tex`
  - keep equations, assumptions, baselines, and plots aligned with the roadmap and methodology notes.
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
3. migrate the agreed methodology into the main docs and paper notes;
4. define the paired-data schema, latent decomposition, and conditioning inputs;
5. formalize mismatch and the MFMC-SINDy estimator;
6. implement the smallest end-to-end staged pipeline;
7. align the test suite with the new backbone;
8. build the minimum baseline matrix;
9. redesign minimal examples and tutorial notebooks around the chosen backbone;
10. run ablations and comparisons;
11. write the paper around the finalized evidence.

## Recommended First Working Session

In the next pass, I recommend we do only the following:

- migrate [docs/methodology_shared_latent.md](/Users/filippozacchei/Library/CloudStorage/OneDrive-PolitecnicodiMilano/Documenti/Projects/2026_MFSINDY2/mfaesindy/docs/methodology_shared_latent.md:1) into the main methodology path;
- decide whether the default model has a private latent block or keeps only a shared core;
- define the paired CFD/experiment metadata contract;
- define the first SINDy library and latent derivative estimator;
- specify the exact MFMC regression moments and control-variate estimator;
- pin down the first baseline suite and evaluation metrics;
- decide which existing tests will be extended first;
- decide which minimal example becomes the first end-to-end smoke benchmark.

That will turn the methodology into an implementable research plan.

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
