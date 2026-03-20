# Multi Fidelity Latent Dynamics Identification

`mfaesindy` is a PyTorch- and PySINDy-based package for multi-fidelity latent dynamics identification.

## Package Layout

- `src/mfaesindy/config.py`: typed experiment and training configuration.
- `src/mfaesindy/data/`: trajectory data containers and loading-facing types.
- `src/mfaesindy/models/`: multi-fidelity autoencoder modules.
- `src/mfaesindy/mfmc.py`: MFMC-style estimators for latent regression moments.
- `src/mfaesindy/dynamics/`: sparse latent regression helpers built around PySINDy.
- `src/mfaesindy/losses.py`: reconstruction, alignment, and regularization losses.
- `src/mfaesindy/training/`: training scaffolding for the alternating optimization loop.
- `tests/`: initial unit tests for configuration and MFMC estimator shapes.

## Next Implementation Steps

1. Add proper partial weight-sharing in the autoencoder rather than the current placeholder split encoders/decoders.
2. Implement trajectory datasets and dataloaders for paired and unpaired LF/HF samples.
3. Build the alternating training loop that couples PyTorch updates with PySINDy/STLSQ coefficient updates.
4. Add experiment configuration files and reproducible training entry points.
