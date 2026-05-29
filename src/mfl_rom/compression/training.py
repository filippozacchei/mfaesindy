"""
Training utilities for snapshot-based multifidelity autoencoders.

Overview
--------
This module provides reusable training primitives while deliberately leaving
the *total* objective to the caller.

The main idea is to expose four SGD-friendly data streams:

1. ``lf``
   Independent LF snapshots for reconstruction training.

2. ``hf``
   Independent HF snapshots for reconstruction training.

3. ``mf_state``
   Paired LF/HF snapshots, aligned in time, for state-level latent alignment.

4. ``mf_sindy``
   Paired LF/HF temporal transitions for SINDy-style residual regularization.

Why this split?
---------------
- Reconstruction should see snapshots as fully independent samples.
- Latent state alignment only needs paired states, not whole trajectories.
- SINDy-style regularization needs local temporal information, which is why
  it uses transition pairs ``(x_t, x_{t+1}, dt)`` rather than isolated
  snapshots.

Expected model API
------------------
The autoencoder model is expected to implement:

    encode(x, fidelity) -> torch.Tensor
    decode(z, fidelity) -> torch.Tensor
    reconstruct(x, fidelity) -> torch.Tensor

If a future model includes shared and private latent coordinates, it may also
implement:

    split_latent(z) -> tuple[z_shared, z_private]

Expected data API
-----------------
Training uses `TrainingTrajectoryDataset`, which contains:
- mf_samples: list[MFTrajectory]
- lf_samples: list[Trajectory]
- hf_samples: list[Trajectory]

How to use
----------
Define one external dispatcher loss function.

Example: reconstruction on LF/HF, latent state alignment on paired MF states

    def loss_fn(model, batch_kind, batch, device):
        if batch_kind == "lf":
            rec = snapshot_reconstruction_term(model, batch, "LF", device)
            return rec, {
                "loss": float(rec.detach().cpu()),
                "reconstruction": float(rec.detach().cpu()),
            }

        if batch_kind == "hf":
            rec = snapshot_reconstruction_term(model, batch, "HF", device)
            return rec, {
                "loss": float(rec.detach().cpu()),
                "reconstruction": float(rec.detach().cpu()),
            }

        if batch_kind == "mf_state":
            align = state_alignment_term(model, batch, device)
            return align, {
                "loss": float(align.detach().cpu()),
                "state_alignment": float(align.detach().cpu()),
            }

        if batch_kind == "mf_sindy":
            return torch.zeros((), device=device), {"loss": 0.0}

        raise ValueError(f"Unknown batch kind: {batch_kind}")

Example: add a shared SINDy residual term

    coefficients = nn.Parameter(torch.zeros(num_library_terms, latent_dim))

    def library_fn(z):
        ones = torch.ones((z.shape[0], 1), device=z.device, dtype=z.dtype)
        z1 = z[:, :1]
        z2 = z[:, 1:2]
        return torch.cat([ones, z1, z2, z1 * z1, z1 * z2, z2 * z2], dim=1)

    def loss_fn(model, batch_kind, batch, device):
        if batch_kind == "lf":
            rec = snapshot_reconstruction_term(model, batch, "LF", device)
            return rec, {"loss": float(rec.detach().cpu())}

        if batch_kind == "hf":
            rec = snapshot_reconstruction_term(model, batch, "HF", device)
            return rec, {"loss": float(rec.detach().cpu())}

        if batch_kind == "mf_state":
            align = state_alignment_term(model, batch, device)
            loss = 0.1 * align
            return loss, {
                "loss": float(loss.detach().cpu()),
                "state_alignment": float(align.detach().cpu()),
            }

        if batch_kind == "mf_sindy":
            sindy = sindy_residual_term(
                model,
                batch,
                device,
                coefficients=coefficients,
                library_fn=library_fn,
            )
            loss = 0.01 * sindy
            return loss, {
                "loss": float(loss.detach().cpu()),
                "sindy_residual": float(sindy.detach().cpu()),
            }

        raise ValueError(f"Unknown batch kind: {batch_kind}")
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from mfl_rom.data import MFTrajectory, TrainingTrajectoryDataset, Trajectory

BatchKind: TypeAlias = Literal["lf", "hf", "mf_state", "mf_sindy"]
SnapshotBatch: TypeAlias = np.ndarray
LossResult: TypeAlias = tuple[torch.Tensor, dict[str, float]]
LibraryFn: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
LossFn: TypeAlias = Callable[
    [nn.Module, BatchKind, object, torch.device | str],
    LossResult,
]


@dataclass(slots=True)
class BatchSizes:
    """Mini-batch sizes for the four SGD streams."""

    lf: int = 32
    hf: int = 32
    mf_state: int = 32
    mf_sindy: int = 32


@dataclass(slots=True)
class PairedStateBatch:
    """
    Batch of paired LF/HF snapshots aligned at common physical times.

    Shapes
    ------
    lf: (batch_size, *lf_state_shape)
    hf: (batch_size, *hf_state_shape)
    """

    lf: np.ndarray
    hf: np.ndarray


@dataclass(slots=True)
class TemporalPairBatch:
    """
    Batch of paired LF/HF temporal transitions.

    Each row corresponds to one local transition:
    - LF: (x_t, x_{t+1}, dt)
    - HF: (x_t, x_{t+1}, dt)

    Shapes
    ------
    lf_x0: (batch_size, *lf_state_shape)
    lf_x1: (batch_size, *lf_state_shape)
    hf_x0: (batch_size, *hf_state_shape)
    hf_x1: (batch_size, *hf_state_shape)
    dt: (batch_size,)
    """

    lf_x0: np.ndarray
    lf_x1: np.ndarray
    hf_x0: np.ndarray
    hf_x1: np.ndarray
    dt: np.ndarray


def snapshot_batch_to_tensor(
    batch: SnapshotBatch,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Move one snapshot batch to the requested device."""
    return torch.as_tensor(batch, dtype=dtype, device=device)


def trajectory_to_tensor(
    trajectory: Trajectory,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Move one trajectory to the requested device."""
    return torch.as_tensor(trajectory.states, dtype=dtype, device=device)


def paired_state_batch_to_tensors(
    batch: PairedStateBatch,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one paired-state batch to the requested device."""
    lf = torch.as_tensor(batch.lf, dtype=dtype, device=device)
    hf = torch.as_tensor(batch.hf, dtype=dtype, device=device)
    return lf, hf


def temporal_pair_batch_to_tensors(
    batch: TemporalPairBatch,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Move one temporal-pair batch to the requested device."""
    lf_x0 = torch.as_tensor(batch.lf_x0, dtype=dtype, device=device)
    lf_x1 = torch.as_tensor(batch.lf_x1, dtype=dtype, device=device)
    hf_x0 = torch.as_tensor(batch.hf_x0, dtype=dtype, device=device)
    hf_x1 = torch.as_tensor(batch.hf_x1, dtype=dtype, device=device)
    dt = torch.as_tensor(batch.dt, dtype=dtype, device=device)
    return lf_x0, lf_x1, hf_x0, hf_x1, dt


def extract_shared_latent(
    model: nn.Module,
    latent: torch.Tensor,
) -> torch.Tensor:
    """
    Extract shared latent coordinates.

    For shared-only models, the full latent tensor is returned.
    For shared+private models, the model may expose `split_latent`.
    """
    if hasattr(model, "split_latent"):
        z_shared, _ = model.split_latent(latent)
        return z_shared
    return latent


def finite_difference_latent(
    z0: torch.Tensor,
    z1: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a latent time derivative from batched transitions.

    Parameters
    ----------
    z0, z1
        Latent states with shape `(batch_size, latent_dim)`.
    dt
        Time increments with shape `(batch_size,)`.
    """
    if z0.shape != z1.shape:
        raise ValueError("z0 and z1 must have the same shape.")
    if dt.ndim != 1 or dt.shape[0] != z0.shape[0]:
        raise ValueError("dt must have shape (batch_size,).")
    return (z1 - z0) / dt.unsqueeze(-1)


def snapshot_reconstruction_term(
    model: nn.Module,
    batch: SnapshotBatch,
    fidelity: str,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Reconstruction loss on an independent snapshot batch.

    Parameters
    ----------
    batch
        Snapshot batch with shape `(batch_size, ...)`.
    fidelity
        Fidelity label used by the model.
    """
    x = snapshot_batch_to_tensor(batch, device=device)
    x_hat = model.reconstruct(x, fidelity)
    return F.mse_loss(x_hat, x)


def state_alignment_term(
    model: nn.Module,
    batch: PairedStateBatch,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Shared-latent alignment loss on paired LF/HF snapshots.

    This is the SGD-friendly state-level MF alignment term.
    """
    x_lf, x_hf = paired_state_batch_to_tensors(batch, device=device)

    z_lf = extract_shared_latent(model, model.encode(x_lf, "LF"))
    z_hf = extract_shared_latent(model, model.encode(x_hf, "HF"))

    return F.mse_loss(z_lf, z_hf)


def sindy_residual_term(
    model: nn.Module,
    batch: TemporalPairBatch,
    device: torch.device | str,
    coefficients: torch.Tensor,
    library_fn: LibraryFn,
) -> torch.Tensor:
    """
    Shared SINDy residual on paired LF/HF temporal transitions.

    This is the SGD-friendly dynamics-aware term. For each batch element:
    1. encode LF and HF states at times t and t+dt
    2. extract shared latent coordinates
    3. estimate local latent derivatives by finite differences
    4. evaluate the residual `z_dot - Theta(z) Xi`

    The same coefficient matrix `Xi` is used for both LF and HF.
    """
    lf_x0, lf_x1, hf_x0, hf_x1, dt = temporal_pair_batch_to_tensors(
        batch,
        device=device,
    )

    z_lf_0 = extract_shared_latent(model, model.encode(lf_x0, "LF"))
    z_lf_1 = extract_shared_latent(model, model.encode(lf_x1, "LF"))
    z_hf_0 = extract_shared_latent(model, model.encode(hf_x0, "HF"))
    z_hf_1 = extract_shared_latent(model, model.encode(hf_x1, "HF"))

    zdot_lf = finite_difference_latent(z_lf_0, z_lf_1, dt)
    zdot_hf = finite_difference_latent(z_hf_0, z_hf_1, dt)

    theta_lf = library_fn(z_lf_0)
    theta_hf = library_fn(z_hf_0)

    if theta_lf.ndim != 2 or theta_hf.ndim != 2:
        raise ValueError("library_fn must return a 2D tensor.")
    if theta_lf.shape[1] != coefficients.shape[0]:
        raise ValueError(
            "library output dimension must match coefficients.shape[0]."
        )
    if zdot_lf.shape[1] != coefficients.shape[1]:
        raise ValueError(
            "coefficients.shape[1] must match the shared latent dimension."
        )

    residual_lf = zdot_lf - theta_lf @ coefficients
    residual_hf = zdot_hf - theta_hf @ coefficients

    return 0.5 * (
        torch.mean(residual_lf.pow(2)) + torch.mean(residual_hf.pow(2))
    )


def collect_fidelity_snapshots(
    dataset: TrainingTrajectoryDataset,
    fidelity: Literal["LF", "HF"],
) -> list[np.ndarray]:
    """
    Collect independent snapshots for snapshot-level SGD.

    Snapshots from paired MF samples are included together with snapshots from
    the corresponding single-fidelity-only pool.
    """
    snapshots: list[np.ndarray] = []

    if fidelity == "LF":
        for sample in dataset.mf_samples:
            snapshots.extend(sample.lf_trajectory.states)
        for trajectory in dataset.lf_samples:
            snapshots.extend(trajectory.states)
        return snapshots

    if fidelity == "HF":
        for sample in dataset.mf_samples:
            snapshots.extend(sample.hf_trajectory.states)
        for trajectory in dataset.hf_samples:
            snapshots.extend(trajectory.states)
        return snapshots

    raise ValueError("fidelity must be either 'LF' or 'HF'.")


def resample_state_sequence(
    time: np.ndarray,
    states: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    """
    Resample a state trajectory onto a target time grid using linear
    interpolation.

    This operates on arbitrary state tensors by flattening the non-time axes,
    interpolating each feature independently, and reshaping back.
    """
    time = np.asarray(time, dtype=float)
    states = np.asarray(states)
    target_time = np.asarray(target_time, dtype=float)

    if time.ndim != 1 or target_time.ndim != 1:
        raise ValueError("time and target_time must be one-dimensional.")
    if states.shape[0] != time.shape[0]:
        raise ValueError("states and time must have matching first dimension.")

    flat_states = states.reshape(states.shape[0], -1)
    resampled_flat = np.empty(
        (target_time.shape[0], flat_states.shape[1]),
        dtype=states.dtype,
    )

    for feature_index in range(flat_states.shape[1]):
        resampled_flat[:, feature_index] = np.interp(
            target_time,
            time,
            flat_states[:, feature_index],
        )

    return resampled_flat.reshape(target_time.shape[0], *states.shape[1:])


def paired_common_time_grid(
    sample: MFTrajectory,
    target_steps: int | None = None,
) -> np.ndarray:
    """
    Build a common physical time grid over the overlap of one LF/HF pair.
    """
    t0 = max(sample.lf_trajectory.time[0], sample.hf_trajectory.time[0])
    tf = min(sample.lf_trajectory.time[-1], sample.hf_trajectory.time[-1])
    if tf <= t0:
        raise ValueError("LF/HF trajectories must have overlapping time spans.")

    if target_steps is None:
        target_steps = max(
            sample.lf_trajectory.num_steps,
            sample.hf_trajectory.num_steps,
        )
    if target_steps < 2:
        raise ValueError("target_steps must be at least 2.")

    return np.linspace(t0, tf, target_steps)


def collect_paired_state_examples(
    dataset: TrainingTrajectoryDataset,
    target_steps: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Collect aligned LF/HF state pairs for snapshot-level MF alignment.
    """
    examples: list[tuple[np.ndarray, np.ndarray]] = []

    for sample in dataset.mf_samples:
        common_time = paired_common_time_grid(sample, target_steps=target_steps)
        lf_states = resample_state_sequence(
            sample.lf_trajectory.time,
            sample.lf_trajectory.states,
            common_time,
        )
        hf_states = resample_state_sequence(
            sample.hf_trajectory.time,
            sample.hf_trajectory.states,
            common_time,
        )
        examples.extend(zip(lf_states, hf_states, strict=True))

    return examples


def collect_paired_temporal_examples(
    dataset: TrainingTrajectoryDataset,
    target_steps: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Collect paired LF/HF temporal transitions for SGD-compatible SINDy losses.
    """
    examples: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
    ] = []

    for sample in dataset.mf_samples:
        common_time = paired_common_time_grid(sample, target_steps=target_steps)
        lf_states = resample_state_sequence(
            sample.lf_trajectory.time,
            sample.lf_trajectory.states,
            common_time,
        )
        hf_states = resample_state_sequence(
            sample.hf_trajectory.time,
            sample.hf_trajectory.states,
            common_time,
        )

        dt = float(common_time[1] - common_time[0])
        for index in range(common_time.shape[0] - 1):
            examples.append(
                (
                    lf_states[index],
                    lf_states[index + 1],
                    hf_states[index],
                    hf_states[index + 1],
                    dt,
                )
            )

    return examples


def iterate_snapshot_batches(
    snapshots: list[np.ndarray],
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None = None,
) -> Iterator[SnapshotBatch]:
    """Yield independent snapshot batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(snapshots) == 0:
        return

    indices = np.arange(len(snapshots))
    if shuffle:
        rng = np.random.default_rng() if rng is None else rng
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield np.stack([snapshots[index] for index in batch_indices], axis=0)


def iterate_paired_state_batches(
    examples: list[tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None = None,
) -> Iterator[PairedStateBatch]:
    """Yield paired LF/HF state batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(examples) == 0:
        return

    indices = np.arange(len(examples))
    if shuffle:
        rng = np.random.default_rng() if rng is None else rng
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        lf = np.stack([examples[index][0] for index in batch_indices], axis=0)
        hf = np.stack([examples[index][1] for index in batch_indices], axis=0)
        yield PairedStateBatch(lf=lf, hf=hf)


def iterate_paired_temporal_batches(
    examples: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
    ],
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None = None,
) -> Iterator[TemporalPairBatch]:
    """Yield paired LF/HF temporal-transition batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(examples) == 0:
        return

    indices = np.arange(len(examples))
    if shuffle:
        rng = np.random.default_rng() if rng is None else rng
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        lf_x0 = np.stack(
            [examples[index][0] for index in batch_indices],
            axis=0,
        )
        lf_x1 = np.stack(
            [examples[index][1] for index in batch_indices],
            axis=0,
        )
        hf_x0 = np.stack(
            [examples[index][2] for index in batch_indices],
            axis=0,
        )
        hf_x1 = np.stack(
            [examples[index][3] for index in batch_indices],
            axis=0,
        )
        dt = np.asarray([examples[index][4] for index in batch_indices])
        yield TemporalPairBatch(
            lf_x0=lf_x0,
            lf_x1=lf_x1,
            hf_x0=hf_x0,
            hf_x1=hf_x1,
            dt=dt,
        )


def iterate_training_batches(
    dataset: TrainingTrajectoryDataset,
    batch_sizes: BatchSizes,
    *,
    shuffle: bool,
    rng: np.random.Generator | None = None,
    mf_target_steps: int | None = None,
) -> Iterator[tuple[BatchKind, object]]:
    """
    Yield all SGD streams used by the training loop.

    Current policy:
    1. LF snapshot batches
    2. HF snapshot batches
    3. MF paired-state batches
    4. MF paired-temporal batches
    """
    lf_snapshots = collect_fidelity_snapshots(dataset, "LF")
    hf_snapshots = collect_fidelity_snapshots(dataset, "HF")
    paired_states = collect_paired_state_examples(
        dataset,
        target_steps=mf_target_steps,
    )
    paired_temporal = collect_paired_temporal_examples(
        dataset,
        target_steps=mf_target_steps,
    )

    yield from (
        ("lf", batch)
        for batch in iterate_snapshot_batches(
            lf_snapshots,
            batch_sizes.lf,
            shuffle=shuffle,
            rng=rng,
        )
    )
    yield from (
        ("hf", batch)
        for batch in iterate_snapshot_batches(
            hf_snapshots,
            batch_sizes.hf,
            shuffle=shuffle,
            rng=rng,
        )
    )
    yield from (
        ("mf_state", batch)
        for batch in iterate_paired_state_batches(
            paired_states,
            batch_sizes.mf_state,
            shuffle=shuffle,
            rng=rng,
        )
    )
    yield from (
        ("mf_sindy", batch)
        for batch in iterate_paired_temporal_batches(
            paired_temporal,
            batch_sizes.mf_sindy,
            shuffle=shuffle,
            rng=rng,
        )
    )


def _accumulate_metrics(
    metric_sums: dict[str, float],
    metric_counts: dict[str, int],
    prefix: str,
    metrics: dict[str, float],
) -> None:
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}"
        metric_sums[full_key] = metric_sums.get(full_key, 0.0) + float(value)
        metric_counts[full_key] = metric_counts.get(full_key, 0) + 1


def _finalize_metrics(
    metric_sums: dict[str, float],
    metric_counts: dict[str, int],
) -> dict[str, float]:
    return {
        key: metric_sums[key] / metric_counts[key]
        for key in metric_sums
    }


def train_epoch(
    model: nn.Module,
    dataset: TrainingTrajectoryDataset,
    optimizer: Optimizer,
    device: torch.device | str,
    loss_fn: LossFn,
    batch_sizes: BatchSizes,
    *,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
    mf_target_steps: int | None = None,
) -> dict[str, float]:
    """
    Run one SGD epoch over LF snapshots, HF snapshots, MF paired states, and
    MF paired temporal transitions.
    """
    model.train()

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    for batch_kind, batch in iterate_training_batches(
        dataset,
        batch_sizes,
        shuffle=shuffle,
        rng=rng,
        mf_target_steps=mf_target_steps,
    ):
        optimizer.zero_grad()

        loss, metrics = loss_fn(model, batch_kind, batch, device)
        loss.backward()
        optimizer.step()

        _accumulate_metrics(metric_sums, metric_counts, batch_kind, metrics)

    return _finalize_metrics(metric_sums, metric_counts)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataset: TrainingTrajectoryDataset,
    device: torch.device | str,
    loss_fn: LossFn,
    batch_sizes: BatchSizes,
    *,
    mf_target_steps: int | None = None,
) -> dict[str, float]:
    """
    Evaluate one epoch without optimization.
    """
    model.eval()

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    for batch_kind, batch in iterate_training_batches(
        dataset,
        batch_sizes,
        shuffle=False,
        rng=None,
        mf_target_steps=mf_target_steps,
    ):
        _, metrics = loss_fn(model, batch_kind, batch, device)
        _accumulate_metrics(metric_sums, metric_counts, batch_kind, metrics)

    return _finalize_metrics(metric_sums, metric_counts)


def fit(
    model: nn.Module,
    dataset: TrainingTrajectoryDataset,
    optimizer: Optimizer,
    device: torch.device | str,
    loss_fn: LossFn,
    num_epochs: int,
    batch_sizes: BatchSizes,
    *,
    val_dataset: TrainingTrajectoryDataset | None = None,
    val_loss_fn: LossFn | None = None,
    rng: np.random.Generator | None = None,
    mf_target_steps: int | None = None,
    verbose: bool = True,
    scheduler: LRScheduler | None = None,
    print_every: int = 1,
) -> list[dict[str, float]]:
    """
    Minimal training driver for fully SGD-compatible multifidelity training.
    """
    history: list[dict[str, float]] = []
    val_loss_fn = loss_fn if val_loss_fn is None else val_loss_fn

    for epoch in tqdm(range(num_epochs), disable=not verbose):
        train_metrics = train_epoch(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            batch_sizes=batch_sizes,
            shuffle=True,
            rng=rng,
            mf_target_steps=mf_target_steps,
        )

        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
        }

        if val_dataset is not None:
            val_metrics = evaluate_epoch(
                model=model,
                dataset=val_dataset,
                device=device,
                loss_fn=val_loss_fn,
                batch_sizes=batch_sizes,
                mf_target_steps=mf_target_steps,
            )
            record.update(
                {f"val_{key}": value for key, value in val_metrics.items()}
            )
        
        record["lr"] = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step()

        history.append(record)

        if verbose and (
            (epoch + 1) % print_every == 0
            or epoch == 0
            or epoch + 1 == num_epochs
        ):
            summary_keys = [
                key for key in record
                if key != "epoch" and key.endswith("/loss")
            ]
            if not summary_keys:
                summary_keys = [key for key in record if key != "epoch"]

            summary = " | ".join(
                f"{key}={record[key]:.4e}"
                for key in summary_keys
                if isinstance(record[key], (int, float))
            )
            print(f"Epoch {epoch + 1:4d}/{num_epochs}: {summary}")

    return history
