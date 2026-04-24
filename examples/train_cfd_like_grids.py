"""Train the multi-fidelity latent Neural ODE model on different-grid CFD-like data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mf_lnode import (
    ExperimentConfig,
    FidelityWeightScheduler,
    LatentNeuralODEModel,
    LossComposer,
    MultiFidelityScaler,
    Trainer,
    WindowedTrajectoryDataset,
    build_cfd_grid_multifidelity_dataset,
    collate_windowed_groups,
    seed_everything,
    split_group_dataset,
)
from mf_lnode.data.datasets import MultiFidelityTrajectoryDataset, MultiFidelityTrajectorySample
from mf_lnode.evaluation import (
    plot_architecture_breakdown,
    plot_cfd_dataset_overview,
    plot_cfd_fidelity_gap,
    plot_cfd_full_trajectory_rollout,
    plot_cfd_measurement_overview,
    plot_cfd_rollout_error_profile,
    plot_cfd_test_problem_summary,
    plot_cfd_window_rollout,
    plot_latent_diagnostics,
    plot_method_overview,
    plot_parameter_coverage,
    plot_test_metrics,
    plot_training_strategy_overview,
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    """Parse simple runtime arguments for the CFD-like grid experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="artifacts/cfd_like_grids")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args()


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _configure_example_defaults(config: ExperimentConfig, quick: bool) -> None:
    """Apply presentation-oriented defaults for the CFD-like example."""

    config.cfd_grid.low_grid_shape = (16, 16)
    config.cfd_grid.high_grid_shape = (32, 32)
    config.cfd_grid.trajectory_length = 72
    config.cfd_grid.time_end = 10.0
    config.data.window_size = 8
    config.data.rollout_horizon = 16
    config.model.window_size = config.data.window_size
    config.model.parameter_dim = config.cfd_grid.parameter_dim
    config.model.latent_dim = 12
    config.model.adapter_dim = 64
    config.model.encoder_hidden_dim = 128
    config.model.decoder_hidden_dim = 128
    config.model.dynamics_hidden_dim = 96
    config.model.hierarchical_decoder = False
    config.loss.discrepancy_weight = 0.0
    config.training.batch_size = 6
    config.training.transition_epochs = max(4, min(config.training.epochs - 1, 10))

    if quick:
        config.cfd_grid.trajectory_length = 48
        config.cfd_grid.time_end = 7.5
        config.data.window_size = 6
        config.data.rollout_horizon = 10
        config.model.window_size = config.data.window_size
        config.training.batch_size = 4
        config.training.transition_epochs = max(2, min(config.training.epochs - 1, 4))


def _select_representative_group(
    raw_dataset: MultiFidelityTrajectoryDataset,
    scaled_dataset: MultiFidelityTrajectoryDataset,
) -> tuple[int, MultiFidelityTrajectorySample, MultiFidelityTrajectorySample]:
    candidate_indices = [
        index
        for index, group in enumerate(raw_dataset.groups)
        if {"low", "high"}.issubset(group.trajectories)
    ]
    if candidate_indices:
        selected_index = candidate_indices[len(candidate_indices) // 2]
    elif len(raw_dataset) > 0:
        selected_index = len(raw_dataset) // 2
    else:
        raise RuntimeError("The test dataset is empty.")
    return selected_index, raw_dataset[selected_index], scaled_dataset[selected_index]


def _select_representative_batch(
    dataset: WindowedTrajectoryDataset,
    group_index: int,
) -> dict[str, object]:
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample.group_index == group_index and len(sample.windows) > 1:
            return collate_windowed_groups([sample])
    for index in range(len(dataset)):
        sample = dataset[index]
        if len(sample.windows) > 1:
            return collate_windowed_groups([sample])
    if len(dataset) == 0:
        raise RuntimeError("The test windowed dataset is empty.")
    return collate_windowed_groups([dataset[0]])


def _grid_shape_from_sample(sample: Any) -> tuple[int, int]:
    return int(sample.metadata["grid_height"]), int(sample.metadata["grid_width"])


def _build_window_rollout_bundle(
    batch_device: dict[str, Any],
    outputs: dict[str, Any],
    raw_group: MultiFidelityTrajectorySample,
    scaler: MultiFidelityScaler,
) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    for fidelity, output in outputs["per_fidelity"].items():
        sample = raw_group.trajectories[fidelity]
        grid_height, grid_width = _grid_shape_from_sample(sample)
        predictions = scaler.inverse_transform(fidelity, output["predictions"][0].detach().cpu())
        targets = scaler.inverse_transform(fidelity, output["targets"][0].detach().cpu())
        context_targets = scaler.inverse_transform(
            fidelity,
            batch_device["windows"][fidelity]["context_observations"][0].detach().cpu(),
        )
        bundle[fidelity] = {
            "grid_height": grid_height,
            "grid_width": grid_width,
            "context_times": batch_device["windows"][fidelity]["context_times"][0].detach().cpu(),
            "context_targets": context_targets,
            "target_times": batch_device["windows"][fidelity]["target_times"][0].detach().cpu(),
            "targets": targets,
            "predictions": predictions,
        }
    return bundle


def _build_full_rollout_bundle(
    model: LatentNeuralODEModel,
    raw_group: MultiFidelityTrajectorySample,
    scaled_group: MultiFidelityTrajectorySample,
    scaler: MultiFidelityScaler,
    window_size: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for fidelity, scaled_sample in scaled_group.trajectories.items():
            raw_sample = raw_group.trajectories[fidelity]
            context_times = scaled_sample.times[:window_size].unsqueeze(0).to(device)
            context_observations = scaled_sample.observations[:window_size].unsqueeze(0).to(device)
            parameters = scaled_sample.parameter.unsqueeze(0).to(device)
            rollout_times = scaled_sample.times[window_size - 1 :].unsqueeze(0).to(device)

            z0 = model.encode(
                fidelity=fidelity,
                context_times=context_times,
                context_observations=context_observations,
                parameters=parameters,
            )
            latent_trajectory = model.solver.integrate(
                model.dynamics,
                z0,
                rollout_times,
                parameters,
            )
            decoded = model.decode(fidelity, latent_trajectory, parameters)
            prediction = scaler.inverse_transform(
                fidelity,
                decoded["prediction"][0].detach().cpu(),
            )
            grid_height, grid_width = _grid_shape_from_sample(raw_sample)

            bundle[fidelity] = {
                "pairing_id": raw_group.pairing_id or f"group-{fidelity}",
                "parameter": raw_sample.parameter.detach().cpu(),
                "grid_height": grid_height,
                "grid_width": grid_width,
                "full_times": raw_sample.times.detach().cpu(),
                "full_targets": raw_sample.observations.detach().cpu(),
                "context_times": raw_sample.times[:window_size].detach().cpu(),
                "context_targets": raw_sample.observations[:window_size].detach().cpu(),
                "rollout_times": raw_sample.times[window_size - 1 :].detach().cpu(),
                "rollout_targets": raw_sample.observations[window_size - 1 :].detach().cpu(),
                "prediction": prediction,
                "latent_trajectory": latent_trajectory[0].detach().cpu(),
                "z0": z0[0].detach().cpu(),
            }
    return bundle


def _write_artifact_index(
    output_dir: Path,
    metrics_path: Path,
    figure_specs: list[dict[str, str | Path]],
) -> Path:
    index_path = output_dir / "artifact_index.md"
    lines = [
        "# CFD-Like Grid Example Artifacts",
        "",
        f"- Metrics JSON: `{metrics_path.name}`",
        "",
        "## Figures",
    ]
    section_order = [
        "Problem and Method",
        "Data and Training",
        "Results",
    ]
    for section in section_order:
        section_specs = [spec for spec in figure_specs if spec["section"] == section]
        if not section_specs:
            continue
        lines.append("")
        lines.append(f"### {section}")
        for spec in section_specs:
            figure_path = spec["path"]
            title = spec["title"]
            description = spec["description"]
            lines.append(f"- `{figure_path.name}`: **{title}**. {description}")
    lines.append("")
    index_path.write_text("\n".join(lines))
    return index_path


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    config.seed = args.seed
    config.cfd_grid.seed = args.seed
    config.training.epochs = args.epochs
    config.training.device = args.device
    config.training.output_dir = args.output_dir
    config.training.tensorboard = args.tensorboard
    _configure_example_defaults(config, quick=args.quick)

    seed_everything(config.seed)

    raw_groups = build_cfd_grid_multifidelity_dataset(config.cfd_grid)
    train_raw, val_raw, test_raw = split_group_dataset(
        raw_groups,
        train_fraction=config.data.train_fraction,
        val_fraction=config.data.val_fraction,
        seed=config.seed,
    )
    scaler = MultiFidelityScaler.fit_from_groups(train_raw.groups)

    train_dataset = MultiFidelityTrajectoryDataset(scaler.transform_groups(train_raw.groups))
    val_dataset = MultiFidelityTrajectoryDataset(scaler.transform_groups(val_raw.groups))
    test_dataset = MultiFidelityTrajectoryDataset(scaler.transform_groups(test_raw.groups))

    config.model.fidelity_dims = train_dataset.fidelity_dims

    train_windows = WindowedTrajectoryDataset(
        group_dataset=train_dataset,
        window_size=config.data.window_size,
        rollout_horizon=config.data.rollout_horizon,
        future_context_shift=config.data.future_context_shift,
    )
    val_windows = WindowedTrajectoryDataset(
        group_dataset=val_dataset,
        window_size=config.data.window_size,
        rollout_horizon=config.data.rollout_horizon,
        future_context_shift=config.data.future_context_shift,
    )
    test_windows = WindowedTrajectoryDataset(
        group_dataset=test_dataset,
        window_size=config.data.window_size,
        rollout_horizon=config.data.rollout_horizon,
        future_context_shift=config.data.future_context_shift,
    )

    model = LatentNeuralODEModel.from_config(config.model)
    loss_composer = LossComposer(config.loss)
    scheduler = FidelityWeightScheduler(
        initial_weights=config.loss.fidelity_weights_start,
        final_weights=config.loss.fidelity_weights_end,
        transition_epochs=config.training.transition_epochs,
        warmup_epochs=config.training.warmup_epochs,
    )
    trainer = Trainer(
        config=config.training,
        model=model,
        loss_composer=loss_composer,
        fidelity_weight_scheduler=scheduler,
    )

    history = trainer.fit(train_windows, val_windows)
    final_weights = scheduler.weights_at(config.training.epochs)
    test_metrics = trainer.evaluate(test_windows, fidelity_weights=final_weights)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2))
    fidelity_weight_history = [scheduler.weights_at(epoch) for epoch in range(len(history))]

    representative_group_index, representative_raw_group, representative_scaled_group = _select_representative_group(
        test_raw,
        test_dataset,
    )
    preview_batch = _select_representative_batch(test_windows, representative_group_index)
    preview_batch_device = _move_to_device(preview_batch, torch.device(config.training.device))

    model.eval()
    with torch.no_grad():
        preview_outputs = model.rollout(preview_batch_device)

    window_bundle = _build_window_rollout_bundle(
        batch_device=preview_batch_device,
        outputs=preview_outputs,
        raw_group=representative_raw_group,
        scaler=scaler,
    )
    full_rollout_bundle = _build_full_rollout_bundle(
        model=model,
        raw_group=representative_raw_group,
        scaled_group=representative_scaled_group,
        scaler=scaler,
        window_size=config.data.window_size,
        device=torch.device(config.training.device),
    )

    figure_specs = [
        {
            "section": "Problem and Method",
            "title": "Test problem",
            "path": plot_cfd_test_problem_summary(
                test_raw.groups,
                output_dir / "01_test_problem.png",
            ),
            "description": "CFD-like different-grid problem statement, representative low/high snapshots, and global field amplitude.",
        },
        {
            "section": "Problem and Method",
            "title": "Measurement definition",
            "path": plot_cfd_measurement_overview(
                test_raw.groups,
                output_dir / "02_measurement_overview.png",
            ),
            "description": "How the field is measured on coarse and fine grids, how snapshots are vectorized, and how the cross-grid bias appears.",
        },
        {
            "section": "Problem and Method",
            "title": "Method overview",
            "path": plot_method_overview(
                config.model,
                output_dir / "03_method_overview.png",
            ),
            "description": "High-level dataflow from multi-fidelity windows to shared latent dynamics and fidelity-specific outputs.",
        },
        {
            "section": "Problem and Method",
            "title": "Architecture breakdown",
            "path": plot_architecture_breakdown(
                config.model,
                output_dir / "04_architecture_breakdown.png",
            ),
            "description": "Module-level view of the encoder, shared latent space, latent Neural ODE, decoder, and different-grid output heads.",
        },
        {
            "section": "Problem and Method",
            "title": "Training strategy",
            "path": plot_training_strategy_overview(
                config,
                history,
                fidelity_weight_history,
                output_dir / "05_training_strategy.png",
            ),
            "description": "Objective composition, curriculum schedule, windowing setup, and optimization protocol.",
        },
        {
            "section": "Data and Training",
            "title": "Dataset overview",
            "path": plot_cfd_dataset_overview(
                test_raw.groups,
                output_dir / "06_dataset_overview.png",
            ),
            "description": "Representative field snapshots across held-out grouped samples at different times and fidelities.",
        },
        {
            "section": "Data and Training",
            "title": "Parameter coverage",
            "path": plot_parameter_coverage(
                {
                    "train": train_raw.groups,
                    "val": val_raw.groups,
                    "test": test_raw.groups,
                },
                output_dir / "07_parameter_coverage.png",
            ),
            "description": "Parameter-space coverage and split composition for the CFD-like experiment.",
        },
        {
            "section": "Data and Training",
            "title": "Training dashboard",
            "path": plot_training_history(
                history,
                output_dir / "08_training_dashboard.png",
                fidelity_weight_history=fidelity_weight_history,
            ),
            "description": "Optimization dashboard with loss breakdowns, per-fidelity MSE, and curriculum weights.",
        },
        {
            "section": "Results",
            "title": "Window rollout",
            "path": plot_cfd_window_rollout(
                window_bundle,
                output_dir / "09_window_rollout.png",
            ),
            "description": "Short-window field rollout showing context, target, prediction, error map, and short-window RMSE.",
        },
        {
            "section": "Results",
            "title": "Full-trajectory rollout",
            "path": plot_cfd_full_trajectory_rollout(
                full_rollout_bundle,
                output_dir / "10_full_trajectory_rollout.png",
            ),
            "description": "Long-horizon rollout snapshots at multiple times for both coarse and fine grids.",
        },
        {
            "section": "Results",
            "title": "Rollout error profile",
            "path": plot_cfd_rollout_error_profile(
                full_rollout_bundle,
                output_dir / "11_rollout_error_profile.png",
            ),
            "description": "Time-resolved field RMSE, relative L2, and amplitude tracking for each fidelity.",
        },
        {
            "section": "Results",
            "title": "Fidelity gap analysis",
            "path": plot_cfd_fidelity_gap(
                full_rollout_bundle,
                output_dir / "12_fidelity_gap.png",
            ),
            "description": "Cross-grid target and predicted fidelity gaps after upsampling low fidelity to the high-fidelity grid.",
        },
        {
            "section": "Results",
            "title": "Latent diagnostics",
            "path": plot_latent_diagnostics(
                full_rollout_bundle,
                output_path=output_dir / "13_latent_diagnostics.png",
            ),
            "description": "Latent phase portrait, coordinate traces, norms, and cross-fidelity latent distance.",
        },
        {
            "section": "Results",
            "title": "Test metrics",
            "path": plot_test_metrics(
                metrics=test_metrics,
                output_path=output_dir / "14_test_metrics.png",
            ),
            "description": "Held-out metric summary across fidelities and global training objectives.",
        },
    ]
    artifact_index = _write_artifact_index(output_dir, metrics_path, figure_specs)

    print("CFD-like grid training complete.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved artifact index to: {artifact_index}")
    print("Saved plots:")
    for spec in figure_specs:
        print(f"  {spec['path']} :: {spec['title']} :: {spec['description']}")
    print("Per-fidelity rollout preview:")
    for fidelity, fidelity_output in preview_outputs["per_fidelity"].items():
        print(
            f"  {fidelity}: predictions={tuple(fidelity_output['predictions'].shape)} "
            f"targets={tuple(fidelity_output['targets'].shape)}"
        )
    print("Test metrics:")
    for key, value in sorted(test_metrics.items()):
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
