from __future__ import annotations

from pathlib import Path

import torch

from mf_lnode import LatentNeuralODEModel, collate_windowed_groups, split_group_dataset
from mf_lnode.evaluation.plots import (
    plot_architecture_breakdown,
    plot_cfd_dataset_overview,
    plot_cfd_fidelity_gap,
    plot_cfd_full_trajectory_rollout,
    plot_cfd_measurement_overview,
    plot_cfd_rollout_error_profile,
    plot_cfd_test_problem_summary,
    plot_cfd_window_rollout,
    plot_method_overview,
)


def test_cfd_plotting_utilities_emit_files(
    cfd_groups,
    windowed_cfd_datasets,
    scaled_cfd_datasets,
    tiny_cfd_config,
    tmp_path,
):
    model = LatentNeuralODEModel.from_config(tiny_cfd_config.model)
    batch = collate_windowed_groups([windowed_cfd_datasets["test"][0]])
    outputs = model(batch)
    train_raw, val_raw, test_raw = split_group_dataset(cfd_groups, seed=tiny_cfd_config.seed)

    representative_group = next(group for group in cfd_groups if {"low", "high"}.issubset(group.trajectories))
    scaler = scaled_cfd_datasets["scaler"]
    window_bundle = {}
    for fidelity, output in outputs["per_fidelity"].items():
        sample = representative_group.trajectories[fidelity]
        window_bundle[fidelity] = {
            "grid_height": int(sample.metadata["grid_height"]),
            "grid_width": int(sample.metadata["grid_width"]),
            "context_times": batch["windows"][fidelity]["context_times"][0],
            "context_targets": scaler.inverse_transform(
                fidelity,
                batch["windows"][fidelity]["context_observations"][0],
            ),
            "target_times": batch["windows"][fidelity]["target_times"][0],
            "targets": scaler.inverse_transform(fidelity, output["targets"][0]),
            "predictions": scaler.inverse_transform(fidelity, output["predictions"][0]),
        }

    full_rollout = {}
    for fidelity, sample in representative_group.trajectories.items():
        grid_height = int(sample.metadata["grid_height"])
        grid_width = int(sample.metadata["grid_width"])
        rollout_targets = sample.observations[3:]
        full_rollout[fidelity] = {
            "pairing_id": representative_group.pairing_id,
            "parameter": sample.parameter,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "full_times": sample.times,
            "full_targets": sample.observations,
            "context_times": sample.times[:4],
            "context_targets": sample.observations[:4],
            "rollout_times": sample.times[3:],
            "rollout_targets": rollout_targets,
            "prediction": rollout_targets.clone(),
            "latent_trajectory": torch.randn(rollout_targets.shape[0], tiny_cfd_config.model.latent_dim),
            "z0": torch.randn(tiny_cfd_config.model.latent_dim),
        }

    paths = [
        plot_cfd_test_problem_summary(cfd_groups, tmp_path / "cfd_test_problem.png"),
        plot_cfd_measurement_overview(cfd_groups, tmp_path / "cfd_measurement_overview.png"),
        plot_method_overview(tiny_cfd_config.model, tmp_path / "cfd_method_overview.png"),
        plot_architecture_breakdown(tiny_cfd_config.model, tmp_path / "cfd_architecture_breakdown.png"),
        plot_cfd_dataset_overview(test_raw.groups, tmp_path / "cfd_dataset_overview.png"),
        plot_cfd_window_rollout(window_bundle, tmp_path / "cfd_window_rollout.png"),
        plot_cfd_full_trajectory_rollout(full_rollout, tmp_path / "cfd_full_rollout.png"),
        plot_cfd_rollout_error_profile(full_rollout, tmp_path / "cfd_rollout_error_profile.png"),
        plot_cfd_fidelity_gap(full_rollout, tmp_path / "cfd_fidelity_gap.png"),
    ]

    for path in paths:
        assert Path(path).exists()
