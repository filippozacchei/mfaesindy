"""MFMC-style estimators for latent regression moments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class MomentEstimate:
    c_xx: torch.Tensor
    c_xy: torch.Tensor


def empirical_cross_moment(theta: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    return theta.T @ dz / theta.shape[0]


def empirical_feature_moment(theta: torch.Tensor) -> torch.Tensor:
    return theta.T @ theta / theta.shape[0]


def mfmc_moment_estimate(
    theta_lf: torch.Tensor,
    dz_lf: torch.Tensor,
    theta_hf: torch.Tensor,
    dz_hf: torch.Tensor,
    theta_lf_paired: torch.Tensor,
    dz_lf_paired: torch.Tensor,
    a_xx: float = 1.0,
    a_xy: float = 1.0,
) -> MomentEstimate:
    c_xy_lf = empirical_cross_moment(theta_lf, dz_lf)
    c_xy_hf = empirical_cross_moment(theta_hf, dz_hf)
    c_xy_lf_paired = empirical_cross_moment(theta_lf_paired, dz_lf_paired)

    c_xx_lf = empirical_feature_moment(theta_lf)
    c_xx_hf = empirical_feature_moment(theta_hf)
    c_xx_lf_paired = empirical_feature_moment(theta_lf_paired)

    return MomentEstimate(
        c_xx=c_xx_lf + a_xx * (c_xx_hf - c_xx_lf_paired),
        c_xy=c_xy_lf + a_xy * (c_xy_hf - c_xy_lf_paired),
    )
