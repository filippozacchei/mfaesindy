import torch

from mfaesindy.mfmc import mfmc_moment_estimate


def test_mfmc_moment_estimate_shapes() -> None:
    theta_lf = torch.randn(8, 5)
    dz_lf = torch.randn(8, 2)
    theta_hf = torch.randn(4, 5)
    dz_hf = torch.randn(4, 2)
    theta_lf_paired = torch.randn(4, 5)
    dz_lf_paired = torch.randn(4, 2)

    estimate = mfmc_moment_estimate(
        theta_lf=theta_lf,
        dz_lf=dz_lf,
        theta_hf=theta_hf,
        dz_hf=dz_hf,
        theta_lf_paired=theta_lf_paired,
        dz_lf_paired=dz_lf_paired,
    )

    assert estimate.c_xx.shape == (5, 5)
    assert estimate.c_xy.shape == (5, 2)
