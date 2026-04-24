from __future__ import annotations

import torch

from mf_lnode.data.scalers import MultiFidelityScaler, TensorStandardScaler


def test_tensor_standard_scaler_round_trip():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaler = TensorStandardScaler().fit([tensor])
    restored = scaler.inverse_transform(scaler.transform(tensor))
    assert torch.allclose(restored, tensor)


def test_multifidelity_scaler_transforms_group_data(synthetic_groups):
    scaler = MultiFidelityScaler.fit_from_groups(synthetic_groups)
    transformed = scaler.transform_group(synthetic_groups[0])
    for fidelity, sample in transformed.trajectories.items():
        restored = scaler.inverse_transform(fidelity, sample.observations)
        original = synthetic_groups[0].trajectories[fidelity].observations
        assert torch.allclose(restored, original, atol=1e-5)

