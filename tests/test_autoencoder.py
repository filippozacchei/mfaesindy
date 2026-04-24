from __future__ import annotations

import torch

from mf_lnode.models.autoencoder import (
    DiscrepancyDecoder,
    FidelityDecoderHead,
    FidelityEncoderHead,
    SharedLatentProjector,
    SharedDecoder,
    SharedEncoder,
)


def test_autoencoder_components_return_expected_shapes():
    adapted = torch.randn(3, 4, 5)
    parameters = torch.randn(3, 2)
    encoder = SharedEncoder(window_size=4, adapter_dim=5, parameter_dim=2, hidden_dim=7)
    hidden = encoder(adapted, parameters)
    latent_projector = SharedLatentProjector(hidden_dim=7, latent_dim=3)
    latent = latent_projector(hidden)
    fidelity_latent = FidelityEncoderHead(hidden_dim=7, latent_dim=3)(hidden)
    decoder_hidden = SharedDecoder(latent_dim=3, parameter_dim=2, hidden_dim=7)(
        latent.unsqueeze(1).expand(-1, 2, -1),
        parameters,
    )
    prediction = FidelityDecoderHead(hidden_dim=7, output_dim=2)(decoder_hidden)
    discrepancy = DiscrepancyDecoder(hidden_dim=7, output_dim=2)(decoder_hidden)
    assert hidden.shape == (3, 7)
    assert latent.shape == (3, 3)
    assert fidelity_latent.shape == (3, 3)
    assert prediction.shape == (3, 2, 2)
    assert discrepancy.shape == (3, 2, 2)
