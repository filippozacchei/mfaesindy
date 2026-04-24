from __future__ import annotations

import mf_lnode


def test_public_api_exports_expected_symbols():
    assert hasattr(mf_lnode, "LatentNeuralODEModel")
    assert hasattr(mf_lnode, "Trainer")
    assert hasattr(mf_lnode, "LossComposer")

