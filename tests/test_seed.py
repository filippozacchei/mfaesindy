from __future__ import annotations

import torch

from mf_lnode.utils.seed import seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(17)
    first = torch.randn(4)
    seed_everything(17)
    second = torch.randn(4)
    assert torch.allclose(first, second)

