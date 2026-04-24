from __future__ import annotations

from mf_lnode.training.schedulers import FidelityWeightScheduler


def test_fidelity_weight_scheduler_interpolates_linearly():
    scheduler = FidelityWeightScheduler(
        initial_weights={"low": 1.0, "high": 0.5},
        final_weights={"low": 0.5, "high": 2.0},
        transition_epochs=4,
    )
    first = scheduler.weights_at(0)
    middle = scheduler.weights_at(2)
    last = scheduler.weights_at(8)
    assert first["low"] == 1.0
    assert middle["high"] > first["high"]
    assert last["high"] == 2.0

