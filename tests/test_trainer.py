from __future__ import annotations

from mf_lnode import FidelityWeightScheduler, LatentNeuralODEModel, LossComposer, Trainer


def test_trainer_fits_and_saves_checkpoint(windowed_datasets, tiny_config):
    model = LatentNeuralODEModel.from_config(tiny_config.model)
    composer = LossComposer(tiny_config.loss)
    scheduler = FidelityWeightScheduler(
        initial_weights=tiny_config.loss.fidelity_weights_start,
        final_weights=tiny_config.loss.fidelity_weights_end,
        transition_epochs=tiny_config.training.transition_epochs,
    )
    trainer = Trainer(
        config=tiny_config.training,
        model=model,
        loss_composer=composer,
        fidelity_weight_scheduler=scheduler,
    )
    history = trainer.fit(windowed_datasets["train"], windowed_datasets["val"])
    checkpoint_path = trainer.output_dir / tiny_config.training.checkpoint_name
    assert len(history) == 1
    assert checkpoint_path.exists()
    metrics = trainer.evaluate(windowed_datasets["test"])
    assert "total_loss" in metrics
    checkpoint = trainer.load_checkpoint(checkpoint_path)
    assert "model_state" in checkpoint

