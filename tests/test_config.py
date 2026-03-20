from mfaesindy.config import ExperimentConfig, SINDyConfig


def test_experiment_config_defaults() -> None:
    config = ExperimentConfig.model_validate(
        {
            "autoencoder": {
                "input_dim": 4,
                "latent_dim": 2,
            }
        }
    )

    assert config.autoencoder.input_dim == 4
    assert config.autoencoder.latent_dim == 2
    assert isinstance(config.sindy, SINDyConfig)
