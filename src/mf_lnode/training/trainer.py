"""Training loop, callbacks, and checkpointing."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mf_lnode.configs.schema import TrainingConfig
from mf_lnode.data.datasets import WindowedTrajectoryDataset, collate_windowed_groups
from mf_lnode.evaluation.metrics import compute_per_fidelity_metrics
from mf_lnode.losses.core import LossComposer
from mf_lnode.training.schedulers import FidelityWeightScheduler


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(value, device) for value in batch]
    return batch


class TrainingCallback:
    """Minimal callback interface for extending the trainer."""

    def on_train_start(self, trainer: "Trainer") -> None:  # pragma: no cover - default no-op
        pass

    def on_epoch_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:  # pragma: no cover
        pass

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:  # pragma: no cover
        pass

    def on_train_end(self, trainer: "Trainer") -> None:  # pragma: no cover - default no-op
        pass


class Trainer:
    """Model training helper with checkpointing and fidelity weight scheduling."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        loss_composer: LossComposer,
        fidelity_weight_scheduler: FidelityWeightScheduler | None = None,
        callbacks: list[TrainingCallback] | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.loss_composer = loss_composer
        self.fidelity_weight_scheduler = fidelity_weight_scheduler
        self.callbacks = callbacks or []
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.use_amp and self.device.type == "cuda",
        )
        self.history: list[dict[str, float]] = []
        self.best_validation_loss = float("inf")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = self._maybe_build_writer()

    def _maybe_build_writer(self) -> Any:
        if not self.config.tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return None
        return SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

    def _as_loader(self, data: WindowedTrajectoryDataset | DataLoader, shuffle: bool) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        return DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_windowed_groups,
        )

    def _autocast_context(self) -> Any:
        if self.config.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        fidelity_weights: dict[str, float] | None,
    ) -> dict[str, float]:
        self.model.train(train)
        aggregate: defaultdict[str, float] = defaultdict(float)
        batches = 0

        for batch in loader:
            batch = _move_to_device(batch, self.device)
            with self._autocast_context():
                outputs = self.model(batch)
                total_loss, loss_metrics = self.loss_composer(
                    outputs,
                    fidelity_weights=fidelity_weights,
                )

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.max_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            aggregate["total_loss"] += float(total_loss.detach().cpu())
            for key, value in loss_metrics.items():
                aggregate[key] += value
            for key, value in compute_per_fidelity_metrics(outputs).items():
                aggregate[key] += value
            batches += 1

        if batches == 0:
            return {}
        return {key: value / batches for key, value in aggregate.items()}

    def save_checkpoint(self, epoch: int, metrics: dict[str, float], filename: str | None = None) -> Path:
        """Persist model, optimizer, and trainer state."""

        path = self.output_dir / (filename or self.config.checkpoint_name)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "training_config": asdict(self.config),
                "metrics": metrics,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore model and optimizer state from disk."""

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint

    def fit(
        self,
        train_data: WindowedTrajectoryDataset | DataLoader,
        val_data: WindowedTrajectoryDataset | DataLoader | None = None,
    ) -> list[dict[str, float]]:
        """Train the model and optionally track validation metrics."""

        train_loader = self._as_loader(train_data, shuffle=True)
        val_loader = self._as_loader(val_data, shuffle=False) if val_data is not None else None

        for callback in self.callbacks:
            callback.on_train_start(self)

        for epoch in range(self.config.epochs):
            fidelity_weights = (
                self.fidelity_weight_scheduler.weights_at(epoch)
                if self.fidelity_weight_scheduler is not None
                else None
            )
            train_metrics = self._run_epoch(train_loader, train=True, fidelity_weights=fidelity_weights)
            record = {f"train/{key}": value for key, value in train_metrics.items()}
            record["epoch"] = float(epoch + 1)

            if val_loader is not None:
                validation_metrics = self._run_epoch(
                    val_loader,
                    train=False,
                    fidelity_weights=fidelity_weights,
                )
                record.update({f"val/{key}": value for key, value in validation_metrics.items()})
                validation_loss = validation_metrics.get("total_loss", float("inf"))
                for callback in self.callbacks:
                    callback.on_validation_end(self, validation_metrics)
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss
                    self.save_checkpoint(epoch=epoch + 1, metrics=record)
            else:
                self.save_checkpoint(epoch=epoch + 1, metrics=record, filename=f"epoch-{epoch + 1}.pt")

            self.history.append(record)
            self._log_metrics(record, epoch + 1)
            for callback in self.callbacks:
                callback.on_epoch_end(self, record)

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        for callback in self.callbacks:
            callback.on_train_end(self)
        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        data: WindowedTrajectoryDataset | DataLoader,
        fidelity_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Evaluate the model on a dataset or dataloader."""

        loader = self._as_loader(data, shuffle=False)
        return self._run_epoch(loader, train=False, fidelity_weights=fidelity_weights)

    def _log_metrics(self, metrics: dict[str, float], epoch: int) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            if key == "epoch":
                continue
            self.writer.add_scalar(key, value, epoch)
