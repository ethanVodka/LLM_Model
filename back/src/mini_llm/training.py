"""次トークン予測モデルの学習・評価・保存処理。"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from mini_llm.model import MiniDecoderLM


@dataclass(frozen=True)
class TrainingConfig:
    """学習結果を再現するためのハイパーパラメータ。"""

    batch_size: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    eval_interval: int
    seed: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """YAMLから型を検証して学習設定を作る。"""

        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("training config must be a mapping")

        integer_fields = ("batch_size", "max_steps", "eval_interval", "seed")
        for field in integer_fields:
            value = raw.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{field} must be an integer")

        float_fields = ("learning_rate", "weight_decay", "grad_clip_norm")
        for field in float_fields:
            value = raw.get(field)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"{field} must be a number")

        return cls(
            batch_size=raw["batch_size"],
            max_steps=raw["max_steps"],
            learning_rate=float(raw["learning_rate"]),
            weight_decay=float(raw["weight_decay"]),
            grad_clip_norm=float(raw["grad_clip_norm"]),
            eval_interval=raw["eval_interval"],
            seed=raw["seed"],
        )


@dataclass(frozen=True)
class TrainingMetrics:
    """1回の評価タイミングで記録する値。"""

    step: int
    train_loss: float
    validation_loss: float
    grad_norm: float


@dataclass(frozen=True)
class TrainingResult:
    """学習完了後にCLIやテストが参照する結果。"""

    final_step: int
    final_train_loss: float
    final_validation_loss: float
    checkpoint_path: Path


def set_seed(seed: int) -> None:
    """Python、NumPy、PyTorchの乱数を同じseedへ固定する。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def language_model_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """`[batch, sequence, vocab]`の予測と正解IDから平均損失を返す。"""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab_size]")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match logits batch and sequence dimensions")
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def evaluate(
    model: MiniDecoderLM,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> float:
    """validation全体の1トークンあたり平均損失を返す。"""

    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss = language_model_loss(model(inputs), targets)
        batch_token_count = targets.numel()
        loss_sum += loss.item() * batch_token_count
        token_count += batch_token_count

    model.train(was_training)
    if token_count == 0:
        raise ValueError("validation dataset must not be empty")
    return loss_sum / token_count


def train_model(
    model: MiniDecoderLM,
    train_dataset: Dataset[tuple[Tensor, Tensor]],
    validation_dataset: Dataset[tuple[Tensor, Tensor]],
    config: TrainingConfig,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    on_evaluate: Callable[[TrainingMetrics], None] | None = None,
) -> TrainingResult:
    """AdamWで指定step数を学習し、最終状態と設定を保存する。"""

    if not isinstance(train_dataset, Sized) or len(train_dataset) == 0:
        raise ValueError("training dataset must not be empty")
    if not isinstance(validation_dataset, Sized) or len(validation_dataset) == 0:
        raise ValueError("validation dataset must not be empty")

    set_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
    batches = _repeat_batches(train_loader)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    final_metrics: TrainingMetrics | None = None

    for step in range(1, config.max_steps + 1):
        model.train()
        inputs, targets = next(batches)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model(inputs), targets)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.max_steps:
            final_metrics = TrainingMetrics(
                step=step,
                train_loss=loss.item(),
                validation_loss=evaluate(model, validation_loader, device),
                grad_norm=float(grad_norm),
            )
            if on_evaluate is not None:
                on_evaluate(final_metrics)

    if final_metrics is None:
        raise AssertionError("training completed without evaluation")

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "step": final_metrics.step,
            "model_config": asdict(model.config),
            "training_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": asdict(final_metrics),
        },
        destination,
    )
    return TrainingResult(
        final_step=final_metrics.step,
        final_train_loss=final_metrics.train_loss,
        final_validation_loss=final_metrics.validation_loss,
        checkpoint_path=destination,
    )


def _repeat_batches(
    data_loader: DataLoader[tuple[Tensor, Tensor]],
) -> Iterator[tuple[Tensor, Tensor]]:
    """小さなデータセットをmax_stepsまで繰り返す。"""

    while True:
        yield from data_loader
