"""次トークン予測モデルの学習・評価・再開・保存処理。"""

from __future__ import annotations

import json
import math
import random
import shutil
from collections.abc import Callable, Mapping, Sized
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
    checkpoint_interval: int
    seed: int

    def __post_init__(self) -> None:
        positive_integers = {
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """YAMLから型を検証して学習設定を作る。"""

        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("training config must be a mapping")

        integer_fields = (
            "batch_size",
            "max_steps",
            "eval_interval",
            "checkpoint_interval",
            "seed",
        )
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
            checkpoint_interval=raw["checkpoint_interval"],
            seed=raw["seed"],
        )


@dataclass(frozen=True)
class TrainingMetrics:
    """1回の評価タイミングで記録する値。"""

    step: int
    tokens_seen: int
    train_loss: float
    validation_loss: float
    perplexity: float
    grad_norm: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    """学習完了後にCLIやテストが参照する結果。"""

    initial_step: int
    final_step: int
    final_train_loss: float
    final_validation_loss: float
    checkpoint_path: Path
    metrics_path: Path | None
    exact_resume: bool


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
    resume_from: str | Path | None = None,
    metrics_path: str | Path | None = None,
    on_evaluate: Callable[[TrainingMetrics], None] | None = None,
) -> TrainingResult:
    """AdamWで指定stepまで学習し、乱数状態を含む再開可能な状態を保存する。"""

    train_size = _dataset_size(train_dataset, "training")
    _dataset_size(validation_dataset, "validation")
    set_seed(config.seed)
    batch_generator = torch.Generator().manual_seed(config.seed)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    initial_step = 0
    exact_resume = True
    if resume_from is not None:
        initial_step, exact_resume = _restore_training_state(
            Path(resume_from),
            model=model,
            optimizer=optimizer,
            config=config,
            batch_generator=batch_generator,
            train_size=train_size,
            device=device,
        )
    if initial_step >= config.max_steps:
        raise ValueError("max_steps must be greater than the checkpoint step")

    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
    destination = Path(checkpoint_path)
    metric_destination = Path(metrics_path) if metrics_path is not None else None
    _prepare_metrics_file(metric_destination, resume=resume_from is not None)
    final_metrics: TrainingMetrics | None = None

    for step in range(initial_step + 1, config.max_steps + 1):
        model.train()
        inputs, targets = _sample_batch(
            train_dataset,
            train_size=train_size,
            batch_size=config.batch_size,
            generator=batch_generator,
        )
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model(inputs), targets)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        should_evaluate = step % config.eval_interval == 0 or step == config.max_steps
        if should_evaluate:
            validation_loss = evaluate(model, validation_loader, device)
            final_metrics = TrainingMetrics(
                step=step,
                tokens_seen=step * config.batch_size * targets.size(1),
                train_loss=loss.item(),
                validation_loss=validation_loss,
                perplexity=math.exp(validation_loss),
                grad_norm=float(grad_norm),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
            _append_metrics(metric_destination, final_metrics)
            if on_evaluate is not None:
                on_evaluate(final_metrics)

        should_checkpoint = (
            step % config.checkpoint_interval == 0 or step == config.max_steps
        )
        if should_checkpoint:
            _save_training_state(
                destination,
                step=step,
                model=model,
                optimizer=optimizer,
                config=config,
                metrics=final_metrics,
                batch_generator=batch_generator,
            )

    if final_metrics is None:
        raise AssertionError("training completed without evaluation")
    return TrainingResult(
        initial_step=initial_step,
        final_step=final_metrics.step,
        final_train_loss=final_metrics.train_loss,
        final_validation_loss=final_metrics.validation_loss,
        checkpoint_path=destination,
        metrics_path=metric_destination,
        exact_resume=exact_resume,
    )


def _dataset_size(dataset: Dataset[tuple[Tensor, Tensor]], name: str) -> int:
    if not isinstance(dataset, Sized) or len(dataset) == 0:
        raise ValueError(f"{name} dataset must not be empty")
    return len(dataset)


def _sample_batch(
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    train_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    indices = torch.randint(0, train_size, (batch_size,), generator=generator)
    examples = [dataset[int(index)] for index in indices]
    return (
        torch.stack([inputs for inputs, _ in examples]),
        torch.stack([targets for _, targets in examples]),
    )


def _restore_training_state(
    path: Path,
    *,
    model: MiniDecoderLM,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    batch_generator: torch.Generator,
    train_size: int,
    device: torch.device,
) -> tuple[int, bool]:
    raw: Any = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("training checkpoint must be a mapping")
    step = raw.get("step")
    model_config = raw.get("model_config")
    model_state = raw.get("model_state_dict")
    optimizer_state = raw.get("optimizer_state_dict")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    if model_config != asdict(model.config):
        raise ValueError("checkpoint model_config must match the current model")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint must contain model and optimizer states")
    _validate_resume_config(raw.get("training_config"), config)

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    torch_rng_state = raw.get("torch_rng_state")
    batch_generator_state = raw.get("batch_generator_state")
    if isinstance(torch_rng_state, Tensor) and isinstance(batch_generator_state, Tensor):
        torch.set_rng_state(torch_rng_state.cpu())
        batch_generator.set_state(batch_generator_state.cpu())
        cuda_rng_states = raw.get("cuda_rng_states")
        if device.type == "cuda" and isinstance(cuda_rng_states, list):
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_states])
        return step, True

    # schema v1には乱数状態がないため、バッチ列だけをseedから再構成する。
    for _ in range(step):
        torch.randint(0, train_size, (config.batch_size,), generator=batch_generator)
    return step, False


def _validate_resume_config(saved_config: object, current: TrainingConfig) -> None:
    if not isinstance(saved_config, Mapping):
        raise ValueError("checkpoint training_config must be a mapping")
    immutable_fields = (
        "batch_size",
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "seed",
    )
    for field in immutable_fields:
        if saved_config.get(field) != getattr(current, field):
            raise ValueError(f"checkpoint training_config mismatch: {field}")


def _save_training_state(
    destination: Path,
    *,
    step: int,
    model: MiniDecoderLM,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    metrics: TrainingMetrics | None,
    batch_generator: torch.Generator,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    step_destination = destination.parent / f"step_{step:06d}{destination.suffix}"
    torch.save(
        {
            "schema_version": 2,
            "step": step,
            "model_config": asdict(model.config),
            "training_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": asdict(metrics) if metrics is not None else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "batch_generator_state": batch_generator.get_state(),
        },
        step_destination,
    )
    shutil.copyfile(step_destination, destination)


def _prepare_metrics_file(path: Path | None, *, resume: bool) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        path.write_text("", encoding="utf-8")
    elif not path.exists():
        path.touch()


def _append_metrics(path: Path | None, metrics: TrainingMetrics) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(asdict(metrics), ensure_ascii=False, sort_keys=True) + "\n")
