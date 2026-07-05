"""Qwen QLoRA実験を再現する設定。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class QLoRAConfig:
    model_name: str
    model_revision: str
    dataset_path: Path
    output_dir: Path
    cache_dir: Path
    max_length: int
    validation_fraction: float
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    warmup_steps: int
    eval_interval: int
    save_interval: int
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_name or len(self.model_revision) != 40:
            raise ValueError("model_name and a 40-character model_revision are required")
        positive_values = {
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "save_interval": self.save_interval,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.learning_rate <= 0.0 or not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("learning_rate and lora_dropout are invalid")
        if self.warmup_steps < 0 or not self.target_modules:
            raise ValueError("warmup_steps and target_modules are invalid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> QLoRAConfig:
        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("QLoRA config must be a mapping")
        target_modules = raw.get("target_modules")
        if not isinstance(target_modules, list) or not all(
            isinstance(module, str) and module for module in target_modules
        ):
            raise ValueError("target_modules must be a list of strings")
        return cls(
            model_name=_string(raw, "model_name"),
            model_revision=_string(raw, "model_revision"),
            dataset_path=Path(_string(raw, "dataset_path")),
            output_dir=Path(_string(raw, "output_dir")),
            cache_dir=Path(_string(raw, "cache_dir")),
            max_length=_integer(raw, "max_length"),
            validation_fraction=_number(raw, "validation_fraction"),
            batch_size=_integer(raw, "batch_size"),
            gradient_accumulation_steps=_integer(raw, "gradient_accumulation_steps"),
            max_steps=_integer(raw, "max_steps"),
            learning_rate=_number(raw, "learning_rate"),
            warmup_steps=_integer(raw, "warmup_steps"),
            eval_interval=_integer(raw, "eval_interval"),
            save_interval=_integer(raw, "save_interval"),
            seed=_integer(raw, "seed"),
            lora_rank=_integer(raw, "lora_rank"),
            lora_alpha=_integer(raw, "lora_alpha"),
            lora_dropout=_number(raw, "lora_dropout"),
            target_modules=tuple(target_modules),
        )


def _string(raw: dict[object, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(raw: dict[object, object], field: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(raw: dict[object, object], field: str) -> float:
    value = raw.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    return float(value)
