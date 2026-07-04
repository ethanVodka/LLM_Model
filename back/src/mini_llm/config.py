from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(
            self.vocab_size,
            self.context_length,
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.d_ff,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with Path(path).open(encoding="utf-8") as file:
            values: dict[str, Any] = yaml.safe_load(file)
        return cls(**values)

