"""学習済みチェックポイントの読み込みと自己回帰生成。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from mini_llm.config import ModelConfig
from mini_llm.model import MiniDecoderLM


@dataclass(frozen=True)
class GenerationConfig:
    """生成長とサンプリング方法を再現する設定。"""

    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int | None = 50
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")


@dataclass(frozen=True)
class LoadedCheckpoint:
    """推論に必要なモデルと学習step。"""

    model: MiniDecoderLM
    step: int


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> LoadedCheckpoint:
    """自プロジェクトのチェックポイントを安全モードで読み込む。"""

    raw: Any = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=True,
    )
    if not isinstance(raw, dict):
        raise ValueError("checkpoint must be a mapping")

    model_config_raw = raw.get("model_config")
    state_dict = raw.get("model_state_dict")
    step = raw.get("step")
    if not isinstance(model_config_raw, dict):
        raise ValueError("checkpoint model_config must be a mapping")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError("checkpoint step must be an integer")

    model = MiniDecoderLM(ModelConfig(**model_config_raw))
    model.load_state_dict(state_dict)
    model.to(device)
    return LoadedCheckpoint(model=model, step=step)


def generate_token_ids(
    model: MiniDecoderLM,
    prompt_ids: Sequence[int],
    config: GenerationConfig,
    *,
    eos_token_id: int,
    device: torch.device,
) -> list[int]:
    """プロンプトの後ろへ1トークンずつ予測結果を追加する。"""

    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if any(token_id < 0 or token_id >= model.config.vocab_size for token_id in prompt_ids):
        raise ValueError("prompt token id is outside the model vocabulary")
    if not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError("eos_token_id is outside the model vocabulary")

    generated = list(prompt_ids)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                # contextを超えた場合は、生成に必要な直近の履歴だけを渡す。
                context = generated[-model.config.context_length :]
                input_ids = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
                next_logits = model(input_ids)[0, -1]
                next_token_id = _select_next_token(next_logits, config, generator)
                generated.append(next_token_id)
                if next_token_id == eos_token_id:
                    break
    finally:
        model.train(was_training)
    return generated


def _select_next_token(
    logits: Tensor,
    config: GenerationConfig,
    generator: torch.Generator,
) -> int:
    if config.temperature == 0.0:
        return int(torch.argmax(logits).item())

    scaled_logits = logits / config.temperature
    if config.top_k is not None:
        top_k = min(config.top_k, scaled_logits.numel())
        top_values = torch.topk(scaled_logits, top_k).values
        scaled_logits = scaled_logits.masked_fill(scaled_logits < top_values[-1], -torch.inf)
    probabilities = torch.softmax(scaled_logits, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())
