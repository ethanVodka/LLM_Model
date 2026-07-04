from __future__ import annotations

import torch
from torch import Tensor, nn

from mini_llm.config import ModelConfig


class MiniDecoderLM(nn.Module):
    """学習過程を追いやすい最小構成のDecoder-only言語モデル。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        sequence_length = input_ids.size(1)
        if sequence_length > self.config.context_length:
            raise ValueError("sequence exceeds context_length")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sequence_length,
            device=input_ids.device,
        )
        hidden = self.blocks(hidden, mask=causal_mask, is_causal=True)
        return self.lm_head(self.final_norm(hidden))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

