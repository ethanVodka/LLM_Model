"""学習用Decoder-only Transformerのモデル定義。"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from mini_llm.config import ModelConfig


class MiniDecoderLM(nn.Module):
    """次のトークンを予測する、最小構成のDecoder-only言語モデル。

    処理の流れは「トークンID → 埋め込み → Transformerブロック → 語彙ごとの
    logits」である。PyTorchのTransformerEncoderを使うが、未来を隠す因果マスクを
    渡すことで、EncoderではなくGPT系と同じDecoder-only自己回帰モデルとして動く。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # token_embedding: 各トークンIDをd_model次元の意味表現へ変換する。
        # position_embedding: 並び順を持たないAttentionへ位置情報を加える。
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)

        # 1層分の設計図を作り、TransformerEncoderがn_layers個に複製する。
        # norm_first=TrueはAttention/FFNの前に正規化するPre-LN構成で、
        # 深いモデルでも勾配を安定させやすい。
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pre-LNは現在のnested tensor最適化対象外なので明示的に無効化し、
        # PyTorchの警告を避ける。通常のTensorによる計算結果は変わらない。
        self.blocks = nn.TransformerEncoder(
            layer,
            config.n_layers,
            enable_nested_tensor=False,
        )

        # 最終表現を安定させ、各位置について「次に来るトークン」のスコアへ変換する。
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # EmbeddingのPyTorch既定初期化はWeight Tying時のlogitsを過大にするため、
        # GPT系の小型モデルで一般的な小さい標準偏差で全重みを初期化する。
        self.apply(self._initialize_weights)

        # 入力Embeddingと出力層の重みを共有するWeight Tying。
        # 語彙×隠れ次元の大きな行列を1つ減らし、入出力のトークン表現も対応させる。
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        """トークンID列から、各位置の次トークン予測logitsを計算する。

        Args:
            input_ids: `[batch, sequence]` 形状の整数トークンID。

        Returns:
            `[batch, sequence, vocab_size]` 形状の未正規化スコア。
            学習時は通常、末尾次元にCross Entropy Lossを適用する。
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        sequence_length = input_ids.size(1)
        if sequence_length > self.config.context_length:
            raise ValueError("sequence exceeds context_length")

        # positionsは[sequence]。位置Embeddingは[sequence, d_model]となり、
        # token_embeddingの[batch, sequence, d_model]へbatch方向にbroadcastされる。
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        # causal_maskの上三角（未来位置）を-infにしてAttention対象から除外する。
        # これにより位置tの予測が、正解となる未来トークンを覗くことを防ぐ。
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sequence_length,
            device=input_ids.device,
        )
        hidden = self.blocks(hidden, mask=causal_mask, is_causal=True)
        return cast(Tensor, self.lm_head(self.final_norm(hidden)))

    def parameter_count(self) -> int:
        """共有済みの重みを重複計上せず、モデルの全パラメータ数を返す。"""

        return sum(parameter.numel() for parameter in self.parameters())
