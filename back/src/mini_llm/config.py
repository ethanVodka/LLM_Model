"""モデル構造を決めるハイパーパラメータの定義と読み込み処理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Decoder-only Transformerの構造を表す変更不可の設定。

    Attributes:
        vocab_size: トークナイザーが扱うトークンIDの総数。
        context_length: 1回の推論・学習で参照できる最大トークン数。
        d_model: 各トークンを表現する埋め込みベクトルの次元数。
        n_heads: Multi-Head Attentionで並列に計算するヘッド数。
        n_layers: 積み重ねるTransformerブロック数。
        d_ff: 各ブロック内のFeed Forward Networkの中間次元数。
        dropout: 学習時に過学習を抑えるため無効化する要素の割合。

    frozen=Trueにより、モデル作成後に設定が書き換わって構造と記録が
    食い違う事故を防ぐ。
    """

    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """モデルを作る前に、構造として成立しない値を拒否する。"""

        # 各Attentionヘッドへ同じ幅で分割するため、d_modelはn_headsの倍数が必要。
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        # 次元数が0以下ではEmbeddingやLinearレイヤーを構築できない。
        if min(
            self.vocab_size,
            self.context_length,
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.d_ff,
        ) <= 0:
            raise ValueError("model dimensions must be positive")

        # 1.0では全要素が無効になるため、上限を含めない。
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """YAMLのキーと値から検証済みの設定オブジェクトを作る。"""

        with Path(path).open(encoding="utf-8") as file:
            values: dict[str, Any] = yaml.safe_load(file)

        # **valuesでYAMLの各キーをdataclassの同名引数へ対応させる。
        # 未知のキーや必須キー不足もTypeErrorとなるため、設定ミスを早期検出できる。
        return cls(**values)
