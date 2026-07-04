"""モデル構成と実行環境を確認するスモークテスト用CLI。"""

import os

import torch

from mini_llm import MiniDecoderLM, ModelConfig


def main() -> None:
    """設定からモデルを構築し、学習開始前に基本情報を表示する。"""

    # 環境変数で設定を差し替えられるため、コードを変えずにモデル規模を比較できる。
    config_path = os.getenv("MODEL_CONFIG", "configs/model/tiny.yaml")
    config = ModelConfig.from_yaml(config_path)
    model = MiniDecoderLM(config)

    # ここでは利用可能な計算デバイスを報告するだけで、学習や推論はまだ行わない。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"config={config_path}")
    print(f"parameters={model.parameter_count():,}")
    print(f"device={device}")


if __name__ == "__main__":
    # `python -m mini_llm.cli` として直接実行された場合の入口。
    main()
