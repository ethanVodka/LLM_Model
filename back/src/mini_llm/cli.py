import os

import torch

from mini_llm import MiniDecoderLM, ModelConfig


def main() -> None:
    config_path = os.getenv("MODEL_CONFIG", "configs/model/tiny.yaml")
    config = ModelConfig.from_yaml(config_path)
    model = MiniDecoderLM(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"config={config_path}")
    print(f"parameters={model.parameter_count():,}")
    print(f"device={device}")


if __name__ == "__main__":
    main()

