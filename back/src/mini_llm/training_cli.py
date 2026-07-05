"""小型言語モデルを学習するCLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mini_llm.config import ModelConfig
from mini_llm.dataset import NextTokenDataset
from mini_llm.model import MiniDecoderLM
from mini_llm.training import TrainingConfig, TrainingMetrics, set_seed, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-llm-train")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/tiny.yaml"))
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training/tiny.yaml"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/tiny"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/tiny/latest.pt"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_config = ModelConfig.from_yaml(args.model_config)
    training_config = TrainingConfig.from_yaml(args.training_config)
    metadata = _load_metadata(args.dataset_dir / "metadata.json")
    _validate_compatibility(model_config, metadata)

    device = _resolve_device(args.device)
    # モデル初期化も再現対象に含めるため、構築よりseedを固定する。
    set_seed(training_config.seed)
    model = MiniDecoderLM(model_config)
    train_dataset = NextTokenDataset(args.dataset_dir / "train.npy")
    validation_dataset = NextTokenDataset(args.dataset_dir / "validation.npy")

    print(f"device={device.type}")
    print(f"parameters={model.parameter_count():,}")
    result = train_model(
        model,
        train_dataset,
        validation_dataset,
        training_config,
        args.checkpoint,
        device=device,
        on_evaluate=_print_metrics,
    )
    print(f"checkpoint={result.checkpoint_path}")


def _load_metadata(path: Path) -> dict[str, Any]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dataset metadata must be a mapping")
    return raw


def _validate_compatibility(model_config: ModelConfig, metadata: dict[str, Any]) -> None:
    vocab_size = metadata.get("vocab_size")
    context_length = metadata.get("context_length")
    if vocab_size != model_config.vocab_size:
        raise ValueError("dataset vocab_size must match model vocab_size")
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        raise ValueError("dataset context_length must be an integer")
    if context_length > model_config.context_length:
        raise ValueError("dataset context_length must not exceed model context_length")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(value)


def _print_metrics(metrics: TrainingMetrics) -> None:
    print(
        f"step={metrics.step} "
        f"train_loss={metrics.train_loss:.4f} "
        f"validation_loss={metrics.validation_loss:.4f} "
        f"grad_norm={metrics.grad_norm:.4f}"
    )


if __name__ == "__main__":
    main()
