"""チェックポイントを固定条件で評価するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_llm.evaluation import EvaluationConfig, evaluate_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-llm-evaluate")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/tiny.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/tiny/latest.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tiny.json"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/datasets/tiny/validation.npy"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        nargs="+",
        default=[Path("data/samples/evaluation_prompts.jsonl")],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/tiny/report.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    report = evaluate_checkpoint(
        EvaluationConfig.from_yaml(args.config),
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        validation_path=args.validation,
        prompt_paths=args.prompts,
        output_path=args.output,
        device=device,
    )
    validation_loss = report["validation_loss"]
    perplexity = report["perplexity"]
    if not isinstance(validation_loss, float) or not isinstance(perplexity, float):
        raise ValueError("evaluation report metrics must be floats")
    print(f"device={device.type}")
    print(f"checkpoint_step={report['checkpoint_step']}")
    print(f"validation_loss={validation_loss:.4f}")
    print(f"perplexity={perplexity:.2f}")
    print(f"report={args.output}")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(value)


if __name__ == "__main__":
    main()
