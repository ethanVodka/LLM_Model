"""次トークン予測データセットを準備・確認するCLI。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mini_llm.config import ModelConfig
from mini_llm.corpus import iter_jsonl_records
from mini_llm.dataset import DataConfig, prepare_dataset, prepare_sft_dataset
from mini_llm.tokenizer import load_tokenizer

DEFAULT_DATASET_DIR = Path("artifacts/datasets/tiny")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-llm-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="prepare train and validation data")
    prepare_parser.add_argument("--config", type=Path, default=Path("configs/data/tiny.yaml"))
    prepare_parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/tiny.yaml"),
    )
    prepare_parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tiny.json"),
    )
    prepare_parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=[Path("data/samples/tokenizer_corpus.jsonl")],
    )
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)

    sft_parser = subparsers.add_parser(
        "prepare-sft",
        help="prepare assistant-response masked train and validation data",
    )
    sft_parser.add_argument("--config", type=Path, required=True)
    sft_parser.add_argument("--model-config", type=Path, required=True)
    sft_parser.add_argument("--tokenizer", type=Path, required=True)
    sft_parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    sft_parser.add_argument("--output-dir", type=Path, required=True)

    inspect_parser = subparsers.add_parser("inspect", help="print dataset metadata and shapes")
    inspect_parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in {"prepare", "prepare-sft"}:
        data_config = DataConfig.from_yaml(args.config)
        model_config = ModelConfig.from_yaml(args.model_config)
        if data_config.context_length > model_config.context_length:
            raise ValueError("data context_length must not exceed model context_length")

        tokenizer = load_tokenizer(args.tokenizer)
        if tokenizer.get_vocab_size() != model_config.vocab_size:
            raise ValueError("tokenizer vocab_size must match model vocab_size")
        records = list(iter_jsonl_records(args.corpus))
        prepare = prepare_sft_dataset if args.command == "prepare-sft" else prepare_dataset
        metadata = prepare(
            data_config,
            records,
            tokenizer,
            args.output_dir,
            tokenizer_path=args.tokenizer,
            corpus_paths=args.corpus,
        )
        print(f"dataset={args.output_dir}")
        print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        return

    if args.command == "inspect":
        metadata = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        if metadata.get("objective") == "assistant-response":
            train = np.load(
                args.dataset_dir / "train_inputs.npy", mmap_mode="r", allow_pickle=False
            )
            validation = np.load(
                args.dataset_dir / "validation_inputs.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        else:
            train = np.load(args.dataset_dir / "train.npy", mmap_mode="r", allow_pickle=False)
            validation = np.load(
                args.dataset_dir / "validation.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        print(f"train_shape={train.shape}")
        print(f"validation_shape={validation.shape}")
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
