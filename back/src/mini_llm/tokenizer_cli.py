"""BPEトークナイザーを学習・確認するコマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mini_llm.config import ModelConfig
from mini_llm.corpus import iter_jsonl_texts
from mini_llm.tokenizer import (
    TokenizerConfig,
    load_tokenizer,
    train_tokenizer,
)

DEFAULT_TOKENIZER_PATH = Path("artifacts/tokenizer/tiny.json")


def build_parser() -> argparse.ArgumentParser:
    """サブコマンドと引数を定義する。"""

    parser = argparse.ArgumentParser(prog="mini-llm-tokenizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and save a BPE tokenizer")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/tiny.yaml"),
    )
    train_parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/tiny.yaml"),
    )
    train_parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=[Path("data/samples/tokenizer_corpus.jsonl")],
    )
    train_parser.add_argument("--output", type=Path, default=DEFAULT_TOKENIZER_PATH)

    encode_parser = subparsers.add_parser("encode", help="encode text into token ids")
    encode_parser.add_argument("text")
    encode_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    encode_parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="do not add <bos> and <eos>",
    )

    decode_parser = subparsers.add_parser("decode", help="decode token ids into text")
    decode_parser.add_argument("token_ids", type=int, nargs="+")
    decode_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    decode_parser.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help="include special tokens in decoded text",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """指定されたサブコマンドを実行する。"""

    args = build_parser().parse_args(argv)
    if args.command == "train":
        tokenizer_config = TokenizerConfig.from_yaml(args.config)
        model_config = ModelConfig.from_yaml(args.model_config)
        tokenizer = train_tokenizer(
            tokenizer_config,
            iter_jsonl_texts(args.corpus),
            args.output,
            expected_vocab_size=model_config.vocab_size,
        )
        print(f"tokenizer={args.output}")
        print(f"vocab_size={tokenizer.get_vocab_size():,}")
        return

    tokenizer = load_tokenizer(args.tokenizer)
    if args.command == "encode":
        encoding = tokenizer.encode(args.text, add_special_tokens=not args.no_special_tokens)
        print(json.dumps(encoding.ids, ensure_ascii=False))
        return
    if args.command == "decode":
        print(
            tokenizer.decode(
                args.token_ids,
                skip_special_tokens=not args.keep_special_tokens,
            )
        )
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
