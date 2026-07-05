"""学習済み小型言語モデルから文字列を生成するCLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from mini_llm.conversation import format_chat_prompt
from mini_llm.inference import GenerationConfig, generate_token_ids, load_checkpoint
from mini_llm.tokenizer import load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-llm-generate")
    parser.add_argument("prompt")
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
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="format the prompt as a user message and print only the assistant response",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    loaded = load_checkpoint(args.checkpoint, device=device)
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.get_vocab_size() != loaded.model.config.vocab_size:
        raise ValueError("tokenizer vocab_size must match checkpoint vocab_size")

    bos_token_id = tokenizer.token_to_id("<bos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("tokenizer must define <bos> and <eos> tokens")

    prompt = format_chat_prompt(args.prompt) if args.chat else args.prompt
    prompt_ids = [
        bos_token_id,
        *tokenizer.encode(prompt, add_special_tokens=False).ids,
    ]
    generated_ids = generate_token_ids(
        loaded.model,
        prompt_ids,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        ),
        eos_token_id=eos_token_id,
        device=device,
    )

    print(f"device={device.type}")
    print(f"checkpoint_step={loaded.step}")
    output_ids = generated_ids[len(prompt_ids) :] if args.chat else generated_ids
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print(f"text={_console_safe(decoded)}")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(value)


def _console_safe(value: str, *, encoding: str | None = None) -> str:
    """Byte-level生成の不完全な文字をWindows端末でも安全に表示する。"""

    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    return value.encode(target_encoding, errors="backslashreplace").decode(target_encoding)


if __name__ == "__main__":
    main()
