"""Qwen QLoRA用messages JSONLを生成するCLI。"""

import argparse
from pathlib import Path

from mini_llm.qwen_data import prepare_qwen_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="mini-llm-qwen-data")
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=[Path("data/processed/sft_v1/corpus.jsonl")],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/qwen_sft_v1/conversations.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/data/qwen_sft_v1/report.json"),
    )
    args = parser.parse_args()
    report = prepare_qwen_dataset(args.corpus, args.output, args.report)
    print(f"dataset={args.output}")
    print(f"records={report['record_count']}")
    print(f"characters={report['character_count']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
