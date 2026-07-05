"""出典manifestから学習コーパスを準備するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_llm.corpus_prepare import CorpusPrepareConfig, prepare_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-llm-corpus")
    parser.add_argument("--config", type=Path, default=Path("configs/data/corpus_v1.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/v1/corpus.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/data/v1/report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = prepare_corpus(
        CorpusPrepareConfig.from_yaml(args.config),
        args.output,
        args.report,
    )
    print(f"corpus={args.output}")
    print(f"records={report['output_record_count']}")
    print(f"characters={report['character_count']}")
    print(f"duplicates_removed={report['duplicate_record_count']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
