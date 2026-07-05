"""Wikipedia抜粋をrole付き質問回答データへ変換するCLI。"""

import argparse
from pathlib import Path

from mini_llm.qa_prepare import prepare_qa_corpus


def main() -> None:
    parser = argparse.ArgumentParser(prog="mini-llm-qa")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/wikipedia_ja_v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/wikipedia_qa_v1.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/data/wikipedia_qa_v1.json"),
    )
    args = parser.parse_args()
    report = prepare_qa_corpus(args.input, args.output, args.report)
    print(f"dataset={args.output}")
    print(f"records={report['record_count']}")
    print(f"characters={report['character_count']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
