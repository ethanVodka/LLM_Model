"""日本語Wikipediaの固定タイトル集を取得するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_llm.wikipedia_import import WikipediaImportConfig, import_wikipedia


def main() -> None:
    parser = argparse.ArgumentParser(prog="mini-llm-wikipedia")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/wikipedia_ja_v1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/wikipedia_ja_v1.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/data/wikipedia_ja_v1.json"),
    )
    args = parser.parse_args()
    report = import_wikipedia(
        WikipediaImportConfig.from_yaml(args.config),
        args.output,
        args.report,
    )
    print(f"dataset={args.output}")
    print(f"records={report['record_count']}")
    print(f"characters={report['character_count']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
