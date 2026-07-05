"""人手作成の意図テンプレートを会話JSONLへ展開するCLI。"""

import argparse
from pathlib import Path

from mini_llm.conversation_prepare import (
    ConversationCorpusConfig,
    prepare_conversation_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="mini-llm-conversations")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/conversation_intents_v1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/conversation_intents_v1.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/data/conversation_intents_v1.json"),
    )
    args = parser.parse_args()
    report = prepare_conversation_corpus(
        ConversationCorpusConfig.from_yaml(args.config),
        args.output,
        args.report,
    )
    print(f"dataset={args.output}")
    print(f"records={report['record_count']}")
    print(f"characters={report['character_count']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
