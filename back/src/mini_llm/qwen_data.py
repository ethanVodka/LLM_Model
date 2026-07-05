"""role付きコーパスをHugging Face chat messages形式へ変換する。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mini_llm.corpus import iter_jsonl_records


@dataclass(frozen=True)
class ChatExample:
    id: str
    system: str
    user: str
    assistant: str


def parse_chat_record(record_id: str, text: str) -> ChatExample:
    """単一ターンのsystem／user／assistant本文を分離して検証する。"""

    if not text.startswith("<system>"):
        raise ValueError(f"chat record must start with <system>: {record_id}")
    system_and_rest = text.removeprefix("<system>")
    system, user_marker, user_and_assistant = system_and_rest.partition("<user>")
    user, assistant_marker, assistant = user_and_assistant.partition("<assistant>")
    if not user_marker or not assistant_marker:
        raise ValueError(f"chat record must contain user and assistant roles: {record_id}")
    if any(marker in assistant for marker in ("<system>", "<user>", "<assistant>")):
        raise ValueError(f"chat record must contain exactly one turn: {record_id}")
    if not system.strip() or not user.strip() or not assistant.strip():
        raise ValueError(f"chat record roles must not be empty: {record_id}")
    return ChatExample(
        id=record_id,
        system=system.strip(),
        user=user.strip(),
        assistant=assistant.strip(),
    )


def prepare_qwen_dataset(
    corpus_paths: list[str | Path],
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, object]:
    """検証済み会話をQwen chat templateへ渡せるJSONLとして保存する。"""

    examples = [
        parse_chat_record(record.id, record.text)
        for record in iter_jsonl_records(corpus_paths)
    ]
    if not examples:
        raise ValueError("Qwen dataset must not be empty")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for example in examples:
            file.write(
                json.dumps(
                    {
                        "id": example.id,
                        "messages": [
                            {"role": "system", "content": example.system},
                            {"role": "user", "content": example.user},
                            {"role": "assistant", "content": example.assistant},
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    report: dict[str, object] = {
        "schema_version": 1,
        "record_count": len(examples),
        "character_count": sum(
            len(example.system) + len(example.user) + len(example.assistant)
            for example in examples
        ),
        "source_paths": [str(Path(path)) for path in corpus_paths],
    }
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
