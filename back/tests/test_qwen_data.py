import json
from pathlib import Path

import pytest

from mini_llm.qwen_data import parse_chat_record, prepare_qwen_dataset


def test_parses_single_turn_chat_roles() -> None:
    example = parse_chat_record(
        "chat-1",
        "<system>短く回答します。<user>こんにちは<assistant>こんにちは。",
    )

    assert example.system == "短く回答します。"
    assert example.user == "こんにちは"
    assert example.assistant == "こんにちは。"


def test_rejects_chat_without_assistant_role() -> None:
    with pytest.raises(ValueError, match="user and assistant"):
        parse_chat_record("invalid", "<system>回答します。<user>質問")


def test_writes_qwen_messages_jsonl(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "id": "chat-1",
                "text": "<system>短く回答します。<user>質問<assistant>回答",
                "source": "project-original",
                "license": "project-original",
                "language": "ja",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "qwen.jsonl"

    report = prepare_qwen_dataset(
        [corpus_path],
        output_path,
        tmp_path / "report.json",
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["record_count"] == 1
    assert saved["messages"] == [
        {"role": "system", "content": "短く回答します。"},
        {"role": "user", "content": "質問"},
        {"role": "assistant", "content": "回答"},
    ]
