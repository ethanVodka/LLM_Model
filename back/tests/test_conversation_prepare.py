import json
from pathlib import Path

from mini_llm.conversation_prepare import (
    ConversationCorpusConfig,
    ConversationIntent,
    prepare_conversation_corpus,
)
from mini_llm.corpus import iter_jsonl_records


def test_expands_all_prompt_response_combinations(tmp_path: Path) -> None:
    output_path = tmp_path / "conversations.jsonl"
    report = prepare_conversation_corpus(
        ConversationCorpusConfig(
            (
                ConversationIntent(
                    name="greeting",
                    prompts=("こんにちは", "やあ"),
                    responses=("こんにちは。", "やあ、元気です。"),
                ),
            )
        ),
        output_path,
        tmp_path / "report.json",
    )
    records = list(iter_jsonl_records([output_path]))

    assert report["record_count"] == 4
    assert {record.id for record in records} == {
        "intent-greeting-01-01",
        "intent-greeting-01-02",
        "intent-greeting-02-01",
        "intent-greeting-02-02",
    }
    assert all("<assistant>" in record.text for record in records)
    saved_report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved_report["quality_status"] == "human-authored-templates"
