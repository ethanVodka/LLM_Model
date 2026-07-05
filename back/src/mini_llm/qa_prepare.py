"""帰属情報付きの知識文書から会話形式の質問回答を作る。"""

from __future__ import annotations

import json
from pathlib import Path

from mini_llm.corpus import iter_jsonl_records


def prepare_qa_corpus(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, object]:
    """Wikipediaの記事名を質問、抜粋を回答とし、元の帰属を引き継ぐ。"""

    records: list[dict[str, str]] = []
    for source_record in iter_jsonl_records([input_path]):
        if source_record.license != "CC-BY-SA-4.0" or source_record.attribution is None:
            raise ValueError(
                f"QA source must contain CC-BY-SA attribution: {source_record.id}"
            )
        title, separator, body = source_record.text.partition("\n")
        if not separator or not title.strip() or not body.strip():
            raise ValueError(f"QA source must start with title and body: {source_record.id}")
        conversation = (
            "<system>事実に基づいて日本語で簡潔に回答します。"
            f"<user>{title.strip()}とは何ですか？"
            f"<assistant>{body.strip()}"
        )
        records.append(
            {
                "id": f"qa-{source_record.id}",
                "text": conversation,
                "source": "jawikipedia-derived-qa",
                "license": source_record.license,
                "language": "ja",
                "attribution": source_record.attribution,
                "revision": source_record.revision or "unknown",
            }
        )
    if not records:
        raise ValueError("QA preparation produced no records")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for output_record in records:
            file.write(json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n")
    report: dict[str, object] = {
        "schema_version": 1,
        "record_count": len(records),
        "character_count": sum(len(record["text"]) for record in records),
        "license": "CC-BY-SA-4.0",
        "quality_status": "automatically-derived",
    }
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
