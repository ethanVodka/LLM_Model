"""学習コーパスの共通JSONL形式と検証処理。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_CORPUS_FIELDS = ("id", "text", "source", "license", "language")


@dataclass(frozen=True)
class CorpusRecord:
    """出典と利用条件を追跡できる1件の学習文書。"""

    id: str
    text: str
    source: str
    license: str
    language: str


def iter_jsonl_records(paths: Sequence[str | Path]) -> Iterator[CorpusRecord]:
    """JSONLを読み、必須メタデータを検証した文書を順番に返す。"""

    seen_ids: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    raw: Any = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error
                if not isinstance(raw, dict):
                    raise ValueError(f"record must be a mapping at {path}:{line_number}")

                invalid_fields = [
                    field
                    for field in REQUIRED_CORPUS_FIELDS
                    if not isinstance(raw.get(field), str) or not raw[field].strip()
                ]
                if invalid_fields:
                    fields = ", ".join(invalid_fields)
                    raise ValueError(
                        f"fields must be non-empty strings at {path}:{line_number}: {fields}"
                    )

                record = CorpusRecord(
                    id=raw["id"],
                    text=raw["text"],
                    source=raw["source"],
                    license=raw["license"],
                    language=raw["language"],
                )
                if record.id in seen_ids:
                    raise ValueError(f"duplicate corpus id at {path}:{line_number}: {record.id}")
                seen_ids.add(record.id)
                yield record


def iter_jsonl_texts(paths: Sequence[str | Path]) -> Iterator[str]:
    """トークナイザー学習用に、検証済み文書から本文だけを返す。"""

    for record in iter_jsonl_records(paths):
        yield record.text
