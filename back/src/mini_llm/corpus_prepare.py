"""出典manifestに従う学習コーパスの正規化・検査・重複除去。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from mini_llm.corpus import CorpusRecord, iter_jsonl_records

SENSITIVE_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github-token": re.compile(r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "openai-key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}


@dataclass(frozen=True)
class CorpusSource:
    """1つの入力データの出典と利用条件。"""

    path: Path
    source: str
    license: str
    acquired_at: str

    def __post_init__(self) -> None:
        if not self.source or not self.license:
            raise ValueError("source and license must not be empty")
        try:
            date.fromisoformat(self.acquired_at)
        except ValueError as error:
            raise ValueError("acquired_at must use YYYY-MM-DD format") from error


@dataclass(frozen=True)
class CorpusPrepareConfig:
    """複数データを1つの学習コーパスへ統合する条件。"""

    sources: tuple[CorpusSource, ...]
    allowed_licenses: tuple[str, ...]
    min_characters: int
    max_characters: int
    normalize_nfkc: bool
    deduplicate: bool

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("at least one corpus source is required")
        if not self.allowed_licenses:
            raise ValueError("allowed_licenses must not be empty")
        if self.min_characters <= 0:
            raise ValueError("min_characters must be positive")
        if self.max_characters < self.min_characters:
            raise ValueError("max_characters must not be less than min_characters")
        for source in self.sources:
            if source.license not in self.allowed_licenses:
                raise ValueError(f"source license is not allowed: {source.license}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> CorpusPrepareConfig:
        """YAML manifestを検証してコーパス準備設定を作る。"""

        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("corpus config must be a mapping")
        raw_sources = raw.get("sources")
        if not isinstance(raw_sources, list):
            raise ValueError("sources must be a list")
        sources = tuple(_parse_source(value, index) for index, value in enumerate(raw_sources))

        licenses = raw.get("allowed_licenses")
        if not isinstance(licenses, list) or not all(
            isinstance(value, str) and value for value in licenses
        ):
            raise ValueError("allowed_licenses must be a list of non-empty strings")
        min_characters = _require_integer(raw, "min_characters")
        max_characters = _require_integer(raw, "max_characters")
        normalize_nfkc = raw.get("normalize_nfkc")
        deduplicate = raw.get("deduplicate")
        if not isinstance(normalize_nfkc, bool) or not isinstance(deduplicate, bool):
            raise ValueError("normalize_nfkc and deduplicate must be booleans")
        return cls(
            sources=sources,
            allowed_licenses=tuple(licenses),
            min_characters=min_characters,
            max_characters=max_characters,
            normalize_nfkc=normalize_nfkc,
            deduplicate=deduplicate,
        )


def prepare_corpus(
    config: CorpusPrepareConfig,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, object]:
    """レコードを検査・正規化し、学習用JSONLと監査レポートを保存する。"""

    prepared: list[CorpusRecord] = []
    seen_ids: set[str] = set()
    seen_text_hashes: set[str] = set()
    input_count = 0
    duplicate_count = 0
    filtered_count = 0

    for source_config in config.sources:
        for record in iter_jsonl_records([source_config.path]):
            input_count += 1
            if record.id in seen_ids:
                raise ValueError(f"duplicate corpus id across sources: {record.id}")
            seen_ids.add(record.id)
            _validate_provenance(record, source_config)
            text = _normalize_text(record.text, use_nfkc=config.normalize_nfkc)
            _reject_sensitive_text(record.id, text)
            if not config.min_characters <= len(text) <= config.max_characters:
                filtered_count += 1
                continue
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if config.deduplicate and text_hash in seen_text_hashes:
                duplicate_count += 1
                continue
            seen_text_hashes.add(text_hash)
            prepared.append(replace(record, text=text))

    if not prepared:
        raise ValueError("corpus preparation produced no records")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for record in prepared:
            file.write(
                json.dumps(
                    _record_to_dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    report: dict[str, object] = {
        "schema_version": 1,
        "input_record_count": input_count,
        "output_record_count": len(prepared),
        "duplicate_record_count": duplicate_count,
        "filtered_record_count": filtered_count,
        "character_count": sum(len(record.text) for record in prepared),
        "languages": dict(sorted(Counter(record.language for record in prepared).items())),
        "licenses": dict(sorted(Counter(record.license for record in prepared).items())),
        "sources": [
            {
                "path": str(source.path),
                "source": source.source,
                "license": source.license,
                "acquired_at": source.acquired_at,
                "sha256": _sha256_file(source.path),
            }
            for source in config.sources
        ],
        "output_sha256": _sha256_file(destination),
    }
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_source(value: object, index: int) -> CorpusSource:
    if not isinstance(value, dict):
        raise ValueError(f"sources[{index}] must be a mapping")
    fields = ("path", "source", "license", "acquired_at")
    for field in fields:
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"sources[{index}].{field} must be a non-empty string")
    return CorpusSource(
        path=Path(value["path"]),
        source=value["source"],
        license=value["license"],
        acquired_at=value["acquired_at"],
    )


def _require_integer(raw: dict[object, object], field: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _normalize_text(text: str, *, use_nfkc: bool) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if use_nfkc:
        normalized = unicodedata.normalize("NFKC", normalized)
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _validate_provenance(record: CorpusRecord, source: CorpusSource) -> None:
    if record.source != source.source or record.license != source.license:
        raise ValueError(f"record provenance does not match manifest: {record.id}")


def _reject_sensitive_text(record_id: str, text: str) -> None:
    for name, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"sensitive pattern detected in {record_id}: {name}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_to_dict(record: CorpusRecord) -> dict[str, str]:
    output = {
        "id": record.id,
        "text": record.text,
        "source": record.source,
        "license": record.license,
        "language": record.language,
    }
    if record.attribution is not None:
        output["attribution"] = record.attribution
    if record.revision is not None:
        output["revision"] = record.revision
    return output
