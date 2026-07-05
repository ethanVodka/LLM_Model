import json
from pathlib import Path

import pytest

from mini_llm.corpus import iter_jsonl_records
from mini_llm.corpus_prepare import CorpusPrepareConfig, CorpusSource, prepare_corpus


def write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def make_record(record_id: str, text: str, *, source: str = "project-original") -> dict[str, str]:
    return {
        "id": record_id,
        "text": text,
        "source": source,
        "license": "project-original",
        "language": "mixed",
    }


def make_config(path: Path) -> CorpusPrepareConfig:
    return CorpusPrepareConfig(
        sources=(
            CorpusSource(
                path=path,
                source="project-original",
                license="project-original",
                acquired_at="2026-07-05",
            ),
        ),
        allowed_licenses=("project-original",),
        min_characters=10,
        max_characters=1000,
        normalize_nfkc=True,
        deduplicate=True,
    )


def test_normalizes_filters_and_deduplicates_corpus(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    write_records(
        source_path,
        [
            make_record("first", "ＡＢＣ sample text  \r\n"),
            make_record("duplicate", "ABC sample text"),
            make_record("short", "short"),
        ],
    )
    output_path = tmp_path / "processed" / "corpus.jsonl"
    report_path = tmp_path / "report.json"

    report = prepare_corpus(make_config(source_path), output_path, report_path)
    records = list(iter_jsonl_records([output_path]))

    assert [record.id for record in records] == ["first"]
    assert records[0].text == "ABC sample text"
    assert report["input_record_count"] == 3
    assert report["output_record_count"] == 1
    assert report["duplicate_record_count"] == 1
    assert report["filtered_record_count"] == 1
    assert len(report["output_sha256"]) == 64  # type: ignore[arg-type]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_rejects_manifest_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    write_records(source_path, [make_record("mismatch", "long enough text", source="other")])

    with pytest.raises(ValueError, match="provenance"):
        prepare_corpus(
            make_config(source_path),
            tmp_path / "output.jsonl",
            tmp_path / "report.json",
        )


def test_rejects_sensitive_values(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    write_records(
        source_path,
        [make_record("secret", "credential AKIAABCDEFGHIJKLMNOP must not be stored")],
    )

    with pytest.raises(ValueError, match="sensitive pattern"):
        prepare_corpus(
            make_config(source_path),
            tmp_path / "output.jsonl",
            tmp_path / "report.json",
        )
