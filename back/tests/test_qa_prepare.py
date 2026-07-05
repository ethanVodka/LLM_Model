import json
from pathlib import Path

import pytest

from mini_llm.corpus import iter_jsonl_records
from mini_llm.qa_prepare import prepare_qa_corpus


def write_source(path: Path, *, include_attribution: bool) -> None:
    record = {
        "id": "jawiki-1-2",
        "text": "人工知能\n人間の知的活動を計算機で扱う研究分野です。",
        "source": "jawikipedia",
        "license": "CC-BY-SA-4.0",
        "language": "ja",
        "revision": "2",
    }
    if include_attribution:
        record["attribution"] = "https://ja.wikipedia.org/w/index.php?oldid=2"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def test_creates_role_based_qa_and_preserves_attribution(tmp_path: Path) -> None:
    source_path = tmp_path / "wikipedia.jsonl"
    write_source(source_path, include_attribution=True)
    output_path = tmp_path / "qa.jsonl"

    report = prepare_qa_corpus(source_path, output_path, tmp_path / "report.json")
    record = next(iter_jsonl_records([output_path]))

    assert report["record_count"] == 1
    assert "<user>人工知能とは何ですか？<assistant>" in record.text
    assert record.source == "jawikipedia-derived-qa"
    assert record.attribution == "https://ja.wikipedia.org/w/index.php?oldid=2"
    assert record.revision == "2"


def test_rejects_qa_source_without_attribution(tmp_path: Path) -> None:
    source_path = tmp_path / "wikipedia.jsonl"
    write_source(source_path, include_attribution=False)

    with pytest.raises(ValueError, match="attribution"):
        prepare_qa_corpus(source_path, tmp_path / "qa.jsonl", tmp_path / "report.json")
