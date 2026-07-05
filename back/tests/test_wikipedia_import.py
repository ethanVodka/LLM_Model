from pathlib import Path

from mini_llm.corpus import iter_jsonl_records
from mini_llm.wikipedia_import import WikipediaImportConfig, import_wikipedia


def test_imports_extract_with_revision_attribution(tmp_path: Path) -> None:
    requested_urls: list[str] = []

    def request_json(url: str, user_agent: str) -> object:
        requested_urls.append(url)
        assert user_agent.startswith("LearningMiniLLM/")
        return {
            "query": {
                "pages": [
                    {
                        "pageid": 123,
                        "title": "人工知能",
                        "extract": "人間の知的活動を計算機で扱う研究分野です。",
                        "revisions": [{"revid": 456, "timestamp": "2026-07-05T00:00:00Z"}],
                    },
                    {"title": "missing", "missing": True},
                ]
            }
        }

    output_path = tmp_path / "wikipedia.jsonl"
    report = import_wikipedia(
        WikipediaImportConfig(
            endpoint="https://ja.wikipedia.org/w/api.php",
            user_agent="LearningMiniLLM/0.1 (https://example.com/project)",
            license="CC-BY-SA-4.0",
            language="ja",
            batch_size=20,
            extract_characters=1200,
            intro_only=True,
            request_delay_seconds=0.0,
            titles=("人工知能", "missing"),
        ),
        output_path,
        tmp_path / "report.json",
        request_json=request_json,
    )
    records = list(iter_jsonl_records([output_path]))

    assert len(requested_urls) == 1
    assert "prop=extracts%7Crevisions" in requested_urls[0]
    assert report["record_count"] == 1
    assert report["missing_titles"] == ["missing"]
    assert records[0].source == "jawikipedia"
    assert records[0].license == "CC-BY-SA-4.0"
    assert records[0].revision == "456"
    assert records[0].attribution == (
        "https://ja.wikipedia.org/w/index.php?title="
        "%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD&oldid=456"
    )


def test_batches_requests_sequentially(tmp_path: Path) -> None:
    request_count = 0

    def request_json(_: str, __: str) -> object:
        nonlocal request_count
        request_count += 1
        return {
            "query": {
                "pages": [
                    {
                        "pageid": request_count,
                        "title": f"page-{request_count}",
                        "extract": "sufficient extract text",
                        "revisions": [{"revid": request_count}],
                    }
                ]
            }
        }

    import_wikipedia(
        WikipediaImportConfig(
            endpoint="https://ja.wikipedia.org/w/api.php",
            user_agent="LearningMiniLLM/0.1 (https://example.com/project)",
            license="CC-BY-SA-4.0",
            language="ja",
            batch_size=2,
            extract_characters=1200,
            intro_only=True,
            request_delay_seconds=0.0,
            titles=("one", "two", "three"),
        ),
        tmp_path / "output.jsonl",
        tmp_path / "report.json",
        request_json=request_json,
    )

    assert request_count == 2
