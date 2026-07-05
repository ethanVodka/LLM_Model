from pathlib import Path

from mini_llm.corpus import iter_jsonl_records
from mini_llm.project_code_import import ProjectCodeConfig, import_project_code


def test_imports_python_and_typescript_with_source_attribution(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    (source_dir / "sample.ts").write_text(
        "export const value: number = 42\n",
        encoding="utf-8",
    )
    (source_dir / "ignored.txt").write_text("ignore", encoding="utf-8")
    output_path = tmp_path / "code.jsonl"

    report = import_project_code(
        ProjectCodeConfig(
            root=tmp_path,
            repository_url="https://example.com/repository",
            revision="main",
            license="project-original",
            include=("src/**/*.py", "src/**/*.ts"),
            max_file_characters=1000,
        ),
        output_path,
        tmp_path / "report.json",
    )
    records = list(iter_jsonl_records([output_path]))

    assert report["record_count"] == 2
    assert {record.language for record in records} == {"python", "typescript"}
    assert all(record.attribution is not None for record in records)
    assert all(record.revision is not None and len(record.revision) == 64 for record in records)
