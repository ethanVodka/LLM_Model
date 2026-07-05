"""プロジェクト自身のPython・TypeScriptを出典付きコーパスへ変換する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectCodeConfig:
    root: Path
    repository_url: str
    revision: str
    license: str
    include: tuple[str, ...]
    max_file_characters: int

    def __post_init__(self) -> None:
        if not self.repository_url.startswith("https://"):
            raise ValueError("repository_url must use HTTPS")
        if not self.revision or not self.license or not self.include:
            raise ValueError("revision, license, and include must not be empty")
        if self.max_file_characters <= 0:
            raise ValueError("max_file_characters must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectCodeConfig:
        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("project code config must be a mapping")
        include = raw.get("include")
        if not isinstance(include, list) or not all(
            isinstance(pattern, str) and pattern for pattern in include
        ):
            raise ValueError("include must be a list of non-empty glob patterns")
        return cls(
            root=Path(_require_string(raw, "root")),
            repository_url=_require_string(raw, "repository_url"),
            revision=_require_string(raw, "revision"),
            license=_require_string(raw, "license"),
            include=tuple(include),
            max_file_characters=_require_integer(raw, "max_file_characters"),
        )


def import_project_code(
    config: ProjectCodeConfig,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, object]:
    """明示globに一致するUTF-8ソースだけをJSONLへ保存する。"""

    root = config.root.resolve()
    files = sorted(
        {
            path.resolve()
            for pattern in config.include
            for path in config.root.glob(pattern)
            if path.is_file()
        }
    )
    if not files:
        raise ValueError("project code import matched no files")

    records: list[dict[str, str]] = []
    skipped_files: list[str] = []
    for path in files:
        try:
            relative_path = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"source path escapes project root: {path}") from error
        language = _language_for_path(path)
        if language is None:
            continue
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        if not text or len(text) > config.max_file_characters:
            skipped_files.append(relative_path.as_posix())
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        attribution = (
            f"{config.repository_url.rstrip('/')}/blob/{config.revision}/"
            f"{relative_path.as_posix()}"
        )
        path_hash = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()
        records.append(
            {
                "id": f"project-code-{path_hash[:16]}",
                "text": text,
                "source": "project-source-code",
                "license": config.license,
                "language": language,
                "attribution": attribution,
                "revision": content_hash,
            }
        )
    if not records:
        raise ValueError("project code import produced no records")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    report: dict[str, object] = {
        "schema_version": 1,
        "record_count": len(records),
        "character_count": sum(len(record["text"]) for record in records),
        "languages": {
            language: sum(record["language"] == language for record in records)
            for language in sorted({record["language"] for record in records})
        },
        "skipped_files": skipped_files,
    }
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _language_for_path(path: Path) -> str | None:
    return {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
    }.get(path.suffix.lower())


def _require_string(raw: dict[object, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_integer(raw: dict[object, object], field: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value
