"""MediaWiki APIから帰属情報付きの日本語Wikipedia抜粋を取得する。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import yaml

JsonRequest = Callable[[str, str], object]


@dataclass(frozen=True)
class WikipediaImportConfig:
    """取得先と対象記事を固定する設定。"""

    endpoint: str
    user_agent: str
    license: str
    language: str
    batch_size: int
    extract_characters: int
    intro_only: bool
    request_delay_seconds: float
    titles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.endpoint.startswith("https://"):
            raise ValueError("endpoint must use HTTPS")
        if "http" not in self.user_agent or "(" not in self.user_agent:
            raise ValueError("user_agent must identify the project and contact URL")
        if self.license != "CC-BY-SA-4.0":
            raise ValueError("Wikipedia text must be recorded as CC-BY-SA-4.0")
        if not 1 <= self.batch_size <= 20:
            raise ValueError("batch_size must be between 1 and 20")
        if not 1 <= self.extract_characters <= 1200:
            raise ValueError("extract_characters must be between 1 and 1200")
        if not self.intro_only and self.batch_size != 1:
            raise ValueError("full extracts require batch_size 1")
        if not 0.0 <= self.request_delay_seconds <= 10.0:
            raise ValueError("request_delay_seconds must be between 0 and 10")
        if not self.titles or len(set(self.titles)) != len(self.titles):
            raise ValueError("titles must be non-empty and unique")

    @classmethod
    def from_yaml(cls, path: str | Path) -> WikipediaImportConfig:
        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("Wikipedia import config must be a mapping")
        string_fields = ("endpoint", "user_agent", "license", "language")
        values: dict[str, str] = {}
        for field in string_fields:
            value = raw.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
            values[field] = value
        titles = raw.get("titles")
        if not isinstance(titles, list) or not all(
            isinstance(title, str) and title for title in titles
        ):
            raise ValueError("titles must be a list of non-empty strings")
        intro_only = raw.get("intro_only")
        request_delay_seconds = raw.get("request_delay_seconds")
        if not isinstance(intro_only, bool):
            raise ValueError("intro_only must be a boolean")
        if not isinstance(request_delay_seconds, int | float) or isinstance(
            request_delay_seconds, bool
        ):
            raise ValueError("request_delay_seconds must be a number")
        return cls(
            endpoint=values["endpoint"],
            user_agent=values["user_agent"],
            license=values["license"],
            language=values["language"],
            batch_size=_require_integer(raw, "batch_size"),
            extract_characters=_require_integer(raw, "extract_characters"),
            intro_only=intro_only,
            request_delay_seconds=float(request_delay_seconds),
            titles=tuple(titles),
        )


def import_wikipedia(
    config: WikipediaImportConfig,
    output_path: str | Path,
    report_path: str | Path,
    *,
    request_json: JsonRequest | None = None,
) -> dict[str, object]:
    """逐次APIリクエストで記事抜粋と固定revision帰属URLを保存する。"""

    requester = request_json or _request_json
    records: list[dict[str, str]] = []
    missing_titles: list[str] = []
    batches = _batched(config.titles, config.batch_size)
    for batch_index, titles in enumerate(batches):
        response = requester(_build_url(config, titles), config.user_agent)
        pages = _extract_pages(response)
        returned_titles = set()
        for page in pages:
            record = _page_to_record(page, config)
            if record is None:
                continue
            records.append(record)
            returned_titles.add(record["title"])
        # redirect先はタイトルが変わるため、APIがmissingと返したものだけを後段で記録する。
        missing_titles.extend(
            str(page.get("title", ""))
            for page in pages
            if isinstance(page, dict) and "missing" in page
        )
        if not returned_titles and not missing_titles:
            raise ValueError("Wikipedia API returned no usable pages")
        if request_json is None and batch_index < len(batches) - 1:
            time.sleep(config.request_delay_seconds)

    if not records:
        raise ValueError("Wikipedia import produced no records")
    records.sort(key=lambda record: record["id"])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            output_record = {key: value for key, value in record.items() if key != "title"}
            file.write(json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n")

    report: dict[str, object] = {
        "schema_version": 1,
        "endpoint": config.endpoint,
        "license": config.license,
        "requested_title_count": len(config.titles),
        "record_count": len(records),
        "character_count": sum(len(record["text"]) for record in records),
        "missing_titles": sorted(title for title in missing_titles if title),
        "revision_ids": [record["revision"] for record in records],
    }
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_url(config: WikipediaImportConfig, titles: Sequence[str]) -> str:
    parameters = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts|revisions",
        "explaintext": "1",
        "exchars": str(config.extract_characters),
        "rvprop": "ids|timestamp",
        "redirects": "1",
        "titles": "|".join(titles),
    }
    if config.intro_only:
        parameters["exintro"] = "1"
    query = urlencode(parameters)
    return f"{config.endpoint}?{query}"


def _request_json(url: str, user_agent: str) -> object:
    for attempt in range(3):
        request = Request(url, headers={"User-Agent": user_agent})
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310 - HTTPSは設定で検証済み。
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code not in {429, 503} or attempt == 2:
                raise
            retry_after = error.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after is not None else 2**attempt
            time.sleep(min(wait_seconds, 30.0))
    raise AssertionError("unreachable retry loop")


def _extract_pages(response: object) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        raise ValueError("Wikipedia API response must be a mapping")
    query = response.get("query")
    if not isinstance(query, dict) or not isinstance(query.get("pages"), list):
        raise ValueError("Wikipedia API response is missing query.pages")
    pages: list[dict[str, Any]] = []
    for page in query["pages"]:
        if not isinstance(page, dict):
            raise ValueError("Wikipedia API pages must be mappings")
        pages.append(page)
    return pages


def _page_to_record(
    page: dict[str, Any],
    config: WikipediaImportConfig,
) -> dict[str, str] | None:
    if "missing" in page:
        return None
    page_id = page.get("pageid")
    title = page.get("title")
    extract = page.get("extract")
    revisions = page.get("revisions")
    if not isinstance(page_id, int) or not isinstance(title, str) or not isinstance(extract, str):
        raise ValueError("Wikipedia page is missing id, title, or extract")
    if not isinstance(revisions, list) or not revisions or not isinstance(revisions[0], dict):
        raise ValueError("Wikipedia page is missing revision metadata")
    revision_id = revisions[0].get("revid")
    if not isinstance(revision_id, int):
        raise ValueError("Wikipedia revision id must be an integer")
    cleaned_extract = extract.strip()
    if not cleaned_extract:
        return None
    attribution_url = (
        f"https://ja.wikipedia.org/w/index.php?title={quote(title.replace(' ', '_'))}"
        f"&oldid={revision_id}"
    )
    return {
        "id": f"jawiki-{page_id}-{revision_id}",
        "title": title,
        "text": f"{title}\n{cleaned_extract}",
        "source": "jawikipedia",
        "license": config.license,
        "language": config.language,
        "attribution": attribution_url,
        "revision": str(revision_id),
    }


def _batched(values: Sequence[str], size: int) -> list[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _require_integer(raw: dict[object, object], field: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value
