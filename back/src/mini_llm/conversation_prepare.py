"""意図ごとの人手作成表現から再現可能な会話SFTコーパスを作る。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mini_llm.conversation import DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class ConversationIntent:
    name: str
    prompts: tuple[str, ...]
    responses: tuple[str, ...]


@dataclass(frozen=True)
class ConversationCorpusConfig:
    intents: tuple[ConversationIntent, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConversationCorpusConfig:
        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict) or not isinstance(raw.get("intents"), list):
            raise ValueError("conversation config must contain an intents list")

        intents: list[ConversationIntent] = []
        seen_names: set[str] = set()
        for raw_intent in raw["intents"]:
            if not isinstance(raw_intent, dict):
                raise ValueError("each conversation intent must be a mapping")
            name = _require_string(raw_intent, "name")
            if name in seen_names:
                raise ValueError(f"duplicate conversation intent: {name}")
            seen_names.add(name)
            intents.append(
                ConversationIntent(
                    name=name,
                    prompts=_require_strings(raw_intent, "prompts"),
                    responses=_require_strings(raw_intent, "responses"),
                )
            )
        if not intents:
            raise ValueError("conversation intents must not be empty")
        return cls(tuple(intents))


def prepare_conversation_corpus(
    config: ConversationCorpusConfig,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, object]:
    """同じ意図の質問と回答を全組み合わせし、role付きJSONLへ保存する。"""

    records: list[dict[str, str]] = []
    intent_counts: dict[str, int] = {}
    for intent in config.intents:
        for prompt_index, prompt in enumerate(intent.prompts, start=1):
            for response_index, response in enumerate(intent.responses, start=1):
                _reject_role_tokens(prompt, intent.name)
                _reject_role_tokens(response, intent.name)
                records.append(
                    {
                        "id": (
                            f"intent-{intent.name}-{prompt_index:02d}-{response_index:02d}"
                        ),
                        "text": (
                            f"<system>{DEFAULT_SYSTEM_PROMPT}"
                            f"<user>{prompt}<assistant>{response}"
                        ),
                        "source": "project-original",
                        "license": "project-original",
                        "language": "ja",
                    }
                )
        intent_counts[intent.name] = len(intent.prompts) * len(intent.responses)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    report: dict[str, object] = {
        "schema_version": 1,
        "record_count": len(records),
        "character_count": sum(len(record["text"]) for record in records),
        "intent_counts": intent_counts,
        "quality_status": "human-authored-templates",
    }
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _require_string(raw: dict[object, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_strings(raw: dict[object, object], field: str) -> tuple[str, ...]:
    value = raw.get(field)
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{field} must contain at least two strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    return tuple(value)


def _reject_role_tokens(value: str, intent_name: str) -> None:
    if any(token in value for token in ("<system>", "<user>", "<assistant>")):
        raise ValueError(f"role token is not allowed in intent text: {intent_name}")
