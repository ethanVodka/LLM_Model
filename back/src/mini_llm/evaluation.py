"""固定データとプロンプトによるモデル評価。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from mini_llm.dataset import NextTokenDataset
from mini_llm.inference import GenerationConfig, generate_token_ids, load_checkpoint
from mini_llm.tokenizer import load_tokenizer
from mini_llm.training import evaluate


@dataclass(frozen=True)
class EvaluationConfig:
    """検証損失と固定生成を同じ条件で比較する設定。"""

    batch_size: int
    max_new_tokens: int
    temperature: float
    top_k: int | None
    seed: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            seed=self.seed,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationConfig:
        """YAMLから型を検証して評価設定を作る。"""

        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("evaluation config must be a mapping")

        integer_fields = ("batch_size", "max_new_tokens", "seed")
        for field in integer_fields:
            value = raw.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{field} must be an integer")

        temperature = raw.get("temperature")
        if not isinstance(temperature, int | float) or isinstance(temperature, bool):
            raise ValueError("temperature must be a number")
        top_k = raw.get("top_k")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool)):
            raise ValueError("top_k must be an integer or null")
        return cls(
            batch_size=raw["batch_size"],
            max_new_tokens=raw["max_new_tokens"],
            temperature=float(temperature),
            top_k=top_k,
            seed=raw["seed"],
        )


@dataclass(frozen=True)
class EvaluationPrompt:
    """生成品質を継続比較する1件の入力。"""

    id: str
    prompt: str
    category: str
    language: str


def iter_evaluation_prompts(paths: Sequence[str | Path]) -> Iterator[EvaluationPrompt]:
    """JSONLから重複のない固定プロンプトを読み込む。"""

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
                    raise ValueError(f"prompt must be a mapping at {path}:{line_number}")

                fields = ("id", "prompt", "category", "language")
                invalid_fields = [
                    field
                    for field in fields
                    if not isinstance(raw.get(field), str) or not raw[field].strip()
                ]
                if invalid_fields:
                    raise ValueError(
                        f"fields must be non-empty strings at {path}:{line_number}: "
                        + ", ".join(invalid_fields)
                    )
                if raw["id"] in seen_ids:
                    raise ValueError(f"duplicate prompt id at {path}:{line_number}: {raw['id']}")
                seen_ids.add(raw["id"])
                yield EvaluationPrompt(
                    id=raw["id"],
                    prompt=raw["prompt"],
                    category=raw["category"],
                    language=raw["language"],
                )


def perplexity_from_loss(loss: float) -> float:
    """1トークンあたりのCross EntropyをPerplexityへ変換する。"""

    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("loss must be a finite non-negative number")
    try:
        return math.exp(loss)
    except OverflowError as error:
        raise ValueError("loss is too large to represent perplexity") from error


def evaluate_checkpoint(
    config: EvaluationConfig,
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    validation_path: str | Path,
    prompt_paths: Sequence[str | Path],
    output_path: str | Path,
    device: torch.device,
) -> dict[str, object]:
    """損失・Perplexity・固定生成を計算し、監査可能なJSONへ保存する。"""

    checkpoint_file = Path(checkpoint_path)
    tokenizer_file = Path(tokenizer_path)
    validation_file = Path(validation_path)
    loaded = load_checkpoint(checkpoint_file, device=device)
    tokenizer = load_tokenizer(tokenizer_file)
    if tokenizer.get_vocab_size() != loaded.model.config.vocab_size:
        raise ValueError("tokenizer vocab_size must match checkpoint vocab_size")

    validation_dataset = NextTokenDataset(validation_file)
    sequence_length = validation_dataset.sequences.shape[1] - 1
    if sequence_length > loaded.model.config.context_length:
        raise ValueError("validation sequence exceeds model context_length")
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
    validation_loss = evaluate(loaded.model, validation_loader, device)

    bos_token_id = tokenizer.token_to_id("<bos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("tokenizer must define <bos> and <eos> tokens")

    generation_config = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        seed=config.seed,
    )
    prompt_results: list[dict[str, object]] = []
    prompts = list(iter_evaluation_prompts(prompt_paths))
    if not prompts:
        raise ValueError("at least one evaluation prompt is required")
    for prompt in prompts:
        prompt_ids = [
            bos_token_id,
            *tokenizer.encode(prompt.prompt, add_special_tokens=False).ids,
        ]
        generated_ids = generate_token_ids(
            loaded.model,
            prompt_ids,
            generation_config,
            eos_token_id=eos_token_id,
            device=device,
        )
        completion_ids = generated_ids[len(prompt_ids) :]
        prompt_results.append(
            {
                "id": prompt.id,
                "prompt": prompt.prompt,
                "category": prompt.category,
                "language": prompt.language,
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                "generated_token_count": len(completion_ids),
            }
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint_step": loaded.step,
        "checkpoint_sha256": _sha256_file(checkpoint_file),
        "tokenizer_sha256": _sha256_file(tokenizer_file),
        "validation_sha256": _sha256_file(validation_file),
        "prompt_set_sha256": _sha256_files(prompt_paths),
        "model_config": asdict(loaded.model.config),
        "evaluation_config": asdict(config),
        "validation_sequence_count": len(validation_dataset),
        "validation_token_count": len(validation_dataset) * sequence_length,
        "validation_loss": validation_loss,
        "perplexity": perplexity_from_loss(validation_loss),
        "prompt_results": prompt_results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_files(paths: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    for path_value in paths:
        content = Path(path_value).read_bytes()
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()
