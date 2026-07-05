import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from mini_llm.config import ModelConfig
from mini_llm.evaluation import (
    EvaluationConfig,
    evaluate_checkpoint,
    iter_evaluation_prompts,
    perplexity_from_loss,
)
from mini_llm.model import MiniDecoderLM
from mini_llm.tokenizer import TokenizerConfig, train_tokenizer


def test_converts_cross_entropy_to_perplexity() -> None:
    assert perplexity_from_loss(math.log(8.0)) == pytest.approx(8.0)


def test_evaluates_checkpoint_and_writes_reproducible_report(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(
        TokenizerConfig(vocab_size=280, min_frequency=1),
        [
            "日本語の文章を評価します。",
            "def add(a: int, b: int) -> int:\n    return a + b",
            "type Status = 'idle' | 'training' | 'complete'",
        ],
        tokenizer_path,
    )
    model = MiniDecoderLM(
        ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            context_length=4,
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
        )
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "step": 3,
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    validation_path = tmp_path / "validation.npy"
    validation = np.arange(20, dtype=np.uint16).reshape(4, 5)
    np.save(validation_path, validation, allow_pickle=False)
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "ja",
                        "prompt": "こんにちは",
                        "category": "conversation",
                        "language": "ja",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "py",
                        "prompt": "def add",
                        "category": "code-completion",
                        "language": "python",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "evaluation" / "report.json"

    report = evaluate_checkpoint(
        EvaluationConfig(
            batch_size=2,
            max_new_tokens=2,
            temperature=0.0,
            top_k=10,
            seed=42,
        ),
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        validation_path=validation_path,
        prompt_paths=[prompts_path],
        output_path=output_path,
        device=torch.device("cpu"),
    )

    assert report["checkpoint_step"] == 3
    assert report["validation_sequence_count"] == 4
    assert report["validation_token_count"] == 16
    assert isinstance(report["validation_loss"], float)
    assert isinstance(report["perplexity"], float)
    assert len(report["checkpoint_sha256"]) == 64  # type: ignore[arg-type]
    assert len(report["prompt_results"]) == 2  # type: ignore[arg-type]
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_rejects_duplicate_evaluation_prompt_ids(tmp_path: Path) -> None:
    prompts_path = tmp_path / "prompts.jsonl"
    record = {
        "id": "same",
        "prompt": "sample",
        "category": "test",
        "language": "en",
    }
    prompts_path.write_text(
        f"{json.dumps(record)}\n{json.dumps(record)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        list(iter_evaluation_prompts([prompts_path]))


def test_rejects_invalid_evaluation_config() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        EvaluationConfig(
            batch_size=0,
            max_new_tokens=8,
            temperature=0.0,
            top_k=None,
            seed=42,
        )
