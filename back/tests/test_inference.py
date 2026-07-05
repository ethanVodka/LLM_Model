from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from mini_llm.config import ModelConfig
from mini_llm.inference import GenerationConfig, generate_token_ids, load_checkpoint
from mini_llm.inference_cli import _console_safe
from mini_llm.model import MiniDecoderLM


def make_model() -> MiniDecoderLM:
    model = MiniDecoderLM(
        ModelConfig(
            vocab_size=16,
            context_length=4,
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
        )
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    return model


def test_loads_model_checkpoint_in_safe_mode(tmp_path: Path) -> None:
    model = make_model()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "step": 12,
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    loaded = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert loaded.step == 12
    assert loaded.model.config == model.config
    assert torch.equal(loaded.model.token_embedding.weight, model.token_embedding.weight)


def test_generates_greedily_and_stops_at_eos() -> None:
    model = make_model()

    generated = generate_token_ids(
        model,
        [2, 5],
        GenerationConfig(max_new_tokens=5, temperature=0.0, top_k=None, seed=42),
        eos_token_id=0,
        device=torch.device("cpu"),
    )

    assert generated == [2, 5, 0]


def test_sampling_is_reproducible_and_crops_long_context() -> None:
    model = make_model()
    config = GenerationConfig(max_new_tokens=4, temperature=1.0, top_k=8, seed=7)

    first = generate_token_ids(
        model,
        [1, 2, 3, 4, 5, 6],
        config,
        eos_token_id=15,
        device=torch.device("cpu"),
    )
    second = generate_token_ids(
        model,
        [1, 2, 3, 4, 5, 6],
        config,
        eos_token_id=15,
        device=torch.device("cpu"),
    )

    assert first == second
    assert len(first) == 10


def test_rejects_invalid_generation_config() -> None:
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(max_new_tokens=1, temperature=-0.1)


def test_escapes_characters_unsupported_by_windows_console() -> None:
    assert _console_safe("text�", encoding="cp932") == r"text\ufffd"
