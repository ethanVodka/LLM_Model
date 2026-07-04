import pytest
import torch

from mini_llm import MiniDecoderLM, ModelConfig


@pytest.fixture
def model() -> MiniDecoderLM:
    config = ModelConfig(
        vocab_size=256,
        context_length=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )
    return MiniDecoderLM(config)


def test_returns_logits_for_each_token(model: MiniDecoderLM) -> None:
    input_ids = torch.randint(0, 256, (2, 12))

    logits = model(input_ids)

    assert logits.shape == (2, 12, 256)


def test_rejects_sequence_over_context_length(model: MiniDecoderLM) -> None:
    input_ids = torch.randint(0, 256, (1, 33))

    with pytest.raises(ValueError, match="context_length"):
        model(input_ids)

