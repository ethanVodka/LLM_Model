import pytest

from mini_llm import ModelConfig


def test_rejects_incompatible_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            vocab_size=256,
            context_length=32,
            d_model=30,
            n_heads=4,
            n_layers=2,
            d_ff=64,
        )

