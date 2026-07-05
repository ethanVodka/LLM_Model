from pathlib import Path

import numpy as np
import pytest
import torch

from mini_llm.config import ModelConfig
from mini_llm.dataset import NextTokenDataset
from mini_llm.model import MiniDecoderLM
from mini_llm.training import TrainingConfig, language_model_loss, train_model


def make_model() -> MiniDecoderLM:
    return MiniDecoderLM(
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


def save_dataset(path: Path, rows: int) -> NextTokenDataset:
    values = np.arange(rows * 5, dtype=np.uint16).reshape(rows, 5) % 16
    np.save(path, values, allow_pickle=False)
    return NextTokenDataset(path)


def test_calculates_language_model_loss() -> None:
    logits = torch.zeros((2, 3, 4))
    targets = torch.zeros((2, 3), dtype=torch.long)

    loss = language_model_loss(logits, targets)

    assert loss.item() == pytest.approx(np.log(4))


def test_trains_model_and_saves_reproducible_checkpoint(tmp_path: Path) -> None:
    train_dataset = save_dataset(tmp_path / "train.npy", rows=8)
    validation_dataset = save_dataset(tmp_path / "validation.npy", rows=4)
    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"
    model = make_model()
    original_weights = model.token_embedding.weight.detach().clone()

    result = train_model(
        model,
        train_dataset,
        validation_dataset,
        TrainingConfig(
            batch_size=2,
            max_steps=2,
            learning_rate=0.001,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            eval_interval=1,
            seed=42,
        ),
        checkpoint_path,
        device=torch.device("cpu"),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert result.final_step == 2
    assert np.isfinite(result.final_train_loss)
    assert np.isfinite(result.final_validation_loss)
    assert checkpoint["step"] == 2
    assert checkpoint["model_config"]["vocab_size"] == 16
    assert not torch.equal(original_weights, model.token_embedding.weight)


def test_rejects_invalid_training_config() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(
            batch_size=2,
            max_steps=1,
            learning_rate=0.0,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            eval_interval=1,
            seed=42,
        )


def test_rejects_mismatched_loss_shapes() -> None:
    with pytest.raises(ValueError, match="targets"):
        language_model_loss(torch.zeros((2, 3, 4)), torch.zeros((2, 2), dtype=torch.long))
