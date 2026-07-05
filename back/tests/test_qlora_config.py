from pathlib import Path

import pytest

from mini_llm.qlora_config import QLoRAConfig


def test_loads_reproducible_qlora_config(tmp_path: Path) -> None:
    path = tmp_path / "qlora.yaml"
    path.write_text(
        """model_name: Qwen/Qwen3-1.7B
model_revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
dataset_path: data/conversations.jsonl
output_dir: artifacts/adapters/test
cache_dir: artifacts/models/huggingface
max_length: 256
validation_fraction: 0.1
batch_size: 1
gradient_accumulation_steps: 8
max_steps: 10
learning_rate: 0.0002
warmup_steps: 1
eval_interval: 5
save_interval: 10
seed: 42
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: [q_proj, v_proj]
""",
        encoding="utf-8",
    )

    config = QLoRAConfig.from_yaml(path)

    assert config.model_name == "Qwen/Qwen3-1.7B"
    assert config.max_length == 256
    assert config.target_modules == ("q_proj", "v_proj")


def test_rejects_unpinned_model_revision() -> None:
    with pytest.raises(ValueError, match="model_revision"):
        QLoRAConfig(
            model_name="Qwen/Qwen3-1.7B",
            model_revision="main",
            dataset_path=Path("data.jsonl"),
            output_dir=Path("output"),
            cache_dir=Path("cache"),
            max_length=256,
            validation_fraction=0.1,
            batch_size=1,
            gradient_accumulation_steps=8,
            max_steps=10,
            learning_rate=0.0002,
            warmup_steps=1,
            eval_interval=5,
            save_interval=10,
            seed=42,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=("q_proj",),
        )
