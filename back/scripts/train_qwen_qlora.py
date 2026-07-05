"""RTX 3060 Ti 8GB向けQwen3-1.7B 4-bit QLoRA学習スクリプト。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from mini_llm.qlora_config import QLoRAConfig

IGNORE_INDEX = -100


@dataclass(frozen=True)
class MessagesExample:
    id: str
    system: str
    user: str
    assistant: str


class QwenSFTDataset(Dataset[dict[str, Tensor]]):
    """Qwen chat templateを適用し、assistant回答だけを正解にするDataset。"""

    def __init__(
        self,
        examples: Sequence[MessagesExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        example = self.examples[index]
        prompt_messages = [
            {"role": "system", "content": example.system},
            {"role": "user", "content": example.user},
        ]
        full_messages = [
            *prompt_messages,
            {"role": "assistant", "content": example.assistant},
        ]
        prompt_ids = _chat_token_ids(
            self.tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        full_ids = _chat_token_ids(
            self.tokenizer,
            full_messages,
            add_generation_prompt=False,
        )[: self.max_length]
        if len(full_ids) <= len(prompt_ids):
            raise ValueError(f"assistant response does not fit max_length: {example.id}")
        labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _chat_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    token_ids: Any = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if not isinstance(token_ids, list) or not all(isinstance(value, int) for value in token_ids):
        raise ValueError("chat template must return a list of token ids")
    return token_ids


def collate_batch(
    examples: Sequence[dict[str, Tensor]],
    *,
    pad_token_id: int,
) -> dict[str, Tensor]:
    max_length = max(example["input_ids"].numel() for example in examples)
    input_rows: list[Tensor] = []
    attention_rows: list[Tensor] = []
    label_rows: list[Tensor] = []
    for example in examples:
        padding = max_length - example["input_ids"].numel()
        input_rows.append(
            torch.nn.functional.pad(example["input_ids"], (0, padding), value=pad_token_id)
        )
        attention_rows.append(
            torch.nn.functional.pad(example["attention_mask"], (0, padding), value=0)
        )
        label_rows.append(
            torch.nn.functional.pad(example["labels"], (0, padding), value=IGNORE_INDEX)
        )
    return {
        "input_ids": torch.stack(input_rows),
        "attention_mask": torch.stack(attention_rows),
        "labels": torch.stack(label_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/qwen3_1_7b_qlora.yaml"),
    )
    args = parser.parse_args()
    config = QLoRAConfig.from_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")
    _set_seed(config.seed)

    examples = _load_examples(config.dataset_path)
    train_examples, validation_examples = _split_examples(
        examples,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        cache_dir=config.cache_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Qwen tokenizer must define a padding token")

    train_dataset = QwenSFTDataset(train_examples, tokenizer, config.max_length)
    validation_dataset = QwenSFTDataset(
        validation_examples,
        tokenizer,
        config.max_length,
    )
    collate = lambda batch: collate_batch(batch, pad_token_id=tokenizer.pad_token_id)  # noqa: E731
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        cache_dir=config.cache_dir,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            revision=config.model_revision,
            target_modules=list(config.target_modules),
        ),
    )
    model.print_trainable_parameters()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    train_iterator = _repeat(train_loader)
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, config.max_steps + 1):
        model.train()
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            batch = _move_batch(next(train_iterator), model.device)
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += float(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        should_evaluate = step % config.eval_interval == 0 or step == config.max_steps
        if should_evaluate:
            validation_loss = _evaluate(model, validation_loader)
            metric = {
                "step": step,
                "train_loss": accumulated_loss,
                "validation_loss": validation_loss,
                "perplexity": math.exp(min(validation_loss, 20.0)),
                "grad_norm": float(grad_norm),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            }
            _append_jsonl(metrics_path, metric)
            print(json.dumps(metric, ensure_ascii=False, sort_keys=True), flush=True)

        if step % config.save_interval == 0 or step == config.max_steps:
            adapter_path = config.output_dir / f"step_{step:04d}"
            model.save_pretrained(adapter_path, safe_serialization=True)
            tokenizer.save_pretrained(adapter_path)

    report = {
        "schema_version": 1,
        "config": asdict(config),
        "train_record_count": len(train_examples),
        "validation_record_count": len(validation_examples),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "final_adapter": str(config.output_dir / f"step_{config.max_steps:04d}"),
    }
    (config.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def _evaluate(model: Any, data_loader: DataLoader[dict[str, Tensor]]) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for batch in data_loader:
        moved = _move_batch(batch, model.device)
        outputs = model(**moved)
        count = int(torch.count_nonzero(moved["labels"] != IGNORE_INDEX))
        loss_sum += float(outputs.loss) * count
        token_count += count
    if token_count == 0:
        raise ValueError("validation dataset contains no assistant tokens")
    return loss_sum / token_count


def _load_examples(path: Path) -> list[MessagesExample]:
    examples: list[MessagesExample] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            raw: Any = json.loads(line)
            if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
                raise ValueError(f"invalid conversation at {path}:{line_number}")
            messages = raw.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError(f"conversation must contain three messages: {raw['id']}")
            contents = [_message_content(message, raw["id"]) for message in messages]
            roles = [
                message.get("role") if isinstance(message, dict) else None
                for message in messages
            ]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"conversation roles are invalid: {raw['id']}")
            examples.append(MessagesExample(raw["id"], *contents))
    if len(examples) < 2:
        raise ValueError("QLoRA dataset requires at least two examples")
    return examples


def _message_content(message: object, record_id: str) -> str:
    if not isinstance(message, dict):
        raise ValueError(f"conversation message must be a mapping: {record_id}")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError(f"conversation content must not be empty: {record_id}")
    return content


def _split_examples(
    examples: Sequence[MessagesExample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[MessagesExample], list[MessagesExample]]:
    shuffled = sorted(examples, key=lambda example: example.id)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    return shuffled[validation_count:], shuffled[:validation_count]


def _repeat(data_loader: DataLoader[dict[str, Tensor]]) -> Iterator[dict[str, Tensor]]:
    while True:
        yield from data_loader


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
