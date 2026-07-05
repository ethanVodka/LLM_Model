"""トークン列を次トークン予測用データセットへ変換する処理。"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from tokenizers import Tokenizer
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

from mini_llm.corpus import CorpusRecord

IGNORE_INDEX = -100


@dataclass(frozen=True)
class DataConfig:
    """文書分割と固定長系列化を再現する設定。"""

    context_length: int
    validation_fraction: float
    seed: int

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        """YAMLから型を検証してデータ設定を作る。"""

        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("data config must be a mapping")

        context_length = raw.get("context_length")
        validation_fraction = raw.get("validation_fraction")
        seed = raw.get("seed")
        if not isinstance(context_length, int) or isinstance(context_length, bool):
            raise ValueError("context_length must be an integer")
        if not isinstance(validation_fraction, int | float) or isinstance(
            validation_fraction, bool
        ):
            raise ValueError("validation_fraction must be a number")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        return cls(context_length, float(validation_fraction), seed)


def split_records(
    records: Sequence[CorpusRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[CorpusRecord], list[CorpusRecord]]:
    """文書単位で決定的にtrainとvalidationへ分割する。"""

    if len(records) < 2:
        raise ValueError("at least two corpus records are required")
    shuffled = sorted(records, key=lambda record: record.id)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    return train, validation


def tokenize_records(records: Sequence[CorpusRecord], tokenizer: Tokenizer) -> list[int]:
    """文書ごとに境界トークンを付け、1本のトークン列へ連結する。"""

    token_ids: list[int] = []
    for record in records:
        token_ids.extend(tokenizer.encode(record.text, add_special_tokens=True).ids)
    return token_ids


def build_sequences(token_ids: Sequence[int], context_length: int) -> NDArray[np.uint16]:
    """1トークン重なる`context_length + 1`の学習窓を作る。"""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if any(token_id < 0 or token_id > np.iinfo(np.uint16).max for token_id in token_ids):
        raise ValueError("token ids must fit in uint16")

    window_size = context_length + 1
    windows = [
        token_ids[start : start + window_size]
        for start in range(0, len(token_ids) - context_length, context_length)
    ]
    if not windows:
        return np.empty((0, window_size), dtype=np.uint16)
    return np.asarray(windows, dtype=np.uint16)


def prepare_dataset(
    config: DataConfig,
    records: Sequence[CorpusRecord],
    tokenizer: Tokenizer,
    output_dir: str | Path,
    *,
    tokenizer_path: str | Path,
    corpus_paths: Sequence[str | Path],
) -> dict[str, object]:
    """分割・ID化・系列化を行い、NumPy配列と監査用メタデータを保存する。"""

    train_records, validation_records = split_records(
        records,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_tokens = tokenize_records(train_records, tokenizer)
    validation_tokens = tokenize_records(validation_records, tokenizer)
    train_sequences = build_sequences(train_tokens, config.context_length)
    validation_sequences = build_sequences(validation_tokens, config.context_length)
    if len(train_sequences) == 0 or len(validation_sequences) == 0:
        raise ValueError(
            "both splits must contain at least one sequence; "
            "reduce context_length or provide more corpus records"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "train.npy", train_sequences, allow_pickle=False)
    np.save(destination / "validation.npy", validation_sequences, allow_pickle=False)

    tokenizer_file = Path(tokenizer_path)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "context_length": config.context_length,
        "validation_fraction": config.validation_fraction,
        "seed": config.seed,
        "vocab_size": tokenizer.get_vocab_size(),
        "dtype": "uint16",
        "tokenizer_sha256": _sha256_file(tokenizer_file),
        "corpus_sha256": _sha256_files(corpus_paths),
        "train_record_ids": [record.id for record in train_records],
        "validation_record_ids": [record.id for record in validation_records],
        "train_token_count": len(train_tokens),
        "validation_token_count": len(validation_tokens),
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def prepare_sft_dataset(
    config: DataConfig,
    records: Sequence[CorpusRecord],
    tokenizer: Tokenizer,
    output_dir: str | Path,
    *,
    tokenizer_path: str | Path,
    corpus_paths: Sequence[str | Path],
) -> dict[str, object]:
    """会話文書を固定長化し、assistant回答部分だけを学習対象として保存する。"""

    train_records, validation_records = split_records(
        records,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_inputs, train_targets = _build_sft_arrays(
        train_records,
        tokenizer,
        config.context_length,
    )
    validation_inputs, validation_targets = _build_sft_arrays(
        validation_records,
        tokenizer,
        config.context_length,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "train_inputs.npy", train_inputs, allow_pickle=False)
    np.save(destination / "train_targets.npy", train_targets, allow_pickle=False)
    np.save(destination / "validation_inputs.npy", validation_inputs, allow_pickle=False)
    np.save(destination / "validation_targets.npy", validation_targets, allow_pickle=False)

    metadata: dict[str, object] = {
        "schema_version": 2,
        "objective": "assistant-response",
        "context_length": config.context_length,
        "validation_fraction": config.validation_fraction,
        "seed": config.seed,
        "vocab_size": tokenizer.get_vocab_size(),
        "input_dtype": "int64",
        "target_dtype": "int64",
        "ignore_index": IGNORE_INDEX,
        "tokenizer_sha256": _sha256_file(Path(tokenizer_path)),
        "corpus_sha256": _sha256_files(corpus_paths),
        "train_record_ids": [record.id for record in train_records],
        "validation_record_ids": [record.id for record in validation_records],
        "train_sequence_count": len(train_inputs),
        "validation_sequence_count": len(validation_inputs),
        "train_target_token_count": int(np.count_nonzero(train_targets != IGNORE_INDEX)),
        "validation_target_token_count": int(
            np.count_nonzero(validation_targets != IGNORE_INDEX)
        ),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _build_sft_arrays(
    records: Sequence[CorpusRecord],
    tokenizer: Tokenizer,
    context_length: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    assistant_token_id = tokenizer.token_to_id("<assistant>")
    pad_token_id = tokenizer.token_to_id("<pad>")
    if assistant_token_id is None or pad_token_id is None:
        raise ValueError("SFT tokenizer must define <pad> and <assistant>")

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for record in records:
        token_ids = tokenizer.encode(record.text, add_special_tokens=True).ids
        assistant_positions = [
            index for index, token_id in enumerate(token_ids) if token_id == assistant_token_id
        ]
        if len(assistant_positions) != 1:
            raise ValueError(f"SFT record must contain exactly one <assistant>: {record.id}")
        assistant_index = assistant_positions[0]
        sequence = token_ids[: context_length + 1]
        if assistant_index >= len(sequence) - 1:
            raise ValueError(f"SFT answer does not fit in context_length: {record.id}")

        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        masked_targets = [
            token_id if index >= assistant_index else IGNORE_INDEX
            for index, token_id in enumerate(target_ids)
        ]
        padding_length = context_length - len(input_ids)
        inputs.append(input_ids + [pad_token_id] * padding_length)
        targets.append(masked_targets + [IGNORE_INDEX] * padding_length)

    return (
        np.asarray(inputs, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


class NextTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """保存済み系列から入力と1トークン先の正解を返すPyTorch Dataset。"""

    def __init__(self, path: str | Path) -> None:
        sequences = np.load(Path(path), mmap_mode="r", allow_pickle=False)
        if sequences.ndim != 2 or sequences.shape[1] < 2:
            raise ValueError("dataset must have shape [samples, context_length + 1]")
        self.sequences: NDArray[np.uint16] = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # 読み取り専用memmapの警告を避けるため、独立したint64配列へコピーする。
        row = np.array(self.sequences[index], dtype=np.int64, copy=True)
        tokens = from_numpy(row)
        return tokens[:-1], tokens[1:]


class MaskedNextTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """入力配列と、非回答部分を無視する正解配列を読み込むDataset。"""

    def __init__(self, inputs_path: str | Path, targets_path: str | Path) -> None:
        inputs = np.load(Path(inputs_path), mmap_mode="r", allow_pickle=False)
        targets = np.load(Path(targets_path), mmap_mode="r", allow_pickle=False)
        if inputs.ndim != 2 or inputs.shape != targets.shape or inputs.shape[1] < 1:
            raise ValueError("masked dataset inputs and targets must have the same 2D shape")
        self.inputs: NDArray[np.int64] = inputs
        self.targets: NDArray[np.int64] = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        inputs = np.array(self.inputs[index], dtype=np.int64, copy=True)
        targets = np.array(self.targets[index], dtype=np.int64, copy=True)
        return from_numpy(inputs), from_numpy(targets)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_files(paths: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    for path_value in paths:
        digest.update(Path(path_value).read_bytes())
    return digest.hexdigest()
