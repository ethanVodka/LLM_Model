import json
from pathlib import Path

import numpy as np
import pytest

from mini_llm.corpus import CorpusRecord, iter_jsonl_records
from mini_llm.dataset import (
    IGNORE_INDEX,
    DataConfig,
    MaskedNextTokenDataset,
    NextTokenDataset,
    build_sequences,
    prepare_dataset,
    prepare_sft_dataset,
    split_records,
)
from mini_llm.tokenizer import TokenizerConfig, train_tokenizer


def make_records(count: int = 10) -> list[CorpusRecord]:
    return [
        CorpusRecord(
            id=f"record-{index:02d}",
            text=f"文書番号{index}の学習サンプルです。value = {index}",
            source="project-original",
            license="project-original",
            language="mixed",
        )
        for index in range(count)
    ]


def test_splits_records_deterministically_without_overlap() -> None:
    records = make_records()

    first_train, first_validation = split_records(
        records,
        validation_fraction=0.2,
        seed=42,
    )
    second_train, second_validation = split_records(
        list(reversed(records)),
        validation_fraction=0.2,
        seed=42,
    )

    assert [record.id for record in first_train] == [record.id for record in second_train]
    assert [record.id for record in first_validation] == [
        record.id for record in second_validation
    ]
    assert {record.id for record in first_train}.isdisjoint(
        record.id for record in first_validation
    )


def test_builds_overlapping_next_token_sequences() -> None:
    sequences = build_sequences(list(range(11)), context_length=4)

    assert sequences.tolist() == [
        [0, 1, 2, 3, 4],
        [4, 5, 6, 7, 8],
    ]


def test_prepares_dataset_and_shifts_targets(tmp_path: Path) -> None:
    records = make_records(12)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(
        TokenizerConfig(vocab_size=300, min_frequency=1),
        [record.text for record in records],
        tokenizer_path,
    )
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("corpus hash input", encoding="utf-8")
    output_dir = tmp_path / "dataset"

    metadata = prepare_dataset(
        DataConfig(context_length=8, validation_fraction=0.25, seed=7),
        records,
        tokenizer,
        output_dir,
        tokenizer_path=tokenizer_path,
        corpus_paths=[corpus_path],
    )
    dataset = NextTokenDataset(output_dir / "train.npy")
    inputs, targets = dataset[0]

    assert len(dataset) == metadata["train_sequence_count"]
    assert inputs.shape == targets.shape == (8,)
    assert inputs[1:].tolist() == targets[:-1].tolist()
    assert int(inputs.max()) < tokenizer.get_vocab_size()
    assert (output_dir / "validation.npy").exists()
    assert json.loads((output_dir / "metadata.json").read_text(encoding="utf-8")) == metadata


def test_rejects_duplicate_corpus_ids(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    record = {
        "id": "duplicate",
        "text": "sample",
        "source": "project-original",
        "license": "project-original",
        "language": "en",
    }
    corpus_path.write_text(
        f"{json.dumps(record)}\n{json.dumps(record)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        list(iter_jsonl_records([corpus_path]))


def test_rejects_invalid_data_config() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        DataConfig(context_length=64, validation_fraction=1.0, seed=42)


def test_empty_token_stream_has_no_sequences() -> None:
    sequences = build_sequences([], context_length=8)

    assert sequences.shape == (0, 9)
    assert sequences.dtype == np.uint16


def test_prepares_sft_dataset_masking_prompt_tokens(tmp_path: Path) -> None:
    records = [
        CorpusRecord(
            id=f"chat-{index}",
            text=(
                "<system>簡潔に回答します。"
                f"<user>質問{index}<assistant>回答{index}です。"
            ),
            source="project-original",
            license="project-original",
            language="ja",
        )
        for index in range(4)
    ]
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(
        TokenizerConfig(
            vocab_size=290,
            min_frequency=1,
            special_tokens=(
                "<pad>",
                "<unk>",
                "<bos>",
                "<eos>",
                "<system>",
                "<user>",
                "<assistant>",
            ),
        ),
        [record.text for record in records],
        tokenizer_path,
    )
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("corpus hash input", encoding="utf-8")
    output_dir = tmp_path / "sft"

    metadata = prepare_sft_dataset(
        DataConfig(context_length=48, validation_fraction=0.25, seed=42),
        records,
        tokenizer,
        output_dir,
        tokenizer_path=tokenizer_path,
        corpus_paths=[corpus_path],
    )
    dataset = MaskedNextTokenDataset(
        output_dir / "train_inputs.npy",
        output_dir / "train_targets.npy",
    )
    inputs, targets = dataset[0]
    assistant_token_id = tokenizer.token_to_id("<assistant>")
    assert assistant_token_id is not None
    assistant_index = inputs.tolist().index(assistant_token_id)

    assert metadata["objective"] == "assistant-response"
    assert targets[:assistant_index].tolist() == [IGNORE_INDEX] * assistant_index
    assert targets[assistant_index].item() != IGNORE_INDEX
    assert targets[-1].item() == IGNORE_INDEX


def test_rejects_sft_record_without_assistant_role(tmp_path: Path) -> None:
    records = [
        CorpusRecord(
            id="invalid-chat",
            text="<user>質問だけです。",
            source="project-original",
            license="project-original",
            language="ja",
        ),
        CorpusRecord(
            id="valid-chat",
            text="<user>質問<assistant>回答",
            source="project-original",
            license="project-original",
            language="ja",
        ),
    ]
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(
        TokenizerConfig(
            vocab_size=290,
            min_frequency=1,
            special_tokens=(
                "<pad>",
                "<unk>",
                "<bos>",
                "<eos>",
                "<system>",
                "<user>",
                "<assistant>",
            ),
        ),
        [record.text for record in records],
        tokenizer_path,
    )

    with pytest.raises(ValueError, match="exactly one"):
        prepare_sft_dataset(
            DataConfig(context_length=32, validation_fraction=0.5, seed=1),
            records,
            tokenizer,
            tmp_path / "sft",
            tokenizer_path=tokenizer_path,
            corpus_paths=[tokenizer_path],
        )
