import json
from pathlib import Path

import pytest

from mini_llm.tokenizer import (
    DEFAULT_SPECIAL_TOKENS,
    TokenizerConfig,
    iter_jsonl_texts,
    load_tokenizer,
    train_tokenizer,
)


def test_trains_and_round_trips_japanese_and_code(tmp_path: Path) -> None:
    config = TokenizerConfig(
        vocab_size=280,
        min_frequency=1,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )
    texts = [
        "日本語の文章をトークンへ変換します。",
        "def hello(name: str) -> str:\n    return f'Hello, {name}'",
        "type Status = 'idle' | 'training' | 'complete'",
    ]
    output_path = tmp_path / "tokenizer.json"

    trained = train_tokenizer(config, texts, output_path, expected_vocab_size=280)
    loaded = load_tokenizer(output_path)

    assert trained.get_vocab_size() == 280
    assert [loaded.token_to_id(token) for token in DEFAULT_SPECIAL_TOKENS] == [0, 1, 2, 3]
    sample = "日本語とPython: print('こんにちは')"
    encoding = loaded.encode(sample)
    assert encoding.ids[0] == 2
    assert encoding.ids[-1] == 3
    assert loaded.decode(encoding.ids) == sample


def test_nfkc_normalizes_compatible_characters(tmp_path: Path) -> None:
    config = TokenizerConfig(vocab_size=268, min_frequency=1)
    tokenizer = train_tokenizer(config, ["ABC abc 123"], tmp_path / "tokenizer.json")

    assert tokenizer.encode("ＡＢＣ", add_special_tokens=False).ids == tokenizer.encode(
        "ABC", add_special_tokens=False
    ).ids


def test_rejects_vocab_size_mismatch(tmp_path: Path) -> None:
    config = TokenizerConfig(vocab_size=270, min_frequency=1)

    with pytest.raises(ValueError, match="must match"):
        train_tokenizer(
            config,
            ["sample text"],
            tmp_path / "tokenizer.json",
            expected_vocab_size=512,
        )


def test_reads_jsonl_texts(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": "1",
            "text": "first",
            "source": "project-original",
            "license": "project-original",
            "language": "en",
        },
        {
            "id": "2",
            "text": "second",
            "source": "project-original",
            "license": "project-original",
            "language": "en",
        },
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    assert list(iter_jsonl_texts([corpus_path])) == ["first", "second"]


def test_rejects_corpus_without_license(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "id": "1",
                "text": "sample",
                "source": "project-original",
                "language": "en",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="license"):
        list(iter_jsonl_texts([corpus_path]))


def test_training_is_deterministic(tmp_path: Path) -> None:
    config = TokenizerConfig(vocab_size=280, min_frequency=1)
    texts = [
        "deterministic tokenizer training",
        "同じ入力から同じ語彙を作ります。",
        "const value: number = 42",
    ]

    first = train_tokenizer(config, texts, tmp_path / "first.json")
    second = train_tokenizer(config, texts, tmp_path / "second.json")

    assert first.to_str() == second.to_str()
