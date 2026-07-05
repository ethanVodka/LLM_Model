"""日本語とソースコードを扱うByte-level BPEトークナイザー。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

DEFAULT_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


@dataclass(frozen=True)
class TokenizerConfig:
    """BPE学習に必要な再現可能な設定。"""

    vocab_size: int
    min_frequency: int
    special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS

    def __post_init__(self) -> None:
        minimum_vocab_size = len(ByteLevel.alphabet()) + len(self.special_tokens)
        if self.vocab_size < minimum_vocab_size:
            raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")
        if self.min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if self.special_tokens[: len(DEFAULT_SPECIAL_TOKENS)] != DEFAULT_SPECIAL_TOKENS:
            raise ValueError(f"special_tokens must start with {DEFAULT_SPECIAL_TOKENS}")
        if any(not token for token in self.special_tokens):
            raise ValueError("special_tokens must not contain empty strings")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TokenizerConfig:
        """YAMLから型を検証し、トークナイザー設定を作る。"""

        with Path(path).open(encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
        if not isinstance(raw, dict):
            raise ValueError("tokenizer config must be a mapping")

        vocab_size = raw.get("vocab_size")
        min_frequency = raw.get("min_frequency")
        special_tokens = raw.get("special_tokens")
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise ValueError("vocab_size must be an integer")
        if not isinstance(min_frequency, int) or isinstance(min_frequency, bool):
            raise ValueError("min_frequency must be an integer")
        if not isinstance(special_tokens, list) or not all(
            isinstance(token, str) for token in special_tokens
        ):
            raise ValueError("special_tokens must be a list of strings")

        return cls(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=tuple(special_tokens),
        )


def create_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """未学習のByte-level BPEパイプラインを構築する。"""

    pad_token, unk_token, bos_token, eos_token = config.special_tokens[:4]
    tokenizer = Tokenizer(BPE(unk_token=unk_token))

    # NFKCで全角英数字などの互換文字を統一する。大文字小文字はコード上で意味を持つため保持する。
    tokenizer.normalizer = NFKC()
    # ByteLevelは任意のUTF-8入力を256種類のbyte表現へ分解でき、日本語とコードを同じ方式で扱える。
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 文頭・文末トークンを自動付与し、モデルが文の境界を学べるようにする。
    tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[
            (pad_token, 0),
            (unk_token, 1),
            (bos_token, 2),
            (eos_token, 3),
        ],
    )
    return tokenizer


def train_tokenizer(
    config: TokenizerConfig,
    texts: Iterable[str],
    output_path: str | Path,
    *,
    expected_vocab_size: int | None = None,
) -> Tokenizer:
    """テキスト列でBPEを学習し、検証済みtokenizer.jsonを保存する。"""

    if expected_vocab_size is not None and config.vocab_size != expected_vocab_size:
        raise ValueError(
            "tokenizer vocab_size must match model vocab_size: "
            f"{config.vocab_size} != {expected_vocab_size}"
        )

    tokenizer = create_tokenizer(config)
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        show_progress=False,
        special_tokens=list(config.special_tokens),
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != config.vocab_size:
        raise ValueError(
            "corpus is too small to build the configured vocabulary: "
            f"expected {config.vocab_size}, got {actual_vocab_size}"
        )
    _validate_special_token_ids(tokenizer, config.special_tokens)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(destination))
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """保存済みトークナイザーを読み、必須特殊トークンを検証する。"""

    tokenizer = Tokenizer.from_file(str(path))
    _validate_special_token_ids(tokenizer, DEFAULT_SPECIAL_TOKENS)
    return tokenizer


def _validate_special_token_ids(tokenizer: Tokenizer, special_tokens: Sequence[str]) -> None:
    """特殊トークンIDが学習や環境によらず0から固定されていることを確認する。"""

    for expected_id, token in enumerate(special_tokens):
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise ValueError(
                f"special token {token} must have id {expected_id}, got {actual_id}"
            )
