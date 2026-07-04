"""学習用ミニLLMの中核パッケージ。"""

from mini_llm.config import ModelConfig
from mini_llm.model import MiniDecoderLM

__all__ = ["MiniDecoderLM", "ModelConfig"]

