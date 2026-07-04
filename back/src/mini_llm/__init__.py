"""学習用ミニLLMの公開API。"""

from mini_llm.config import ModelConfig
from mini_llm.model import MiniDecoderLM

# 利用側が内部のファイル配置を意識せず、`from mini_llm import ...` で参照できるようにする。
__all__ = ["MiniDecoderLM", "ModelConfig"]
