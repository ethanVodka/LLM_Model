"""Qwen3-1.7B + LoRAを既存FastAPI契約で公開するGPUサーバー。"""

from __future__ import annotations

import argparse
from pathlib import Path
from threading import Lock

import torch
import uvicorn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mini_llm.api import GenerateRequest, GenerateResponse, create_app
from mini_llm.conversation import DEFAULT_SYSTEM_PROMPT
from mini_llm.qlora_config import QLoRAConfig


class QwenGenerationService:
    """4-bit基盤モデルとLoRA adapterを1度だけ読み込み、生成要求を直列化する。"""

    def __init__(self, config: QLoRAConfig, adapter_path: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen service requires a CUDA GPU")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            cache_dir=config.cache_dir,
            local_files_only=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            cache_dir=config.cache_dir,
            local_files_only=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        self.device = self.model.device
        self.checkpoint_step = _adapter_step(adapter_path)
        self.lock = Lock()

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        with self.lock, torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.temperature > 0.0,
                temperature=request.temperature if request.temperature > 0.0 else None,
                top_p=0.8 if request.temperature > 0.0 else None,
                top_k=request.top_k if request.temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = model_inputs["input_ids"].shape[1]
        completion_ids = generated[0, prompt_length:]
        return GenerateResponse(
            generated_text=self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            ).strip(),
            generated_token_count=int(completion_ids.numel()),
            checkpoint_step=self.checkpoint_step,
        )


def _adapter_step(path: Path) -> int:
    name = path.name
    if name.startswith("step_") and name.removeprefix("step_").isdigit():
        return int(name.removeprefix("step_"))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/qwen3_1_7b_qlora.yaml"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("artifacts/adapters/qwen3_1_7b_ja/step_0040"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    service = QwenGenerationService(QLoRAConfig.from_yaml(args.config), args.adapter)
    uvicorn.run(create_app(service=service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
