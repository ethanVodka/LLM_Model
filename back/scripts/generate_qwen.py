"""Qwen3-1.7Bの基盤モデルまたはLoRA adapterで会話生成する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mini_llm.conversation import DEFAULT_SYSTEM_PROMPT
from mini_llm.qlora_config import QLoRAConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/qwen3_1_7b_qlora.yaml"),
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen 4-bit generation requires a CUDA GPU")
    config = QLoRAConfig.from_yaml(args.config)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        cache_dir=config.cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        cache_dir=config.cache_dir,
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
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.0,
            temperature=args.temperature if args.temperature > 0.0 else None,
            top_p=0.8 if args.temperature > 0.0 else None,
            top_k=20 if args.temperature > 0.0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_length = model_inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()
    print(f"model={config.model_name}@{config.model_revision}")
    print(f"adapter={args.adapter}")
    print(f"gpu_memory_gib={torch.cuda.max_memory_allocated() / (1024**3):.2f}")
    print(f"text={response}")


if __name__ == "__main__":
    main()
