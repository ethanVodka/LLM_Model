"""学習済みモデルの生成機能を公開するFastAPIアプリケーション。"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer

from mini_llm.conversation import format_chat_prompt
from mini_llm.inference import GenerationConfig, generate_token_ids, load_checkpoint
from mini_llm.model import MiniDecoderLM
from mini_llm.tokenizer import load_tokenizer


class GenerateRequest(BaseModel):
    """ブラウザから受け取る生成条件。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=50, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=1024)
    seed: int = 42


class GenerateResponse(BaseModel):
    """生成文と再現・表示に必要な最小メタデータ。"""

    generated_text: str
    generated_token_count: int
    checkpoint_step: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class TextGenerator(Protocol):
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """検証済み条件で文字列を生成する。"""


@dataclass(frozen=True)
class ApiSettings:
    """API起動時に読み込むファイルと実行環境。"""

    checkpoint_path: Path
    tokenizer_path: Path
    device: str
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> ApiSettings:
        origins = os.getenv("API_CORS_ORIGINS", "http://localhost:5173")
        return cls(
            checkpoint_path=Path(
                os.getenv("CHECKPOINT_PATH", "artifacts/checkpoints/chat_demo/latest.pt")
            ),
            tokenizer_path=Path(
                os.getenv("TOKENIZER_PATH", "artifacts/tokenizer/chat_demo.json")
            ),
            device=os.getenv("DEVICE", "auto"),
            cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
        )


class GenerationService:
    """共有モデルへの同時アクセスを直列化する生成サービス。"""

    def __init__(
        self,
        model: MiniDecoderLM,
        tokenizer: Tokenizer,
        checkpoint_step: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.checkpoint_step = checkpoint_step
        self.device = device
        self.lock = Lock()

    @classmethod
    def load(cls, settings: ApiSettings) -> GenerationService:
        device = _resolve_device(settings.device)
        loaded = load_checkpoint(settings.checkpoint_path, device=device)
        tokenizer = load_tokenizer(settings.tokenizer_path)
        if tokenizer.get_vocab_size() != loaded.model.config.vocab_size:
            raise ValueError("tokenizer vocab_size must match checkpoint vocab_size")
        return cls(loaded.model, tokenizer, loaded.step, device)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        bos_token_id = self.tokenizer.token_to_id("<bos>")
        eos_token_id = self.tokenizer.token_to_id("<eos>")
        if bos_token_id is None or eos_token_id is None:
            raise ValueError("tokenizer must define <bos> and <eos> tokens")

        chat_prompt = format_chat_prompt(request.prompt)
        prompt_ids = [
            bos_token_id,
            *self.tokenizer.encode(chat_prompt, add_special_tokens=False).ids,
        ]
        with self.lock:
            generated_ids = generate_token_ids(
                self.model,
                prompt_ids,
                GenerationConfig(
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    seed=request.seed,
                ),
                eos_token_id=eos_token_id,
                device=self.device,
            )
        completion_ids = generated_ids[len(prompt_ids) :]
        return GenerateResponse(
            generated_text=self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            ).strip(),
            generated_token_count=len(generated_ids) - len(prompt_ids),
            checkpoint_step=self.checkpoint_step,
        )


def create_app(
    *,
    service: TextGenerator | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    """本番では起動時にモデルを読み、テストでは代替実装を渡す。"""

    api_settings = settings or ApiSettings.from_env()
    active_service = service

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal active_service
        if active_service is None:
            active_service = GenerationService.load(api_settings)
        yield

    application = FastAPI(
        title="Learning Mini LLM API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(api_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=active_service is not None)

    @application.post("/api/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        if active_service is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return active_service.generate(request)

    return application


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cpu", "cuda"}:
        raise ValueError("DEVICE must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(value)


app = create_app()
