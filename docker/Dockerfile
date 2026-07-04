FROM ghcr.io/astral-sh/uv:0.11.26 AS uv

FROM python:3.12.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

COPY --from=uv /uv /uvx /bin/
COPY back/pyproject.toml back/uv.lock back/README.md /workspace/back/
COPY back/src /workspace/back/src
RUN uv sync --project /workspace/back --frozen --extra cpu

COPY back/tests /workspace/back/tests
COPY configs /workspace/configs

CMD ["pytest", "back/tests"]
