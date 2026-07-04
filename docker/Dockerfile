FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY back/pyproject.toml back/README.md /workspace/back/
COPY back/src /workspace/back/src
RUN python -m pip install --upgrade pip \
    && python -m pip install -e "/workspace/back[dev]"

COPY back/tests /workspace/back/tests
COPY configs /workspace/configs

CMD ["pytest", "back/tests"]

