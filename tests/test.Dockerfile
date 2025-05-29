ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /build 

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* 

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv $HOME/.local/bin/uv /usr/local/bin/ 
COPY pyproject.toml /build
COPY uv.lock /build

RUN uv venv 
RUN uv sync --group test --no-install-project
