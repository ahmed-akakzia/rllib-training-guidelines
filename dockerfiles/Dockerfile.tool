FROM alpine:3.18

RUN apk add --no-cache \
    python3 \
    py3-pip \
    git

COPY pyproject.toml .

RUN git init  \
    && git config --global --add safe.directory /tmp \
    && pip install uv==0.9.5 \
    && uv sync --dev

WORKDIR /app