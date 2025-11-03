# IMPORTANT: This Dockerfile is to be used if you chose python 3.12 as your python version. 
# Please update the base image if you chose a different python version.
FROM python:3.12-slim AS builder

WORKDIR /app

COPY ../pyproject.toml /app/pyproject.toml

RUN pip install uv==0.9.5 && \
    uv venv && \
    uv sync

COPY . .

RUN uv pip install -e .

# IMPORTANT: This Dockerfile is to be used if you chose python 3.12 as your python version. 
# Please update the base image if you chose a different python version.
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app /app

ENTRYPOINT ["bash", "-c"]
