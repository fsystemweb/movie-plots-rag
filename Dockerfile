# MCP server image (docker compose service `mcp-server`). Build context: the repository root.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app

# Dependencies first (cached until pyproject.toml / uv.lock change), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config.yaml ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Non-root; /models holds the FastEmbed model cache (a named volume in docker-compose.yml).
RUN useradd --create-home --uid 10001 mcp && mkdir -p /models && chown mcp /models
USER mcp

EXPOSE 8000
CMD ["python", "-m", "movie_rag.mcp_server"]
