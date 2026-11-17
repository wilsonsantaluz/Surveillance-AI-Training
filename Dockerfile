FROM ghcr.io/astral-sh/uv:0.13.8 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=10 uv sync --locked --no-dev --extra inference --extra api

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SURVEILLANCE_HOST=0.0.0.0 \
    SURVEILLANCE_PORT=8000 \
    SURVEILLANCE_MODEL=/models/model.onnx \
    SURVEILLANCE_MANIFEST=/models/manifest.json

RUN useradd --create-home --uid 10001 app && mkdir /models && chown app:app /models
USER app
EXPOSE 8000
ENTRYPOINT ["surveillance-ml-serve"]
