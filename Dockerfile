# Use Python 3.14
FROM python:3.14-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/

WORKDIR /app

# Configure uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# Copy application
COPY . .

# Install the project
RUN uv sync --locked

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "src/telecharger_commons.py"]