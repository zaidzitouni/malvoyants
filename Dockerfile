# Python 3.14 sur Debian Bookworm (variante figée : les noms des paquets
# système ci-dessous sont garantis, contrairement au tag "slim" seul qui
# peut changer de version de Debian).
FROM python:3.14-slim-bookworm

# Install uv (version figée)
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/

# Dépendances système :
#  - tesseract-ocr        moteur OCR (anglais inclus)
#  - tesseract-ocr-fra    français
#  - tesseract-ocr-ara    arabe (plaques bilingues marocaines)
#  - libglib2.0-0         requis par opencv-python-headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

# Install dependencies first (couche mise en cache tant que pyproject.toml
# et uv.lock ne changent pas). --no-dev : pas de Jupyter dans l'image.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Copy application (data/ et results/ sont exclus par .dockerignore : montés en volumes)
COPY . .

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# L'image lance n'importe quel script du projet :
#   docker compose run --rm ocr scripts/telecharger_commons.py --preset
#   docker compose run --rm ocr scripts/etape2_pretraitement.py --ocr
ENTRYPOINT ["python"]
CMD ["scripts/etape2_pretraitement.py", "--help"]
