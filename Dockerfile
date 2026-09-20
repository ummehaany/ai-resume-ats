# Backend image (FastAPI + spaCy + sentence-transformers + WeasyPrint).
# Build:  docker build -t ai-resume-ats-api .
# Run:    docker run --rm -p 8000:8000 --env-file .env ai-resume-ats-api
# NOTE: this Dockerfile was written but could NOT be built in the environment where the project was
# fixed (no Docker available there) - do a local build once before relying on it.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

# System libraries WeasyPrint needs (Pango/HarfBuzz) + a font so PDFs have readable text.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel bundles ~2 GB of CUDA libraries the server never uses).
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-backend.txt .
RUN pip install -r requirements-backend.txt

# Download the embedding model at build time so the container starts fast and works offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY backend ./backend

# Run as an unprivileged user; the model cache stays readable.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app && chmod -R a+rX /opt/hf-cache
USER appuser

# Platforms such as Render / Cloud Run / Fly inject $PORT. ONE worker on purpose: the rate limiter and the
# concurrency limits are per-process, and every extra worker would load its own copy of the models (~1.5 GB).
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/api/v1/health' % os.environ.get('PORT','8000'),timeout=4)"
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
