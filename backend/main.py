from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core import config
from backend.core.middleware import BodySizeLimitMiddleware
from backend.utils.file_utils import logger   # also configures logging
from backend.api.routes import router


def load_spacy_model():
    """Load the primary spaCy model, fall back to the small one, or fail with instructions."""
    import spacy
    for name in (config.SPACY_MODEL_PRIMARY, config.SPACY_MODEL_SECONDARY):
        try:
            nlp = spacy.load(name)
            logger.info(f'Loaded spaCy model {name}')
            return nlp
        except OSError:
            logger.warning(f'spaCy model {name!r} is not installed')
    raise RuntimeError(
        'No spaCy model is installed. Install one with:\n'
        f'    python -m spacy download {config.SPACY_MODEL_PRIMARY}\n'
        '(The Dockerfile does this automatically.)'
    )


def load_embedder():
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
    except Exception as exc:
        raise RuntimeError(
            f'Could not load the sentence-transformer model {config.SENTENCE_TRANSFORMER_MODEL!r} '
            f'({type(exc).__name__}). The first start needs internet access to download it (~90 MB); '
            'afterwards it is cached.'
        ) from exc
    logger.info(f'Loaded SentenceTransformer {config.SENTENCE_TRANSFORMER_MODEL}')
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Starting ATS Resume Analyzer API...')

    problems = config.get_missing_config()
    if problems:
        raise RuntimeError(
            'Missing required configuration - the API cannot start:\n  - '
            + '\n  - '.join(problems)
            + '\nSet these as environment variables (or in .env). See .env.example.'
        )

    app.state.nlp = load_spacy_model()
    app.state.embedder = load_embedder()
    logger.info('All models loaded. API is ready to serve requests.')

    yield

    logger.info('Shutting down the API')


app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan,
    docs_url='/docs' if config.ENABLE_DOCS else None,
    redoc_url='/redoc' if config.ENABLE_DOCS else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,          # auth uses a Bearer header, not cookies
    allow_methods=['GET', 'POST', 'DELETE', 'OPTIONS'],
    allow_headers=['Authorization', 'Content-Type'],
)
app.add_middleware(
    BodySizeLimitMiddleware,
    limits={
        '/api/v1/analyze-resume': config.MAX_FILE_SIZE_BYTES + config.MAX_JD_CHARS * 4 + 64 * 1024,
        '/api/v1/generate-pdf': config.MAX_PDF_REQUEST_BYTES,
    },
)

app.include_router(router)


@app.get('/')
async def root():
    return {
        'name': 'ATS Resume Analyzer API',
        'version': config.APP_VERSION,
        'endpoints': {
            'POST   /api/v1/analyze-resume': 'Analyze a resume',
            'GET    /api/v1/history': 'Get user history',
            'DELETE /api/v1/history/{id}': 'Delete a history entry',
            'GET    /api/v1/history/{id}/pdf': 'PDF report for a saved analysis',
            'GET    /api/v1/health': 'Health check',
            'POST   /api/v1/generate-pdf': 'Generate PDF report from analysis data',
        },
    }


if __name__ == '__main__':
    # Local convenience only (run from the project root: `python -m backend.main`). For servers use
    # `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` (see Dockerfile) - one worker, no reload.
    import os
    import uvicorn
    uvicorn.run('backend.main:app', host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '8000')))
