"""Central configuration for the ATS Resume Scorer backend.

Everything that changes between machines/environments is read from environment
variables (optionally loaded from a local ``.env`` file). See ``.env.example``
for the full list. Values that are pure business logic (score weights, file
types) live here as constants.
"""
import os
from pathlib import Path
from typing import List

# Load .env from the project root (two levels up from this file) explicitly.
# In production (Docker / hosting dashboards) real environment variables are used
# and the .env file simply does not exist.
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parents[2] / '.env'
    load_dotenv(_ENV_PATH)
except ImportError:  # python-dotenv is optional at runtime
    pass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, '').strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f'Environment variable {name} must be an integer, got {raw!r}')


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return default
    return raw in ('1', 'true', 'yes', 'on')


def _parse_origins(raw: str) -> List[str]:
    """Comma-separated origins -> normalised list (no trailing slash, no blanks)."""
    origins = []
    for item in raw.split(','):
        item = item.strip().rstrip('/')
        if item:
            origins.append(item)
    return origins


# ── API metadata ────────────────────────────────────────────────────────────
APP_TITLE = 'ATS RESUME ANALYZER API'
APP_VERSION = '2.0.0'
APP_DESCRIPTION = 'Analyse resumes against a job description using NLP + an LLM'
ENABLE_DOCS = _env_bool('ENABLE_DOCS', True)   # set false in production to hide /docs and /redoc

# ── CORS ────────────────────────────────────────────────────────────────────
# Comma-separated list of origins allowed to call the API from a browser.
# NOTE: the bundled Streamlit frontend calls this API from its *server*, so CORS
# does not apply to it; this only matters if you add a browser-based client.
ALLOWED_ORIGINS = _parse_origins(
    os.getenv('ALLOWED_ORIGINS', 'http://localhost:8501,http://127.0.0.1:8501')
)

# ── Uploads ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024   # zip-bomb guard for .docx
MAX_JD_CHARS = _env_int('MAX_JD_CHARS', 15000)
MAX_RESUME_TEXT_CHARS = _env_int('MAX_RESUME_TEXT_CHARS', 20000)  # text sent to the LLM
MAX_PDF_REQUEST_BYTES = 1024 * 1024               # body limit for /generate-pdf

# Supported document types, detected from file *content* (not the filename).
# 'doc' (legacy Word) is recognised only so we can return a clear message.
SUPPORTED_MIME_TYPES = {
    'application/pdf': 'pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/msword': 'doc',
}
ACCEPTED_UPLOAD_EXTENSIONS = ('pdf', 'docx')     # what the UI offers

# ── NLP models ──────────────────────────────────────────────────────────────
SPACY_MODEL_PRIMARY = 'en_core_web_md'    # better accuracy
SPACY_MODEL_SECONDARY = 'en_core_web_sm'  # smaller fallback
SENTENCE_TRANSFORMER_MODEL = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')

# ── Scoring ─────────────────────────────────────────────────────────────────
# Maximum points of each component (they add up to 100).
SCORE_COMPONENT_MAX = {
    'formatting': 20.0,
    'keywords': 25.0,
    'content': 25.0,
    'skill_validation': 15.0,
    'ats_compatibility': 15.0,
}
# How the overall score blends the components (fractions of the final 0-100 score):
#   40% skills&keywords (60% keywords + 40% skill validation), 30% content,
#   15% formatting, 15% ATS compatibility. See README "How the score is calculated".
SCORE_BLEND = {
    'skills_and_keywords': 0.40,
    'content': 0.30,
    'formatting': 0.15,
    'ats_compatibility': 0.15,
}
SCORE_KEYWORDS_SHARE_OF_SKILLS = 0.60      # remaining 0.40 is skill validation

JD_KEYWORD_WEIGHT = 0.6
JD_SEMANTIC_WEIGHT = 0.4

# ── External services ───────────────────────────────────────────────────────
SUPABASE_URL = os.getenv('SUPABASE_URL', '').strip()
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', '').strip()      # public "anon" key; RLS protects data
SUPABASE_JWT_SECRET = os.getenv('SUPABASE_JWT_SECRET', '').strip()  # only for legacy HS256 projects
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '').strip()
# Groq shut down llama-3.3-70b-versatile on 2026-08-16 (console.groq.com/docs/deprecations); their recommended
# replacement, openai/gpt-oss-120b, is a current production model. Override with GROQ_MODEL if Groq changes this again.
GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b').strip()
# gpt-oss models are reasoning models: their reasoning tokens count against the completion budget, so keep the effort
# low (resume -> JSON extraction needs no deep reasoning) and leave room for the JSON itself. Only sent to gpt-oss.
GROQ_REASONING_EFFORT = os.getenv('GROQ_REASONING_EFFORT', 'low').strip().lower()
GROQ_MAX_TOKENS = _env_int('GROQ_MAX_TOKENS', 8192)
GROQ_TIMEOUT_SECONDS = _env_int('GROQ_TIMEOUT_SECONDS', 45)
SUPABASE_TIMEOUT_SECONDS = _env_int('SUPABASE_TIMEOUT_SECONDS', 10)

# ── Abuse / resource protection (in-memory, per server process) ─────────────
ANALYZE_RATE_LIMIT_PER_MINUTE = _env_int('ANALYZE_RATE_LIMIT_PER_MINUTE', 5)   # per user
ANALYZE_RATE_LIMIT_PER_DAY = _env_int('ANALYZE_RATE_LIMIT_PER_DAY', 30)        # per user
GLOBAL_ANALYSES_PER_DAY = _env_int('GLOBAL_ANALYSES_PER_DAY', 500)             # all users (protects Groq quota)
PDF_RATE_LIMIT_PER_MINUTE = _env_int('PDF_RATE_LIMIT_PER_MINUTE', 10)          # per user
MAX_CONCURRENT_ANALYSES = _env_int('MAX_CONCURRENT_ANALYSES', 2)   # simultaneous parse+LLM+embedding jobs
MAX_CONCURRENT_PDFS = _env_int('MAX_CONCURRENT_PDFS', 2)
BUSY_WAIT_SECONDS = _env_int('BUSY_WAIT_SECONDS', 20)   # how long a request waits for a free analysis slot

# ── History ─────────────────────────────────────────────────────────────────
HISTORY_DEFAULT_LIMIT = 50
HISTORY_MAX_LIMIT = 100

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').strip().upper()
LOG_FILE = os.getenv('LOG_FILE', '').strip()   # empty = console only (recommended for containers)


def get_missing_config() -> List[str]:
    """Return human-readable problems with required configuration (empty list = OK)."""
    problems = []
    if not GROQ_API_KEY:
        problems.append('GROQ_API_KEY is not set (get one at https://console.groq.com)')
    if not SUPABASE_URL:
        problems.append('SUPABASE_URL is not set (Supabase dashboard -> Project Settings -> API)')
    if not SUPABASE_ANON_KEY:
        problems.append('SUPABASE_ANON_KEY is not set (Supabase dashboard -> Project Settings -> API)')
    return problems
