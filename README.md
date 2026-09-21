# ATS Resume Scorer

A web app that scores how well a resume fits ATS (applicant tracking system) screening, optionally against a job description, and returns actionable feedback. FastAPI backend + Streamlit frontend, spaCy and Sentence Transformers for NLP, the Groq API (OpenAI gpt-oss-120b) to parse resumes and job descriptions into structured data, and Supabase for sign-in and saved history.

> **Credit.** This project is based on the course project **"AI Resume ATS" by Shradha Khapra** - <https://github.com/shradha-khapra/ai-resume-ats>. The original architecture, scoring design, prompts and UI come from that repository. This copy contains bug fixes, security hardening, tests and deployment files added on top of it (see [What changed](#what-changed-from-the-original-course-code)). The original repository does not include a LICENSE file: follow your course's attribution/usage rules, keep this credit in place, and do not add a license of your own unless the course allows it.

## What it does

1. Sign in (email/password or Google), upload a resume (**PDF or DOCX, max 5 MB**) and optionally paste a job description (or upload it as a `.txt` file).
2. The backend extracts the text, has Groq turn it into structured data (skills, experience, projects, action verbs, keywords), and scores it.
3. You get an overall ATS score, a breakdown by category, strengths, critical issues, suggestions, detailed per-issue feedback, and - when a job description is given - matched/missing keywords and a match percentage.
4. Analyses are saved to your account (history page) and can be exported as a PDF report or deleted.

## How the score is calculated

`overall = 40% skills/keywords + 30% content + 15% formatting + 15% ATS compatibility`, then small adjustments:

| Part | What is measured |
|---|---|
| Skills/keywords (40%) | 60% keyword coverage (resume keywords, skills, and - with a JD - match against the JD) + 40% "skill validation": whether each listed skill is actually evidenced in a project or job description (embedding similarity) |
| Content (30%) | number of distinct action verbs, and quantified achievements (%, $, counts) |
| Formatting (15%) | which sections are present and filled in (summary, experience, education, skills, projects) and how many bullet points are used |
| ATS compatibility (15%) | starts at 15 and loses points for a street address/postal code on the resume (privacy), table-drawing characters, and very short sections; +1 for a resume with experience and more than 5 skills |
| Adjustments | +0/+1/+2 for skill validation; with a JD, -0/5/10/15 for missing JD keywords |

The response includes `scoring_notes` that state exactly which adjustments were applied.

A resume that yields no text still receives the ATS-compatibility baseline (about 15 points) - a known simplification.

**Not evaluated:** grammar and spelling. The original code awarded every resume 10 free "perfect grammar" points using a checker that did not exist; that was removed, so scores are honest but lower than the original code's. Privacy/location detection is regex-only (street addresses and postal codes); it does not detect city or country.

The score is a *heuristic* to guide improvements. It is not what any real ATS computes, and parts of it depend on an LLM's output.

## Tech stack and layout

```
backend/    FastAPI app: routes, auth (Supabase JWT), scoring, parsing, PDF export, rate limiting
frontend/   Streamlit app (independent of backend code: talks to the API over HTTP, own requirements)
supabase/   schema.sql - table + Row Level Security policies (run once)
tests/      pytest suite (mocked) + tests/integration (opt-in, real services)
jupyter notebooks/   research/dataset prep, not used at runtime
Dockerfile  backend image
```

## Requirements

* **Python 3.11** (see `.python-version`; other versions are untested).
* A [Groq](https://console.groq.com) API key and a [Supabase](https://supabase.com) project (both have free tiers).
* **RAM:** the backend needs roughly 1-1.5 GB (measured ~0.8 GB with the spaCy model and PyTorch loaded and a tiny stand-in embedding model; the real MiniLM model adds more). A 512 MB host will not work; plan for 2 GB.
* **System libraries for PDF export (WeasyPrint):** Debian/Ubuntu `sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 fonts-dejavu-core`; Fedora `sudo dnf install pango`; macOS `brew install pango`; Windows: see the [WeasyPrint docs](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html). Without them the app runs but `/generate-pdf` returns an error.
* Internet on first start: the embedding model `all-MiniLM-L6-v2` (~90 MB) is downloaded once and cached.

## Setup (local)

```bash
python3.11 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt                          # backend + frontend, pinned versions
```

`requirements-backend.txt` also installs the spaCy model wheel (`en_core_web_md`), so no separate download step is needed. On Linux, install CPU-only PyTorch first to avoid a ~2 GB download: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`.

### 1. Supabase

1. Create a project. In **SQL Editor**, run the whole file `supabase/schema.sql` (safe to re-run; it never deletes data). It creates the `analyses` table with Row Level Security so users can only see, add and delete their own rows.
2. Run `supabase/rls_selftest.sql` the same way; the last line must read `ALL RLS CHECKS PASSED` (it creates two temporary users inside a transaction and rolls everything back).
3. **Project Settings -> API**: copy the *Project URL* and the **anon public** key. Never use the `service_role` key anywhere in this project.
4. **Authentication -> URL Configuration**: set *Site URL* to your frontend URL and add it (e.g. `http://localhost:8501`) to *Redirect URLs*.
5. Optional Google sign-in: create an OAuth client in Google Cloud Console whose *Authorized redirect URI* is `https://<your-project-ref>.supabase.co/auth/v1/callback`, enable the Google provider in Supabase (Authentication -> Providers) with that client ID/secret, and make sure the frontend's `AUTH_REDIRECT_URL` (or `[google_oauth] redirect_uri`) equals the app URL and is in Supabase's *Redirect URLs*.
6. Recommended for anything public: turn on email confirmation and CAPTCHA/rate limits under Authentication.
7. Legacy projects that still sign tokens with the HS256 "JWT secret" must also set `SUPABASE_JWT_SECRET` for the backend; new projects (asymmetric signing keys) need nothing extra.

### 2. Environment variables

```bash
cp .env.example .env                                    # backend + frontend values, placeholders only
cp frontend/.streamlit/secrets.toml.example frontend/.streamlit/secrets.toml   # optional alternative for the frontend
```

Required for the backend: `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`. The backend refuses to start with a clear message if one is missing. Every optional setting (CORS origins, rate limits, log file, model name, docs on/off ...) is documented in `.env.example`. Both the backend and the frontend read `.env` from the project root automatically. `.env` and `secrets.toml` are git-ignored - never commit them.

### 3. Run

```bash
# terminal 1 (from the project root; .env is loaded automatically)
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# terminal 2
streamlit run frontend/streamlit_app.py                 # http://localhost:8501
```

Health check: `GET http://localhost:8000/api/v1/health`. Interactive API docs: `/docs` (turn off with `ENABLE_DOCS=false`).

**Run a single backend worker.** The rate limiter and concurrency limits live in process memory; extra workers multiply memory use (each loads its own models) and split the limits.

## Security and privacy notes

* Every API route except `/` and `/health` requires a valid Supabase access token; history rows are additionally protected by Postgres Row Level Security because the backend queries as the signed-in user.
* Uploads are validated by content (not file name), size-limited (5 MB, streamed), DOCX zip-bombs and huge PDFs are rejected, and per-user rate limits apply to analysis and PDF export. Limits are per server process.
* PDF export escapes all user text and blocks every network/file fetch during rendering (no SSRF / local-file reads).
* **Resume text is sent to Groq** (a third party) for parsing. Tell your users, and check Groq's data terms before using real resumes in production. Resume text is not written to server logs.
* Prompt-injection inside a resume can influence the LLM-parsed fields (and therefore the score); the prompts treat resume text as untrusted data, which reduces but does not eliminate that.
* `/generate-pdf` renders the analysis data sent by the client, so a user can generate a PDF containing whatever numbers they submit. Use the saved-history PDF endpoint when the report must reflect a real analysis.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                   # ~137 tests + 8 opt-in ones (skipped), ~15 s, no network, no API keys needed
RUN_INTEGRATION=1 pytest tests/integration -v -s   # opt-in: REAL Groq, REAL Supabase, REAL MiniLM (see each file's header)
```

Real-service checks, in the order to run them once you have configured `.env` (each file's header lists the variables it needs):

1. `supabase/rls_selftest.sql` - paste into the Supabase SQL Editor after `schema.sql`; it must end with `ALL RLS CHECKS PASSED` (it rolls everything back).
2. `tests/integration/test_real_minilm.py` - downloads and loads the real embedding model, prints load time and memory.
3. `tests/integration/test_real_services.py` - real Groq parsing/error handling and a two-user Supabase isolation check.
4. `tests/integration/test_real_fullstack.py` - the whole flow (sign-in, analysis with JD, history, PDF, delete, cross-user access denied) against the real services.

`supabase/rls_selftest.sql` and the test wiring were validated on a local PostgreSQL 16 / local stand-ins, **not** against real Supabase, Groq or Hugging Face.

The default suite **mocks** Groq (canned JSON), Supabase (an in-memory PostgREST simulator, including RLS-style filtering) and the heavy models (tiny fakes). It does use real PDF/DOCX parsing, real WeasyPrint rendering, real spaCy for the model-loading test, and the real FastAPI/Streamlit apps. `tests/integration/` is the only place that talks to real Groq/Supabase and is the only thing that proves the SQL policies in `supabase/schema.sql` work on a real database.

## Deployment

The two halves deploy separately:

* **Frontend (Streamlit):** Streamlit Community Cloud works. Point it at `frontend/streamlit_app.py`; because that file lives in `frontend/`, Community Cloud uses `frontend/requirements.txt` (it looks in the main file's folder first, then the repo root). Add the contents of `frontend/.streamlit/secrets.toml.example` under *Secrets*, with `[backend] url` set to your deployed API and `[google_oauth] redirect_uri` set to the app's URL.
* **Backend (FastAPI):** any host that runs a Docker container with **>= 2 GB RAM**. Build with `docker build -t ai-resume-ats-api .` and set `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` (plus `SUPABASE_JWT_SECRET` for legacy projects) as the platform's secrets, and `ALLOWED_ORIGINS` to the frontend URL. Run **one** instance/worker (limits are in memory). The container listens on `$PORT` (default 8000) and reports health at `/api/v1/health`. Note: the `Dockerfile` has not been built by the author of these fixes - build it once locally first.
* Then in Supabase add the deployed frontend URL to *Redirect URLs* / *Site URL*.

Hosts with free tiers generally cannot run the backend (512 MB RAM); see the deployment notes delivered with this fix for options and costs.

## What changed from the original course code

Fixed a SyntaxError that stopped the backend from starting; corrected the LLM-response parsing; replaced the service-role-key design with per-user access + RLS (schema included); removed fake grammar/location scoring; wired strengths / critical issues / suggestions end to end; fixed job-description matching that ignored the resume's skills list; hardened PDF export, uploads, auth errors, CORS and logging; added rate limiting, non-blocking request handling, pinned dependencies, Dockerfile, `.env.example`, tests and this documentation. Removed the unused `recommendation_engine.py` (dead code that depended on the removed grammar checker; it remains in git history at commit `c5b6b6d`).

## Known limitations

* Scores are heuristic and partly LLM-dependent; grammar is not checked.
* Rate limits are per process and reset on restart.
* Signing in with Google from the Streamlit app uses a shared auth client (a limitation of the PKCE flow in Streamlit); simultaneous Google sign-ins by different users are not safe on one server. Email/password sign-in is not affected.
* Only PDF and DOCX are accepted (legacy `.doc` is not supported).
