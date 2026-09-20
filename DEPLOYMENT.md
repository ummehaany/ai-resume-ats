# Deployment notes

Prices and free-tier limits below were read from the providers' own pages on 2026-09-20 and change often - re-check before relying on them. **Nothing in this file has been deployed or tested on those platforms.**

## What the app needs

| | Backend (FastAPI) | Frontend (Streamlit) |
|---|---|---|
| Runtime | Python 3.11 in a Docker image (`Dockerfile`) | Python 3.11, `frontend/requirements.txt` (small) |
| RAM | **>= 2 GB** (about 0.8 GB measured with spaCy + PyTorch loaded and a tiny stand-in embedding model; the real MiniLM model adds more; PDF rendering and concurrent analyses add more) | small |
| Disk | an image of a few GB (estimate - CPU PyTorch + models; not built here); nothing persisted | none |
| Instances | **exactly one** (rate limits and concurrency limits are in memory) | any |
| Secrets | `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, (legacy: `SUPABASE_JWT_SECRET`) | `SUPABASE_URL`, `SUPABASE_ANON_KEY`, backend URL, OAuth redirect URL |
| Never | the Supabase `service_role` key anywhere | |

Uploads are processed in memory and never written to disk; nothing is stored except the analysis JSON in the user's Supabase row. PDF reports are generated on demand and streamed back.

## Platforms

* **Frontend - Streamlit Community Cloud (free).** Main file `frontend/streamlit_app.py`. Community Cloud looks for the requirements file in the main file's folder first and then the repo root, so `frontend/requirements.txt` is used (the root `requirements.txt` is only a convenience file for local development). Paste `frontend/.streamlit/secrets.toml.example` (filled in) into the app's *Secrets*, and pick Python 3.11 in *Advanced settings*.
* **Backend - options** (all need a container with >= 2 GB):
  * *Google Cloud Run* - free tier per month: 180,000 vCPU-seconds, 360,000 GiB-seconds, 2 M requests (needs a billing account; set a budget alert). Use 2 GiB memory, 1 vCPU, `--max-instances=1`, and expect a 30-90 s cold start after idle (models load at startup) unless you keep `--min-instances=1` (not free). Best low-cost fit for a portfolio project.
  * *Fly.io* - roughly $11-12/month for a 2 GB shared-CPU machine; always available and simple.
  * *Render* - the free web service has 512 MB (too small; it also sleeps after 15 min); the 2 GB "Standard" instance is $25/month.
  * *Hugging Face Spaces* - the free CPU Basic hardware has 16 GB RAM, but its documentation says Docker Spaces require a paid (PRO) account, so it is not a free option for this backend.
* Frontend and backend on different hosts is fine: set `ALLOWED_ORIGINS` on the backend to the frontend's URL and `[backend] url` in the frontend secrets to the backend's URL.

## After the first deploy

1. `GET <backend>/api/v1/health` returns `{"status":"healthy","nlp_loaded":true,"embedder_loaded":true}`.
2. In Supabase: *Authentication -> URL Configuration* -> Site URL = frontend URL, Redirect URLs include it; run `supabase/schema.sql` if you have not.
3. Sign in on the frontend, run one analysis, open History, export the PDF.
4. Run the opt-in real-service tests once: `RUN_INTEGRATION=1 pytest tests/integration -v` (see the file header for the test-user variables).
5. Consider `ENABLE_DOCS=false` on the backend, and Supabase email confirmation + CAPTCHA.
