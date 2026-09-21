# Deploying the backend to Google Cloud Run

Nothing in this guide has been run for you. Every command below is for you to run after you have reviewed it.
Replace the `UPPER_CASE` placeholders. Commands assume macOS/zsh and the project root as the working directory.

## 0. What this deploys

Only the FastAPI backend (see `Dockerfile`: it copies `requirements-backend.txt` and `backend/`).
The Streamlit frontend stays on Streamlit Community Cloud and reaches this API over HTTPS.

Sizing (why these numbers):

- Memory: **2 GiB**. The API holds spaCy `en_core_web_md`, PyTorch and MiniLM in RAM. Peak measured locally with a
  stand-in model was about 1.26 GB; the real MiniLM should be roughly 1.4-1.5 GB (an estimate, not measured). 1 GiB will be OOM-killed.
- CPU: **1 vCPU** with startup CPU boost. Model loading is the slow part, not request handling.
- Instances: **max 1**. Rate limits and concurrency guards are held in process memory, so a second instance would
  silently double every limit.
- `--allow-unauthenticated`: required, because the Streamlit app calls the API directly with a Supabase user token.
  The API itself verifies that token on every protected route.

## 1. One-time Google Cloud setup (needs your login and billing)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-central1          # pick the region closest to your users

# Billing must already be enabled on the project (console.cloud.google.com/billing).
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com
```

Recommended: create a budget alert (Billing > Budgets & alerts) before deploying anything.

## 2. Store secrets in Secret Manager

Type the value when prompted (input is hidden and never lands in shell history):

```bash
read -rs "?Paste GROQ_API_KEY: " GROQ && echo
printf %s "$GROQ" | gcloud secrets create groq-api-key --data-file=-
unset GROQ
```

If your Supabase project signs user tokens with the legacy HS256 secret (see section 7), also:

```bash
read -rs "?Paste SUPABASE_JWT_SECRET: " JWT && echo
printf %s "$JWT" | gcloud secrets create supabase-jwt-secret --data-file=-
unset JWT
```

Let Cloud Run's runtime service account read them:

```bash
PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
for s in groq-api-key supabase-jwt-secret; do
  gcloud secrets add-iam-policy-binding "$s" \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor
done
```

(Skip `supabase-jwt-secret` in the loop if you did not create it.)

## 3. Build the image (Cloud Build, produces the correct linux/amd64 image)

Build in the cloud rather than on an Apple Silicon Mac, which would produce an arm64 image Cloud Run cannot use.
`.gcloudignore` keeps `.env`, the frontend, tests and notebooks out of the upload.

```bash
REGION=us-central1
PROJECT=$(gcloud config get-value project)
IMAGE=$REGION-docker.pkg.dev/$PROJECT/ats/ai-resume-ats-api

gcloud artifacts repositories create ats --repository-format=docker --location=$REGION
gcloud builds submit --tag "$IMAGE" --timeout=30m .
```

The first build is slow (PyTorch, spaCy model, MiniLM download). Later builds reuse cached layers.

## 4. Deploy

Load your existing Supabase values from `.env` into the shell without printing them:

```bash
set -a; source .env; set +a

gcloud run deploy ai-resume-ats-api \
  --image "$IMAGE" \
  --allow-unauthenticated \
  --memory 2Gi --cpu 1 --cpu-boost \
  --min-instances 0 --max-instances 1 \
  --concurrency 10 --timeout 120 \
  --startup-probe httpGet.path=/api/v1/health,httpGet.port=8080,periodSeconds=10,failureThreshold=24,timeoutSeconds=5 \
  --set-env-vars "SUPABASE_URL=$SUPABASE_URL,SUPABASE_ANON_KEY=$SUPABASE_ANON_KEY,ENABLE_DOCS=false,LOG_LEVEL=INFO" \
  --set-secrets "GROQ_API_KEY=groq-api-key:latest"
```

With the legacy JWT secret, use this `--set-secrets` line instead:

```
  --set-secrets "GROQ_API_KEY=groq-api-key:latest,SUPABASE_JWT_SECRET=supabase-jwt-secret:latest"
```

Cloud Run injects `PORT=8080`; the Dockerfile's start command honours it. The startup probe allows up to 240 s
(24 x 10 s), which is the maximum Cloud Run permits and is enough for model loading.

Environment summary for the host:

| Name | Kind | Required | Notes |
|---|---|---|---|
| `SUPABASE_URL` | env var | yes | `https://<ref>.supabase.co` |
| `SUPABASE_ANON_KEY` | env var | yes | anon/publishable key only, never the service-role key |
| `GROQ_API_KEY` | secret | yes | |
| `SUPABASE_JWT_SECRET` | secret | only for HS256 projects | see section 7 |
| `ENABLE_DOCS` | env var | no | keep `false` in production |
| `LOG_LEVEL` | env var | no | |
| `ALLOWED_ORIGINS` | env var | no | Streamlit calls the API server-side, so CORS is normally irrelevant |

## 5. Get the URL and connect Streamlit

```bash
gcloud run services describe ai-resume-ats-api --format='value(status.url)'
# e.g. https://ai-resume-ats-api-abc123-uc.a.run.app
```

Check it:

```bash
curl https://YOUR-URL/api/v1/health
```

In Streamlit Community Cloud: app > Settings > Secrets, add (the `[backend]` table, exactly as the frontend reads it):

```toml
[backend]
url = "https://ai-resume-ats-api-abc123-uc.a.run.app"
```

Then reboot the app. Do not add a trailing slash.

Check the startup log for the auth line:

```bash
gcloud run services logs read ai-resume-ats-api --limit 50
```

Look for `Auth check: ...`. An `ERROR` there means every sign-in will fail with 401 until `SUPABASE_JWT_SECRET` is set.

## 6. Cold starts and cost

With `--min-instances 0` the service scales to zero. The first request after idle reloads the models, typically
30-90 s (an estimate; measure it). The frontend's history timeout is 30 s and PDF timeout 60 s, so a cold first
request may show a timeout error; retrying works once the instance is warm. Options:

- `--min-instances 1`: no cold starts, but you pay for an always-on 2 GiB instance (order of $19/month idle - verify on the Cloud Run pricing page).
- Leave at 0 and accept an occasional slow first request (free tier likely covers light use; verify current limits).
- A Cloud Scheduler ping to `/api/v1/health` every few minutes keeps it warm but has the same effect as min-instances and adds its own cost (unverified).

Artifact Registry charges a small storage fee for the image (a few GB).

## 7. Which Supabase token type do you have?

Open `https://<your-ref>.supabase.co/auth/v1/.well-known/jwks.json` in a browser.

- `keys` has entries: asymmetric signing keys; the API verifies tokens through that endpoint, nothing more to set.
- `keys` is empty: legacy HS256 project; set `SUPABASE_JWT_SECRET` (Supabase dashboard > Project Settings > JWT Keys / API).

The API logs which case it found at startup.

## 8. Supabase database (needs you; the API never changes your schema)

1. In the Supabase SQL Editor run `supabase/schema.sql` if you have not already (creates the `analyses` table and Row Level Security).
2. Optionally run `supabase/rls_selftest.sql`. It inserts throwaway users into `auth.users` and **rolls everything back**, but review it before running on a real project.

## 9. Checks you can run on your own Mac before deploying

```bash
brew install pango                       # WeasyPrint needs it; missing Pango is what fails the PDF tests
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-backend.txt -r requirements-dev.txt
python -m spacy download en_core_web_md
pytest                                    # expected: all pass, a few skipped
```

On Apple Silicon, if WeasyPrint still cannot find Pango: `export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`.

Local Docker smoke test (needs Docker Desktop; builds native arm64, so this checks the Dockerfile but is not the image you deploy):

```bash
docker build -t ai-resume-ats-api .
docker run --rm -p 8000:8000 -e PORT=8000 --env-file .env ai-resume-ats-api
curl http://localhost:8000/api/v1/health
```

Real-service integration tests (they use your real Supabase project and Groq quota; create two dedicated test users first):

```bash
RUN_INTEGRATION=1 IT_USER_A_EMAIL=... IT_USER_A_PASSWORD=... IT_USER_B_EMAIL=... IT_USER_B_PASSWORD=... \
  pytest tests/integration
```

## 10. Rollback / cleanup

```bash
gcloud run services delete ai-resume-ats-api          # stop all charges from the service
gcloud artifacts repositories delete ats --location=us-central1
gcloud secrets delete groq-api-key
```
