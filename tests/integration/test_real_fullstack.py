"""OPT-IN full-stack test: real Groq + real Supabase (auth + database + RLS) + real spaCy/MiniLM + real WeasyPrint.

    set -a; . ./.env; set +a        # GROQ_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
    export IT_USER_A_EMAIL=... IT_USER_A_PASSWORD=... IT_USER_B_EMAIL=... IT_USER_B_PASSWORD=...
    RUN_INTEGRATION=1 pytest tests/integration/test_real_fullstack.py -v -s

Use TWO throw-away, email-confirmed test users in a TEST Supabase project. The test only creates and deletes
its own synthetic analysis row; it never lists, reads or deletes anything else.
"""
import os

import httpx
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.getenv('RUN_INTEGRATION') != '1', reason='set RUN_INTEGRATION=1')]

REQUIRED = ['GROQ_API_KEY', 'SUPABASE_URL', 'SUPABASE_ANON_KEY', 'IT_USER_A_EMAIL', 'IT_USER_A_PASSWORD', 'IT_USER_B_EMAIL', 'IT_USER_B_PASSWORD']

RESUME_HTML = """<html><body><h1>Jane Doe</h1><p>jane@example.com | +1 555 123 4567</p>
<h2>Summary</h2><p>Backend engineer with 3 years of experience building Python APIs.</p>
<h2>Experience</h2><p>Software Engineer, Acme Corp, 2022 - Present</p>
<ul><li>Built REST APIs with FastAPI serving 10K requests/day</li><li>Reduced query latency by 40% with PostgreSQL indexes</li><li>Deployed services with Docker</li></ul>
<h2>Education</h2><p>B.Sc. Computer Science, State University, 2021</p>
<h2>Skills</h2><p>Python, FastAPI, PostgreSQL, Docker</p></body></html>"""


def _sign_in(email, password):
    url, anon = os.environ['SUPABASE_URL'].rstrip('/'), os.environ['SUPABASE_ANON_KEY']
    r = httpx.post(f'{url}/auth/v1/token?grant_type=password', headers={'apikey': anon}, json={'email': email, 'password': password}, timeout=20)
    assert r.status_code == 200, f'sign-in failed (HTTP {r.status_code}) - is the test user created and confirmed?'
    return r.json()['access_token']


def test_full_flow_with_real_services():
    missing = [n for n in REQUIRED if not os.getenv(n)]
    if missing:
        pytest.skip('missing env: ' + ', '.join(missing))
    import weasyprint
    from fastapi.testclient import TestClient
    from backend.main import app

    pdf = weasyprint.HTML(string=RESUME_HTML).write_pdf()
    tok_a = _sign_in(os.environ['IT_USER_A_EMAIL'], os.environ['IT_USER_A_PASSWORD'])
    tok_b = _sign_in(os.environ['IT_USER_B_EMAIL'], os.environ['IT_USER_B_PASSWORD'])
    ha, hb = {'Authorization': f'Bearer {tok_a}'}, {'Authorization': f'Bearer {tok_b}'}

    with TestClient(app) as client:                      # runs the real lifespan: loads real spaCy + real MiniLM
        assert client.get('/api/v1/health').json()['embedder_loaded'] is True
        assert client.post('/api/v1/analyze-resume', files={'resume': ('r.pdf', pdf, 'application/pdf')}).status_code == 401
        r = client.post('/api/v1/analyze-resume', headers=ha, files={'resume': ('it-test.pdf', pdf, 'application/pdf')},
                        data={'job_description': 'Backend engineer: Python, FastAPI, AWS, Terraform.'})
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0 < body['ats_score'] <= 100 and body['strengths'] and body['job_title']
        assert 'Python' not in body['missing_keywords']

        history = client.get('/api/v1/history', headers=ha).json()
        mine = next(h for h in history if h['filename'] == 'it-test.pdf')
        try:
            assert client.get(f"/api/v1/history/{mine['id']}/pdf", headers=ha).content[:5] == b'%PDF-'
            # user B must not see, export or delete A's analysis (Row Level Security)
            assert all(h['id'] != mine['id'] for h in client.get('/api/v1/history', headers=hb).json())
            assert client.get(f"/api/v1/history/{mine['id']}/pdf", headers=hb).status_code == 404
            assert client.delete(f"/api/v1/history/{mine['id']}", headers=hb).status_code == 404
        finally:
            assert client.delete(f"/api/v1/history/{mine['id']}", headers=ha).status_code in (200, 204)
        assert all(h['id'] != mine['id'] for h in client.get('/api/v1/history', headers=ha).json())
