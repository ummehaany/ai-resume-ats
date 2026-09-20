"""End-to-end API tests through FastAPI's TestClient.

MOCKED: Groq (mock_llm), Supabase (fake_supabase), spaCy/MiniLM (fakes).
REAL:   FastAPI routing/validation, JWT verification, file parsing (real PDF/DOCX), scoring,
        PDF rendering, rate limiting, request-size limits, threadpool offloading.
"""
import asyncio
import time

import httpx
import pytest

from backend.core import config
from tests.conftest import USER_A, USER_B, auth_header, make_token

SAMPLE_JD = 'We need a Senior Backend Engineer with Python, FastAPI, AWS and Terraform experience.'


def analyze(client, pdf, user=USER_A, jd=None, filename='resume.pdf'):
    data = {'job_description': jd} if jd is not None else {}
    return client.post('/api/v1/analyze-resume', files={'resume': (filename, pdf, 'application/pdf')}, data=data, headers=auth_header(user))


# ── auth ────────────────────────────────────────────────────────────────────
def test_health_is_public_and_reports_models(client):
    r = client.get('/api/v1/health')
    assert r.status_code == 200 and r.json() == {'status': 'healthy', 'nlp_loaded': True, 'embedder_loaded': True}


@pytest.mark.parametrize('method, path', [
    ('post', '/api/v1/analyze-resume'), ('get', '/api/v1/history'),
    ('delete', '/api/v1/history/11111111-1111-4111-8111-111111111111'), ('post', '/api/v1/generate-pdf'),
    ('get', '/api/v1/history/11111111-1111-4111-8111-111111111111/pdf'),
])
def test_every_protected_route_requires_a_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_bad_tokens_are_rejected(client):
    for token in ['garbage', make_token(expires_in=-10), make_token(audience='anon'), make_token(secret='x' * 40)]:
        r = client.get('/api/v1/history', headers={'Authorization': f'Bearer {token}'})
        assert r.status_code == 401, token
        assert 'x' * 40 not in r.text


def test_expired_token_message(client):
    r = client.get('/api/v1/history', headers={'Authorization': f'Bearer {make_token(expires_in=-10)}'})
    assert r.status_code == 401 and 'expired' in r.json()['detail'].lower()


def test_unsupported_jwt_algorithm_is_rejected(client):
    import jwt
    # 'none' algorithm token must never be accepted
    token = jwt.encode({'sub': USER_A, 'aud': 'authenticated'}, key=None, algorithm='none')
    assert client.get('/api/v1/history', headers={'Authorization': f'Bearer {token}'}).status_code == 401


# ── analysis: happy paths (Groq mocked) ─────────────────────────────────────
def test_analyze_pdf_without_job_description(client, resume_pdf, mock_llm, fake_supabase):
    r = analyze(client, resume_pdf)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < body['ATS_score'] <= 100 and body['ATS_score'] == body['ats_score']
    assert set(body['component_scores']) == {'formatting', 'keywords', 'content', 'skill_validation', 'ats_compatibility'}
    assert body['jd_comparison'] is None and body['job_title'] is None
    assert len(mock_llm) == 1                                   # exactly one Groq call for a resume-only analysis
    assert body['strengths'], 'strengths must be populated from real findings'
    assert body['scoring_notes'] and any('Grammar' in n for n in body['scoring_notes'])
    assert isinstance(body['critical_issues'], list) and isinstance(body['suggestions'], list)
    assert body['warnings'] == []


def test_analyze_with_job_description(client, resume_pdf, mock_llm):
    r = analyze(client, resume_pdf, jd=SAMPLE_JD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(mock_llm) == 2                                   # resume + JD
    assert body['job_title'] == 'Senior Backend Engineer'
    jd = body['jd_comparison']
    assert jd and 0 <= jd['match_percentage'] <= 100
    assert 'AWS' in jd['missing_keywords'] and 'Terraform' in jd['missing_keywords']
    assert 'Python' not in jd['missing_keywords'] and 'FastAPI' not in jd['missing_keywords']   # listed on the resume
    assert 'Python' in jd['matched_keywords']


def test_analyze_docx(client, resume_docx, mock_llm):
    r = client.post('/api/v1/analyze-resume', files={'resume': ('r.docx', resume_docx, 'application/octet-stream')}, headers=auth_header())
    assert r.status_code == 200, r.text


def test_critical_issues_and_suggestions_reflect_real_findings(client, resume_pdf, monkeypatch):
    """A resume whose parse has no skills/experience must surface real, non-empty critical issues + suggestions."""
    import json
    from backend.services import groq_parser
    sparse = {'name': 'X', 'skills': [], 'experience': [], 'projects': [], 'education': []}
    monkeypatch.setattr(groq_parser, '_get_client', lambda: object())
    monkeypatch.setattr(groq_parser, '_call_groq', lambda c, s, u: json.dumps(sparse))
    body = analyze(client, resume_pdf).json()
    assert 'Missing or Weak Skills Section' in body['critical_issues']
    assert body['suggestions'] and all(isinstance(s, str) and s for s in body['suggestions'])
    assert 'Missing or Weak Skills Section' in body['issues_summary']


def test_analysis_is_saved_and_listed_in_history(client, resume_pdf, mock_llm, fake_supabase):
    analyze(client, resume_pdf, jd=SAMPLE_JD)
    assert len(fake_supabase.rows) == 1
    row = fake_supabase.rows[0]
    assert row['user_id'] == USER_A and row['filename'] == 'resume.pdf'
    # the DB call was made as the user (their token + the anon key), never a service key
    req = fake_supabase.requests[0]
    assert req.headers['apikey'] == 'test-anon-key' and req.headers['authorization'].startswith('Bearer ')
    assert 'strengths' in row['analysis_result'] and 'scoring_notes' in row['analysis_result']

    hist = client.get('/api/v1/history', headers=auth_header()).json()
    assert len(hist) == 1 and hist[0]['job_title'] == 'Senior Backend Engineer'
    assert 'Software Engineer' != hist[0]['job_title']         # the old hard-coded fake value is gone


def test_history_job_title_is_none_without_jd(client, resume_pdf, mock_llm):
    analyze(client, resume_pdf)
    assert client.get('/api/v1/history', headers=auth_header()).json()[0]['job_title'] is None


def test_history_save_failure_is_reported_not_hidden(client, resume_pdf, mock_llm, fake_supabase):
    fake_supabase.fail_with = 500
    r = analyze(client, resume_pdf)
    assert r.status_code == 200
    assert any('could not be saved' in w for w in r.json()['warnings'])


# ── invalid input / error handling ──────────────────────────────────────────
def test_non_resume_file_is_rejected(client, mock_llm):
    r = client.post('/api/v1/analyze-resume', files={'resume': ('a.pdf', b'hello world this is text', 'application/pdf')}, headers=auth_header())
    assert r.status_code == 422 and 'Unsupported' in r.json()['detail']
    assert mock_llm == []                                       # the LLM was never called


def test_empty_file_is_rejected(client, mock_llm):
    r = client.post('/api/v1/analyze-resume', files={'resume': ('a.pdf', b'', 'application/pdf')}, headers=auth_header())
    assert r.status_code == 422 and 'empty' in r.json()['detail'].lower()


def test_legacy_doc_is_rejected_with_guidance(client, mock_llm):
    ole = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' + b'\x00' * 600
    r = client.post('/api/v1/analyze-resume', files={'resume': ('a.doc', ole, 'application/msword')}, headers=auth_header())
    assert r.status_code == 422 and 'docx' in r.json()['detail'].lower()


def test_corrupt_pdf_is_rejected(client, mock_llm):
    r = client.post('/api/v1/analyze-resume', files={'resume': ('a.pdf', b'%PDF-1.4 broken', 'application/pdf')}, headers=auth_header())
    assert r.status_code == 422


def test_oversized_upload_is_rejected_before_it_is_read(client, mock_llm):
    big = b'%PDF-1.4' + b'0' * (7 * 1024 * 1024)
    r = client.post('/api/v1/analyze-resume', files={'resume': ('a.pdf', big, 'application/pdf')}, headers=auth_header())
    assert r.status_code == 413
    assert mock_llm == []


def test_overlong_job_description_is_rejected(client, resume_pdf, mock_llm):
    r = analyze(client, resume_pdf, jd='x' * (config.MAX_JD_CHARS + 1))
    assert r.status_code == 422 and 'too long' in r.json()['detail']


def test_llm_outage_returns_502_without_internal_details(client, resume_pdf, monkeypatch):
    from backend.services import groq_parser
    monkeypatch.setattr(groq_parser, '_get_client', lambda: object())

    def boom(c, s, u):
        raise groq_parser.LLMServiceError('The AI service is temporarily unavailable. Please try again shortly.')
    monkeypatch.setattr(groq_parser, '_call_groq', boom)
    r = analyze(client, resume_pdf)
    assert r.status_code == 502 and 'temporarily unavailable' in r.json()['detail']


def test_llm_rate_limit_returns_503(client, resume_pdf, monkeypatch):
    from backend.services import groq_parser
    monkeypatch.setattr(groq_parser, '_get_client', lambda: object())

    def busy(c, s, u):
        raise groq_parser.LLMBusyError('The AI service is busy right now. Please try again in a minute.')
    monkeypatch.setattr(groq_parser, '_call_groq', busy)
    r = analyze(client, resume_pdf)
    assert r.status_code == 503 and r.headers.get('retry-after')


def test_unexpected_pipeline_error_does_not_leak_internals(client, resume_pdf, monkeypatch):
    import backend.services.resume_analyzer as ra

    def crash(**kwargs):
        raise RuntimeError('secret internal path /srv/app/private.py password=hunter2')
    monkeypatch.setattr(ra, 'analyze_full_resume', crash)
    r = analyze(client, resume_pdf)
    assert r.status_code == 500 and 'hunter2' not in r.text and 'private.py' not in r.text


def test_missing_groq_key_gives_clear_error_via_api(client, resume_pdf, monkeypatch):
    from backend.services import groq_parser
    monkeypatch.setattr(config, 'GROQ_API_KEY', '')
    r = analyze(client, resume_pdf)
    assert r.status_code == 502 and 'not configured' in r.json()['detail']


# ── history, ownership, deletion ────────────────────────────────────────────
def test_users_cannot_see_or_delete_each_others_history(client, resume_pdf, mock_llm, fake_supabase):
    analyze(client, resume_pdf, user=USER_A)
    row_id = fake_supabase.rows[0]['id']

    assert client.get('/api/v1/history', headers=auth_header(USER_B)).json() == []
    r = client.delete(f'/api/v1/history/{row_id}', headers=auth_header(USER_B))
    assert r.status_code == 404
    assert len(fake_supabase.rows) == 1                          # untouched
    assert client.get(f'/api/v1/history/{row_id}/pdf', headers=auth_header(USER_B)).status_code == 404

    assert client.delete(f'/api/v1/history/{row_id}', headers=auth_header(USER_A)).status_code == 200
    assert fake_supabase.rows == []


def test_delete_unknown_or_malformed_id_is_404(client):
    for bad in ['does-not-exist', '1;drop table', '00000000-0000-4000-8000-000000000000']:
        assert client.delete(f'/api/v1/history/{bad}', headers=auth_header()).status_code == 404


def test_history_is_paged_and_bounded(client, resume_pdf, mock_llm, fake_supabase):
    for i in range(5):
        analyze(client, resume_pdf)
    assert len(client.get('/api/v1/history?limit=2', headers=auth_header()).json()) == 2
    assert len(client.get('/api/v1/history?limit=2&offset=4', headers=auth_header()).json()) == 1
    assert client.get('/api/v1/history?limit=100000', headers=auth_header()).status_code == 422
    assert client.get('/api/v1/history?limit=0', headers=auth_header()).status_code == 422
    listing = [r for r in fake_supabase.requests if r.method == 'GET'][-1]
    assert 'limit' in dict(listing.url.params)                   # the query sent to the DB is always limited


def test_database_outage_on_history_is_a_502_not_an_empty_list(client, fake_supabase):
    fake_supabase.fail_with = 500
    assert client.get('/api/v1/history', headers=auth_header()).status_code == 502


# ── PDF endpoints ───────────────────────────────────────────────────────────
def analysis_payload(client, resume_pdf):
    return analyze(client, resume_pdf).json()


def test_generate_pdf_endpoint_returns_a_real_pdf(client, resume_pdf, mock_llm):
    payload = analysis_payload(client, resume_pdf)
    r = client.post('/api/v1/generate-pdf', json=payload, headers=auth_header())
    assert r.status_code == 200 and r.headers['content-type'] == 'application/pdf'
    assert r.content.startswith(b'%PDF') and len(r.content) > 5000


def test_history_pdf_endpoint(client, resume_pdf, mock_llm, fake_supabase):
    analyze(client, resume_pdf)
    row_id = fake_supabase.rows[0]['id']
    r = client.get(f'/api/v1/history/{row_id}/pdf', headers=auth_header())
    assert r.status_code == 200 and r.content.startswith(b'%PDF')


def test_generate_pdf_with_malicious_payload_is_safe(client, resume_pdf, mock_llm):
    payload = analysis_payload(client, resume_pdf)
    payload['interpretation'] = '<img src="file:///etc/passwd"><script>1</script>'
    payload['strengths'] = ['<iframe src="http://169.254.169.254/latest/meta-data/"></iframe>']
    r = client.post('/api/v1/generate-pdf', json=payload, headers=auth_header())
    assert r.status_code == 200 and r.content.startswith(b'%PDF')


def test_generate_pdf_body_size_limit(client):
    r = client.post('/api/v1/generate-pdf', content=b'{"x": "' + b'a' * (config.MAX_PDF_REQUEST_BYTES + 10) + b'"}',
                    headers={**auth_header(), 'Content-Type': 'application/json'})
    assert r.status_code == 413


def test_generate_pdf_rejects_invalid_payload(client):
    assert client.post('/api/v1/generate-pdf', json={'nope': 1}, headers=auth_header()).status_code == 422


# ── rate limiting & resource protection ─────────────────────────────────────
def test_per_user_rate_limit(client, resume_pdf, mock_llm, monkeypatch):
    monkeypatch.setattr(config, 'ANALYZE_RATE_LIMIT_PER_MINUTE', 2)
    assert analyze(client, resume_pdf).status_code == 200
    assert analyze(client, resume_pdf).status_code == 200
    r = analyze(client, resume_pdf)
    assert r.status_code == 429 and int(r.headers['retry-after']) > 0
    assert len(mock_llm) == 2                                    # the blocked request never reached Groq
    assert analyze(client, resume_pdf, user=USER_B).status_code == 200      # other users unaffected


def test_global_daily_cap(client, resume_pdf, mock_llm, monkeypatch):
    monkeypatch.setattr(config, 'GLOBAL_ANALYSES_PER_DAY', 2)
    assert analyze(client, resume_pdf, user=USER_A).status_code == 200
    assert analyze(client, resume_pdf, user=USER_B).status_code == 200
    assert analyze(client, resume_pdf, user=USER_A).status_code == 429


def test_unauthenticated_requests_do_not_consume_rate_limit(client, resume_pdf, mock_llm, monkeypatch):
    monkeypatch.setattr(config, 'ANALYZE_RATE_LIMIT_PER_MINUTE', 1)
    client.post('/api/v1/analyze-resume', files={'resume': ('a.pdf', resume_pdf, 'application/pdf')})
    assert analyze(client, resume_pdf).status_code == 200


def test_pdf_rate_limit(client, resume_pdf, mock_llm, monkeypatch):
    payload = analysis_payload(client, resume_pdf)
    monkeypatch.setattr(config, 'PDF_RATE_LIMIT_PER_MINUTE', 1)
    assert client.post('/api/v1/generate-pdf', json=payload, headers=auth_header()).status_code == 200
    assert client.post('/api/v1/generate-pdf', json=payload, headers=auth_header()).status_code == 429


def test_slow_analysis_does_not_block_other_requests(resume_pdf, mock_llm, fake_supabase, monkeypatch):
    """Regression for 'sync AI work blocks the event loop': while one analysis is busy for ~1s,
    /health must still answer immediately, and two analyses must overlap rather than serialise."""
    import backend.services.resume_analyzer as ra
    from backend.main import app
    from tests.conftest import FakeEmbedder, FakeNLP
    real = ra.analyze_full_resume

    def slow(**kwargs):
        time.sleep(1.0)
        return real(**kwargs)
    monkeypatch.setattr(ra, 'analyze_full_resume', slow)
    app.state.nlp, app.state.embedder = FakeNLP(), FakeEmbedder()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as ac:
            files = {'resume': ('r.pdf', resume_pdf, 'application/pdf')}
            t0 = time.perf_counter()
            slow_calls = [asyncio.create_task(ac.post('/api/v1/analyze-resume', files=files, headers=auth_header(u))) for u in (USER_A, USER_B)]
            await asyncio.sleep(0.3)
            t1 = time.perf_counter()
            health = await ac.get('/api/v1/health')
            health_latency = time.perf_counter() - t1
            results = await asyncio.gather(*slow_calls)
            return health, health_latency, results, time.perf_counter() - t0

    health, health_latency, results, total = asyncio.run(scenario())
    assert health.status_code == 200 and health_latency < 0.5, f'event loop was blocked ({health_latency:.2f}s)'
    assert [r.status_code for r in results] == [200, 200]
    assert total < 1.9, f'analyses ran serially ({total:.2f}s)'


def test_busy_server_returns_503_instead_of_queueing_forever(resume_pdf, mock_llm, fake_supabase, monkeypatch):
    import backend.services.resume_analyzer as ra
    from backend.main import app
    from tests.conftest import FakeEmbedder, FakeNLP
    monkeypatch.setattr(config, 'MAX_CONCURRENT_ANALYSES', 1)
    monkeypatch.setattr(config, 'BUSY_WAIT_SECONDS', 0.2)
    real = ra.analyze_full_resume
    monkeypatch.setattr(ra, 'analyze_full_resume', lambda **kw: (time.sleep(0.8), real(**kw))[1])
    app.state.nlp, app.state.embedder = FakeNLP(), FakeEmbedder()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as ac:
            files = {'resume': ('r.pdf', resume_pdf, 'application/pdf')}
            first = asyncio.create_task(ac.post('/api/v1/analyze-resume', files=files, headers=auth_header(USER_A)))
            await asyncio.sleep(0.2)
            second = await ac.post('/api/v1/analyze-resume', files=files, headers=auth_header(USER_B))
            return second, await first

    second, first = asyncio.run(scenario())
    assert second.status_code == 503 and second.headers.get('retry-after')
    assert first.status_code == 200


# ── misc hardening ──────────────────────────────────────────────────────────
def test_filename_is_sanitised_before_storage(client, resume_pdf, mock_llm, fake_supabase):
    analyze(client, resume_pdf, filename='../../etc/pass<script>wd\n.pdf')
    stored = fake_supabase.rows[0]['filename']
    assert '/' not in stored and '<' not in stored and '\n' not in stored


def test_cors_only_allows_configured_origins(client):
    ok = client.options('/api/v1/history', headers={'Origin': 'http://localhost:8501', 'Access-Control-Request-Method': 'GET'})
    bad = client.options('/api/v1/history', headers={'Origin': 'https://evil.example', 'Access-Control-Request-Method': 'GET'})
    assert ok.headers.get('access-control-allow-origin') == 'http://localhost:8501'
    assert 'access-control-allow-origin' not in bad.headers


def test_server_logs_contain_no_resume_content(client, resume_pdf, mock_llm, caplog):
    """Regression: the INFO log used to print the first 100 characters of the parsed resume summary."""
    import logging
    with caplog.at_level(logging.DEBUG):
        r = analyze(client, resume_pdf, jd=SAMPLE_JD)
    assert r.status_code == 200
    for private in ('Backend engineer with 3 years', 'jane@example.com', '555 123 4567', 'Acme Corp', 'Jane Doe'):
        assert private not in caplog.text, f'{private!r} leaked into the logs'


def test_supabase_error_details_are_not_logged(client, resume_pdf, mock_llm, fake_supabase, caplog):
    """PostgREST error 'details' contain the offending row (= the user's analysis); only code+message may be logged."""
    import logging, httpx
    from backend.database import supabase_db
    secret = 'Failing row contains (Jane Doe, jane@example.com)'
    fake_supabase.handler_orig = fake_supabase.handler
    fake_supabase.handler = lambda req: httpx.Response(400, json={'code': '23514', 'message': 'check violation', 'details': secret})
    with caplog.at_level(logging.DEBUG):
        r = analyze(client, resume_pdf)
    assert secret not in caplog.text and 'jane@example.com' not in caplog.text
    assert '23514' in caplog.text
