"""OPT-IN REAL-SERVICE TESTS. These call the real Groq API and a real Supabase project.

They are skipped unless RUN_INTEGRATION=1 and the needed credentials are present in the environment
(load your .env first, e.g. `set -a; . ./.env; set +a`). They are NOT part of the default run and
were NOT run by the assistant that wrote them (no access to those services from its sandbox).

    RUN_INTEGRATION=1 pytest tests/integration -v

Supabase two-user RLS test needs two throw-away confirmed test users:
    IT_USER_A_EMAIL / IT_USER_A_PASSWORD / IT_USER_B_EMAIL / IT_USER_B_PASSWORD
(create them in the Supabase dashboard -> Authentication -> Users; use a test project, not production).
"""
import os

import httpx
import pytest

RUN = os.getenv('RUN_INTEGRATION') == '1'
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not RUN, reason='set RUN_INTEGRATION=1 to run real-service tests')]

SAMPLE = """Jane Doe
jane@example.com | +1 555 123 4567
Summary: Backend engineer with 3 years building Python APIs.
Skills: Python, FastAPI, PostgreSQL, Docker
Experience
Acme Corp - Software Engineer (2021-2024)
- Built REST APIs serving 2M requests/day, cutting latency by 35%
- Led migration to Docker, reducing deploy time by 60%
Education
B.Sc. Computer Science, State University, 2021
"""


def _need(*names):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        pytest.skip(f'missing env: {", ".join(missing)}')


def test_real_groq_parses_resume():
    _need('GROQ_API_KEY')
    from backend.services import groq_parser
    result = groq_parser.parse_resume(SAMPLE)
    assert isinstance(result, dict)
    assert 'python' in ' '.join(map(str, result.get('skills', []))).lower()
    assert 'jane@example.com' in str(result.get('email', '')).lower()


def test_real_groq_parses_job_description():
    _need('GROQ_API_KEY')
    from backend.services import groq_parser
    result = groq_parser.parse_job_description('Senior Backend Engineer. Required: Python, FastAPI, AWS. Preferred: Kubernetes.')
    assert 'python' in ' '.join(map(str, result.get('required_skills', []))).lower()


def test_real_groq_makes_one_request_per_parse(monkeypatch):
    """A well-formed answer must cost exactly ONE Groq call (the retry only fires on unreadable output)."""
    _need('GROQ_API_KEY')
    from backend.services import groq_parser
    calls = []
    real = groq_parser._call_groq
    monkeypatch.setattr(groq_parser, '_call_groq', lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    groq_parser.parse_resume(SAMPLE)
    assert len(calls) == 1


def test_real_groq_bad_key_is_a_safe_error(monkeypatch):
    """Needs network access to Groq but NOT a real key: a wrong key must give a clean error that never echoes the key."""
    from backend.core import config
    from backend.services import groq_parser
    fake_key = 'invalid-key-for-testing-0123456789'
    monkeypatch.setattr(config, 'GROQ_API_KEY', fake_key)
    monkeypatch.setattr(groq_parser, '_client', None)
    with pytest.raises(groq_parser.LLMServiceError) as exc:
        groq_parser.parse_resume(SAMPLE)
    assert fake_key not in str(exc.value)


def _sign_in(email, password):
    url, anon = os.environ['SUPABASE_URL'].rstrip('/'), os.environ['SUPABASE_ANON_KEY']
    r = httpx.post(f'{url}/auth/v1/token?grant_type=password', headers={'apikey': anon},
                   json={'email': email, 'password': password}, timeout=20)
    assert r.status_code == 200, f'sign-in failed (HTTP {r.status_code})'
    return r.json()['access_token']


def test_real_supabase_rls_isolates_users():
    """Verifies the REAL supabase/schema.sql policies: user B must not read, delete, or forge rows of user A."""
    _need('SUPABASE_URL', 'SUPABASE_ANON_KEY', 'IT_USER_A_EMAIL', 'IT_USER_A_PASSWORD', 'IT_USER_B_EMAIL', 'IT_USER_B_PASSWORD')
    import asyncio
    import jwt
    from backend.database import supabase_db as db

    token_a = _sign_in(os.environ['IT_USER_A_EMAIL'], os.environ['IT_USER_A_PASSWORD'])
    token_b = _sign_in(os.environ['IT_USER_B_EMAIL'], os.environ['IT_USER_B_PASSWORD'])
    uid_a = jwt.decode(token_a, options={'verify_signature': False})['sub']

    async def scenario():
        aid = await db.save_analysis(uid_a, token_a, 'it.pdf', {'ats_score': 50, 'keyword_match': 0, 'missing_keywords': []})
        assert aid
        try:
            assert await db.get_analysis(aid, uid_a, token_a) is not None       # owner can read
            assert await db.get_analysis(aid, uid_a, token_b) is None           # B cannot read A's row (RLS)
            assert await db.get_user_history(uid_a, token_b) == []              # nor list it
            assert await db.delete_analysis(aid, uid_a, token_b) is False       # nor delete it
            with pytest.raises(db.DatabaseError):                               # nor insert a row owned by A
                await db.save_analysis(uid_a, token_b, 'forged.pdf', {'ats_score': 1})
            assert await db.get_analysis(aid, uid_a, token_a) is not None       # still there
        finally:
            await db.delete_analysis(aid, uid_a, token_a)

    asyncio.run(scenario())
