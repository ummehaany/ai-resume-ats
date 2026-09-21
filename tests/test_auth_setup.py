"""Startup diagnostics for sign-in token verification (legacy HS256 secret vs. asymmetric signing keys).

Why this exists: if the Supabase project signs user tokens with the legacy HS256 secret and SUPABASE_JWT_SECRET is not
set on the server, EVERY request gets a 401 - which the Streamlit UI shows as "session expired". These tests pin down
that the cause is reported clearly in the server log (never in the HTTP response).
"""
import logging

import httpx
import pytest

from backend import main
from backend.core import config
from tests.conftest import USER_A, make_token


@pytest.fixture
def logs(caplog):
    """The app logger has propagate=False, so attach pytest's capture handler to it directly."""
    logger = logging.getLogger('ats_resume_scorer')
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger='ats_resume_scorer')
    yield caplog
    logger.removeHandler(caplog.handler)


def _jwks_response(keys):
    return lambda url, timeout=None: httpx.Response(200, json={'keys': keys}, request=httpx.Request('GET', url))


def _text(caplog):
    return ' | '.join(f'{r.levelname}: {r.getMessage()}' for r in caplog.records)


def test_no_signing_keys_and_no_secret_is_logged_as_an_error(monkeypatch, logs):
    monkeypatch.setattr(config, 'SUPABASE_JWT_SECRET', '')
    monkeypatch.setattr(httpx, 'get', _jwks_response([]))
    main.check_auth_setup()
    text = _text(logs)
    assert 'ERROR' in text and 'SUPABASE_JWT_SECRET' in text and 'HTTP 401' in text


def test_no_signing_keys_but_secret_set_is_fine(monkeypatch, logs):
    monkeypatch.setattr(config, 'SUPABASE_JWT_SECRET', 'x' * 40)
    monkeypatch.setattr(httpx, 'get', _jwks_response([]))
    main.check_auth_setup()
    assert not [r for r in logs.records if r.levelno >= logging.WARNING]


def test_asymmetric_keys_found_is_fine(monkeypatch, logs):
    monkeypatch.setattr(config, 'SUPABASE_JWT_SECRET', '')
    monkeypatch.setattr(httpx, 'get', _jwks_response([{'kid': 'k1', 'kty': 'EC'}]))
    main.check_auth_setup()
    assert '1 Supabase signing key' in _text(logs)
    assert not [r for r in logs.records if r.levelno >= logging.ERROR]


def test_unreachable_supabase_only_warns_and_never_raises(monkeypatch, logs):
    def boom(url, timeout=None):
        raise httpx.ConnectError('no route')
    monkeypatch.setattr(httpx, 'get', boom)
    main.check_auth_setup()          # must not raise: this is a diagnostic, not a startup gate
    assert 'WARNING' in _text(logs) and 'ConnectError' in _text(logs)


def test_hs256_token_without_server_secret_is_401_and_the_cause_is_logged(client, monkeypatch, logs):
    monkeypatch.setattr(config, 'SUPABASE_JWT_SECRET', '')
    r = client.get('/api/v1/history', headers={'Authorization': f'Bearer {make_token(USER_A)}'})
    assert r.status_code == 401 and r.json()['detail'] == 'Invalid token'      # the client learns nothing about the setup
    assert 'SUPABASE_JWT_SECRET is not configured' in _text(logs)
    assert any(rec.levelno == logging.ERROR for rec in logs.records)
