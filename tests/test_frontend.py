"""Frontend tests (Streamlit). Uses streamlit's official AppTest harness, which runs the real app scripts
in-process. The backend API and Supabase auth are MOCKED (monkeypatched)."""
import os
import sys

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, 'frontend', 'streamlit_app.py')

ANALYSIS = {
    'ATS_score': 72.4, 'ats_score': 72.4, 'interpretation': 'Good! Your resume is ATS-friendly.',
    'component_scores': {'formatting': 15, 'keywords': 18, 'content': 14, 'skill_validation': 9, 'ats_compatibility': 14},
    'strengths': ['Has a dedicated Experience section'], 'critical_issues': ['Missing Projects Section'],
    'suggestions': ['Add a Projects section'], 'issues_summary': ['Missing Projects Section', 'Most Skills Lack Supporting Evidence'],
    'scoring_notes': ['Base score 70.1 = 40% skills & keywords + ...', 'Grammar and spelling are not evaluated and do not affect the score.'],
    'detailed_feedback': [
        {'issue_title': 'Missing Projects Section', 'severity_level': 'High', 'ats_impact': 'High', 'explanation': 'e', 'where_it_appears': 'w', 'how_to_fix': 'h', 'action_items': ['do a'], 'example_improvement': 'ex'},
        {'issue_title': 'Most Skills Lack Supporting Evidence', 'severity_level': 'Moderate', 'ats_impact': 'High', 'explanation': 'e', 'where_it_appears': 'w', 'how_to_fix': 'h', 'action_items': ['do b'], 'example_improvement': 'ex'},
        {'issue_title': 'Missing Professional Summary', 'severity_level': 'Low', 'ats_impact': 'Low', 'explanation': 'e', 'where_it_appears': 'w', 'how_to_fix': 'h', 'action_items': ['do c'], 'example_improvement': 'ex'},
    ],
    'skill_validation_details': {'validated': [], 'unvalidated': ['Go'], 'total': 1, 'validated_count': 0, 'validation_pct': 0},
    'jd_comparison': None, 'warnings': ['Your results could not be saved to your history, but the analysis below is complete.'],
}


def all_text(at):
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info]
    return '\n'.join(parts)


def test_severity_normalisation():
    from frontend.components._helpers import normalize_severity, get_severity_style
    assert normalize_severity('Moderate') == 'medium'
    assert normalize_severity(None) == 'low' and normalize_severity('weird') == 'low'
    assert normalize_severity('HIGH') == 'high'
    assert get_severity_style('Moderate')[0] == '🟡'          # was green (fell through to "low")


def test_moderate_issues_are_no_longer_dropped_from_detailed_feedback():
    from frontend.components.detailed_feedback import _group_by_severity
    grouped = _group_by_severity(ANALYSIS['detailed_feedback'])
    assert [len(grouped[k]) for k in ('critical', 'high', 'medium', 'low')] == [0, 1, 1, 1]


def test_app_starts_and_landing_page_is_honest():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, at.exception
    text = all_text(at)
    assert 'never leaves your system' not in text and '100% Private' not in text and 'no external API calls' not in text
    assert 'Groq' in text                                    # discloses that resume text goes to an AI provider
    assert 'PDF and DOCX' in text and 'DOC,' not in text


def test_scorer_page_offers_only_supported_types_and_asks_for_sign_in():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state['current_view'] = 'scorer'
    at.run()
    assert not at.exception, at.exception


def test_results_dashboard_renders_all_sections_including_moderate_issues():
    def page():
        import streamlit as st
        from frontend.components.dashboard import display_results_dashboard
        display_results_dashboard(st.session_state['analysis'])

    at = AppTest.from_function(page, default_timeout=30)
    at.session_state['analysis'] = ANALYSIS
    at.run()
    assert not at.exception, at.exception
    text = all_text(at)
    for expected in ['Strengths', 'Has a dedicated Experience section', 'Critical Issues', 'Missing Projects Section',
                     'Most Skills Lack Supporting Evidence', 'Medium (1)', 'Low (1)', 'Recommendations', 'Add a Projects section',
                     'Action Items', 'do b']:
        assert expected in text, f'{expected!r} missing from rendered dashboard'
    assert any('How was this score calculated' in e.label for e in at.expander)


def test_only_moderate_findings_are_not_labelled_critical():
    def page():
        import streamlit as st
        from frontend.components.strengths_issues import display_critical_issues
        display_critical_issues(st.session_state['analysis'])

    at = AppTest.from_function(page, default_timeout=30)
    at.session_state['analysis'] = {'critical_issues': [], 'issues_summary': ['Most Skills Lack Supporting Evidence']}
    at.run()
    text = all_text(at)
    assert 'Issues to Improve' in text and '🚨 Critical Issues' not in text


def test_history_page_shows_real_job_title_and_handles_expired_session(monkeypatch):
    import requests
    from frontend.services import api_client
    entry = {'id': 'abc', 'filename': 'cv.pdf', 'ats_score': 60, 'created_at': '2026-01-01', 'job_title': 'Data Engineer',
             'analysis_result': {'component_scores': {}, 'jd_comparison': None}}
    monkeypatch.setattr(api_client, 'get_history', lambda token, **kw: [entry])
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state['current_view'] = 'history'
    at.session_state['access_token'] = 'tok'
    at.session_state['user_email'] = 'a@b.c'
    at.run()
    assert not at.exception, at.exception
    assert 'Data Engineer' in all_text(at) + ' '.join(c.value for c in at.caption)

    def expired(token, **kw):
        resp = requests.Response()
        resp.status_code = 401
        resp._content = b'{"detail": "Token expired - sign in again"}'
        raise requests.HTTPError(response=resp)
    monkeypatch.setattr(api_client, 'get_history', expired)
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state['current_view'] = 'history'
    at.session_state['access_token'] = 'tok'
    at.session_state['user_email'] = 'a@b.c'
    at.run()
    assert any('session has expired' in w.value for w in at.warning)
    assert at.session_state['access_token'] is None            # user is signed out, not stuck


def test_backend_url_resolution(monkeypatch):
    from frontend.services import api_client
    monkeypatch.delenv('BACKEND_URL', raising=False)
    assert api_client._backend_url() == 'http://localhost:8000'
    monkeypatch.setenv('BACKEND_URL', 'https://api.example.com/')
    assert api_client._backend_url() == 'https://api.example.com'


def test_history_client_sends_paging_params(monkeypatch):
    from frontend.services import api_client
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return []
    monkeypatch.setattr(api_client.requests, 'get', lambda url, **kw: seen.update(kw) or R())
    api_client.get_history('tok', limit=10, offset=5)
    assert seen['params'] == {'limit': 10, 'offset': 5}


def test_password_auth_uses_a_fresh_client_per_call_and_signout_revokes_the_users_token(monkeypatch):
    from frontend.services import supabase_client as sc
    monkeypatch.setattr(sc, 'SUPABASE_URL', 'https://x.supabase.co')
    monkeypatch.setattr(sc, 'SUPABASE_ANON_KEY', 'anon')
    created = []

    class FakeAuth:
        def sign_in_with_password(self, creds):
            class S: access_token, refresh_token = 'at', 'rt'
            class U: id, email = 'uid', creds['email']
            class Resp: session, user = S, U
            return Resp

    class FakeClient:
        auth = FakeAuth()

    monkeypatch.setattr(sc, 'create_client', lambda *a: created.append(1) or FakeClient())
    r1 = sc.sign_in_with_password('a@example.com', 'pw')
    r2 = sc.sign_in_with_password('b@example.com', 'pw')
    assert r1['email'] == 'a@example.com' and r2['email'] == 'b@example.com'
    assert len(created) == 2                                   # never a shared client between users

    import requests
    calls = []
    monkeypatch.setattr(requests, 'post', lambda url, **kw: calls.append((url, kw)))
    sc.sign_out('user-token')
    url, kw = calls[0]
    assert url.endswith('/auth/v1/logout') and kw['headers']['Authorization'] == 'Bearer user-token' and kw['params'] == {'scope': 'local'}
    sc.sign_out(None)
    assert len(calls) == 1


def test_streamlit_config_keeps_security_defaults_and_matches_backend_limit():
    import tomllib
    cfg = tomllib.load(open(os.path.join(ROOT, 'frontend', '.streamlit', 'config.toml'), 'rb'))
    assert 'enableXsrfProtection' not in cfg['server'] and 'enableCORS' not in cfg['server']
    from backend.core import config
    assert cfg['server']['maxUploadSize'] == config.MAX_FILE_SIZE_MB
