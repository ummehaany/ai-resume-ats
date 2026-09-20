"""Regression tests for the Groq parser (audit item 3). Groq is MOCKED here."""
import json
import logging

import pytest

from backend.core import config
from backend.services import groq_parser as gp

GOOD = json.dumps({'name': 'A', 'skills': ['python']})


def script_llm(monkeypatch, replies):
    """Make _call_groq return/raise `replies` in order; returns the list of prompts it received."""
    prompts = []

    def fake(client, system_prompt, user_prompt):
        prompts.append(user_prompt)
        reply = replies[len(prompts) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(gp, '_get_client', lambda: object())
    monkeypatch.setattr(gp, '_call_groq', fake)
    return prompts


def test_valid_json_makes_exactly_one_call(monkeypatch):
    """The original code made a SECOND call even when the first reply was valid."""
    prompts = script_llm(monkeypatch, [GOOD, GOOD])
    result = gp.parse_resume('some resume text')
    assert len(prompts) == 1
    assert result['skills'] == ['python']


def test_invalid_then_valid_retries_once_with_strict_prompt(monkeypatch):
    """The original code crashed with TypeError when the first reply was invalid."""
    prompts = script_llm(monkeypatch, ['definitely not json', GOOD])
    result = gp.parse_resume('resume')
    assert len(prompts) == 2
    assert prompts[1].startswith('Your previous response was not valid JSON')
    assert result['skills'] == ['python']


def test_invalid_twice_raises_safe_error_without_leaking_content(monkeypatch, caplog):
    secret = 'SECRET-RESUME-CONTENT-jane@example.com'
    script_llm(monkeypatch, [f'nope {secret}', f'still nope {secret}'])
    with caplog.at_level(logging.DEBUG, logger='ats_resume_scorer'):
        with pytest.raises(gp.LLMResponseError) as exc_info:
            gp.parse_resume('resume')
    assert secret not in str(exc_info.value)
    assert secret not in caplog.text


@pytest.mark.parametrize('reply', [
    '```json\n{"name": "A", "skills": ["x"]}\n```',
    'Sure! Here is the JSON:\n{"name": "A", "skills": ["x"]}\nHope that helps',
    '```\n{"name": "A", "skills": ["x"]}\n```',
])
def test_tolerates_fences_and_prose(monkeypatch, reply):
    prompts = script_llm(monkeypatch, [reply])
    assert gp.parse_resume('resume')['skills'] == ['x']
    assert len(prompts) == 1


def test_json_that_is_not_an_object_counts_as_invalid(monkeypatch):
    prompts = script_llm(monkeypatch, ['["a", "b"]', GOOD])
    gp.parse_resume('resume')
    assert len(prompts) == 2


def test_llm_output_shapes_are_sanitised(monkeypatch):
    weird = json.dumps({
        'skills': ['Python', 5, None, '  ', {'x': 1}, 'SQL'],
        'keywords': 'not a list',
        'experience': [{'job_title': 'Dev', 'duration_months': '18'}, 'junk', {'duration_months': True}, {'duration_months': -4}],
        'projects': [{'title': 'P', 'technologies': 'python'}],
        'email': ['x'], 'professional_summary': None,
    })
    script_llm(monkeypatch, [weird])
    r = gp.parse_resume('resume')
    assert r['skills'] == ['Python', 'SQL']
    assert r['keywords'] == []
    assert [e['duration_months'] for e in r['experience']] == [18, 0, 0]
    assert r['projects'][0]['technologies'] == []
    assert r['email'] is None and r['professional_summary'] == ''


def test_prompt_handles_braces_and_contains_date_and_delimiters(monkeypatch):
    prompts = script_llm(monkeypatch, [GOOD])
    gp.parse_resume('code sample {"a": {1}} and {curly}')
    assert '{curly}' in prompts[0]
    assert '<resume>' in prompts[0]
    from datetime import date
    assert date.today().isoformat() in prompts[0]


def test_resume_text_is_truncated_before_being_sent(monkeypatch):
    prompts = script_llm(monkeypatch, [GOOD])
    gp.parse_resume('§' * (config.MAX_RESUME_TEXT_CHARS + 5000))
    assert prompts[0].count('§') == config.MAX_RESUME_TEXT_CHARS


class _ProviderError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class _ExplodingClient:
    def __init__(self, exc):
        self.chat = self
        self.completions = self
        self._exc = exc

    def create(self, **kwargs):
        raise self._exc


def test_rate_limit_from_provider_maps_to_busy_error_without_leaking_secrets(caplog):
    exc = _ProviderError('Rate limit reached for key sk-SUPERSECRET', status_code=429)
    with caplog.at_level(logging.DEBUG, logger='ats_resume_scorer'):
        with pytest.raises(gp.LLMBusyError) as e:
            gp._call_groq(_ExplodingClient(exc), 's', 'u')
    assert 'sk-SUPERSECRET' not in str(e.value) and 'sk-SUPERSECRET' not in caplog.text


def test_other_provider_errors_map_to_service_error_without_leaking_secrets(caplog):
    exc = _ProviderError('boom Authorization: Bearer sk-SUPERSECRET', status_code=500)
    with caplog.at_level(logging.DEBUG, logger='ats_resume_scorer'):
        with pytest.raises(gp.LLMServiceError) as e:
            gp._call_groq(_ExplodingClient(exc), 's', 'u')
    assert 'sk-SUPERSECRET' not in str(e.value) and 'sk-SUPERSECRET' not in caplog.text


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.setattr(config, 'GROQ_API_KEY', '')
    gp._client = None
    with pytest.raises(gp.LLMServiceError, match='GROQ_API_KEY'):
        gp._get_client()


def test_job_description_parse_is_normalised(monkeypatch):
    script_llm(monkeypatch, [json.dumps({'job_title': 'X', 'required_skills': 'oops', 'keywords': ['a', 1]})])
    r = gp.parse_job_description('jd')
    assert r['required_skills'] == [] and r['keywords'] == ['a'] and r['job_title'] == 'X'
