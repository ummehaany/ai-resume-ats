"""Configuration, model loading and startup behaviour (audit items 2 and 5)."""
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.core import config


def test_spacy_fallback_model_name_is_valid():
    # Was '"en_core_web_sm' (stray quote) - the fallback could never load.
    assert config.SPACY_MODEL_SECONDARY == 'en_core_web_sm'
    assert config.SPACY_MODEL_PRIMARY == 'en_core_web_md'


def test_allowed_origins_parsing_strips_trailing_slashes_and_blanks():
    assert config._parse_origins('https://a.example/, http://b.example ,, ') == ['https://a.example', 'http://b.example']
    assert all(not o.endswith('/') for o in config.ALLOWED_ORIGINS)
    assert not any('streamlit.app' in o for o in config.ALLOWED_ORIGINS)     # course author's URL removed


def test_bad_integer_env_var_gives_clear_error(monkeypatch):
    monkeypatch.setenv('SOME_LIMIT', 'abc')
    with pytest.raises(RuntimeError, match='SOME_LIMIT'):
        config._env_int('SOME_LIMIT', 5)


def test_missing_config_is_listed_by_name(monkeypatch):
    monkeypatch.setattr(config, 'GROQ_API_KEY', '')
    monkeypatch.setattr(config, 'SUPABASE_URL', '')
    monkeypatch.setattr(config, 'SUPABASE_ANON_KEY', '')
    problems = ' '.join(config.get_missing_config())
    assert 'GROQ_API_KEY' in problems and 'SUPABASE_URL' in problems and 'SUPABASE_ANON_KEY' in problems


def test_service_role_key_is_no_longer_required_or_read():
    assert not hasattr(config, 'SUPABASE_KEY')


def test_backend_refuses_to_start_without_required_config(monkeypatch):
    monkeypatch.setattr(config, 'GROQ_API_KEY', '')
    monkeypatch.setattr(config, 'SUPABASE_URL', '')
    with pytest.raises(RuntimeError) as exc:
        with TestClient(main.app):
            pass
    message = str(exc.value)
    assert 'GROQ_API_KEY' in message and 'SUPABASE_URL' in message and '.env.example' in message


def test_missing_spacy_models_give_actionable_message(monkeypatch):
    import spacy

    def no_model(name):
        raise OSError(f'[E050] Can\'t find model {name}')
    monkeypatch.setattr(spacy, 'load', no_model)
    with pytest.raises(RuntimeError) as exc:
        main.load_spacy_model()
    assert 'python -m spacy download en_core_web_md' in str(exc.value)


def test_spacy_falls_back_to_small_model(monkeypatch):
    import spacy
    loaded = []

    def load(name):
        loaded.append(name)
        if name == 'en_core_web_md':
            raise OSError('missing')
        return object()
    monkeypatch.setattr(spacy, 'load', load)
    main.load_spacy_model()
    assert loaded == ['en_core_web_md', 'en_core_web_sm']


def test_real_spacy_model_loads_and_is_usable():
    spacy = pytest.importorskip('spacy')
    if not (spacy.util.is_package('en_core_web_md') or spacy.util.is_package('en_core_web_sm')):
        pytest.skip('no spaCy model installed')
    nlp = main.load_spacy_model()
    doc = nlp('Jane Doe built APIs for Acme Corp in London.')
    assert len(list(doc.noun_chunks)) > 0


def test_embedder_load_failure_is_explained(monkeypatch):
    import sentence_transformers

    def fail(name):
        raise OSError('cannot reach hub')
    monkeypatch.setattr(sentence_transformers, 'SentenceTransformer', fail)
    with pytest.raises(RuntimeError, match='internet access'):
        main.load_embedder()


def test_log_file_failure_never_stops_the_app(monkeypatch, tmp_path):
    """Original code created backend/logs at import time; on a read-only host that crashed the import."""
    import logging
    from backend.utils import file_utils
    bad = tmp_path / 'afile'
    bad.write_text('x')
    monkeypatch.setattr(config, 'LOG_FILE', str(bad / 'sub' / 'app.log'))    # parent is a FILE -> OSError
    logger = logging.getLogger('ats_resume_scorer')
    saved = list(logger.handlers)
    logger.handlers.clear()
    try:
        file_utils.configure_logging()        # must not raise
        assert logger.handlers                # console handler still installed
    finally:
        logger.handlers[:] = saved


def test_importing_the_backend_does_not_create_a_logs_directory():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert not os.path.exists(os.path.join(root, 'backend', 'logs'))


def test_docs_can_be_disabled():
    assert config.ENABLE_DOCS in (True, False)
