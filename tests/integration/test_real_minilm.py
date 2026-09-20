"""OPT-IN: the REAL sentence-transformers model (all-MiniLM-L6-v2) and the real spaCy model.

    RUN_INTEGRATION=1 pytest tests/integration/test_real_minilm.py -v -s

Needs internet the first time (downloads ~90 MB from huggingface.co, then cached). Nothing is faked here.
The printed numbers (load time, peak memory) are what to use when sizing a server.
"""
import os
import platform
import resource
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.getenv('RUN_INTEGRATION') != '1', reason='set RUN_INTEGRATION=1')]


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if platform.system() == 'Darwin' else peak / 1024   # macOS reports bytes, Linux KiB


@pytest.fixture(scope='module')
def embedder():
    from backend.main import load_embedder
    t0 = time.perf_counter()
    model = load_embedder()
    print(f'\n[MiniLM] load time: {time.perf_counter() - t0:.1f}s, peak RSS so far: {_peak_rss_mb():.0f} MB')
    return model


def test_real_model_embeddings_are_384_dim_and_semantic(embedder):
    a, b, c = (embedder.encode(t) for t in ('Python backend developer', 'Software engineer building APIs in Python', 'Professional baker of sourdough bread'))
    assert a.shape == (384,)
    cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    assert cos(a, b) > cos(a, c)


def test_real_models_job_description_matching_and_memory(embedder):
    from backend.main import load_spacy_model
    from backend.services.jd_matcher import compare_resume_with_jd
    nlp = load_spacy_model()
    result = compare_resume_with_jd(
        resume_text='Backend engineer. Built REST APIs in Python and FastAPI, deployed with Docker on AWS.',
        resume_keywords=['REST APIs', 'microservices'], resume_skills=['Python', 'FastAPI', 'Docker', 'AWS'],
        jd_text='Senior backend engineer: Python, FastAPI, AWS, Terraform, REST APIs.',
        jd_keywords=['Python', 'FastAPI', 'AWS', 'Terraform', 'REST APIs'], embedder=embedder, nlp=nlp)
    print(f'[MiniLM] match={result["match_percentage"]:.1f} semantic={result["semantic_similarity"]:.2f} missing={result["missing_keywords"]}')
    print(f'[MiniLM] peak RSS with spaCy + MiniLM loaded: {_peak_rss_mb():.0f} MB')
    assert result['missing_keywords'] == ['Terraform']
    assert result['semantic_similarity'] > 0.4
    assert _peak_rss_mb() < 3000
