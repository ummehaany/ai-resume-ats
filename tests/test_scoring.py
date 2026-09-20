"""Scoring, privacy detection and feedback (audit items 1 and 6). No external services involved."""
import pytest

from backend.core import config
from backend.services import ats_scorer as sc
from backend.services.feedback_engine import analyze_issues, generate_issues_summary


# ── feedback_engine imports at all (audit item 1: stray text made it a SyntaxError) ──
def test_feedback_engine_module_imports_and_runs():
    issues = analyze_issues(
        resume_text='just some words', parsed_resume={}, skills=[], projects=[], action_verbs=[],
        skill_validation={}, scores={'formatting_score': 2}, contact_info={},
    )
    titles = generate_issues_summary(issues)
    assert 'Missing Work Experience Section' in titles
    assert 'Missing or Weak Skills Section' in titles
    assert 'Poor Resume Formatting' in titles


def test_education_check_is_not_disabled_by_any_number_in_text():
    # Before: the token '20' counted as an "education signal", so a resume containing e.g. "2021" never got flagged.
    issues = analyze_issues(
        resume_text='built a tool in 2021 as a job at company', parsed_resume={}, skills=[], projects=[],
        action_verbs=[], skill_validation={}, scores={'formatting_score': 20}, contact_info={},
    )
    assert 'Missing Education Section' in generate_issues_summary(issues)


# ── privacy / location detection ─────────────────────────────────────────────
@pytest.mark.parametrize('text', [
    'Jane Doe\nKolkata, India\nBuilt pipelines for 10000 users and 25000 events',
    'San Francisco, California | Stanford University, Stanford, CA',
    'Jane Doe | jane@example.com | Kolkata, India\n' + 'Summary line of ordinary text. ' * 20 + '\nImproved throughput 12345 requests; Revenue - 100000 dollars grew',
])
def test_city_names_and_plain_numbers_are_not_flagged(text):
    result = sc.detect_location_info(text)
    assert result['privacy_risk'] == 'none' and result['penalty_applied'] == 0


@pytest.mark.parametrize('text, kind', [
    ('Jane\n12 Park Street\nSomewhere', 'address'),
    ('Jane\n221 Baker Street', 'address'),
    ('Jane\nSan Francisco, CA 94105', 'postal_code'),
    ('Jane\nPIN: 700016', 'postal_code'),
    ('Jane\nZip code 94105', 'postal_code'),
    ('Jane\nKolkata - 700016', 'postal_code'),
])
def test_addresses_and_postal_codes_are_flagged(text, kind):
    result = sc.detect_location_info(text)
    assert kind in {f['type'] for f in result['detected_locations']}
    assert result['penalty_applied'] > 0


def test_address_plus_postal_code_is_high_risk():
    result = sc.detect_location_info('12 Park Street, Kolkata - 700016')
    assert result['privacy_risk'] == 'high'


def test_privacy_finding_becomes_feedback_item():
    loc = sc.detect_location_info('12 Park Street, Kolkata - 700016')
    issues = analyze_issues('x', {}, [], [], [], {}, {'formatting_score': 20}, {}, location_results=loc)
    assert 'Full Address or Postal Code on Resume' in generate_issues_summary(issues)


# ── content score no longer includes free grammar points (audit item 6) ─────
def test_content_score_is_zero_without_verbs_or_metrics():
    assert sc._calc_content_score('plain text without any numbers', []) == 0.0


def test_content_score_can_still_reach_full_marks():
    text = ' '.join(['increased revenue by 40%'] * 12)
    assert sc._calc_content_score(text, ['a'] * 15) == 25.0


def test_content_score_never_exceeds_25():
    assert sc._calc_content_score('50% ' * 100, ['a'] * 100) <= 25.0


# ── overall score is explainable and consistent ─────────────────────────────
PARSED = {
    'experience': [{'job_title': 'Dev', 'description': 'Developed things for 10K users and improved speed by 40%'}],
    'education': [{'degree': 'BSc', 'institution': 'Uni'}],
    'skills': ['a', 'b', 'c', 'd', 'e', 'f'],
    'professional_summary': 'A reasonably long professional summary sentence here.',
    'projects': [{'title': 'p', 'description': 'd'}],
}
VALIDATION = {'validation_percentage': 0.85, 'validation_score': 15 * 0.85}
NO_LOCATION = {'penalty_applied': 0.0}
TEXT = '• Developed things\n• Improved 40%\n• Built x\n' * 3


def test_overall_score_matches_documented_formula():
    r = sc.calculate_overall_score(TEXT, PARSED, PARSED['skills'], ['k'] * 12, ['v'] * 8, VALIDATION, NO_LOCATION)
    cmax = config.SCORE_COMPONENT_MAX
    kw = r['keywords_score'] / cmax['keywords'] * 100
    sv = r['skill_validation_score'] / cmax['skill_validation'] * 100
    expected = (
        (kw * 0.6 + sv * 0.4) * 0.40
        + r['content_score'] / cmax['content'] * 100 * 0.30
        + r['formatting_score'] / cmax['formatting'] * 100 * 0.15
        + r['ats_compatibility_score'] / cmax['ats_compatibility'] * 100 * 0.15
        + 1.0                                     # 85% validated -> +1 bonus
    )
    assert r['overall_score'] == pytest.approx(expected, abs=0.35)   # component scores are rounded to 1 decimal
    assert 0 <= r['overall_score'] <= 100


def test_no_grammar_bonus_or_penalty_anywhere():
    r = sc.calculate_overall_score(TEXT, PARSED, PARSED['skills'], ['k'] * 12, ['v'] * 8, VALIDATION, NO_LOCATION)
    assert 'perfect_grammar' not in r['bonuses'] and 'grammar' not in r['penalties']
    assert any('Grammar and spelling are not evaluated' in n for n in r['scoring_notes'])


def test_missing_jd_keywords_penalty_is_applied_and_explained():
    without_jd = sc.calculate_overall_score(TEXT, PARSED, PARSED['skills'], ['k'] * 12, ['v'] * 8, VALIDATION, NO_LOCATION)
    with_jd = sc.calculate_overall_score(
        TEXT, PARSED, PARSED['skills'], ['k'] * 12, ['v'] * 8, VALIDATION, NO_LOCATION,
        jd_keywords=['terraform', 'kubernetes', 'golang', 'rust', 'haskell'],
    )
    assert with_jd['penalties'].get('missing_jd_keywords') == 15.0
    assert with_jd['overall_score'] < without_jd['overall_score']
    assert any('missing from the resume' in n for n in with_jd['scoring_notes'])


def test_location_penalty_reduces_ats_compatibility_and_is_explained():
    clean = sc.calculate_overall_score(TEXT, PARSED, PARSED['skills'], [], [], VALIDATION, NO_LOCATION)
    loc = sc.detect_location_info('12 Park Street Kolkata - 700016')
    dirty = sc.calculate_overall_score(TEXT, PARSED, PARSED['skills'], [], [], VALIDATION, loc)
    assert dirty['ats_compatibility_score'] < clean['ats_compatibility_score']
    assert any('street address or postal code' in n for n in dirty['scoring_notes'])


def test_skill_validation_reuses_embeddings(monkeypatch):
    from tests.conftest import FakeEmbedder

    class Counting(FakeEmbedder):
        calls = 0
        def encode(self, text, **kw):
            Counting.calls += 1
            return super().encode(text, **kw)

    skills = [f'skill{i}' for i in range(20)]
    projects = [{'title': 'p1', 'description': 'text one'}, {'title': 'p2', 'description': 'text two'}]
    sc.validate_skills_with_projects(skills, projects, [{'description': 'exp text'}], Counting())
    # 20 skills + 3 texts, each embedded once (previously 2 encodes per skill/text pair = 120)
    assert Counting.calls <= 23


def test_jd_comparison_counts_resume_skills_not_only_keywords():
    """Regression: skills listed on the resume (e.g. Python) were reported as 'missing' because only the
    keyword list was compared against the job description."""
    from backend.services.jd_matcher import compare_resume_with_jd
    from tests.conftest import FakeEmbedder, FakeNLP
    result = compare_resume_with_jd(
        resume_text='Python FastAPI developer', resume_keywords=['microservices'], resume_skills=['Python', 'FastAPI'],
        jd_text='Python FastAPI AWS Terraform microservices',
        jd_keywords=['Python', 'FastAPI', 'AWS', 'Terraform', 'microservices'],
        embedder=FakeEmbedder(), nlp=FakeNLP(),
    )
    assert set(result['matched_keywords']) == {'Python', 'FastAPI', 'microservices'}
    assert set(result['missing_keywords']) == {'AWS', 'Terraform'}
