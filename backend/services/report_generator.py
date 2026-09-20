import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from typing import Any, Dict, List

TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))

# SECURITY: autoescape=True HTML-escapes every {{ value }}. Resume-derived text (skills, project names,
# feedback...) and client-supplied JSON end up in these templates, so without escaping a value such as
# '<img src="http://internal-host/...">' would become real HTML that the PDF renderer might fetch.
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce anything to a finite float (stored history data may be missing or malformed)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and number not in (float('inf'), float('-inf')) else default


def _str_list(value: Any) -> List[str]:
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else []

def format_date(value, fmt='%B %d, %Y at %I:%M %p'):
    """Convert ISO timestamp string → human-readable date string."""
    if not value:
        return ''
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return dt.strftime(fmt)
    except Exception:
        return value

env.filters['format_date'] = format_date

def generate_html_reports(analysis_data: Dict) -> Dict[str, str]:
    # 1. Extract timestamp 
    now = datetime.now().isoformat()

    # 2.Overall score + interpretation 
    overall_score = _num(analysis_data.get('ATS_score', 0) or analysis_data.get('ats_score', 0))
    interpretation = str(analysis_data.get('interpretation') or '')
    cs = analysis_data.get('component_scores') or {}
    if hasattr(cs, '__dict__'):      # handle Pydantic model objects
        cs = cs.__dict__

    component_scores = {
        'formatting':       _num(cs.get('formatting', 0)),
        'keywords':         _num(cs.get('keywords', 0)),
        'content':          _num(cs.get('content', 0)),
        'skill_validation': _num(cs.get('skill_validation', 0)),
        'ats_compatibility': _num(cs.get('ats_compatibility', 0)),
    }

    # Progress-bar percentages (used in Report 1's visual breakdown)
    def pct(score, max_score):
        return min(100, max(0, round(score / max_score * 100)))

    component_pct = {
        'formatting':       pct(component_scores['formatting'],       20),
        'keywords':         pct(component_scores['keywords'],         25),
        'content':          pct(component_scores['content'],          25),
        'skill_validation': pct(component_scores['skill_validation'], 15),
        'ats_compatibility': pct(component_scores['ats_compatibility'], 15),
    }

    raw_feedback = analysis_data.get('detailed_feedback') or []
    if not isinstance(raw_feedback, (list, tuple)):
        raw_feedback = []

    # Normalise: each item may be a dict or an IssueDetail Pydantic object
    def to_dict(item):
        if isinstance(item, dict):
            return item
        if hasattr(item, 'model_dump'):
            return item.model_dump()
        return None      # anything else (malformed stored data) is skipped

    detailed_feedback = [d for d in (to_dict(fb) for fb in raw_feedback) if isinstance(d, dict)]

    high_priority   = [fb for fb in detailed_feedback
                       if str(fb.get('severity_level', '')).lower() in ('high',)]
    
    medium_priority = [fb for fb in detailed_feedback
                       if str(fb.get('severity_level', '')).lower() in ('moderate', 'medium')]
    
    low_priority    = [fb for fb in detailed_feedback
                       if str(fb.get('severity_level', '')).lower() in ('low', 'info')]

    strengths = _str_list(analysis_data.get('strengths', []))


    svd_raw = analysis_data.get('skill_validation_details') or {}
    
    if hasattr(svd_raw, 'model_dump'):
        svd_raw = svd_raw.model_dump()

    validated_skills   = svd_raw.get('validated', [])    # [{'skill', 'projects'}]
    unvalidated_skills = svd_raw.get('unvalidated', [])  # ['Flask', ...]
    total_skills       = svd_raw.get('total', len(validated_skills) + len(unvalidated_skills))
    validated_count    = svd_raw.get('validated_count', len(validated_skills))
    validation_pct     = svd_raw.get('validation_pct', 0.0)

    #7. JD comparison (for Report 3) 
    jd_raw = analysis_data.get('jd_match_analysis') or analysis_data.get('jd_comparison')
    if hasattr(jd_raw, 'model_dump'):
        jd_raw = jd_raw.model_dump()


    #8. Score colour (green / orange / red) 
    if overall_score >= 80:
        score_color = '#16a34a'   # green
    elif overall_score >= 60:
        score_color = '#d97706'   # amber
    else:
        score_color = '#dc2626'   # red

    #9. Build shared context dict passed to every template 
    context = {
        'timestamp':          now,
        'overall_score':      overall_score,
        'score_color':        score_color,
        'interpretation':     interpretation,
        'component_scores':   component_scores,
        'component_pct':      component_pct,
        'strengths':          strengths,
        'high_priority':      high_priority,
        'medium_priority':    medium_priority,
        'low_priority':       low_priority,
        'all_feedback':       detailed_feedback,
        # Skill validation
        'validated_skills':   validated_skills,
        'unvalidated_skills': unvalidated_skills,
        'total_skills':       total_skills,
        'validated_count':    validated_count,
        'validation_pct':     validation_pct,
        # JD analysis
        'jd_analysis':        jd_raw,
    }

    return {
        'summary':         env.get_template('summary.html').render(**context),
        'skill_report':    env.get_template('action_items.html').render(**context),
        'jd_report':       env.get_template('quick_actions.html').render(**context),
        'recommendations': env.get_template('jd_comparison.html').render(**context),
    }