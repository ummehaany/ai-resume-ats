import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple

from backend.utils.file_utils import log_warning
from backend.core import config
from backend.utils.matching import fuzzy_match_keywords

# Privacy check patterns. They are deliberately conservative: a city or country name is NOT flagged
# (that is normal and recommended in a resume header); only a full street address or a postal code is.
_STREET_SUFFIXES = (
    'Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl|Nagar|Marg'
)
STREET_ADDRESS_PATTERN = re.compile(
    r'\b\d{1,5}[A-Za-z]?(?:[-/]\d+)?,?\s+(?:[A-Z][A-Za-z.\'-]*\s+){1,4}(?:' + _STREET_SUFFIXES + r')\b'
)
# "City, ST 12345" (US)
US_CITY_ZIP_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b')
# "PIN: 700016", "Zip code 94105", "Postal code: 560001"
LABELLED_POSTAL_PATTERN = re.compile(
    r'\b(?:pin(?:\s*code)?|zip(?:\s*code)?|postal\s*code|postcode)\b[\s:.#-]*\d{5,6}\b', re.IGNORECASE
)
# "Kolkata - 700016" / "Pune, 411001" (India) - header area only, to avoid matching numbers elsewhere
IN_CITY_PIN_PATTERN = re.compile(r'\b[A-Z][a-z]+\s*[-,]\s*[1-9]\d{5}\b')
_HEADER_CHARS = 500


def _tier_score(n: float, tiers:list)-> float:
    for threshold, pts in tiers:
        if n>=threshold:
            return pts
    
    return 0.0

#Location/privacy detection (regex based; no NLP model required)
def detect_location_info(text: str, nlp=None) -> Dict:
    """Detect a full street address or postal code in the resume text.

    ``nlp`` is accepted for backward compatibility and ignored: named-entity
    recognition of place names produced false alarms (any city or university
    location counted against the resume), so it is not used.
    """
    text = text or ''
    findings = []

    for m in STREET_ADDRESS_PATTERN.finditer(text):
        findings.append({'text': m.group().strip(), 'type': 'address', 'start': m.start()})

    postal_matches = list(US_CITY_ZIP_PATTERN.finditer(text)) + list(LABELLED_POSTAL_PATTERN.finditer(text))
    postal_matches += list(IN_CITY_PIN_PATTERN.finditer(text[:_HEADER_CHARS]))
    for m in postal_matches:
        findings.append({'text': m.group().strip(), 'type': 'postal_code', 'start': m.start()})

    has_address = any(f['type'] == 'address' for f in findings)
    has_postal = any(f['type'] == 'postal_code' for f in findings)

    if has_address and has_postal:
        privacy_risk, penalty = 'high', 3.0
    elif has_address or has_postal:
        privacy_risk, penalty = 'medium', 2.0
    else:
        privacy_risk, penalty = 'none', 0.0

    recommendations = []
    if has_address:
        recommendations.append("Remove the full street address - ATS systems don't need it and it is a privacy risk.")
    if has_postal:
        recommendations.append("Remove the postal / ZIP / PIN code - 'City, State' is enough.")

    return {
        'location_found':     bool(findings),
        'detected_locations': findings,
        'privacy_risk':       privacy_risk,
        'recommendations':    recommendations,
        'penalty_applied':    penalty,
    }

def _calculate_semantic_similarity(skill: str, text: str, embedder: SentenceTransformer, encode=None) -> float:
    #similarity = (A · B) / (|A| × |B|)
    if not skill or not text:
        return 0.0
    encode = encode or (lambda t: embedder.encode(t, convert_to_tensor=False))
    try:
        skill_vec  = encode(skill)
        text_vec   = encode(text)

        denom = np.linalg.norm(skill_vec) * np.linalg.norm(text_vec)
        if not denom:
            return 0.0
        similarity = np.dot(skill_vec, text_vec) / denom

        return float(max(0.0, min(1.0, similarity)))
    except Exception as e:
        log_warning(f'Similarity error: {type(e).__name__}', context='ats_scorer')   # no resume-derived text in logs
        return 0.0

def _skill_matches(skill: str, text: str, embedder: SentenceTransformer, threshold: float, encode=None) -> Tuple[bool, float]:

    #fast, o(n) directly check if skill is a substring of the text (case-insensitive)
    if skill.lower() in text.lower():
        return True, 1.0
    
    #slow, semantic similarity check using sentence embeddings
    sim = _calculate_semantic_similarity(skill, text, embedder, encode)
    return sim >= threshold, sim

#Skill validation
def validate_skills_with_projects(
    skills: List[str],
    projects: List[Dict],
    experience_entries: List[Dict],
    embedder: SentenceTransformer,
    threshold: float = 0.6,
) -> Dict:
    
    if not skills:
        return {
            'validated_skills':      [],
            'unvalidated_skills':    [],
            'validation_percentage': 0.0,
            'skill_project_mapping': {},
            'validation_score':      0.0,
        }

    experience_text = ' '.join(
        f"{e.get('job_title', '')} {e.get('company', '')} {e.get('description', '')}"
        for e in experience_entries
        if isinstance(e, dict)
    ).strip()

    validated_skills      = []
    unvalidated_skills    = []
    skill_project_mapping = {}

    # Each distinct text is embedded once per analysis instead of once per (skill, project) pair.
    _cache: Dict[str, object] = {}
    def encode(t: str):
        if t not in _cache:
            _cache[t] = embedder.encode(t, convert_to_tensor=False)
        return _cache[t]

    for skill in skills:
        matching_projects = []
        max_similarity    = 0.0

        for project in projects:
            project_text = f"{project.get('title', '')} {project.get('description', '')}"
            matched, sim = _skill_matches(skill, project_text, embedder, threshold, encode)
            max_similarity = max(max_similarity, sim)

            if matched:
                matching_projects.append(project.get('title', 'Untitled Project'))

        if experience_text:
            matched, sim = _skill_matches(skill, experience_text, embedder, threshold, encode)
            max_similarity = max(max_similarity, sim)
            if matched and 'Experience Section' not in matching_projects:
                matching_projects.append('Experience Section')

        if matching_projects:
            validated_skills.append({'skill': skill, 'projects': matching_projects, 'similarity': max_similarity})
            skill_project_mapping[skill] = matching_projects
        else:
            unvalidated_skills.append(skill)
            skill_project_mapping[skill] = []

    validation_percentage = len(validated_skills) / len(skills)
    validation_score      = validation_percentage * 15.0

    return {
        'validated_skills':      validated_skills,
        'unvalidated_skills':    unvalidated_skills,
        'validation_percentage': validation_percentage,
        'skill_project_mapping': skill_project_mapping,
        'validation_score':      validation_score,
    }

#01: formatting score
def _calc_formatting_score(parsed_resume: Dict, text: str) -> float:

    score = 0.0

    exp_entries  = [e for e in parsed_resume.get('experience', []) if isinstance(e, dict)]
    edu_entries  = [e for e in parsed_resume.get('education', [])  if isinstance(e, dict)]
    skills       = parsed_resume.get('skills', [])
    summary      = parsed_resume.get('professional_summary', '')
    proj_entries = [p for p in parsed_resume.get('projects', [])   if isinstance(p, dict)]

    if exp_entries and any(e.get('job_title') or e.get('description') for e in exp_entries):
        score += 3.0
    if edu_entries:
        score += 2.0
    if len(skills) >= 3:
        score += 2.0
    if len(summary) > 30:
        score += 1.5
    if proj_entries:
        score += 1.5

    bullet_count = sum(
        1 for line in text.split('\n')
        if re.match(r'^\s*[•\-\*\◦]', line) or re.match(r'^\s*\d+\.', line)
    )
    score += _tier_score(bullet_count, [(15,5.0),(10,4.0),(5,3.0),(3,2.0),(1,1.0)])

    filled = sum(1 for has_it in [
        bool(exp_entries), bool(edu_entries), bool(skills),
        bool(summary.strip()), bool(proj_entries),
    ] if has_it)
    score += _tier_score(filled, [(4,5.0),(3,4.0),(2,3.0),(1,2.0)])

    return min(20.0, max(0.0, score))

#02 keyword score
def _calc_keywords_score(
    resume_keywords: List[str],
    skills: List[str],
    jd_keywords: Optional[List[str]] = None,
) -> float:
    score = 0.0

    score += _tier_score(len(resume_keywords), [(20,10.0),(15,8.0),(10,6.0),(5,4.0),(3,2.0)])
    score += _tier_score(len(skills),          [(15,10.0),(10,8.0),(7,6.0),(5,4.0),(3,2.0)])

    if jd_keywords:
        all_resume_terms = list(set(resume_keywords + skills))
        fuzzy_result     = fuzzy_match_keywords(all_resume_terms, jd_keywords, threshold=80)
        match_pct        = len(fuzzy_result['matched']) / len(jd_keywords) if jd_keywords else 0
        score += _tier_score(match_pct, [(0.7,5.0),(0.5,4.0),(0.3,3.0),(0.2,2.0),(0.1,1.0)])
    
    elif len(resume_keywords) >= 10:
        score += 3.0

    return min(25.0, max(0.0, score))

#3. CONTENT QUALITY SCORE  (max 25 = action verbs 15 + quantified achievements 10)
# Grammar/spelling is NOT scored: there is no reliable checker in this project, and the
# original code awarded every resume 10 free "perfect grammar" points.
def _calc_content_score(
    text: str,
    action_verbs: List[str],
) -> float:

    score = 0.0

    score += _tier_score(len(action_verbs), [(15, 15.0), (10, 12.0), (7, 9.0), (5, 6.0), (3, 3.0)])

    number_patterns = [
        r'\d+%',
        r'\$\d+',
        r'\d+[kKmMbB]',
        r'\d+\s*(?:users|customers|clients|projects|hours|days|months|years)',
        r'(?:increased|decreased|improved|reduced|grew|saved)\s+(?:by\s+)?\d+',
    ]
    achievement_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in number_patterns)
    score += _tier_score(achievement_count, [(10, 10.0), (7, 8.0), (5, 6.0), (3, 4.0), (1, 2.0)])

    return min(25.0, max(0.0, score))

#4. SKILL VALIDATION SCORE
def _calc_skill_validation_score(validation_results: Dict) -> float:
    return min(15.0, max(0.0, validation_results.get('validation_score', 0.0)))

#5. ATS COMPATIBILITY SCORE
def _calc_ats_compatibility_score(
    text: str,
    location_results: Dict,
    parsed_resume: Dict,
) -> float:

    score = 15.0

    #dedeuction01
    score -= location_results.get('penalty_applied', 0.0)

    #deduction02
    special_chars = len(re.findall(r'[│┤├┼┴┬╔╗╚╝═║╠╣╦╩╬]', text))
    if special_chars > 20:    score -= 2.0
    elif special_chars > 10:  score -= 1.0

    exp_entries  = [e for e in parsed_resume.get('experience', []) if isinstance(e, dict)]
    edu_entries  = [e for e in parsed_resume.get('education', [])  if isinstance(e, dict)]
    skills_count = len(parsed_resume.get('skills', []))

    exp_desc_len = sum(len(e.get('description', '')) for e in exp_entries)
    edu_desc_len = sum(len((e.get('degree') or '') + (e.get('institution') or '')) for e in edu_entries)  # Handle None to prevent string concatenation errors

    #deduction03
    short_sections = sum([
        bool(exp_entries) and exp_desc_len < 20,
        bool(edu_entries) and edu_desc_len < 20,
        bool(parsed_resume.get('skills')) and skills_count < 2,
    ])
    if short_sections >= 2:    score -= 2.0
    elif short_sections >= 1:  score -= 1.0

    if exp_entries and skills_count > 5:
        score += 1.0

    return min(15.0, max(0.0, score))

#Score aggregation and final interpretation
def calculate_overall_score(
    text: str,
    parsed_resume: Dict,
    skills: List[str],
    keywords: List[str],
    action_verbs: List[str],
    skill_validation_results: Dict,
    location_results: Dict,
    jd_keywords: Optional[List[str]] = None,
    experience_months: int = 0,
) -> Dict:
    """Combine the five component scores into the overall 0-100 ATS score.

    overall = 40% (60% keywords + 40% skill validation) + 30% content
              + 15% formatting + 15% ATS compatibility   (each as a % of its own maximum)
              + skill-validation bonus (0/+1/+2)
              - missing-JD-keyword penalty (0/-5/-10/-15, only when a JD was given)
    """
    formatting_score        = _calc_formatting_score(parsed_resume, text)
    keywords_score          = _calc_keywords_score(keywords, skills, jd_keywords)
    content_score           = _calc_content_score(text, action_verbs)
    skill_validation_score  = _calc_skill_validation_score(skill_validation_results)
    ats_compatibility_score = _calc_ats_compatibility_score(text, location_results, parsed_resume)

    cmax = config.SCORE_COMPONENT_MAX
    formatting_pct        = formatting_score        / cmax['formatting']        * 100.0
    keywords_pct          = keywords_score          / cmax['keywords']          * 100.0
    content_pct           = content_score           / cmax['content']           * 100.0
    skill_validation_pct  = skill_validation_score  / cmax['skill_validation']  * 100.0
    ats_compatibility_pct = ats_compatibility_score / cmax['ats_compatibility'] * 100.0

    kw_share = config.SCORE_KEYWORDS_SHARE_OF_SKILLS
    skills_keywords_pct = keywords_pct * kw_share + skill_validation_pct * (1.0 - kw_share)

    blend = config.SCORE_BLEND
    base_score = (
        skills_keywords_pct   * blend['skills_and_keywords'] +
        content_pct           * blend['content'] +
        formatting_pct        * blend['formatting'] +
        ats_compatibility_pct * blend['ats_compatibility']
    )

    bonuses   = {}
    penalties = {}
    notes: List[str] = [
        f'Base score {base_score:.1f} = 40% skills & keywords + 30% content + 15% formatting + 15% ATS compatibility.',
        'Grammar and spelling are not evaluated and do not affect the score.',
    ]
    score = base_score

    if location_results.get('penalty_applied', 0.0) > 0:
        notes.append(
            f"A street address or postal code was detected: ATS compatibility reduced by "
            f"{location_results['penalty_applied']:.0f} point(s)."
        )

    validation_pct = skill_validation_results.get('validation_percentage', 0.0)
    if validation_pct >= 0.9:
        bonuses['excellent_skill_validation'] = 2.0
    elif validation_pct >= 0.8:
        bonuses['good_skill_validation'] = 1.0
    for name, value in bonuses.items():
        score += value
        notes.append(f'Bonus +{value:.0f}: {name.replace("_", " ")} ({validation_pct * 100:.0f}% of skills backed by evidence).')

    if jd_keywords and len(jd_keywords) > 0:
        all_resume_terms = list(set((keywords or []) + (skills or [])))
        fuzzy_result     = fuzzy_match_keywords(all_resume_terms, jd_keywords, threshold=80)
        missing_pct      = len(fuzzy_result['missing']) / len(jd_keywords)

        deduction = 0.0
        if missing_pct > 0.7:
            deduction = 15.0
        elif missing_pct > 0.5:
            deduction = 10.0
        elif missing_pct > 0.3:
            deduction = 5.0
        if deduction:
            penalties['missing_jd_keywords'] = deduction
            score -= deduction
            notes.append(f'Penalty -{deduction:.0f}: {missing_pct * 100:.0f}% of job-description keywords are missing from the resume.')

    overall_score = min(100.0, max(0.0, score))
    interpretation = _generate_score_interpretation(overall_score)

    return {
        'overall_score':           round(overall_score, 1),
        'formatting_score':        round(formatting_score, 1),
        'keywords_score':          round(keywords_score, 1),
        'content_score':           round(content_score, 1),
        'skill_validation_score':  round(skill_validation_score, 1),
        'ats_compatibility_score': round(ats_compatibility_score, 1),
        'overall_interpretation':  interpretation,
        'penalties':               penalties,
        'bonuses':                 bonuses,
        'scoring_notes':           notes,
    }


#Actionable improvements to enhance ATS performance
def generate_improvements(
    score_results: Dict,
    skill_validation_results: Dict,
) -> List[str]:
    improvements = []

    if 12 <= score_results['formatting_score']       < 16:
        improvements.append('Add more bullet points and improve section organization')
    if 14 <= score_results['keywords_score']          < 20:
        improvements.append('Include more relevant keywords and technical skills')
    if 14 <= score_results['content_score']           < 20:
        improvements.append('Add more quantifiable achievements and action verbs')
    if 7  <= score_results['skill_validation_score']  < 12:
        unvalidated_count = len(skill_validation_results.get('unvalidated_skills', []))
        improvements.append(f'Validate {unvalidated_count} skill(s) by adding relevant project details')
    if 9  <= score_results['ats_compatibility_score'] < 13:
        improvements.append('Simplify formatting for better ATS compatibility')

    return improvements

#Interpretation of overall score
def _generate_score_interpretation(overall_score: float) -> str:
    if overall_score >= 90:    return 'Excellent! Your resume is highly optimized for ATS systems.'
    elif overall_score >= 80:  return 'Great! Your resume should perform well with most ATS systems.'
    elif overall_score >= 70:  return 'Good! Your resume is ATS-friendly with room for minor improvements.'
    elif overall_score >= 60:  return 'Fair. Your resume needs some improvements to be fully ATS-compatible.'
    elif overall_score >= 50:  return 'Below Average. Significant improvements needed for ATS compatibility.'
    else:                      return 'Poor. Your resume requires major revisions to pass ATS screening.'
