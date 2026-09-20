import asyncio
import contextlib
import logging
import os
import re
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from backend.api.auth import AuthenticatedUser, get_current_user
from backend.core import config
from backend.core.rate_limit import limiter
from backend.database import supabase_db
from backend.database.supabase_db import DatabaseError
from backend.models.schemas import (
    AnalysisResponse,
    ComponentScores,
    JDComparison,
    SkillValidationDetails,
)
from backend.services.groq_parser import LLMBusyError, LLMServiceError
from backend.utils.file_utils import FileParsingError, FileValidationError

logger = logging.getLogger('ats_resume_scorer')

router = APIRouter(prefix='/api/v1', tags=['Analysis'])

MIN_RESUME_CHARS = 50


# ── helpers ─────────────────────────────────────────────────────────────────
def _safe_filename(name: Optional[str]) -> str:
    """Untrusted upload name -> short, printable, path-free string (used for display/storage only)."""
    base = os.path.basename((name or '').replace('\\', '/'))
    base = re.sub(r'[^\w.\- ()]', '_', base).strip(' .')
    return base[:120] or 'resume'


def _parse_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail='Analysis not found.')


_semaphores: dict = {}


def _semaphore(name: str, size: int) -> asyncio.Semaphore:
    sem = _semaphores.get(name)
    if sem is None:
        sem = _semaphores[name] = asyncio.Semaphore(max(1, size))
    return sem


@contextlib.asynccontextmanager
async def _slot(name: str, size: int):
    """Bound how many heavy jobs (parsing + LLM + embeddings, PDF rendering) run at once."""
    sem = _semaphore(name, size)
    try:
        await asyncio.wait_for(sem.acquire(), timeout=config.BUSY_WAIT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail='The server is busy analysing other resumes. Please try again in a moment.',
            headers={'Retry-After': '15'},
        )
    try:
        yield
    finally:
        sem.release()


def _rate_limited(retry_after: int, what: str) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=f'Too many {what} requests. Please wait {retry_after} seconds and try again.',
        headers={'Retry-After': str(retry_after)},
    )


def analyze_rate_limit(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    """Per-user and global limits so nobody can burn through the Groq quota."""
    retry_after = limiter.try_acquire([
        (f'analyze:min:{user.user_id}', config.ANALYZE_RATE_LIMIT_PER_MINUTE, 60),
        (f'analyze:day:{user.user_id}', config.ANALYZE_RATE_LIMIT_PER_DAY, 86400),
        ('analyze:global:day', config.GLOBAL_ANALYSES_PER_DAY, 86400),
    ])
    if retry_after:
        raise _rate_limited(retry_after, 'analysis')
    return user


def pdf_rate_limit(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    retry_after = limiter.try_acquire([
        (f'pdf:min:{user.user_id}', config.PDF_RATE_LIMIT_PER_MINUTE, 60),
    ])
    if retry_after:
        raise _rate_limited(retry_after, 'PDF')
    return user


def _db_error(exc: DatabaseError, what: str) -> HTTPException:
    logger.error(f'{what}: {exc}')
    return HTTPException(status_code=502, detail='The database is temporarily unavailable. Please try again.')


# ── routes ──────────────────────────────────────────────────────────────────
@router.post('/analyze-resume', response_model=AnalysisResponse)
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(..., description='Resume file - PDF or DOCX, max 5 MB'),
    job_description: str = Form('', description='Job description text (optional)'),
    user: AuthenticatedUser = Depends(analyze_rate_limit),
):
    jd_text = (job_description or '').strip()
    if len(jd_text) > config.MAX_JD_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f'The job description is too long ({len(jd_text)} characters; the maximum is {config.MAX_JD_CHARS}).',
        )

    nlp = request.app.state.nlp
    embedder = request.app.state.embedder
    filename = _safe_filename(resume.filename)

    # Read at most one byte more than the limit so an oversized upload cannot exhaust memory.
    file_bytes = await resume.read(config.MAX_FILE_SIZE_BYTES + 1)

    from backend.services.resume_parser import parse_resume_file
    from backend.services.resume_analyzer import analyze_full_resume

    async with _slot('analysis', config.MAX_CONCURRENT_ANALYSES):
        try:
            resume_text, _metadata = await run_in_threadpool(parse_resume_file, file_bytes, filename)
        except (FileValidationError, FileParsingError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception:
            logger.exception('Unexpected error while parsing the uploaded resume')
            raise HTTPException(status_code=422, detail='Could not read the resume. Please upload a valid PDF or DOCX file.')

        if len(resume_text.strip()) < MIN_RESUME_CHARS:
            raise HTTPException(status_code=422, detail='The resume contains almost no readable text. Scanned/image-only PDFs are not supported.')
        logger.info(f'Parsed upload: {len(resume_text)} chars extracted')

        try:
            result = await run_in_threadpool(
                analyze_full_resume,
                resume_text=resume_text,
                nlp=nlp,
                embedder=embedder,
                job_description=jd_text,
            )
        except LLMBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={'Retry-After': '30'})
        except LLMServiceError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception:
            logger.exception('Full analysis pipeline failed')
            raise HTTPException(status_code=500, detail='The analysis failed because of an internal error. Please try again.')

    jd_comparison_result = None
    if result.get('jd_comparison'):
        jd = result['jd_comparison']
        jd_comparison_result = JDComparison(
            match_percentage=round(float(jd.get('match_percentage', 0.0)), 1),
            semantic_similarity=round(float(jd.get('semantic_similarity', 0.0)), 3),
            matched_keywords=jd.get('matched_keywords', [])[:20],
            missing_keywords=jd.get('missing_keywords', [])[:15],
            skills_gap=jd.get('skills_gap', [])[:10],
        )

    svd_raw = result.get('skill_validation_details') or {}
    skill_val_details = SkillValidationDetails(
        validated=svd_raw.get('validated', []),
        unvalidated=svd_raw.get('unvalidated', []),
        total=svd_raw.get('total', 0),
        validated_count=svd_raw.get('validated_count', 0),
        validation_pct=svd_raw.get('validation_pct', 0.0),
    )

    response = AnalysisResponse(
        ATS_score=result['ats_score'],
        component_scores=ComponentScores(**result['component_scores']),
        issues_summary=result['issues_summary'],
        detailed_feedback=result.get('detailed_feedback', []),
        jd_match_analysis=jd_comparison_result,
        skill_validation_details=skill_val_details,

        # Fields the Streamlit UI and PDF templates read
        ats_score=result['ats_score'],
        keyword_match=jd_comparison_result.match_percentage if jd_comparison_result else 0.0,
        missing_keywords=result.get('missing_keywords', []),
        matched_keywords=result.get('matched_keywords', []),
        skills=list(result.get('skills', [])[:20]),
        jd_comparison=jd_comparison_result,
        interpretation=result.get('interpretation', ''),
        strengths=result.get('strengths', []),
        critical_issues=result.get('critical_issues', []),
        suggestions=result.get('suggestions', []),
        scoring_notes=result.get('scoring_notes', []),
        job_title=result.get('job_title'),
    )

    try:
        await supabase_db.save_analysis(user.user_id, user.access_token, filename, response.model_dump())
    except DatabaseError as exc:
        logger.warning(f'History save failed (non-blocking): {exc}')
        response.warnings.append('Your results could not be saved to your history, but the analysis below is complete.')

    return response


@router.get('/health')
async def health_check(request: Request):
    """Health check - confirms models are loaded and the API is ready."""
    return {
        'status': 'healthy',
        'nlp_loaded': getattr(request.app.state, 'nlp', None) is not None,
        'embedder_loaded': getattr(request.app.state, 'embedder', None) is not None,
    }


@router.get('/history')
async def get_history(
    limit: int = Query(config.HISTORY_DEFAULT_LIMIT, ge=1, le=config.HISTORY_MAX_LIMIT),
    offset: int = Query(0, ge=0),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Return the signed-in user's past analyses, newest first (paged)."""
    try:
        return await supabase_db.get_user_history(user.user_id, user.access_token, limit=limit, offset=offset)
    except DatabaseError as exc:
        raise _db_error(exc, 'History fetch failed')


@router.delete('/history/{analysis_id}')
async def delete_history_entry(analysis_id: str, user: AuthenticatedUser = Depends(get_current_user)):
    """Delete one analysis from the signed-in user's history."""
    analysis_id = _parse_uuid(analysis_id)
    try:
        deleted = await supabase_db.delete_analysis(analysis_id, user.user_id, user.access_token)
    except DatabaseError as exc:
        raise _db_error(exc, 'History delete failed')
    if not deleted:
        raise HTTPException(status_code=404, detail='Analysis not found or not owned by this user.')
    return {'status': 'deleted', 'id': analysis_id}


async def _render_pdf(analysis_data: dict) -> bytes:
    from backend.services.report_generator import generate_html_reports
    from backend.services.pdf_export import generate_combined_pdf

    def build() -> bytes:
        return generate_combined_pdf(generate_html_reports(analysis_data))

    async with _slot('pdf', config.MAX_CONCURRENT_PDFS):
        try:
            return await run_in_threadpool(build)
        except Exception:
            logger.exception('PDF generation failed')
            raise HTTPException(status_code=500, detail='Failed to generate the PDF report.')


@router.post('/generate-pdf')
async def generate_pdf(data: AnalysisResponse, user: AuthenticatedUser = Depends(pdf_rate_limit)):
    pdf_bytes = await _render_pdf(data.model_dump())
    return Response(
        content=pdf_bytes,
        media_type='application/pdf',
        headers={'Content-Disposition': 'attachment; filename=ats_report.pdf'},
    )


@router.get('/history/{analysis_id}/pdf')
async def generate_history_pdf(analysis_id: str, user: AuthenticatedUser = Depends(pdf_rate_limit)):
    analysis_id = _parse_uuid(analysis_id)
    try:
        entry = await supabase_db.get_analysis(analysis_id, user.user_id, user.access_token)
    except DatabaseError as exc:
        raise _db_error(exc, 'History fetch failed')
    if not entry:
        raise HTTPException(status_code=404, detail='Analysis not found')

    pdf_bytes = await _render_pdf(entry['analysis_result'])
    return Response(
        content=pdf_bytes,
        media_type='application/pdf',
        headers={'Content-Disposition': f'attachment; filename=ats_report_{analysis_id}.pdf'},
    )
