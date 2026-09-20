"""Supabase (PostgREST) access for the ``analyses`` table.

Every call is made *as the signed-in user*: the user's own access token is sent
as the Bearer token together with the public anon key. Postgres Row Level
Security (see ``supabase/schema.sql``) therefore guarantees a user can only
ever read, insert or delete their own rows - even if application code had a bug.
No service-role key is needed or used by the backend.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

from backend.core import config

logger = logging.getLogger('ats_resume_scorer')

TABLE = 'analyses'


class DatabaseError(Exception):
    """Supabase could not be reached, is not configured, or rejected the request."""


def _table_url() -> str:
    return f"{config.SUPABASE_URL.rstrip('/')}/rest/v1/{TABLE}"


def _headers(access_token: str) -> Dict[str, str]:
    if not config.SUPABASE_URL or not config.SUPABASE_ANON_KEY:
        raise DatabaseError('Supabase is not configured on the server.')
    return {
        'apikey': config.SUPABASE_ANON_KEY,
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation',
    }


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=config.SUPABASE_TIMEOUT_SECONDS)


def _check(response: httpx.Response, action: str) -> None:
    if response.status_code >= 400:
        # Server logs only (never sent to the client). Log the error code + message but NOT the "details" field:
        # PostgREST puts the offending row there, which would contain the user's resume analysis.
        try:
            err = response.json()
            hint = f"{err.get('code')}: {str(err.get('message'))[:200]}"
        except ValueError:
            hint = 'non-JSON error body'
        logger.error(f'Supabase {action} failed: HTTP {response.status_code} ({hint})')
        raise DatabaseError(f'Database request failed (HTTP {response.status_code}).')


async def _request(method: str, action: str, access_token: str, **kwargs) -> httpx.Response:
    headers = _headers(access_token)
    try:
        async with _client() as client:
            response = await client.request(method, _table_url(), headers=headers, **kwargs)
    except httpx.HTTPError as exc:
        logger.error(f'Supabase {action} failed: {type(exc).__name__}')
        raise DatabaseError('Could not reach the database.') from None
    _check(response, action)
    return response


def _json_default(o):
    if hasattr(o, 'model_dump'):
        return o.model_dump()
    return str(o)


async def save_analysis(user_id: str, access_token: str, filename: str, analysis_result: Dict) -> Optional[str]:
    """Insert one analysis for ``user_id``. Returns the new row id."""
    serializable_result = json.loads(json.dumps(analysis_result, default=_json_default))
    doc = {
        'user_id': user_id,
        'filename': filename,
        'ats_score': serializable_result.get('ats_score', 0),
        'keyword_match': serializable_result.get('keyword_match', 0),
        'missing_keywords': serializable_result.get('missing_keywords', []),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'analysis_result': serializable_result,
    }
    response = await _request('POST', 'insert', access_token, json=doc)
    rows = response.json()
    if rows:
        inserted_id = str(rows[0].get('id'))
        logger.info(f'Saved analysis {inserted_id}')
        return inserted_id
    return None


def _shape(doc: Dict) -> Dict:
    result = doc.get('analysis_result') or {}
    return {
        'id': str(doc.get('id')),
        'filename': doc.get('filename', 'resume'),
        'job_title': result.get('job_title'),   # only present when a job description was analysed
        'ats_score': doc.get('ats_score', 0),
        'keyword_match': doc.get('keyword_match', 0),
        'missing_keywords': doc.get('missing_keywords', []),
        'created_at': doc.get('created_at', ''),
        'analysis_result': result,
    }


async def get_user_history(user_id: str, access_token: str, limit: int = 50, offset: int = 0) -> List[Dict]:
    limit = max(1, min(int(limit), config.HISTORY_MAX_LIMIT))
    offset = max(0, int(offset))
    response = await _request(
        'GET', 'list', access_token,
        params={
            'user_id': f'eq.{user_id}',
            'order': 'created_at.desc',
            'limit': str(limit),
            'offset': str(offset),
        },
    )
    return [_shape(doc) for doc in response.json()]


async def get_analysis(analysis_id: str, user_id: str, access_token: str) -> Optional[Dict]:
    response = await _request(
        'GET', 'get', access_token,
        params={'id': f'eq.{analysis_id}', 'user_id': f'eq.{user_id}', 'limit': '1'},
    )
    rows = response.json()
    return _shape(rows[0]) if rows else None


async def delete_analysis(analysis_id: str, user_id: str, access_token: str) -> bool:
    """Delete one row. Returns False when nothing matched (unknown id, or not this user's row)."""
    response = await _request(
        'DELETE', 'delete', access_token,
        params={'id': f'eq.{analysis_id}', 'user_id': f'eq.{user_id}'},
    )
    try:
        return len(response.json()) > 0
    except ValueError:   # empty body (HTTP 204) - the delete succeeded
        return True
