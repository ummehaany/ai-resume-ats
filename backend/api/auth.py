import logging
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.core import config

logger = logging.getLogger('ats_resume_scorer')

_bearer_scheme = HTTPBearer(auto_error=False)

_ASYMMETRIC_ALGS = ['ES256', 'RS256']

_jwks_client: jwt.PyJWKClient | None = None


@dataclass(frozen=True)
class AuthenticatedUser:
    """Identity taken from a verified Supabase access token."""
    user_id: str
    access_token: str   # forwarded to Supabase so Row Level Security applies as this user


def _get_jwks_client() -> jwt.PyJWKClient | None:
    global _jwks_client
    if _jwks_client is not None:
        return _jwks_client
    if not config.SUPABASE_URL:
        return None
    jwks_url = f"{config.SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
    _jwks_client = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=3600)
    return _jwks_client


def _verify_token(token: str) -> dict:
    header = jwt.get_unverified_header(token)
    alg = header.get('alg')

    if alg in _ASYMMETRIC_ALGS:
        jwks_client = _get_jwks_client()
        if jwks_client is None:
            raise jwt.InvalidTokenError('SUPABASE_URL not configured - cannot fetch JWKS to verify token')
        signing_key = jwks_client.get_signing_key_from_jwt(token).key
        return jwt.decode(
            token,
            signing_key,
            algorithms=_ASYMMETRIC_ALGS,
            audience='authenticated',
        )

    if alg == 'HS256':
        if not config.SUPABASE_JWT_SECRET:
            raise jwt.InvalidTokenError('HS256 token received but SUPABASE_JWT_SECRET is not configured')
        return jwt.decode(
            token,
            config.SUPABASE_JWT_SECRET,
            algorithms=['HS256'],
            audience='authenticated',
        )

    raise jwt.InvalidTokenError(f'Unsupported JWT algorithm: {alg}')


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={'WWW-Authenticate': 'Bearer'},
    )


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    if creds is None or not creds.credentials:
        raise _unauthorized('Missing Authorization: Bearer <token> header')

    if not config.SUPABASE_URL and not config.SUPABASE_JWT_SECRET:
        logger.error('Neither SUPABASE_URL (for JWKS) nor SUPABASE_JWT_SECRET configured - cannot verify tokens')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Auth not configured on the server',
        )

    try:
        payload = _verify_token(creds.credentials)
    except jwt.ExpiredSignatureError:
        raise _unauthorized('Token expired - sign in again')
    except jwt.InvalidTokenError as exc:
        logger.info(f'Rejected token: {type(exc).__name__}')
        raise _unauthorized('Invalid token')
    except Exception as exc:
        # PyJWKClient can raise network errors while fetching the JWKS document.
        logger.warning(f'JWT verification failed: {type(exc).__name__}')
        raise _unauthorized('Token verification failed')

    user_id = payload.get('sub')
    if not user_id:
        raise _unauthorized('Token missing subject claim')
    return AuthenticatedUser(user_id=str(user_id), access_token=creds.credentials)
