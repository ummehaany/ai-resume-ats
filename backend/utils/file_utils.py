"""Logging setup, shared exception types and small helpers used across the backend."""
import logging
import os
import sys
from typing import Callable, Optional, Tuple, TypeVar

from backend.core import config

logger = logging.getLogger('ats_resume_scorer')


def configure_logging() -> None:
    """Console logging always; a log file only if LOG_FILE is set (and writable)."""
    # Third-party parsers log the document content they are reading at DEBUG level (pdfminer prints the
    # PDF's text objects). Keep them at WARNING no matter how verbose the root logger is made.
    for noisy in ('pdfminer', 'pdfplumber', 'PyPDF2', 'fontTools', 'weasyprint', 'httpcore', 'groq'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if logger.handlers:
        return
    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    if config.LOG_FILE:
        try:
            log_dir = os.path.dirname(os.path.abspath(config.LOG_FILE))
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(config.LOG_FILE)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        except OSError as exc:  # read-only filesystem etc. must never stop the app
            logger.warning(f'Could not open LOG_FILE {config.LOG_FILE!r} ({exc}); logging to console only')


configure_logging()


# ── Exceptions ──────────────────────────────────────────────────────────────
class ATSBaseError(Exception):
    """Base class for errors whose message is safe to show to the user."""
    def __init__(self, message: str, user_message: Optional[str] = None, **kwargs):
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message


class FileUploadError(ATSBaseError):
    pass


class FileValidationError(ATSBaseError):
    """The uploaded file is empty, too large, or not a supported type."""


class FileParsingError(ATSBaseError):
    """The file looks valid but its text could not be extracted."""


class TextExtractionError(ATSBaseError):
    pass


# ── Logging helpers ─────────────────────────────────────────────────────────
def log_error(error: Exception, context: Optional[str] = None, **kwargs) -> None:
    logger.error(f"Error in {context or 'unknown'}: {error}")


def log_warning(message: str, context: Optional[str] = None, **kwargs) -> None:
    logger.warning(f"{context}: {message}" if context else message)


def log_info(message: str, context: Optional[str] = None, **kwargs) -> None:
    logger.info(f"{context}: {message}" if context else message)


T = TypeVar('T')


def with_fallback(
    primary_func: Callable[..., T],
    fallback_func: Callable[..., T],
    *args,
    log_fallback: bool = True,
    **kwargs
) -> Tuple[T, bool]:
    """Run primary_func; if it raises, run fallback_func. Returns (result, used_fallback)."""
    kwargs.pop('error_category', None)
    try:
        return primary_func(*args, **kwargs), False
    except Exception as primary_error:
        if log_fallback:
            log_warning(f"Primary method failed, trying fallback: {primary_error}")
        try:
            return fallback_func(*args, **kwargs), True
        except Exception as fallback_error:
            log_error(fallback_error, context="fallback")
            raise
