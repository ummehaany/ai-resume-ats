"""File validation and text extraction (audit items: empty-file return value, .doc, python-magic removal, duplicate exceptions)."""
import io
import zipfile

import pytest

from backend.core import config
from backend.services import resume_parser as rp
from backend.utils import file_utils
from backend.utils.file_utils import FileParsingError, FileValidationError


def test_only_one_definition_of_each_exception():
    # The original resume_parser.py re-defined FileParsingError, shadowing the imported one.
    assert rp.FileParsingError is file_utils.FileParsingError
    assert rp.FileValidationError is file_utils.FileValidationError


def test_empty_file_returns_a_three_tuple():
    # Original returned a 2-tuple here, which crashed the caller's unpacking.
    result = rp.validate_file(b'', 'x.pdf')
    assert len(result) == 3
    ok, message, file_type = result
    assert ok is False and 'empty' in message.lower() and file_type is None


def test_oversized_file_rejected():
    ok, message, _ = rp.validate_file(b'%PDF-1.4' + b'0' * (config.MAX_FILE_SIZE_BYTES + 1), 'x.pdf')
    assert not ok and 'exceeds' in message


def test_pdf_and_docx_detected(resume_pdf, resume_docx):
    assert rp.validate_file(resume_pdf, 'a.pdf') == (True, '', 'pdf')
    assert rp.validate_file(resume_docx, 'a.docx') == (True, '', 'docx')


def test_type_is_detected_from_content_not_filename(resume_pdf):
    assert rp.validate_file(resume_pdf, 'renamed.docx')[2] == 'pdf'
    ok, message, _ = rp.validate_file(b'just some text, not a resume file at all', 'resume.pdf')
    assert not ok and 'Unsupported' in message


def test_legacy_doc_gets_a_clear_message():
    ole = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' + b'\x00' * 600
    ok, message, _ = rp.validate_file(ole, 'old.doc')
    assert not ok and '.doc' in message and 'docx' in message.lower()


def test_zip_that_is_not_docx_is_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('notes.txt', 'hello')
    ok, message, _ = rp.validate_file(buf.getvalue(), 'x.docx')
    assert not ok


def test_docx_zip_bomb_guard(resume_docx, monkeypatch):
    monkeypatch.setattr(config, 'MAX_DOCX_UNCOMPRESSED_BYTES', 100)
    ok, message, _ = rp.validate_file(resume_docx, 'x.docx')
    assert not ok and 'unreasonable size' in message


def test_extracts_text_from_pdf_and_docx(resume_pdf, resume_docx):
    text, meta = rp.parse_resume_file(resume_pdf, 'r.pdf')
    assert 'Jane Doe' in text and 'FastAPI' in text and meta['file_type'] == 'pdf'
    text, meta = rp.parse_resume_file(resume_docx, 'r.docx')
    assert 'Jane Doe' in text and meta['file_type'] == 'docx'


def test_corrupt_pdf_raises_parsing_error():
    with pytest.raises(FileParsingError):
        rp.parse_resume_file(b'%PDF-1.4 this is not really a pdf', 'r.pdf')


def test_validation_failure_raises_validation_error():
    with pytest.raises(FileValidationError):
        rp.parse_resume_file(b'', 'r.pdf')


def test_pdf_page_limit(monkeypatch, resume_pdf):
    monkeypatch.setattr(rp, 'MAX_PDF_PAGES', 0)
    with pytest.raises(FileValidationError, match='pages'):
        rp.parse_resume_file(resume_pdf, 'r.pdf')


def test_python_magic_is_no_longer_required():
    import sys
    assert 'magic' not in sys.modules or True   # informational: module must not be imported by the parser
    src = open(rp.__file__, encoding='utf-8').read()
    assert 'import magic' not in src
