import io
import zipfile
from typing import Optional, Tuple

import pdfplumber
from docx import Document
import PyPDF2

from backend.utils.file_utils import (
    FileParsingError,
    FileValidationError,
    TextExtractionError,
    log_error,
    log_warning,
    log_info,
    with_fallback,
)

from backend.core import config

MAX_PDF_PAGES = 20   # resumes are 1-4 pages; refuse absurdly large PDFs (CPU/memory protection)

_OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'   # legacy Office (.doc/.xls/.ppt)


def detect_file_type(file_data: bytes) -> Optional[str]:
    """Identify 'pdf' / 'docx' / 'doc' from the file *content*.

    Signature sniffing is used instead of python-magic so no system library
    (libmagic) is needed. Returns None when the type is not recognised.
    """
    if b'%PDF-' in file_data[:1024]:
        return 'pdf'
    if file_data.startswith(_OLE_MAGIC):
        return 'doc'
    if file_data[:4] in (b'PK\x03\x04', b'PK\x05\x06'):
        try:
            with zipfile.ZipFile(io.BytesIO(file_data)) as zf:
                names = set(zf.namelist())
                if 'word/document.xml' in names:
                    total = sum(info.file_size for info in zf.infolist())
                    if total > config.MAX_DOCX_UNCOMPRESSED_BYTES:
                        return 'docx-too-large'
                    return 'docx'
        except zipfile.BadZipFile:
            return None
    return None


def validate_file(file_data: bytes, filename: str) -> Tuple[bool, str, Optional[str]]:
    """Return (is_valid, error_message, file_type). Always a 3-tuple."""
    file_size_bytes = len(file_data)
    if file_size_bytes == 0:
        return False, 'The uploaded file is empty. Please check the file and try again.', None

    if file_size_bytes > config.MAX_FILE_SIZE_BYTES:
        size_mb = file_size_bytes / (1024 * 1024)
        return False, (
            f'File size ({size_mb:.2f} MB) exceeds the maximum of {config.MAX_FILE_SIZE_MB} MB. '
            'Please upload a smaller file or compress your resume.'
        ), None

    file_type = detect_file_type(file_data)
    if file_type == 'docx-too-large':
        return False, 'The DOCX file expands to an unreasonable size and was rejected.', None
    if file_type is None:
        return False, 'Unsupported or unrecognised file type. Please upload a PDF or DOCX resume.', None
    if file_type == 'doc':
        return False, (
            'Legacy Word (.doc) files are not supported. '
            'Please save your resume as .docx or .pdf and upload that instead.'
        ), None

    return True, '', file_type

def _extract_pdf_hyperlinks(file_data: bytes) -> str:
    urls = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_data))
        for page in reader.pages:
            if '/Annots' not in page:
                continue
            for annot_ref in page['/Annots']:
                try:
                    annot = annot_ref.get_object()
                    if annot.get('/Subtype') != '/Link':
                        continue
                    action = annot.get('/A', {})
                    uri = action.get('/URI', '')
                    if uri and isinstance(uri, (str, bytes)):
                        # PyPDF2 may return bytes for URI values
                        if isinstance(uri, bytes):
                            uri = uri.decode('utf-8', errors='ignore')
                        uri = uri.strip()
                        if uri.startswith('http'):
                            urls.append(uri)
                except Exception:
                    pass
    except Exception:
        pass
    return '\n'.join(urls)


def _extract_pdf_with_pdfplumber(file_data: bytes) -> str:
    text = ''
    with pdfplumber.open(io.BytesIO(file_data)) as pdf:
        if len(pdf.pages) > MAX_PDF_PAGES:
            raise FileValidationError(f'PDF has {len(pdf.pages)} pages; the maximum is {MAX_PDF_PAGES}.')
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'

    if not text.strip():
        raise TextExtractionError(
            'pdfplumber extracted no text',
            user_message='No text could be extracted from the PDF.'
        )
    
    hyperlinks = _extract_pdf_hyperlinks(file_data)
    if hyperlinks:
        text = text.strip() + '\n' + hyperlinks

    return text.strip()


def _extract_pdf_with_pypdf2(file_data: bytes) -> str:
    text = ''
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data))
    if len(pdf_reader.pages) > MAX_PDF_PAGES:
        raise FileValidationError(f'PDF has {len(pdf_reader.pages)} pages; the maximum is {MAX_PDF_PAGES}.')
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + '\n'

    if not text.strip():
        raise TextExtractionError(
            'PyPDF2 extracted no text',
            user_message='No text could be extracted from the PDF.'
        )

    hyperlinks = _extract_pdf_hyperlinks(file_data)
    if hyperlinks:
        text = text.strip() + '\n' + hyperlinks

    return text.strip()


def extract_text_from_pdf(file_data: bytes) -> str:
    try: 
        result, used_fallback=with_fallback(
        _extract_pdf_with_pdfplumber, 
        _extract_pdf_with_pypdf2, 
        file_data, 
        log_fallback=True
    )
    
        if used_fallback:
            log_info('PDF EXTRACTION succeded using the PyPDF2 fallback', context='resume_parser')
        return result

    except FileValidationError:
        raise

    except Exception as e:
        log_error(e, context='extract_text_from_pdf')
        raise FileParsingError(
            'Failed to extract text from PDF using both pdfplumber and PyPDF2. '
            'The PDF may be corrupted, password-protected, or contain only scanned images. '
            'Please ensure it contains selectable text.'
        ) from e
    

def extract_text_from_docx(file_data: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_data))
        text_parts = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_parts.append(cell.text)

        text = '\n'.join(text_parts)

        if not text.strip():
            raise FileParsingError(
                'No text could be extracted from the document. '
                'The document may be empty or corrupted.'
            )
        
        try:
            for rel in doc.part.rels.values():
                if 'hyperlink' in rel.reltype.lower():
                    url = rel._target
                    if isinstance(url, str) and url.startswith('http'):
                        text += '\n' + url
        except Exception:
            pass

        log_info(f'Extracted {len(text)} chars from DOCX', context='resume_parser')
        return text.strip()

    except FileParsingError:
        raise   # Re-raise unchanged — don't wrap in another FileParsingError

    except Exception as e:
        log_error(e, context='extract_text_from_docx')
        raise FileParsingError(
            'Failed to extract text from DOCX. '
            'The document may be corrupted or in an unsupported format. '
            'Please try re-saving or converting to PDF.'
        ) from e

def extract_text_from_doc(file_data: bytes) -> str:
    raise FileParsingError(
        'Legacy .doc format is not supported. '
        'Please convert your document to .docx or .pdf and try again. '
        'You can convert using Microsoft Word, Google Docs, or online tools.'
    )

def extract_text(file_data:bytes, file_type:str)->str:
    if file_type=='pdf':
        return extract_text_from_pdf(file_data)
    elif file_type=='docx':
        return extract_text_from_docx(file_data)
    elif file_type=='doc':
        return extract_text_from_doc(file_data)
    else:
        raise FileValidationError(
            f'invalid file type: {file_type}. supported types are: pdf, docx and doc'


        )
    
def parse_resume_file(file_data: bytes, filename:str)->Tuple[str, dict]:
    log_info('parsing uploaded resume', context='parse_resume_file')

    # phase 1: validate file (size, emptiness, real content type)
    is_valid, error_msg, file_type = validate_file(file_data, filename)
    if not is_valid:
        log_warning('validation failed for uploaded file', context='parse_resume_file')
        raise FileValidationError(error_msg)

    #phase02: extraction of file

    try:
        text = extract_text(file_data, file_type)
        log_info(f'Extracted {len(text)} chars', context='parse_resume_file')

    except (FileParsingError, FileValidationError):
        raise   # Re-raise unchanged

    except Exception as e:
        log_error(e, context='parse_resume_file_extraction')
        raise FileParsingError(
            'An unexpected error occurred while processing the file. '
            'Please try again or contact support if the problem persists.'
        ) from e

    metadata = {
        'filename':        filename,
        'file_type':       file_type,
        'file_size_bytes': len(file_data),
        'text_length':     len(text),
        'success':         True,
    }
    return text, metadata
