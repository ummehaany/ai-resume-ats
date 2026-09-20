import logging

try:
    from weasyprint import HTML
    WEASYPRINT_INSTALLED = True
except (ImportError, OSError):   # OSError: the system libraries (Pango/Cairo) are missing
    WEASYPRINT_INSTALLED = False

logger = logging.getLogger('ats_resume_scorer')

_BLOCKED_MESSAGE = 'Loading external resources is disabled during PDF generation'


def _block_all_fetches(url, *args, **kwargs):
    """SECURITY: never let the PDF renderer fetch anything (WeasyPrint < 68 callable style).

    The report templates are fully self-contained (inline CSS, no images or web fonts). Refusing every
    URL makes it impossible for report content to trigger requests to internal services (SSRF) or to
    read local files through file:// URLs, even if some markup slipped past HTML escaping.
    """
    raise ValueError(_BLOCKED_MESSAGE)


def _make_url_fetcher():
    """Return a url_fetcher that refuses everything, for whichever WeasyPrint API is installed."""
    try:
        from weasyprint.urls import URLFetcher      # WeasyPrint >= 68: fetchers are classes
    except ImportError:
        return _block_all_fetches                   # older releases accept a plain callable

    class BlockingFetcher(URLFetcher):
        def fetch(self, url, headers=None):
            raise ValueError(_BLOCKED_MESSAGE)

    return BlockingFetcher(allowed_protocols=set())


def generate_combined_pdf(html_docs: dict[str, str]) -> bytes:
    if not WEASYPRINT_INSTALLED:
        raise ImportError('WeasyPrint (or its Pango/Cairo system libraries) is not available. PDF generation is unavailable.')

    fetcher = _make_url_fetcher()
    documents = [
        HTML(string=html_str, url_fetcher=fetcher).render()
        for html_str in html_docs.values()
    ]
    if not documents:
        raise ValueError('No report documents to render')

    # Merge all rendered documents into the first one.
    first_doc = documents[0]
    for other_doc in documents[1:]:
        first_doc.pages.extend(other_doc.pages)

    return first_doc.write_pdf()
