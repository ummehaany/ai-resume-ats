"""PDF export security and functionality (audit item 9)."""
import http.server
import threading

import pytest

from backend.services import pdf_export
from backend.services.report_generator import generate_html_reports

EVIL = '<img src="http://127.0.0.1:1/ssrf.png"><script>alert(1)</script>'


def evil_analysis():
    return {
        'ATS_score': 55, 'interpretation': EVIL,
        'component_scores': {'formatting': 10, 'keywords': 10, 'content': 10, 'skill_validation': 5, 'ats_compatibility': 10},
        'strengths': [EVIL],
        'detailed_feedback': [{
            'issue_title': EVIL, 'severity_level': 'High', 'ats_impact': EVIL, 'explanation': EVIL,
            'where_it_appears': EVIL, 'how_to_fix': EVIL, 'action_items': [EVIL], 'example_improvement': EVIL,
        }, {'issue_title': 'mod', 'severity_level': 'Moderate', 'explanation': 'e', 'how_to_fix': 'f', 'action_items': []}],
        'skill_validation_details': {'validated': [{'skill': EVIL, 'projects': [EVIL]}], 'unvalidated': [EVIL], 'total': 2, 'validated_count': 1, 'validation_pct': 50},
        'jd_match_analysis': {'match_percentage': 50, 'semantic_similarity': 0.5, 'matched_keywords': [EVIL], 'missing_keywords': [EVIL], 'skills_gap': [EVIL]},
    }


def test_all_user_controlled_values_are_html_escaped():
    docs = generate_html_reports(evil_analysis())
    assert docs, 'no documents generated'
    for name, html in docs.items():
        assert '<img src="http://127.0.0.1' not in html, f'{name}: unescaped <img>'
        assert '<script>alert' not in html, f'{name}: unescaped <script>'
    assert '&lt;img' in docs['summary']


def test_malformed_history_data_does_not_crash_report_generation():
    docs = generate_html_reports({'ats_score': None, 'component_scores': {'formatting': 'abc'}, 'strengths': 'oops', 'detailed_feedback': ['junk', None]})
    assert set(docs) == {'summary', 'skill_report', 'jd_report', 'recommendations'}


def test_moderate_and_high_issues_land_in_the_right_report_sections():
    docs = generate_html_reports(evil_analysis())
    assert 'mod' in docs['recommendations']


def test_renderer_refuses_every_external_url():
    with pytest.raises(ValueError):
        pdf_export._block_all_fetches('http://127.0.0.1/x.png')
    with pytest.raises(ValueError):
        pdf_export._block_all_fetches('file:///etc/passwd')


def test_pdf_render_never_contacts_a_server_even_with_unescaped_markup():
    """Defence in depth: feed RAW malicious HTML straight to the PDF renderer (bypassing escaping)
    and verify a real local HTTP server receives zero requests."""
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'x')
        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        html = (f'<html><head><link rel="stylesheet" href="http://127.0.0.1:{port}/a.css"></head><body>'
                f'<img src="http://127.0.0.1:{port}/b.png"><img src="file:///etc/hostname">'
                f'<div style="background:url(http://127.0.0.1:{port}/c.png)">hi</div></body></html>')
        pdf = pdf_export.generate_combined_pdf({'x': html})
    finally:
        server.shutdown()
    assert pdf.startswith(b'%PDF')
    assert hits == [], f'renderer fetched external resources: {hits}'


def test_control_experiment_default_weasyprint_fetcher_does_leak():
    """Proves the test above is meaningful: with WeasyPrint's DEFAULT fetcher the same markup does hit the server."""
    from weasyprint import HTML
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(404)
            self.end_headers()
        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        HTML(string=f'<img src="http://127.0.0.1:{server.server_address[1]}/leak.png">').render()
    finally:
        server.shutdown()
    assert hits, 'expected the default fetcher to contact the local server'


def test_legitimate_report_renders_to_a_multi_page_pdf():
    analysis = evil_analysis()
    analysis['interpretation'] = 'Good! Your resume is ATS-friendly.'
    pdf = pdf_export.generate_combined_pdf(generate_html_reports(analysis))
    assert pdf.startswith(b'%PDF') and len(pdf) > 5000
    from PyPDF2 import PdfReader
    import io
    assert len(PdfReader(io.BytesIO(pdf)).pages) >= 4    # 4 reports merged
