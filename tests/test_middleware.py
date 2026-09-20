"""BodySizeLimitMiddleware: Content-Length and chunked (no Content-Length) uploads. MOCKED (pure ASGI, no network)."""
import asyncio

from backend.core.middleware import BodySizeLimitMiddleware


async def _echo_app(scope, receive, send):
    total = 0
    while True:
        msg = await receive()
        total += len(msg.get('body', b''))
        if not msg.get('more_body'):
            break
    await send({'type': 'http.response.start', 'status': 200, 'headers': []})
    await send({'type': 'http.response.body', 'body': str(total).encode()})


def _run(headers, chunks, path='/analyze-resume', method='POST', limit=100):
    app = BodySizeLimitMiddleware(_echo_app, {'/analyze-resume': limit})
    sent = []
    queue = list(chunks)

    async def receive():
        body = queue.pop(0) if queue else b''
        return {'type': 'http.request', 'body': body, 'more_body': bool(queue)}

    async def send(message):
        sent.append(message)

    scope = {'type': 'http', 'method': method, 'path': path, 'headers': headers}
    asyncio.run(app(scope, receive, send))
    status = next(m['status'] for m in sent if m['type'] == 'http.response.start')
    return status


def test_declared_too_large_rejected_without_reading():
    assert _run([(b'content-length', b'101')], [b'x' * 101]) == 413


def test_chunked_upload_over_limit_rejected():
    # No Content-Length header: bytes are counted as they stream in.
    assert _run([], [b'x' * 60, b'x' * 60]) == 413


def test_chunked_upload_within_limit_passes():
    assert _run([], [b'x' * 40, b'x' * 40]) == 200


def test_invalid_content_length_is_400():
    assert _run([(b'content-length', b'abc')], [b'x']) == 400


def test_other_paths_and_methods_not_limited():
    assert _run([], [b'x' * 500], path='/health') == 200
    assert _run([], [b'x' * 500], method='GET') == 200
