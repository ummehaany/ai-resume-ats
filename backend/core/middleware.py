"""Request-size limits enforced *before* the body is read into memory."""
import json
from typing import Dict


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Pure-ASGI middleware: rejects oversized POST bodies with HTTP 413.

    ``limits`` maps an exact request path to the maximum body size in bytes. The
    Content-Length header is checked up front; for chunked uploads the bytes are
    counted as they stream in.
    """

    def __init__(self, app, limits: Dict[str, int]):
        self.app = app
        self.limits = limits

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)

        limit = self.limits.get(scope.get('path'))
        if limit is None:
            return await self.app(scope, receive, send)

        headers = {k.lower(): v for k, v in scope.get('headers', [])}
        raw_length = headers.get(b'content-length')
        if raw_length is not None:
            try:
                declared = int(raw_length)
            except ValueError:
                return await self._reject(send, 400, 'Invalid Content-Length header.')
            if declared > limit:
                return await self._reject(send, 413, 'Request body too large.')

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > limit:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message['type'] == 'http.response.start':
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send, 413, 'Request body too large.')

    @staticmethod
    async def _reject(send, status: int, detail: str):
        body = json.dumps({'detail': detail}).encode()
        await send({
            'type': 'http.response.start',
            'status': status,
            'headers': [(b'content-type', b'application/json'), (b'content-length', str(len(body)).encode())],
        })
        await send({'type': 'http.response.body', 'body': body})
