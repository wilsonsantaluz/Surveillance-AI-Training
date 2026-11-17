"""ASGI request-body limit applied before FastAPI parses multipart uploads."""

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    """Reject oversized requests without buffering the whole multipart body."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/predict":
            await self.app(scope, receive, send)
            return
        content_length = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self._reject(send, 400, "Invalid Content-Length")
                return
            if declared_length < 0:
                await self._reject(send, 400, "Invalid Content-Length")
                return
            if declared_length > self.max_body_bytes:
                await self._reject(send, 413, "Request body exceeds the upload limit")
                return

        received_bytes = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal received_bytes, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def limited_send(message: Message) -> None:
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded:
            await self._reject(send, 413, "Request body exceeds the upload limit")

    @staticmethod
    async def _reject(send: Send, status_code: int, detail: str) -> None:
        body = f'{{"detail":"{detail}"}}'.encode()
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
