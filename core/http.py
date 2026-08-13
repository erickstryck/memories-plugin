"""JSON over HTTP, in a single place.

Each adapter used to have its own: the vector store client and the model clients
both repeated request building, error handling and decoding. Duplication like that
is where a fix lands in half the places — which is exactly what happened with the
re-rank scale normalization, which existed in one consumer and not the other.

Stdlib on purpose: this code runs inside a hook fired on every user interaction, and
a missing dependency would turn an environment error into a silent loss of
functionality.
"""
import json
import urllib.error
import urllib.request

from .errors import CoreError


class HttpError(CoreError):
    """A transport failure or an error status. Carries the body, truncated.

    The body matters: without it, `HTTP 400` does not distinguish a missing model
    from a malformed payload from a collection with the wrong dimension, and
    diagnosis turns into trial and error.
    """

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def request_json(url: str, *, method: str = "GET", body=None, headers: dict | None = None,
                 timeout: float = 30.0):
    """Makes the request and decodes JSON. An empty body returns `{}`."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()

            return json.loads(raw.decode()) if raw else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:400]
        raise HttpError(f"HTTP {exc.code} on {method} {url}: {body}",
                        status=exc.code, body=body) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"could not reach {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise HttpError(f"{url} answered with something that is not JSON: {exc}") from exc
    except OSError as exc:
        # Socket `TimeoutError` and the rest of the OSError family. These used to
        # escape raw, and a consumer catching only the domain types died with a
        # traceback instead of degrading.
        raise HttpError(f"network failure on {url}: {type(exc).__name__}: {exc}") from exc


def post_json(url: str, body, *, headers: dict | None = None, timeout: float = 30.0):
    return request_json(url, method="POST", body=body, headers=headers, timeout=timeout)


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}
