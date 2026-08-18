"""What an endpoint said its model's context window was, remembered between processes.

WHY A CACHE AT ALL. The guard needs the window to DECIDE, so it cannot be the thing that
fetches it: that would be circular, and it would put a network call on the path that runs
before every single file read. The hooks that already pay for network fill this; the guard
only ever reads it.

WHY THE KEY IS (ENDPOINT, MODEL). One model name means different windows on different
servers — the same `qwen3.8-27b` is 262,144 on one host and 524,288 on another, measured.
Keying by the name alone lets one deployment answer for another, silently.

WHY A STALE ENTRY IS STILL RETURNED. Falling back to the ceiling table because the value is
a day old makes the guard sleep on a window it already knows. Stale beats absent; the
freshness flag exists so the caller can decide to refresh, not so it can discard.
"""
import json
import os
import time

from .knobs import state_dir

FILENAME = "model-windows.json"

#: A day. The window of a model does not change often, and the cost of being a day late is
#: one refresh; the cost of refreshing constantly is a network call nobody asked for.
TTL_SECONDS = 86400.0


def _path() -> str:
    root = state_dir()
    os.makedirs(root, exist_ok=True)

    return os.path.join(root, FILENAME)


def _key(endpoint: str, model: str) -> str:
    return f"{(endpoint or '').strip()}|{(model or '').strip()}"


def _load() -> dict:
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}

    return data if isinstance(data, dict) else {}


def get(endpoint: str, model: str) -> tuple[int, bool]:
    """`(window, is_fresh)`. `(0, False)` when nothing is known.

    A stale entry comes back with its value and `False` — see the module docstring.
    """
    row = _load().get(_key(endpoint, model))
    if not isinstance(row, dict):
        return 0, False
    try:
        window = int(row.get("window") or 0)
        expires = float(row.get("expires_at") or 0)
    except (TypeError, ValueError):
        return 0, False
    if window <= 0:
        return 0, False

    return window, time.time() < expires


def age_seconds(endpoint: str, model: str) -> float | None:
    """How long ago this entry was written, or `None` when nothing is cached for it.

    Exists for the one diagnostic that has to tell "no window known" apart from "a window
    WAS learned, N hours ago" — without it, both print identically and the user has no way
    to see a broken probe. Derived as `expires_at - TTL_SECONDS`, which assumes the entry
    was written with the default TTL: every real writer (`hosts.hermes.endpoint.
    refresh_window`) does. A caller that wrote with a different `ttl` only skews the
    reported age; `get`'s own freshness bit is unaffected either way.
    """
    row = _load().get(_key(endpoint, model))
    if not isinstance(row, dict):
        return None
    try:
        window = int(row.get("window") or 0)
        expires = float(row.get("expires_at") or 0)
    except (TypeError, ValueError):
        return None
    if window <= 0 or expires <= 0:
        return None

    return max(0.0, time.time() - (expires - TTL_SECONDS))


def put(endpoint: str, model: str, window: int, ttl: float = TTL_SECONDS) -> None:
    """Records a window. A value of zero or less is NOT stored.

    Storing zero would make this cache assert "that endpoint says it does not know", and the
    cascade would stop here instead of falling through to the ceiling table. Absence and
    "answered zero" have to stay different.
    """
    try:
        window = int(window)
    except (TypeError, ValueError):
        return
    if window <= 0:
        return
    data = _load()
    data[_key(endpoint, model)] = {"window": window, "expires_at": time.time() + ttl}
    try:
        # `os.getpid()` in the temp name: two hermes sessions refreshing at once both open
        # a FIXED `.tmp` name, and after the first `os.replace` the loser's still-open
        # descriptor writes into the inode the first one just published. `_load` catches
        # `ValueError` and readers never see a torn file — `os.replace` is atomic — so this
        # already self-heals to a cache miss rather than a corruption anyone observes. Naming
        # the temp file per-process removes the race outright instead of just tolerating it.
        # `core/bindings.py` writes its own state through the same fixed-name pattern; it is
        # not touched here.
        tmp = f"{_path()}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, _path())
    except OSError:
        # A cache that cannot be written is a cache miss, never an error the caller sees.
        return
