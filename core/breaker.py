"""A file-backed circuit breaker, for a dependency that goes away for MINUTES.

The concrete case that motivated it: the cross-encoder runs on a shared GPU. When
another process saturates the card, the outage lasts minutes — and without a breaker
EVERY invocation pays the full timeout to rediscover what the previous one already
knew. On a path fired at every user interaction, that is a direct tax on response
time.

The state is a FILE, not process memory, for two reasons: each hook invocation is a
fresh process (memory does not survive), and the dependency is shared between
sessions (there is only one GPU, so what one session found out holds for the others).

`path=None` IS A VALID CALL, not a caller error. `session_state.load`/`save` already
treat "no writable state directory" as "state unavailable this round, not a failure" —
the same convention lives here so every caller inherits it, rather than each one having
to remember to guard around this class before constructing it. This is not hypothetical:
a caller whose own state directory turned out to be uncreatable once passed that `None`
straight through to `Path(None)`, which raised before the search it was guarding ever
ran — the breaker itself is a convenience, exactly like the state it shares a directory
with, and losing it must never cost the search.
"""
import time
from pathlib import Path


class Breaker:
    def __init__(self, path: str | Path | None, cooldown_seconds: float = 300.0):
        self.path = Path(path) if path is not None else None
        self.cooldown = cooldown_seconds

    def is_open(self) -> float | None:
        """Seconds since the last failure, if we are still inside the cooldown.

        None means "go ahead and try". Any read problem resolves to None on purpose: a
        broken breaker must not become the reason the functionality stops. A caller with
        nowhere to persist this (`path=None`) is the same case, not a special one.
        """
        if self.cooldown <= 0 or self.path is None:
            return None
        try:
            last_ts = float(self.path.read_text().strip())
        except Exception:
            return None
        idle = time.time() - last_ts
        if idle < self.cooldown:
            return idle

        return None

    def arm(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(time.time()))
        except Exception:
            pass

    def clear(self) -> None:
        if self.path is None:
            return
        try:
            self.path.unlink()
        except Exception:
            pass
