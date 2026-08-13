"""A file-backed circuit breaker, for a dependency that goes away for MINUTES.

The concrete case that motivated it: the cross-encoder runs on a shared GPU. When
another process saturates the card, the outage lasts minutes — and without a breaker
EVERY invocation pays the full timeout to rediscover what the previous one already
knew. On a path fired at every user interaction, that is a direct tax on response
time.

The state is a FILE, not process memory, for two reasons: each hook invocation is a
fresh process (memory does not survive), and the dependency is shared between
sessions (there is only one GPU, so what one session found out holds for the others).
"""
import time
from pathlib import Path


class Breaker:
    def __init__(self, path: str | Path, cooldown_seconds: float = 300.0):
        self.path = Path(path)
        self.cooldown = cooldown_seconds

    def is_open(self) -> float | None:
        """Seconds since the last failure, if we are still inside the cooldown.

        None means "go ahead and try". Any read problem resolves to None on purpose: a
        broken breaker must not become the reason the functionality stops.
        """
        if self.cooldown <= 0:
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
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(time.time()))
        except Exception:
            pass

    def clear(self) -> None:
        try:
            self.path.unlink()
        except Exception:
            pass
