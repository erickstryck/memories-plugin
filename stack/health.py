"""Readiness of the three endpoints: whether they are up, and when they get there.

This module owns the question "is the stack UP, and if not, when does it get
there": `http_status` asks one endpoint once, `endpoints` names the health path
of each service and `wait_ready` polls until every target answers 200 or the
deadline passes. It never decides whether the stack is CORRECT — that is
`stack.verify` — and it never touches a runtime: the targets arrive as plain
URLs, which is what lets the installer wait on a freshly started stack and the
`stack status` command wait on a restored one with the same code.

The two llama-servers answer 503 while they load their model and 200 when it
is ready (spec, "Prontidão"), so a 503 in the middle of a start is the expected
state of a healthy stack, and the loop keeps going on it. Qdrant answers 200
on `/readyz`. A port nothing listens on is a connection refusal, which
`http_status` reports as None — the loop treats None like any other non-200,
because a container that has not started listening yet looks exactly like that.

The clock and the status probe are INJECTED the same way the runner is in the
other `stack` modules, so the loop is tested with a script of 503s and 200s
and a clock that advances by hand, and the deadline of ten minutes (the
spec's, for the readiness step) is a constant, not a mystery.
"""
import time
import urllib.error
import urllib.request
from typing import Callable, Mapping

from . import StackError

#: The spec's deadline for the readiness step (its "Verificação e calibração"
#: names 10 minutes): the upper bound the loop will wait before it names the
#: service that never answered.
READY_TIMEOUT_S = 600.0


def http_status(url: str, *, timeout: float = 5.0) -> int | None:
    """The HTTP status of `url`, or None when the connection fails.

    A connection refusal (the container is not listening yet) and a timeout
    are both "not ready", not an error: the caller is a loop that polls, and
    raising here would make a slow start look like a broken one. Only a
    response that could not be read at all is reported as None through the
    same door, so the loop has one shape for "not yet".
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        # A 503 while the model loads is a RESPONSE, and its code is the
        # answer: the loop decides what to do with it.
        return exc.code
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        # A refused port, a timeout, a dead socket: None, and the loop polls
        # again. `URLError` already wraps `ConnectionRefusedError` and the
        # `socket.timeout` behind a slow one, and the `OSError` catch keeps a
        # platform quirk from turning "not yet" into "broken".
        return None


def endpoints(ports: Mapping[str, int]) -> dict[str, str]:
    """The readiness URL of each service, keyed by the catalogue's names.

    Qdrant exposes `/readyz`; the two llama-servers expose `/health` and
    answer 503 until the model is loaded (spec, "Prontidão"). The ports are
    the HOST ports the installer chose, so the keys are `qdrant`, `embed`
    and `rerank`, and a busy port that moved is still named the same.
    """
    return {
        "qdrant": f"http://127.0.0.1:{ports['qdrant']}/readyz",
        "embed": f"http://127.0.0.1:{ports['embed']}/health",
        "rerank": f"http://127.0.0.1:{ports['rerank']}/health",
    }


def wait_ready(targets: Mapping[str, str], *,
               status: Callable[[str], int | None] = http_status,
               clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep,
               timeout: float = READY_TIMEOUT_S,
               interval: float = 1.0) -> dict[str, float]:
    """Poll every target until it answers 200, or raise naming the one that did not.

    The answer is, per target, the clock time it first answered 200 — the
    installer reports it, and a slow load is a fact, not a guess. The loop
    polls in lockstep: one status call per target per round, so a fast
    service does not starve a slow one, and a round that found everything
    ready stops before the next sleep.

    When the deadline passes, the error names the service and its LAST
    answer: a 503 says the model is still loading, a None says nothing is
    listening, and the fix names the command that shows the container's
    output, which is where the real reason is.
    """
    names = list(targets)
    pending = dict(targets)
    ready_at: dict[str, float] = {}
    last_answer: dict[str, int | None] = {}
    started = clock()
    while pending:
        # The deadline is checked BEFORE the poll: a poll that starts past it
        # would spend budget the caller did not give, and the raise below must
        # mean "I waited the whole `timeout` and it never answered 200".
        if clock() - started >= timeout:
            break
        now = clock()
        for name in list(pending):
            answer = status(pending[name])
            last_answer[name] = answer
            if answer == 200:
                ready_at[name] = now
                del pending[name]
        if not pending:
            return ready_at
        sleep(interval)
    for name in names:
        if name in pending:
            answer = last_answer.get(name)
            fixed = "503" if answer == 503 else ("no answer" if answer is None else str(answer))
            raise StackError(
                f"'{name}' was not ready after {timeout:.0f} s (last answer: {fixed})",
                step="health",
                fix="qctx stack status, or the runtime's logs for that container")
    return ready_at
