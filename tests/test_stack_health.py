"""Readiness of the three endpoints: who gets there, and when it never does.

The loop under test (`wait_ready`) is timed with an injected clock, so the test
decides how long the machine takes; `status` is injected too, answering with the
shapes the real endpoints give, each one sourced:

- llama-server `/health` answers 503 while the model loads and 200 when ready —
  the spec's readiness rule (2026-10-05, "Prontidão"), which also says the 503
  is the expected state, not an error;
- Qdrant `/readyz` answers 200 when ready;
- a port nothing listens on is a connection refusal, and `http_status` must
  report that as None (the standard outcome of a failed loopback connect in
  urllib), never as an exception.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError  # noqa: E402
from stack import health  # noqa: E402


class FakeClock:
    """A monotonic clock that advances only by `sleep`'s arguments, never by wall
    time. The real one is `time.monotonic`: a monotonically increasing float
    second counter with no system call inside the loop, so recording it here costs
    the test nothing the machine cannot spend anyway."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds: float):
        self.now += seconds


class ScriptedStatus:
    """Answers, per call, from the list `answers` (the last one repeats), and
    records each call as `(url, timeout)`.

    The signature mirrors `stack.health.http_status` and `wait_ready`'s own
    default (`status(url, *, timeout=5.0)`), so a fake with a wrong shape is a
    loud failure, like the other fakes in this repo.
    """

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, *, timeout: float = 5.0):
        self.calls.append((url, timeout))
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


class TestWaitReady(unittest.TestCase):
    def test_ready_dict_says_who_came_up_even_when_the_wait_raises(self):
        # qdrant answers 200 at once, embed never does: the wait raises naming
        # embed, and the caller's `ready` dict still says qdrant came up (at
        # clock 0), so the caller shows the log tail of embed only.
        clock = FakeClock()
        urls = {"qdrant": "http://127.0.0.1:6333/readyz",
                "embed": "http://127.0.0.1:8003/health"}

        def status(url, *, timeout=5.0):
            return 200 if url.endswith("/readyz") else 503
        ready = {}
        with self.assertRaises(StackError):
            health.wait_ready(urls, status=status, clock=clock, sleep=clock.sleep,
                              timeout=3.0, interval=1.0, ready=ready)
        self.assertEqual(ready, {"qdrant": 0.0})

    def test_503_then_200_is_ready(self):
        url = "http://127.0.0.1:8003/health"
        clock = FakeClock()
        status = ScriptedStatus([503, 200])
        result = health.wait_ready({"embed": url},
                                   status=status, clock=clock, sleep=clock.sleep,
                                   timeout=10.0, interval=1.0)
        # Ready on the second poll, after one interval of the injected clock.
        self.assertEqual(result, {"embed": 1.0})
        self.assertEqual([call[0] for call in status.calls], [url, url])
        self.assertEqual(len(status.calls), 2)

    def test_never_ready_raises_naming_the_service_and_its_last_answer(self):
        clock = FakeClock()
        status = ScriptedStatus([503])
        with self.assertRaises(StackError) as ctx:
            health.wait_ready({"embed": "http://127.0.0.1:8003/health"},
                              status=status, clock=clock, sleep=clock.sleep,
                              timeout=10.0, interval=2.0)
        message = str(ctx.exception)
        self.assertIn("embed", message)
        self.assertIn("503", message)
        self.assertEqual(ctx.exception.step, "health")
        # The loop polled at t=0,2,4,6,8 and then spent the timeout: five answers,
        # ten clock-seconds, not an early give-up and not an extra poll at t=10.
        self.assertEqual(len(status.calls), 5)
        self.assertEqual(clock.now, 10.0)

    def test_no_answer_is_none_not_an_exception(self):
        # A closed port mid-startup answers None; the loop treats it as "not yet"
        # and goes on, until the service answers 200.
        clock = FakeClock()
        status = ScriptedStatus([None, 200])
        result = health.wait_ready({"qdrant": "http://127.0.0.1:6333/readyz"},
                                   status=status, clock=clock, sleep=clock.sleep,
                                   timeout=10.0, interval=1.0)
        self.assertEqual(result, {"qdrant": 1.0})
        self.assertEqual(len(status.calls), 2)


if __name__ == "__main__":
    unittest.main()
