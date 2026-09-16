"""Publishing a state file: what `write_json` must never report as success.

WHY THIS FILE EXISTS. `core/statefile.py` became the SINGLE OWNER of the atomic publish for
six callers, and its docstring says the boolean it returns is what lets `jobs.enqueue` raise
when "losing the write means work that never happens". A return value carrying that much
weight has to be honest about every way the write can fail — not only the ways that raise.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import statefile  # noqa: E402


#: Forces a REAL short write, with no mocking of the code under test.
#:
#: `RLIMIT_FSIZE` makes the kernel stop the write at the limit and raise SIGXFSZ; with that
#: signal ignored, `os.write` simply RETURNS A SMALLER COUNT — which is the documented POSIX
#: behaviour this file is about, and not an error any `except OSError` can see.
_SHORT_WRITE = """
import json, os, resource, signal, sys
sys.path.insert(0, {root!r})
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
from core import statefile
target = {target!r}
payload = {{"pad": "y" * 5000}}
resource.setrlimit(resource.RLIMIT_FSIZE, (2048, resource.RLIM_INFINITY))
try:
    landed = statefile.write_json(target, payload)
finally:
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
print(json.dumps({{"landed": landed, "exists": os.path.exists(target)}}))
"""


def _run_short_write(target: str) -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-c", _SHORT_WRITE.format(root=root, target=target)],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr

    return json.loads(out.stdout.strip().splitlines()[-1])


class TestAWriteThatDidNotLandIsNotReportedAsLanded(unittest.TestCase):
    """A partial write is a FAILED write, and the return value has to say so.

    `os.write` is allowed to write fewer bytes than it was given, and doing so is NOT an
    error: it returns the count and raises nothing. So a `try/except OSError` around it sees
    a perfectly successful call while the file on disk holds half a JSON document, and
    `os.replace` then publishes that half.

    MEASURED before the fix, with the limit at 2048 bytes and a 5012-byte payload:
    `write_json` returned True, 2048 bytes reached the disk, and reading them back raised
    `Unterminated string starting at: line 2 column 7`. Every caller was told the state
    landed — including `jobs.enqueue`, whose docstring promises to raise precisely so that
    "work that never happens" cannot be reported as queued.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_a_truncated_write_is_reported_as_a_FAILURE(self):
        target = os.path.join(self.dir, "state.json")

        self.assertFalse(_run_short_write(target)["landed"],
                         "a write that landed only partially was reported as success")

    def test_a_truncated_write_leaves_NO_UNREADABLE_FILE_behind(self):
        """The half-written bytes must never be published under the real name.

        Returning False and still replacing the target would be the worse half of the bug:
        the caller knows it failed, and every READER afterwards finds a file that exists and
        does not parse.
        """
        target = os.path.join(self.dir, "state.json")
        _run_short_write(target)
        if not os.path.exists(target):
            return                              # nothing published at all: the safe outcome

        with open(target, encoding="utf-8") as fh:
            raw = fh.read()
        try:
            json.loads(raw)
        except ValueError as exc:
            self.fail(f"published a file that does not parse: {exc}")


class TestThePublishedModeDoesNotDependOnWhatWasLyingAround(unittest.TestCase):
    """The mode is a property of what we publish, not of what a crash left behind.

    `os.replace` carries the TEMPORARY's mode onto the target, and the temporary's name is
    deterministic (`<name>.<pid>.tmp`). `os.open(..., O_CREAT | O_TRUNC, 0o600)` does NOT
    re-apply the mode to a file that already exists, so a temporary left behind by an earlier
    crash of the same pid is adopted with ITS permissions and those are what get published.

    MEASURED before the fix: a stale temporary at 0o666 published the daemon claim at 0o666,
    while the module's own comment promises "Creating the temporary with 0o600 fixes every
    caller at once".
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_a_stale_temporary_does_not_donate_its_mode_to_the_published_file(self):
        target = Path(self.dir) / "state.json"
        stale = target.with_suffix(f"{target.suffix}.{os.getpid()}.tmp")
        stale.touch()
        os.chmod(stale, 0o666)

        self.assertTrue(statefile.write_json(target, {"pid": 1}))
        self.assertEqual(stat.S_IMODE(os.stat(target).st_mode), 0o600,
                         "the published file inherited the stale temporary's mode")


class TestAWriteThatStopsMakingProgressIsAFailure(unittest.TestCase):
    """`os.write` returning 0 must end the loop, not spin in it.

    A descriptor that accepts nothing, forever, is not a condition `write_json` can recover
    from — and the loop that guarantees the full write is exactly where "keep trying" becomes
    "hang the caller". The daemon calls this from a host process's `initialize()`, so a hang
    here is a host that never starts.

    Driven through a REAL descriptor that reports 0: a pipe whose read end is closed would
    raise instead, so this uses the one honest way to observe the branch — a fake `os.write`
    substituted for the duration, asserting the loop's own contract rather than the kernel's.
    """

    def test_a_write_that_consumes_nothing_is_reported_as_a_failure(self):
        calls = []

        def stalled(fd, data):
            calls.append(len(data))
            if len(calls) > 50:
                self.fail("the loop kept writing against a descriptor taking nothing")

            return 0

        d = tempfile.mkdtemp()
        target = os.path.join(d, "state.json")
        with unittest.mock.patch.object(os, "write", stalled):
            landed = statefile.write_json(target, {"pad": "x" * 100})

        self.assertFalse(landed, "a write that never progressed was reported as landed")
        self.assertFalse(os.path.exists(target), "it published a file it never wrote")


if __name__ == "__main__":
    unittest.main()
