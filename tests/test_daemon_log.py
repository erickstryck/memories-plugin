"""What the daemon leaves behind to be read, and the version it runs.

The daemon was started with stdout and stderr on /dev/null and its watcher swallowed every
exception with `pass`, so a daemon failing on every cycle looked exactly like one with nothing
to do. It now keeps `daemon.log`: when it started and why it stopped, each job with its
duration and outcome, what the watcher queued and when it re-read an archive, and the
watcher's failures (once per distinct failure, not once per cycle).

It also records the version it runs in `daemon.json`, because both hosts start it and they are
not always on the same version (measured on 2026-09-25: the claude-code install was on a commit
from 8 September while hermes ran 1.0.1), and nothing said which code the daemon was.

NO TEST HERE STARTS A REAL DAEMON: `run` takes its worker and cycle count, `start` its spawn.
"""
import io
import unittest.mock
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import daemon, eventlog, indexer, jobs, lease  # noqa: E402
from core.version import __version__  # noqa: E402
from tests.test_watch import FakeIndex, a_file_on_disk  # noqa: E402


def a_state_dir() -> Path:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return Path(d)


def log_lines() -> list:
    target = eventlog.path(eventlog.DAEMON)

    return target.read_text().splitlines() if target.exists() else []


class TestTheLoopLeavesARecord(unittest.TestCase):
    def setUp(self):
        a_state_dir()
        lease.write("s1", "claude", pid=os.getpid())

    def test_start_and_stop_are_recorded_with_the_version(self):
        daemon.run(lambda job: None, cycles=1, sleep=lambda s: None)
        lines = log_lines()
        self.assertTrue(any(f"version={__version__}" in ln and "start" in ln for ln in lines),
                        lines)
        self.assertIn("stop reason='cycles exhausted'", lines[-1])

    def test_a_loop_ended_by_an_exception_still_records_why(self):
        """`repos daemon stop` sends SIGTERM, which `__main__` turns into SystemExit. Logged
        only on a normal return, the commonest stop left just the start line."""
        def stopped(job):
            raise SystemExit("SIGTERM")

        jobs.enqueue("alpha", "refresh", [])
        with self.assertRaises(SystemExit):
            daemon.run(stopped, cycles=3, sleep=lambda s: None)
        self.assertIn("stop reason='SystemExit: SIGTERM'", log_lines()[-1])

    def test_no_live_lease_is_recorded_as_the_reason_it_stopped(self):
        for path in (eventlog.path("leases")).glob("*.json"):
            path.unlink()
        self.assertEqual(daemon.run(lambda job: None, cycles=1, sleep=lambda s: None),
                         "no live lease")
        self.assertIn("stop reason='no live lease'", log_lines()[-1])

    def test_a_job_that_succeeds_is_recorded_with_its_duration(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        daemon.run(lambda job: None, cycles=1, sleep=lambda s: None)
        [line] = [ln for ln in log_lines() if " job " in ln]
        self.assertRegex(line, r"job repo=alpha kind=index result=done in \d+\.\ds")

    def test_a_job_that_fails_is_recorded_with_its_error(self):
        jobs.enqueue("alpha", "refresh", [])

        def broken(job):
            raise RuntimeError("the archive went away")

        daemon.run(broken, cycles=1, sleep=lambda s: None)
        [line] = [ln for ln in log_lines() if " job " in ln]
        self.assertIn("result=failed", line)
        self.assertIn("RuntimeError: the archive went away", line)

    def test_a_cancelled_job_is_recorded_as_cancelled(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.request_cancel("alpha")
        daemon.run(lambda job: None, cycles=1, sleep=lambda s: None)
        [line] = [ln for ln in log_lines() if " job " in ln]
        self.assertIn("result=cancelled", line)

    def test_the_same_watcher_failure_is_recorded_once_not_every_cycle(self):
        """A failure that repeats every 5 s would fill the log with one fact in an hour."""
        def failing():
            raise ConnectionError("qdrant unreachable")

        daemon.run(lambda job: None, cycles=4, sleep=lambda s: None, watch=failing)
        errors = [ln for ln in log_lines() if "watch failed" in ln]
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("ConnectionError: qdrant unreachable", errors[0])

    def test_a_failure_whose_message_varies_is_still_recorded_once(self):
        """An HTTP error embeds the response body; keyed on the whole message, every cycle
        was a new failure (measured by review: 5 lines in 5 cycles)."""
        count = iter(range(100))

        def failing():
            raise ConnectionError(f"qdrant said {next(count)}")

        daemon.run(lambda job: None, cycles=5, sleep=lambda s: None, watch=failing)
        self.assertEqual(len([ln for ln in log_lines() if "watch failed" in ln]), 1)

    def test_a_watcher_that_recovers_says_so_once(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] <= 2:
                raise ConnectionError("qdrant unreachable")

        daemon.run(lambda job: None, cycles=4, sleep=lambda s: None, watch=flaky)
        self.assertEqual(sum("watch recovered" in ln for ln in log_lines()), 1)

    def test_a_different_watcher_failure_is_recorded_too(self):
        errors = iter([ConnectionError("down"), ValueError("bad page")])

        def failing():
            raise next(errors)

        daemon.run(lambda job: None, cycles=2, sleep=lambda s: None, watch=failing)
        self.assertEqual(sum("watch failed" in ln for ln in log_lines()), 2)

    def test_a_log_that_cannot_be_written_does_not_stop_the_loop(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        ran = []
        # Broken from INSIDE `eventlog`, not by replacing `write`: the daemon relies on
        # `write` keeping its promise, and this is the promise under test.
        with unittest.mock.patch("core.eventlog.path", side_effect=OSError("disk full")):
            daemon.run(lambda job: ran.append(job), cycles=1, sleep=lambda s: None)
        self.assertEqual(len(ran), 1)
        self.assertEqual(jobs.load("alpha")["state"], jobs.DONE)


class TestTheWatcherRecordsWhatItDoes(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_an_archive_read_is_recorded_with_its_size(self):
        watch = indexer.watcher(index=FakeIndex(indexed={"/nonexistent/alpha/a.py"}))
        watch()
        watch()
        reads = [ln for ln in log_lines() if "sources repo=alpha" in ln]
        self.assertEqual(len(reads), 1, "a quiet second cycle must not log a read it skipped")
        self.assertIn("files=1", reads[0])

    def test_a_queued_job_is_recorded(self):
        watch = indexer.watcher(index=FakeIndex(changed=[a_file_on_disk("x = 1\n")]))
        watch()
        watch()
        self.assertTrue(any("enqueue repo=alpha kind=refresh" in ln for ln in log_lines()),
                        log_lines())


class TestStatsReadsWhatTheRealDaemonWrites(unittest.TestCase):
    """`stats` parses lines the daemon and the watcher write as plain strings. Its own tests
    read a hand-typed fixture, so renaming an event (measured by review: `start` to
    `started`) kept every test green and made `stats` report no start and no version. This
    feeds the REAL writers' output to the REAL reader."""

    def test_a_real_run_is_read_back_with_its_start_version_and_jobs(self):
        from core import stats

        a_state_dir()
        lease.write("s1", "claude", pid=os.getpid())
        jobs.enqueue("alpha", "refresh", [])

        def failing():
            raise ConnectionError("down")

        daemon.run(lambda job: None, cycles=2, sleep=lambda s: None, watch=failing)
        d = stats.summarize(Path(os.environ["QCTX_STATE_DIR"]))["daemon"]
        self.assertEqual(d["version"], __version__, d)
        self.assertIsNotNone(d["last_start"], d)
        self.assertEqual(d["jobs"].get("done"), 1, d)
        self.assertEqual(d["watcher_errors"], 1, d)
        self.assertIn("ConnectionError: down", d["last_error"])


class TestTheDaemonRecordsItsVersion(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_start_writes_the_version_into_the_record(self):
        daemon.start(spawn=lambda argv: os.getpid(), argv=["x"])
        record = json.loads(daemon.path().read_text())
        self.assertEqual(record.get("version"), __version__)


class TestStatusShowsTheDaemonsVersion(unittest.TestCase):
    """`repos status` names the version the daemon runs, and warns when it is not this one."""

    def setUp(self):
        a_state_dir()
        from tests.test_cli_repos import load_cli
        self.cli = load_cli()

    def _status(self) -> str:
        from tests.test_cli_repos import Args
        out = io.StringIO()
        with redirect_stdout(out):
            self.cli.cmd_repos_status(Args(json=False), None)

        return out.getvalue()

    def _record(self, **extra):
        entry = {"pid": os.getpid(), "starttime": lease.process_start(os.getpid()) or "",
                 "started_at": 0.0, **extra}
        daemon.path().write_text(json.dumps(entry))

    def test_the_same_version_is_shown_with_no_warning(self):
        self._record(version=__version__)
        text = self._status()
        self.assertIn(f"version {__version__}", text)
        self.assertNotIn("restart", text)

    def test_a_different_version_is_shown_with_the_command_to_restart(self):
        self._record(version="0.9.0")
        text = self._status()
        self.assertIn("version 0.9.0", text)
        self.assertIn("qctx repos daemon stop && qctx repos daemon start", text)

    def test_a_record_from_before_versions_warns_too(self):
        """Every daemon running when this ships was started by code that wrote no version."""
        self._record()
        text = self._status()
        self.assertIn("version unknown", text)
        self.assertIn("qctx repos daemon stop && qctx repos daemon start", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
