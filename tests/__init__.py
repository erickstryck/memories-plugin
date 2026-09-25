"""Test package. Its one job at import time is to contain what the suite leaves behind.

WHY THIS FILE EXISTS. 85 call sites across this suite call `tempfile.mkdtemp()` and none of
them remove what they created — measured: two modules alone left 69 directories behind. That
is not a leak anyone notices on a laptop with a large disk, and this repo learned what it costs
the hard way: `/tmp` is a 16 GB tmpfs here, a run filled it to 96,080 files, and every shell in
the session died — `echo` included. The suite had made the machine unusable.

WHY IT IS FIXED HERE AND NOT AT THE 85 SITES. Editing 85 sites fixes the 85 that exist and
nothing about the 86th. Redirecting `tempfile.tempdir` puts every temporary this suite creates
inside one directory that is removed when the process ends, whoever wrote the test and whenever
they wrote it. Each test still owns its own directory; what it stops owning is the obligation to
remember cleanup, which is the part that was never once honoured.

`TMPDIR` is set as well as `tempfile.tempdir`: the first is inherited by the SUBPROCESSES these
tests spawn (`scripts/hermes_cutover.sh` and the hooks run as real processes), the second is a
Python-level variable they never see. Both are needed to catch both kinds of temporary.

WHAT THIS DOES NOT COVER, stated rather than discovered later: `atexit` does not run when the
process is killed with SIGKILL, so a hard kill still leaves one run directory behind. One
directory per hard kill, named and dated, is a different problem from tens of thousands of
anonymous ones — and `qctx-suite-*` is a glob anyone can clean by hand.
"""
import atexit
import os
import shutil
import tempfile

#: One directory per process, under whatever TMPDIR was in effect when the suite started, so an
#: operator who redirected it keeps their redirection.
_RUN_DIR = tempfile.mkdtemp(prefix=f"qctx-suite-{os.getpid()}-")

tempfile.tempdir = _RUN_DIR
os.environ["TMPDIR"] = _RUN_DIR

#: A DEFAULT STATE DIRECTORY for the whole suite, inside the run directory. The plugin's logs
#: (`recall.log`, `daemon.log`) are written to the state directory by both hosts and the
#: daemon, and a test that isolates a provider through its private attribute, or forgets to
#: pin the variable, would otherwise append to the developer's REAL log: measured, one class
#: added 15 lines per run, and `qctx stats` on that machine then reported rounds that only ever
#: happened inside the suite. Tests that set their own directory still win, since this is only
#: a default; tests that remove it on purpose (to exercise the HOME fallback) still can.
os.environ["QCTX_STATE_DIR"] = os.path.join(_RUN_DIR, "state")


@atexit.register
def _remove_run_dir() -> None:
    """Removes everything the suite created. Never raises: a failure to clean up must not turn
    a green run red, and the operator's own TMPDIR is left exactly as it was found."""
    tempfile.tempdir = None
    shutil.rmtree(_RUN_DIR, ignore_errors=True)
