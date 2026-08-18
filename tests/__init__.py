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


@atexit.register
def _remove_run_dir() -> None:
    """Removes everything the suite created. Never raises: a failure to clean up must not turn
    a green run red, and the operator's own TMPDIR is left exactly as it was found."""
    tempfile.tempdir = None
    shutil.rmtree(_RUN_DIR, ignore_errors=True)
