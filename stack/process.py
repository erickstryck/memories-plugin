"""The command runner the whole stack talks to, isolated from the runtimes.

`Completed` is the result of one command; `Runner` is the contract; `SubprocessRunner`
is the real one. `stack.runtimes` re-exports these three so the earlier tasks and the
tests keep importing them from `stack.runtimes` (review round R2, item m10).
"""
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from typing import Protocol

from . import StackError


@dataclass(frozen=True)
class Completed:
    """The result of one command: a return code and, when captured, the streams."""
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Runner(Protocol):
    def run(self, argv: list[str], *, timeout: float, stream: bool = False) -> Completed: ...


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the command's WHOLE process group, then reap it.

    The group exists because `start_new_session` made the command a session leader.
    A compose provider runs its backend as a child of the command, and a plain
    `subprocess.run` timeout kills only the direct child, leaving that backend alive
    (measured 2026-10-06, review R2 item m2): killing the group is what takes it too.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass


class SubprocessRunner:
    """`Runner` that shells out. A missing binary is a 127, not an exception: the caller
    decides whether an absent engine is an error (it usually is `None` from `engine()`).
    A hung command is never a normal answer: it is a `StackError`, and the command's
    whole process group is killed on the way. An `OSError` at exec is a `StackError`
    naming the command, not a traceback."""

    def __init__(self, which=shutil.which):
        self.which = which

    def run(self, argv: list[str], *, timeout: float, stream: bool = False) -> Completed:
        if self.which(argv[0]) is None:
            return Completed(127, "", f"{argv[0]}: not found")
        try:
            if stream:
                # inherit the terminal's stdout/stderr; the progress bar is the provider's
                proc = subprocess.Popen(argv, start_new_session=True)
            else:
                proc = subprocess.Popen(argv, start_new_session=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise StackError(f"could not run {' '.join(argv)}: {exc}",
                             step="runtime") from None
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            raise StackError(f"timed out after {timeout}s: {' '.join(argv)}",
                             step="runtime") from None
        if stream:
            return Completed(proc.returncode)
        return Completed(proc.returncode,
                         out.decode("utf-8", "replace"),
                         err.decode("utf-8", "replace"))
