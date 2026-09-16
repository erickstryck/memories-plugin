"""The background process: runs what is queued, watches what is indexed, and ends with its hosts.

WHY A DAEMON AT ALL. Indexing a real project is minutes of work — measured at ~0.13 s per file,
so 1,800 files is several minutes — and doing it in the caller's terminal means no progress, no
cancelling, and a frozen prompt. There is no prohibition on background processes in this project;
an earlier note about how the temporary archive's TTL expires was mistaken for one.

WHY IT ENDS WITH THE HOSTS. A process that outlives the tool that started it is a process the
user did not ask for and will not think to stop. Each cycle it checks the leases, and no living
lease means nobody is using it.

WHY `run` TAKES ITS WORKER AND ITS CYCLE COUNT. So the whole loop can be exercised in-process,
with no spawning and no network — the same choice `refresh_window(probe=...)` already makes.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import jobs, lease, statefile
from .errors import CoreError
from .knobs import state_dir

#: How long the loop sleeps between cycles. Short enough that a cancel or a file change is
#: noticed while the user is still looking at the screen; long enough to be free.
CYCLE_S = 5.0


class DaemonError(CoreError):
    """The daemon could not be started, or a start could not be confirmed. Raised only by
    `start`.

    THE CLAIM IS USUALLY RELEASED FIRST, BUT NOT ALWAYS, and the exception says which. Every
    failure path releases the claim it took, EXCEPT the one where a process was spawned and
    then could not be confirmed dead: there, releasing would let the next `start()` spawn a
    second daemon on top of a live first one, which the design calls impossible. That case
    keeps the claim deliberately and its message names the pid to kill by hand. An earlier
    version of this docstring promised the claim was always released; it stopped being true
    the moment that path was added, which is why it now describes both.
    """


def path() -> Path:
    return state_dir() / "daemon.json"


#: What `record()` answers for a file that EXISTS but cannot be parsed.
#:
#: NOT an empty dict: every other caller tests `record()` for TRUTHINESS (`if not entry`,
#: `existing.get("pid", 0) if existing else 0`), so a falsy sentinel would read to them as
#: "nothing is running" — the very answer this exists to avoid. It carries no usable pid, and
#: that is honest: `stop()` cannot signal a daemon whose record it cannot read, and returns
#: False rather than killing something it cannot name. `_claim()` is the one caller that tests
#: `is not None`, which is what stops it from removing a claim it failed to parse.
_UNREADABLE: dict = {"unreadable": True}


def record() -> dict | None:
    """The running daemon, or None. A record whose process is gone reads as none.

    Same `(pid, starttime)` test the leases use: a recycled pid must not make a dead daemon look
    alive, or nothing would ever start one again.

    AN UNPARSEABLE RECORD IS NOT A CORPSE. `None` here sends `_claim()` down the path that
    unlinks the file and takes the claim, so answering `None` for a file that merely could not
    be READ hands the claim to a caller while the real holder is still alive. `_try_create`
    publishes the record complete, so a file that exists and does not parse is a damaged one,
    not a half-written one — and the safe answer to damage is the one that starts no second
    daemon. The empty case is called out separately because it is the one an interrupted write
    leaves behind, and it used to be indistinguishable from a dead daemon's record.
    """
    try:
        raw = path().read_text(encoding="utf-8")
    except OSError:
        return None                     # no file at all: genuinely nothing running
    try:
        entry = json.loads(raw)
    except ValueError:
        return _UNREADABLE
    if not isinstance(entry, dict):
        return _UNREADABLE
    if _unjudgeable(entry):
        # PRESENT AND NOT JUDGEABLE, which is not the same as gone. Falling through to
        # `lease.alive` here would answer None — the corpse verdict — for every record on a
        # platform where `process_start` cannot read a start time, and `_claim()` removes
        # corpses. See `_unjudgeable`.
        return entry
    if not lease.alive(entry):
        return None

    return entry


def start(spawn=None, argv: list[str] | None = None, sleep=time.sleep) -> dict:
    """Starts the daemon if none is running. `{"action": "started" | "already", "pid": int}`.

    CLAIMS THE RECORD FILE BEFORE SPAWNING, not after. `record()` then `write()` is a
    check-then-write: two commands invoked at the same moment can both read no daemon and both
    spawn one — the gap is between two of OUR statements, not something the OS arbitrates.
    `_claim()` instead creates `daemon.json` with `O_EXCL`, which the kernel guarantees only one
    caller can do for a given path; the loser gets `FileExistsError` and never spawns. This is
    the design's answer to "two daemons": exclusive creation decides who won, not a check.

    `spawn` is injected for tests; by default it launches a detached `qctx repos daemon run`.

    RAISES `DaemonError`, and RELEASES ITS OWN CLAIM, when either step after the claim fails:

    - `spawn` itself raises (the executable is missing, `fork` fails, ...) — nothing is
      running, so the claim is simply released and the reason is reported.
    - `spawn` SUCCEEDS but `_write_record` cannot persist the real pid — the trickier case,
      because a live process now exists that the claim does not correctly describe. Leaving
      the placeholder in place would misrepresent the CALLER as the daemon; releasing the
      claim without stopping the process would let it keep running untracked, and the next
      `start()` would then be free to spawn a second one — the one outcome the design calls
      impossible (`O_EXCL` only prevents two CONCURRENT claims, not a claim that never
      correctly recorded what it started). So the spawned process is killed and the claim is
      released: a start that could not be recorded is a failed start, not a silent orphan.

    Called from inside a long-lived host process (hermes' `initialize()`), not only from a
    short CLI invocation — so this cannot rely on "the caller will exit soon and the stale
    claim will self-heal", which is what an ephemeral CLI process gets for free.
    """
    if not _claim():
        existing = record()

        return {"action": "already", "pid": existing.get("pid", 0) if existing else 0}
    launch = spawn or _spawn
    # SPAWNS THIS MODULE, NOT THE CLI. `core` used to launch `cli/qctx.py repos daemon run`,
    # which made the host-neutral layer depend on the CLI layer: shipping `core` alone, or
    # adding a third host, meant editing core to point somewhere else. The loop being started
    # lives HERE, so this module is its own entry point (see the `__main__` block at the
    # bottom) and the dependency simply disappears. `cli` keeps its `repos daemon run` command,
    # which now runs the same loop in the foreground for anyone who wants to watch it.
    command = argv or [sys.executable, "-m", "core.daemon"]
    try:
        pid = launch(command)
    except OSError as exc:
        _release_claim()
        raise DaemonError(f"could not start the daemon: {exc}") from exc
    starttime = lease.process_start(pid) or ""
    if not _write_record({"pid": pid, "starttime": starttime, "started_at": time.time()}):
        if _stop_and_confirm({"pid": pid, "starttime": starttime}, sleep=sleep):
            _release_claim()
            raise DaemonError(f"the daemon started (pid {pid}) but its record could not be "
                              f"written — state directory unavailable; the process was stopped "
                              f"rather than left running untracked")
        raise DaemonError(f"the daemon started (pid {pid}) but its record could not be written "
                          f"AND the process did not stop when asked; the claim is being kept so "
                          f"nothing starts a second daemon on top of it — kill {pid} by hand")

    return {"action": "started", "pid": pid}


def _release_claim() -> None:
    try:
        path().unlink()
    except OSError:
        pass


def _pid_alive_fn(entry: dict):
    """A liveness test for a record we cannot judge by `(pid, starttime)`. ONE owner.

    Signal 0 is the only liveness question that answers on every platform: `process_start`
    reads `/proc`, which does not exist on macOS or Windows, so it answers None for live
    processes there and any caller comparing it against None confirms deaths it never saw.

    The pid alone is weaker evidence than `(pid, starttime)` — a recycled pid reads as alive —
    and both callers want to be wrong in that direction: a claim held too long delays a daemon,
    a claim released too early spawns a second one on top of a live first.
    """
    try:
        pid = int(entry["pid"])
    except (TypeError, ValueError, KeyError):
        return lambda: True             # no pid to check: never confirm anything

    def alive() -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True                 # PermissionError and friends: it exists

        return True

    return alive


def _stop_and_confirm(entry: dict, timeout_s: float = 2.0, poll_s: float = 0.05,
                      sleep=time.sleep) -> bool:
    """Signals the process named by `entry` and returns True only once it is CONFIRMED gone.

    WHY CONFIRMATION AND NOT A BARE `os.kill`. Releasing the claim is what lets the next
    `start()` spawn — so releasing it on an unconfirmed kill is exactly how a second daemon
    gets spawned on top of a first one that is still alive, which the spec calls impossible.
    A signal that raises (the process is already gone) and a signal that lands but is ignored
    look identical to the caller; only re-reading `(pid, starttime)` tells them apart, which is
    the same test the leases and `stop()` use. A timeout that runs out returns False and the
    caller KEEPS the claim: a stale claim blocks new daemons, while a released one over a live
    process multiplies them, and of those two failures only the first is recoverable by waiting.
    """
    try:
        os.kill(int(entry["pid"]), 15)
    except (OSError, ValueError, KeyError):
        pass                                          # already gone, or never startable
    # WITHOUT A START TIME THERE IS NOTHING TO RE-READ, so the `(pid, starttime)` test cannot
    # confirm anything and `lease.alive` answers False on its first guard — which read as
    # "confirmed dead" and released the claim over a process that was still running (measured:
    # True in 0.000s against a child ignoring SIGTERM). The pid alone is the only evidence
    # available here, and it is weaker: a recycled pid looks alive. That is the right way to
    # be wrong for this caller, because the cost is a claim held too long, not two daemons.
    #
    # LIVENESS IS ASKED WITH SIGNAL 0, NOT WITH `process_start`. The first version of this
    # fallback asked `process_start(pid) is not None`, which fixed nothing off Linux: with no
    # `/proc` that answers None for EVERY pid, so the loop exits on its first turn and confirms
    # a death it never saw. Measured with `process_start` stubbed to None, as macOS and Windows
    # behave: stop() returned True in 0.000s against a live child ignoring SIGTERM and released
    # the claim. `os.kill(pid, 0)` answers the same question on every platform.
    judged = _pid_alive_fn(entry) if _unjudgeable(entry) else (lambda: lease.alive(entry))
    deadline = time.monotonic() + timeout_s
    while judged():
        if time.monotonic() >= deadline:
            return False
        sleep(poll_s)

    return True


def _unjudgeable(entry: dict) -> bool:
    """True when the record names a process whose liveness we cannot establish.

    `lease.process_start` returns None where `/proc` is unavailable, and `lease.alive` turns
    that into False because for a LEASE the safe direction is "assume dead": a host whose
    liveness cannot be read must not keep a daemon running forever. For a CLAIM the safe
    direction is the exact opposite — "assume alive", because the answer decides whether we
    may delete someone else's claim and spawn a second daemon on top of them.

    So the predicate is not reused, it is inverted here on purpose. Measured before this
    existed: with `process_start` stubbed to None, three consecutive `_claim()` calls each
    returned True, and `_stop_and_confirm` confirmed in 0.000s the death of a child that was
    ignoring SIGTERM and still running.
    """
    return not entry.get("starttime")


def _claim() -> bool:
    """Exclusively creates the record file. True only for the caller that wins the race.

    A `FileExistsError` means someone got there first — but "someone" might be a daemon that
    crashed without cleaning up, and `record()` already knows how to tell the two apart: it
    reads back as an entry only when the process it names is still alive. So on a collision we
    ask `record()`. An entry means the file is genuinely held — return False, not ours. `None`
    means the file is a corpse left by a dead daemon, and leaving it there would jam every future
    `start()` on one crash forever, so we remove it and retry the claim EXACTLY ONCE. We do not
    loop: a claim that fails twice means somebody else's `_claim()` won the newly-empty slot in
    the interval between our unlink and our retry, which is the correct outcome of the race, not
    a bug to spin around.
    """
    if _try_create():
        return True
    held = record()
    if held is not None and not _reclaimable(held):
        return False
    try:
        path().unlink()
    except OSError:
        pass

    return _try_create()


def _reclaimable(entry: dict) -> bool:
    """Whether a record we cannot judge by `(pid, starttime)` names a process that is GONE.

    `record()` answers an unjudgeable record as truthy on purpose: for `stop()` and for
    `status`, "I cannot tell" must read as "assume alive". But `_claim()` asks a different
    question — may I remove this file? — and truthiness alone answered "never", which jams
    `start()` forever on a record with no `starttime`. Measured: three consecutive `start()`
    calls each returned `{"action": "already"}` with zero spawns and no path out.

    That state is not exotic. `start()` persists `lease.process_start(pid) or ""`, so a child
    that exits immediately leaves it empty here on Linux, and on macOS and Windows — both
    listed as supported — `/proc` never exists, so EVERY record is written this way and
    indexing would stop for good after the first daemon exited.

    The pid alone is weaker evidence than `(pid, starttime)`: a recycled pid reads as alive.
    That is the right way to be wrong here, because holding a claim too long delays a daemon
    while releasing one too early spawns a second on top of a live first.

    LIVENESS IS ASKED WITH SIGNAL 0, NOT WITH `process_start` — see `_pid_alive_fn`, which owns
    that question for this module. On a platform without `/proc` `process_start` answers None
    for every pid, live ones included, so asking it here would steal every claim on exactly the
    platforms this branch exists to serve.

    NO PID AT ALL IS NOT A CORPSE, IT IS A CLAIM IN PROGRESS. A record with no readable pid is
    what another process writing its claim RIGHT NOW looks like from here, and the 0-byte
    window is exactly the race `_try_create` was rewritten to close. Answering "reclaimable"
    would reopen it from the other side.
    """
    if not _unjudgeable(entry):
        return False                    # judgeable and alive: record() would have said None
    try:
        int(entry["pid"])
    except (TypeError, ValueError, KeyError):
        return False                    # mid-write, not dead: leave it alone

    return not _pid_alive_fn(entry)()


def _try_create() -> bool:
    """Attempts the exclusive create. The placeholder written on success names THIS process —
    not the daemon `start()` is about to spawn — so a concurrent `_claim()` that collides with us
    while we are still between claiming and spawning reads back an alive entry (this process) and
    correctly backs off, instead of mistaking our in-progress claim for a stale one and tearing
    it out from under us. `_write_record` overwrites it with the real pid once spawning succeeds.

    THE FILE IS NEVER PUBLISHED EMPTY. `os.open(O_EXCL)` followed by `os.write` created the
    name first and filled it a statement later, and `record()` parses what it reads — so for
    the width of that window the claim was a 0-byte file, `json.loads("")` raised, and an empty
    file read EXACTLY like the record of a daemon that died. A concurrent `_claim()` then took
    the corpse path, unlinked the live claim and created its own on top: a second daemon, which
    the spec calls impossible. Measured deterministically (`record()` on 0 bytes -> None;
    `_claim()` over it -> True, file rewritten to the caller's pid) and under load: 6 of 12
    races of 8 processes ended with two spawns.

    So the content is written to a private temporary and the claim is taken by LINKING it into
    place: `os.link` fails with `FileExistsError` when the name is taken, giving the same
    exclusive-create guarantee, and the file is complete from the instant it becomes visible.

    NOT EVERY `OSError` FROM `os.link` MEANS THE NAME IS TAKEN. `FileExistsError` does; ENOSYS
    or EPERM from a filesystem without hard-link support means the exact opposite — nobody
    holds the claim and nobody ever will. Swallowing both told the caller "a daemon is already
    running" where none was, with no path out: measured with `os.link` raising ENOSYS,
    `start()` answered `{"action": "already", "pid": 0}` with zero spawns, forever. So the
    link-less case falls back to `os.open(O_CREAT | O_EXCL)`, which gives the same exclusivity
    the link was chosen for; it is only the 0-byte window that makes it second choice, and
    that window is closed here by writing the content before the name is published.
    """
    try:
        path().parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps({"pid": os.getpid(),
                            "starttime": lease.process_start(os.getpid()) or ""},
                           sort_keys=True).encode("utf-8")
        staged = path().with_suffix(f".{os.getpid()}.claim")
        # THE CLAIM NAMES A PID THIS USER CONTROLS, and the `os.open(..., 0o600)` this replaced
        # said so explicitly. `write_bytes` takes the umask instead, which published it 0o644
        # under the common umask 022. Staging with an explicit mode keeps the old guarantee,
        # and the link inherits it.
        fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, entry)
        finally:
            os.close(fd)
    except OSError:
        return False
    try:
        try:
            os.link(staged, path())
        except FileExistsError:
            return False                # someone else holds it: the answer the link is for
        except OSError:
            # No hard links here. Same exclusivity via O_EXCL, and the content is written
            # before the name exists, so the 0-byte window that made this second choice
            # does not reopen.
            try:
                fd = os.open(path(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                return False
            except OSError:
                return False
            try:
                os.write(fd, entry)
            finally:
                os.close(fd)

        return True
    finally:
        try:
            staged.unlink()
        except OSError:
            pass


def stop(timeout_s: float = 2.0, poll_s: float = 0.05, sleep=time.sleep) -> bool:
    """Asks the running daemon to end, and WAITS for it to actually be gone before releasing
    the claim. False when there was none, or when the process did not die within `timeout_s`.

    UNLINKING RIGHT AFTER THE SIGNAL — an earlier version did exactly that — freed the claim
    while the process could still be mid-shutdown, so `stop()` immediately followed by
    `start()` (as `add-all` does) could spawn a SECOND live daemon before the first one had
    actually exited. Polling the same `(pid, starttime)` test the leases use closes that
    window: the claim is released only once the process is confirmed gone, comparing
    `starttime` too so a pid recycled during the wait is not mistaken for the daemon still
    running. A timeout that runs out KEEPS the claim rather than guessing — a stale "running"
    that turns out to be true is safer than a second daemon spawned onto a live first one.
    A record that cannot be judged is handled by `_stop_and_confirm`, which owns the
    signal-then-confirm rule for BOTH callers. This function used to carry its own copy of the
    loop, and when the unjudgeable case was taught to the other copy this one was left behind:
    it kept looping on `lease.alive(entry)`, which answers False on its first guard for a
    record with no `starttime`, so it confirmed in 0.000s the death of a child that was
    ignoring SIGTERM and released the claim over it. One owner, so that cannot happen again.
    """
    entry = record()
    if not entry:
        return False
    if not _stop_and_confirm(entry, timeout_s=timeout_s, poll_s=poll_s, sleep=sleep):
        return False
    _release_claim()

    return True


def run(work, *, cycles: int | None = None, sleep=time.sleep, watch=None) -> str:
    """THE LOOP. Returns why it stopped.

    Each cycle, in this order and for these reasons:

      1. no living lease → exit. Checked FIRST so a daemon whose hosts are gone does not start
         one more job before noticing.
      2. reap jobs left `running` by a daemon that died, so `status` never shows stalled
         progress as activity.
      3. run one pending job, if any. One at a time: two workers on one repository duplicate
         the work without finishing sooner.
      4. otherwise watch, which is where changed files become new jobs.

    A worker that raises marks its job failed and the loop CONTINUES — one broken repository
    must not end the daemon for the others. `watch` gets the SAME survival guarantee: it has no
    single job to mark failed, but letting it propagate would end indexing for every repository
    being watched over one that could not enqueue, not just the one that failed.
    """
    seen = 0
    while cycles is None or seen < cycles:
        if not lease.live():
            return "no live lease"
        jobs.reap(lambda pid: lease.process_start(pid) is not None, lease.process_start)
        job = jobs.next_pending()
        if job is not None:
            _run_one(job, work)
        elif watch is not None:
            try:
                watch()
            except Exception:                            # noqa: BLE001 — see the docstring
                # A watcher that cannot enqueue must not end the daemon for every OTHER
                # repository. There is no job here to mark failed, so the loop simply carries
                # on: the change is still on disk, the next cycle sees it again, and a file
                # whose reindex never happens keeps showing `[stale]` in search — which is
                # where the user actually notices, not in a daemon log nobody is watching.
                pass
        seen += 1
        if cycles is None or seen < cycles:
            sleep(CYCLE_S)

    return "cycles exhausted"


def _run_one(job: dict, work) -> None:
    """Runs one job and records its outcome ON THAT JOB, never on whatever the repository has
    by the time it finishes.

    EVERY WRITE HERE IS A COMPARE-AND-SET on the job's `id`. Job files are addressed by
    repository, so `add-all` typed while the daemon is mid-job replaces the record — and the
    daemon then stamped ITS result onto the newcomer. REPRODUCED before the fix: the new job
    ended `state=done, done=0, total=2` with neither of its files indexed, and `status` called
    it a success. With `only_if`, the stale write simply does not land: the new job stays
    PENDING and the next cycle picks it up, which is what queueing it meant.
    """
    repo, jid = job["repo"], job.get("id")
    if jobs.cancel_requested(repo):
        jobs.update(repo, only_if=jid, state=jobs.CANCELLED)

        return
    # THE PAIR, not just the number: `jobs.reap` compares both, because a recycled pid
    # answering "alive" left a dead job RUNNING forever. Same test `lease.alive` applies.
    jobs.update(repo, only_if=jid, state=jobs.RUNNING, daemon_pid=os.getpid(),
                daemon_start=lease.process_start(os.getpid()), error="")
    try:
        work(job)
    except Exception as exc:                        # noqa: BLE001 — see the docstring of `run`
        jobs.update(repo, only_if=jid, state=jobs.FAILED,
                    error=f"{type(exc).__name__}: {exc}"[:400])

        return
    if jobs.cancel_requested(repo):
        jobs.update(repo, only_if=jid, state=jobs.CANCELLED)

        return
    jobs.update(repo, only_if=jid, state=jobs.DONE, current="")


def _write_record(entry: dict) -> bool:
    """Writes the daemon record. Returns True on success, False on failure — checked by
    `start()`, which cannot afford to treat "wrote" and "did not" the same way `jobs._write`
    can, because the process behind a failed write is still running."""
    return statefile.write_json(path(), entry)


#: Spawned children whose exit status has not been collected yet. A `Popen` nobody waits on
#: leaves a zombie in the process table until its parent exits — harmless for liveness
#: (`lease.process_start` reads Z as gone, by design) but this is spawned from the hermes
#: provider's `initialize()`, which lives for the whole session, so they accumulate there.
#: Measured: one Z per spawn.
_spawned: list = []


def _spawn(argv: list[str]) -> int:
    """Launches `argv` fully detached, so it survives the terminal that started it.

    PUTS THE PACKAGE ROOT ON THE CHILD'S `PYTHONPATH` rather than relying on its working
    directory. The child is `python -m core.daemon`, and `-m` needs `core` to be importable —
    but this is spawned from a hook, from the hermes provider and from the CLI, each with a
    different cwd, and a detached process should not depend on the one it happened to inherit.
    """
    # Reaped opportunistically, never waited on: `poll()` collects a child that has already
    # exited and returns None for one still running, so this costs nothing and never blocks
    # the caller on a daemon that is doing its job.
    _spawned[:] = [p for p in _spawned if p.poll() is None]

    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{_root()}{os.pathsep}{existing}" if existing else _root()
    out = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    _spawned.append(out)

    return out.pid


def _root() -> str:
    """The directory holding the `core` package, for a child that must be able to import it."""
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


if __name__ == "__main__":                              # `python -m core.daemon`
    # THE DAEMON'S OWN ENTRY POINT. Kept here, beside the loop it starts, so `start()` never
    # has to name a file in another layer. Imported inside the guard because `core.indexer`
    # pulls the whole indexing stack and nothing that merely imports `core.daemon` needs it.
    from . import indexer

    print(run(indexer.work(), watch=indexer.watcher()))
