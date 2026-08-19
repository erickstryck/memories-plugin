#!/usr/bin/env python3
"""SessionStart: records that this claude-code session is alive.

WHY IT IS ITS OWN HOOK AND NOT A LINE IN THE RECALL ONE. Recall is memory search and a lease is
a sign of life; coupling them would make disabling recall end the daemon under a live host. The
user decided this on 2026-08-18.

WHY SessionStart AND NOT EVERY PROMPT. The lease carries the host's pid and start time, and the
daemon tests whether that process still exists — so it needs writing once, not refreshing. A
per-prompt heartbeat would add work to every turn to answer a question the pid already answers.

IT NEVER FAILS THE SESSION. A hook that can break SessionStart is a hook that gets uninstalled.
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    try:
        from core import daemon, lease

        found = lease.find_host_pid(names=("claude",))
        pid = found[0] if found else os.getppid()
        lease.write(str(payload.get("session_id") or "default"), "claude", pid=pid)
        # The lease alone only records that a host is alive; without this call nothing ever
        # STARTS the daemon outside of someone typing a `repos` command by hand, so watching
        # would only ever last for the session where that happened. `daemon.start()` is
        # idempotent — "already" when one is alive — so this is a call, not a race: it either
        # confirms the existing daemon or starts the one this lease now justifies.
        #
        # QCTX_DAEMON_AUTOSTART_DISABLED="1" skips it, same naming convention as
        # QCTX_CHECKPOINT_DISABLED and QCTX_RECALL_DISABLED elsewhere in this project. This is
        # what keeps a SessionStart integration test from spawning a real, detached background
        # process against whatever Qdrant the environment happens to have configured — the
        # suite sets it for exactly that reason (see tests/test_repos_init.py).
        if os.environ.get("QCTX_DAEMON_AUTOSTART_DISABLED") != "1":
            daemon.start()
    except Exception:                               # noqa: BLE001 — see the module docstring
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)
