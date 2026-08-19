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
        from core import lease

        found = lease.find_host_pid(names=("claude",))
        pid = found[0] if found else os.getppid()
        lease.write(str(payload.get("session_id") or "default"), "claude", pid=pid)
    except Exception:                               # noqa: BLE001 — see the module docstring
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)
