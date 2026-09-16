#!/usr/bin/env python3
"""CHECKPOINT hook: every N interactions, injects the writing procedure.

The write-side counterpart of `recall.py`. This hook stores nothing — it hands the
model the complete procedure at the moment there is accumulated conversation to
distil.

The text is deliberately SELF-SUFFICIENT. A one-line reminder ("save whatever is
durable") produces vague, duplicated, metadata-less memory, and the cost shows up
months later, when a search returns three contradictory versions of the same fact and
nobody knows which one holds. Whoever reads the block has to be able to act without
opening anything else.

Configuration:
    QCTX_CHECKPOINT_INTERVAL   interactions between checkpoints (default 5)
    QCTX_CHECKPOINT_DISABLED   "1" turns it off
    QCTX_STATE_DIR             where to keep the counter
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import knobs, names  # noqa: E402
from core import session_state as st  # noqa: E402
from core.prompts import CHECKPOINT_PROCEDURE as PROCEDURE  # noqa: E402


def env_num(name: str, legacy: str, default: str, kind=int, minimum=None):
    """This host's channel for the shared clamped read in `core/knobs.py`.

    Read at module load, ABOVE `main`'s catch-all: this file had neither the tolerant read nor
    a top-level guard, so `QCTX_CHECKPOINT_INTERVAL=5x` produced a traceback and a non-zero
    exit on EVERY interaction of every session.

    THE SHARED COPY ALSO FIXED A DIVERGENCE. This one carried no `minimum` and the other two
    did, so with a floor of 1 a value of `0` gave 1 in recall and hermes and 0 here, and `-1`
    gave 1, 1 and -1 — the same typo degrading one surface and silencing another. The
    knob-name guards could not see it: identical names were never the question.
    """
    def report(line: str) -> None:
        # ALWAYS BY FILE DESCRIPTOR, never `print(file=sys.stderr)`. Two requirements meet here
        # and both are real: `test_a_malformed_interval_does_not_kill_the_hook` wants the typo
        # visible ("falling back silently hides the typo"), and the protocol this hook prints
        # on stdout must survive a closed fd 2 — where `print(file=sys.stderr)` silently falls
        # back to stdout and corrupts the JSON, which no `try` can catch because writing
        # SUCCEEDS. `os.write(2, ...)` raises on a closed fd instead, so the note is dropped
        # exactly when delivering it would cost the block, and never misdelivered.
        #
        # The comment here used to claim this hook writes no protocol on stdout. It does:
        # `main` ends in `print(json.dumps(...))`.
        try:
            os.write(2, f"checkpoint: {line}\n".encode())
        except OSError:        # noqa: BLE001 — a lost note is cheaper than a lost block
            pass

    return knobs.clamped_num(name, legacy, default, kind, minimum, note=report)


INTERVAL = env_num("QCTX_CHECKPOINT_INTERVAL", "REMEMBER_INTERVAL", "5", int)
STATE_DIR = Path(os.environ.get("QCTX_STATE_DIR") or (Path.home() / ".memories-plugin" / "state"))


def main() -> None:
    """Armoured, like the recall hook.

    This runs on every prompt. Anything it cannot do — an unwritable state directory, a
    full disk — is a reason to inject nothing, never a reason to hand the host a
    traceback and a non-zero exit on every interaction. Nothing here is load-bearing
    enough to be worth failing loudly over: the worst outcome of silence is one skipped
    checkpoint.
    """
    try:
        _run()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — see docstring
        # BY FILE DESCRIPTOR, like the config note above: `print(file=sys.stderr)` falls back
        # to stdout when fd 2 is closed, and this hook's block travels on stdout.
        try:
            os.write(2, f"checkpoint: {type(exc).__name__}: {exc}\n".encode())
        except OSError:        # noqa: BLE001 — a lost note is cheaper than a lost block
            pass


def _run() -> None:
    if os.environ.get("QCTX_CHECKPOINT_DISABLED") == "1":
        return

    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    session = names.safe(data.get("session_id"))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    counter = STATE_DIR / f"checkpoint-{session}.count"

    try:
        n = int(counter.read_text().strip())
    except Exception:
        n = 0
    n += 1
    counter.write_text(str(n))

    if not st.due(n, INTERVAL):
        return  # silent on the intermediate interactions

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": PROCEDURE.format(count=n, interval=INTERVAL),
        }
    }))


if __name__ == "__main__":
    main()
