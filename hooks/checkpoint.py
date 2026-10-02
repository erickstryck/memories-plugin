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
    checkpoint_interval        interactions between checkpoints (default 5, 0 turns it off):
                               `qctx config set checkpoint-interval N`, or the environment's
                               QCTX_CHECKPOINT_INTERVAL, which wins over the file
    QCTX_CHECKPOINT_DISABLED   "1" turns it off
    QCTX_STATE_DIR             where to keep the counter
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config, knobs, names  # noqa: E402
from core import session_state as st  # noqa: E402
from core import statefile  # noqa: E402
from core.prompts import CHECKPOINT_PROCEDURE as PROCEDURE  # noqa: E402


def _note(line: str) -> None:
    """This hook's channel for a configuration value it could not use.

    ALWAYS BY FILE DESCRIPTOR, never `print(file=sys.stderr)`. Two requirements meet here and
    both are real: `test_a_malformed_interval_does_not_kill_the_hook` wants the typo visible
    ("falling back silently hides the typo"), and the protocol this hook prints on stdout must
    survive a closed fd 2, where `print(file=sys.stderr)` silently falls back to stdout and
    corrupts the JSON, which no `try` can catch because writing SUCCEEDS. `os.write(2, ...)`
    raises on a closed fd instead, so the note is dropped exactly when delivering it would
    cost the block, and never misdelivered.
    """
    try:
        os.write(2, f"checkpoint: {line}\n".encode())
    except OSError:        # noqa: BLE001 (a lost note is cheaper than a lost block)
        pass


def interval() -> int:
    """Interactions between checkpoints, resolved by `core.config` like every other setting.

    READ ON EVERY RUN, not at import: a hook is one short process per prompt, so
    `qctx config set checkpoint-interval N` takes effect on the next prompt. It used to be an
    environment variable read here and again in the hermes adapter, two copies of one rule;
    the config is now its only reader, and the environment still wins over the file.
    """
    return config.load(note=_note).checkpoint_interval

# `knobs.state_dir()` and not a fourth copy of this expression: `core/bindings.py`
# already writes down why ("a third copy of where state lives is how the three start\n# to disagree"), and this file was one of the copies. Still a module-level constant
# because a hook is one short process and the directory cannot change under it.
STATE_DIR = knobs.state_dir()


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


def bump(counter: Path) -> int:
    """Increments the session's interaction counter and returns the new value.

    EXTRACTED FROM `_run` SO IT CAN BE DRIVEN DIRECTLY. The counting is one responsibility and
    the hook's stdin/stdout protocol is another; inlining both meant the only way to exercise
    a count was to fake a hook payload on a pipe. It also publishes through `core.statefile`,
    which owns the file mode -- measured on the real machine, the hand-rolled `write_text`
    this replaced published every counter 0o664 under the usual umask, and there were 64 of
    them.

    AN UNREADABLE COUNTER RESTARTS AT ONE rather than raising: a corrupt counter must cost at
    most one early checkpoint, never the user's turn.
    """
    try:
        n = int(counter.read_text().strip())
    except Exception:
        n = 0
    n += 1
    statefile.write_text(counter, str(n))

    return n


def _run() -> None:
    if os.environ.get("QCTX_CHECKPOINT_DISABLED") == "1":
        return

    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    session = names.safe(data.get("session_id"))
    statefile.ensure_dir(STATE_DIR)
    counter = STATE_DIR / f"checkpoint-{session}.count"

    n = bump(counter)
    every = interval()

    if not st.due(n, every):
        return  # silent on the intermediate interactions

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": PROCEDURE.format(count=n, interval=every),
        }
    }))


if __name__ == "__main__":
    main()
