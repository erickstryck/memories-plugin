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

from core import names  # noqa: E402
from core import session_state as st  # noqa: E402
from core.prompts import CHECKPOINT_PROCEDURE as PROCEDURE  # noqa: E402


def env_num(name: str, legacy: str, default: str, kind=int):
    """Read at module load, i.e. BEFORE main's guard — so it must not be able to raise.

    Its sibling `recall.py` learned this the hard way and wrote it down: a malformed
    number in the environment blew up before any of our code ran, and the user got a
    traceback instead of the feature. This file had neither the tolerant read nor a
    top-level guard, so `QCTX_CHECKPOINT_INTERVAL=5x` produced a traceback and a non-zero
    exit on EVERY interaction of every session.

    Shaped like `recall.py`'s `env_num` and `hosts/hermes/__init__.py`'s `_env_num` — the
    same call, the same argument order — because the tolerance test DERIVES the knob list
    from the source of all three files. A knob written in a shape the derivation cannot see
    is a knob nobody checks, and that blind spot is exactly how `int(env(...))` survived in
    `recall.py` through three reviews.
    """
    raw = os.environ.get(name) or os.environ.get(legacy) or default
    try:
        return kind(raw)
    except (TypeError, ValueError):
        print(f"checkpoint: {name}={raw!r} is not a number — using {default}",
              file=sys.stderr)

        return kind(default)


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
        print(f"checkpoint: {type(exc).__name__}: {exc}", file=sys.stderr)


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
