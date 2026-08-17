"""Reading a tuning knob out of the environment without letting a typo kill the host.

The two functions here were, until this module existed, byte-identical copies inside
`hooks/bigfile.py` and `hosts/hermes/bigfile.py`. They are read at IMPORT time, above every
adapter's catch-all, which is exactly what makes them dangerous: `QCTX_BIGFILE_FLOOR_PCT=20%`
raising here takes down a hook that runs before every file read, or — on hermes, whose loader
pre-execs plugin files and swallows the failure at `logger.debug` — the whole provider, with
one debug line as the only symptom.

Deliberately WITHOUT the clamping variant that `hooks/recall.py`, `hooks/checkpoint.py` and
`hosts/hermes/__init__.py` each carry (their `minimum=` argument, which prints a note to
stderr and refuses to leave a caller with nothing to return). Those three still own their own
copies. Adding an unused parameter here to "cover" them would ship a branch no test can
honestly exercise, and this module would then look like the single owner while three files
quietly disagreed with it. Migrating them is its own task, on a live path, with its own review.

`legacy` is not decoration: every knob in this repo answers to a `QCTX_`-prefixed name and to
the older bare one, so an operator who exported the old spelling keeps working.
"""
import os
from pathlib import Path


def state_dir() -> Path:
    """Where this plugin keeps state, honouring QCTX_STATE_DIR.

    It lives here and not beside the code that first needed it: reading one env var and
    building a path is exactly what this module is for, and its previous home talks to Qdrant
    and is imported LAZILY so the common path does not pay for it. A pure helper inside a
    network module forces every caller to choose between an unwanted import and a copy.

    Read at CALL time, not at import: a constant frozen at import would be frozen at a moment
    that varies between hosts.
    """
    return Path(os.environ.get("QCTX_STATE_DIR") or (Path.home() / ".memories-plugin" / "state"))


def env(name: str, legacy: str, default: str) -> str:
    """The value, stripped, or the coded default when there is none worth having.

    BLANK COUNTS AS UNSET, and that is not tidiness. Not every knob here is a number any
    more: the file-read guard's escape marker is TEXT the user types, and
    `QCTX_BIGFILE_ESCAPE="   "` would make the marker a space — a string that appears in
    almost every sentence ever written, so every read would carry the escape and the guard
    would be off while still reporting itself installed. A knob whose absurd value silently
    disables the thing it configures is the shape this repo already clamps for elsewhere.
    """
    for value in (os.environ.get(name), os.environ.get(legacy)):
        if value and value.strip():
            return value.strip()

    return default


def env_num(name: str, legacy: str, default: str, kind=float):
    """The number, or the coded default when the environment holds something that is not one.

    No clamp: the callers here are FRACTIONS, and a 0 means "this criterion never fires",
    which is a coherent thing for a deployer to ask for — unlike the recall ceilings, where a
    0 makes a hook claim an empty archive.
    """
    raw = env(name, legacy, default)
    try:
        return kind(raw)
    except (TypeError, ValueError):
        return kind(default)
