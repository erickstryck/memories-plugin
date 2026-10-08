"""Reading a tuning knob out of the environment without letting a typo kill the host.

The two functions here were, until this module existed, byte-identical copies inside
`hooks/bigfile.py` and `hosts/hermes/bigfile.py`. They are read at IMPORT time, above every
adapter's catch-all, which is exactly what makes them dangerous: `QCTX_BIGFILE_FLOOR_PCT=20%`
raising here takes down a hook that runs before every file read, or — on hermes, whose loader
pre-execs plugin files and swallows the failure at `logger.debug` — the whole provider, with
one debug line as the only symptom.

The clamping variant (`clamped_num`, below) is shared too: `hooks/recall.py` and
`hosts/hermes/__init__.py` each keep only a thin channel for its notes. `hooks/checkpoint.py`
had a third copy until its one knob, the interval, became the `checkpoint_interval` setting
that `core/config.py` resolves. The big-file guard's two thresholds went the same way: they
are the `bigfile_floor_pct` and `bigfile_share_pct` settings, and neither guard reads them
from the environment on its own any more.

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
    return Path(os.environ.get("QCTX_STATE_DIR") or _state_dir_default())


def _state_dir_default() -> Path:
    """The state dir, with the pre-rename dir still honoured.

    A fresh install uses `~/.mnemosine/state`. A pre-rename install has its
    recall/dedup state at `~/.memories-plugin/state`, and forking it into an
    empty new dir on the 2.0.0 update would reset the recall window and the
    rerank breaker mid-history. When the new dir does not exist and the old one
    does, the old one wins (read and written in place). `QCTX_STATE_DIR` still
    overrides both, as before."""
    fresh = Path.home() / ".mnemosine" / "state"
    legacy = Path.home() / ".memories-plugin" / "state"
    if not fresh.exists() and legacy.exists():
        return legacy
    return fresh


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


def as_num(value, default, kind=float):
    """`value` as a number, falling back to `default` when it is not one.

    The tolerance `env_num` gives the environment. (`core/config.py` applies the same rule to
    its numeric fields itself, because it also has to say WHICH value it fell back from, and
    from where.) Bare `int()` on a field a user
    can type is how a single typo took the whole plugin down: every command died on the
    exception, including the one that repairs the file, and in the hermes host the ValueError
    is not a CoreError so the loader swallowed it and the memory provider vanished with one
    debug line. Nothing here decides policy; the caller still chooses the default.
    """
    try:
        return kind(value)
    except (TypeError, ValueError):
        return kind(default)


def env_num(name: str, legacy: str, default: str, kind=float):
    """The number, or the coded default when the environment holds something that is not one.

    No clamp: the callers here are FRACTIONS, and a 0 means "this criterion never fires",
    which is a coherent thing for a deployer to ask for — unlike the recall ceilings, where a
    0 makes a hook claim an empty archive.
    """
    return as_num(env(name, legacy, default), default, kind)


def clamped_num(name: str, legacy: str, default: str, kind=int, minimum=None, *, note=None):
    """`env_num` with a floor, for the knobs where too small is a LIE rather than a setting.

    ONE COPY FOR BOTH HOSTS. This lived three times (`hooks/recall.py`, `hooks/checkpoint.py`
    and `hosts/hermes/__init__.py`) and the copies had already drifted: measured with
    `minimum=1`, a value of `0` gave 1 in recall and hermes and 0 in checkpoint, and `-1` gave
    1, 1 and -1. So the same typo silenced one surface and degraded another, which is the
    class of divergence `tests/test_host_equivalence.py` exists to prevent. The guards that
    existed derived the knob NAMES from all three sources and could not see this, because
    identical names were never the question.

    WHY A FLOOR AND NOT A REFUSAL. `core/retrieval` applies `max_memories` as a slice, so
    measured against three stored memories that all match, `=6` injected 3, `=1` injected 1,
    and `=0` produced "There is no recorded precedent on this subject" on every prompt, about
    an archive that answered. `0` meaning "unlimited" is a common deployer convention, so that
    lie was one typo away. Ignoring an absurd value beats asserting absence on its strength.

    `note` IS THE HOST'S CHANNEL, not this function's. Each surface reports differently and
    must keep doing so: the claude-code hook queues the line for its log AND prints to stderr
    (never stdout, which carries the hook protocol), while hermes prints a prefixed line. What
    belongs here is the DECISION; where the user reads it belongs to the host.
    """
    raw = env(name, legacy, default)
    try:
        value = kind(raw)
    except (TypeError, ValueError):
        # FALLS THROUGH TO THE FLOOR, never returns here. Returning the coded default early
        # would let a malformed value produce a below-floor result — exactly what the floor
        # exists to make impossible. Measured with `raw='abc', default='0', minimum=1`: the
        # early return gave 0, the two copies this replaced both gave 1.
        if note:
            note(f"{name}={raw!r} is not a number — using {default}")
        value = kind(default)
    if minimum is not None and value < minimum:
        if note:
            note(f"{name}={raw!r} would leave nothing to return — using {minimum}")

        return kind(minimum)

    return value
