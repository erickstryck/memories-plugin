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


def env(name: str, legacy: str, default: str) -> str:
    return os.environ.get(name) or os.environ.get(legacy) or default


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
