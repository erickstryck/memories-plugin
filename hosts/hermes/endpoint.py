#!/usr/bin/env python3
"""Which endpoint serves the model, read from the config hermes already keeps.

WHY READ ANOTHER PROJECT'S CONFIG. The window is knowable on this host and nowhere else: the
endpoint that serves the model reports it, and hermes is the only place that records which
endpoint that is. Asking the user to declare it again, in our config, to describe something
his other config already describes, is a second source of truth for one fact.

THE COUPLING IS REAL AND DECLARED. This depends on a config format owned by another project,
which can change. It is the same coupling this adapter already has with hermes' state.db, and
the defence is the same: a test that bites when the shape changes, and a failure that DESCENDS
the cascade rather than breaking. Every path here returns "" or 0, never an exception.

`key_env` NAMES A VARIABLE, IT IS NOT THE SECRET. The config holds the name; the value lives
in the environment the hermes process already has, because it is hermes that loads this
plugin. An endpoint that needs no key is a real case, so a missing variable still yields the
URL.
"""
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import windowcache, windowprobe  # noqa: E402

#: Short: this runs from `prefetch`, which the user is waiting on.
PROBE_TIMEOUT_S = 5.0


def _home(home: str | None = None) -> str:
    """The same resolution hermes uses for subprocesses."""
    return home or os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"),
                                                                 ".hermes")


def from_hermes_config(home: str | None = None) -> tuple[str, str]:
    """`(base_url, api_key)` from hermes' own config, or `("", "")` when it cannot be read.

    Parsed with a regex rather than a YAML library because this package ships stdlib only —
    and because the two lines wanted are flat scalars, not structure. A shape this does not
    recognise yields "", which the caller reads as "no endpoint", which descends the cascade.
    """
    path = os.path.join(_home(home), "config.yaml")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return "", ""
    base = re.search(r"^\s+base_url:\s*(\S+)\s*$", text, re.M)
    if not base:
        return "", ""
    var = re.search(r"^\s+key_env:\s*(\w+)\s*$", text, re.M)

    return base.group(1), (os.environ.get(var.group(1), "") if var else "")


def refresh_window(model: str, *, probe=None) -> int:
    """The window for `model`, probing only when the cache has nothing fresh.

    Called from `prefetch`, which already pays for network. Returns the best value known —
    including a stale one when the probe learns nothing, because an endpoint being down must
    not cost a window we already had.

    `probe` is injected only so tests can drive it without a server.
    """
    base, key = from_hermes_config()
    if not base:
        return 0
    known, fresh = windowcache.get(base, model)
    if fresh:
        return known
    call = probe or windowprobe.probe
    learned = call(base, key, model, timeout=PROBE_TIMEOUT_S)
    if learned > 0:
        windowcache.put(base, model, learned)

        return learned

    return known
