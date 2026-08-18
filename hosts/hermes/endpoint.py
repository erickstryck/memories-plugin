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

SCOPED TO THE ACTIVE `model:` BLOCK, AND NOT THE WHOLE FILE. Measured against a real
`~/.hermes/config.yaml`: the file also carries a `custom_providers:` catalogue — servers
hermes is NOT currently using — that has its own `base_url` and its own `key_env`, sitting
well below the active block. A whole-file search finds whichever one comes first in the
text, which is right only by the coincidence of one file's ordering, not by anything the
format guarantees. So this reads `base_url` and the credential from INSIDE the top-level
`model:` block only; a catalogue elsewhere cannot answer for the endpoint actually in use.

TWO CREDENTIAL FORMS, BOTH NAMING A VARIABLE. The active block was measured using
`api_key: ${VAR}` — `${...}` interpolation — where `key_env: VAR` never appears at all. Both
are accepted; `key_env` is tried first since it already names the variable directly. A literal
`api_key:` (no `${...}`) is used as the key as-is, because hermes itself would use it that way
— but it is never logged, echoed, or placed in an exception message.
"""
import os
import re
import time
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import windowcache, windowprobe  # noqa: E402

#: Short: this runs from `prefetch`, which the user is waiting on.
PROBE_TIMEOUT_S = 5.0

#: How long a probe that learned nothing is left alone before being tried again.
RETRY_AFTER_S = 900.0

#: `(endpoint, model)` -> when it is worth probing again, for probes that learned NOTHING.
#:
#: WHY THIS IS NOT IN THE CACHE FILE. `windowcache.put` refuses to store a zero, and that is
#: correct: absence and "the endpoint answered zero" have to stay distinct, or the cascade
#: would stop at a step that knows nothing. The cost of that correctness is that a failed
#: probe leaves NOTHING behind, so the next turn probes again — with the archive reachable and
#: the model endpoint down, that is PROBE_TIMEOUT_S added to every single turn, forever.
#:
#: This is the missing half, and it belongs in memory rather than on disk because it is a
#: statement about a request that just failed, not a fact about the model. It must not outlive
#: the process, and it must not be read by the other host, which has no endpoint at all. The
#: hermes provider is one long-lived object per process, so an in-process dict is exactly the
#: lifetime wanted.
_RETRY_AFTER: dict[tuple[str, str], float] = {}


def forget_failures() -> None:
    """Drops every back-off. For tests, and for a caller that knows the network changed."""
    _RETRY_AFTER.clear()


def _home(home: str | None = None) -> str:
    """The same resolution hermes uses for subprocesses."""
    return home or os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"),
                                                                 ".hermes")


def _model_block(text: str) -> str:
    """The text INSIDE the top-level `model:` block only, from just after the flush-left
    `model:` line to the next flush-left KEY (or the end of the file).

    A flush-left `#` comment does not end the block. Measured against a real
    `~/.hermes/config.yaml`: it carries 36 flush-left comment lines, none inside the `model:`
    block today — but a lookahead that treated any flush-left non-whitespace as "the next
    key" would close the block the moment one landed there, one hermes upgrade or one user
    note away, and it would fail SILENTLY: `base_url` vanishes from the captured text, the
    cascade falls to the ceiling table, and the guard sleeps with no error anywhere. So the
    boundary is "a flush-left line whose first character is neither whitespace nor `#`" —
    only a real key closes the block; a comment, flush-left or indented, is walked over.

    Scoped like this so a `custom_providers:` catalogue elsewhere in the file — which lists
    servers hermes is NOT currently using — cannot answer for the active endpoint merely
    because a whole-file search reached it first. "" when there is no top-level `model:` key
    at all.

    `[ \t\r]*` and not `[ \t]*` after `model:`: a CRLF config leaves a trailing `\r` before
    the `\n` this looks for, and without it in the class the line never matches at all — the
    same silent, no-error failure a flush-left comment produced before this function existed.
    """
    found = re.search(r"^model:[ \t\r]*\n(.*?)(?=^[^\s#]|\Z)", text, re.M | re.S)

    return found.group(1) if found else ""


def _unquoted(value: str) -> str:
    """Strips one layer of matching single or double quotes. YAML-legal (`base_url:
    "https://x"`), and captured with the quotes still on by a regex that only looks for
    non-whitespace — left as-is, that is a broken URL."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]

    return value


def _credential(block: str) -> str:
    """The credential the active block names, resolved from the environment — or a literal
    `api_key:` value, used as-is because hermes itself would use it that way.

    `key_env` is tried first: it already names the variable directly. `api_key: ${VAR}` is
    the form measured in a real config, where `key_env` does not appear at all; `${...}` is
    unwrapped and resolved the same way. Never logged, echoed, or placed in an exception
    message — whichever form this returns.
    """
    key_env = re.search(r"^\s+key_env:\s*(\w+)\s*$", block, re.M)
    if key_env:
        return os.environ.get(key_env.group(1), "")

    api_key = re.search(r"^\s+api_key:\s*(\S+)\s*$", block, re.M)
    if not api_key:
        return ""
    raw = _unquoted(api_key.group(1))
    interpolated = re.fullmatch(r"\$\{(\w+)\}", raw)
    if interpolated:
        return os.environ.get(interpolated.group(1), "")

    return raw       # a literal already in hermes' own config — never logged past this point


def from_hermes_config(home: str | None = None) -> tuple[str, str]:
    """`(base_url, api_key)` from hermes' own config, or `("", "")` when it cannot be read.

    Parsed with a regex rather than a YAML library because this package ships stdlib only —
    and because the values wanted are flat scalars, not structure. A shape this does not
    recognise yields "", which the caller reads as "no endpoint", which descends the cascade.
    """
    path = os.path.join(_home(home), "config.yaml")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return "", ""
    block = _model_block(text)
    if not block:
        return "", ""
    base = re.search(r"^\s+base_url:\s*(\S+)\s*$", block, re.M)
    if not base:
        return "", ""

    return _unquoted(base.group(1)), _credential(block)


def refresh_window(model: str, *, probe=None) -> int:
    """The window for `model`, probing only when the cache has nothing fresh.

    Called from `prefetch`, which already pays for network. Returns the best value known —
    including a stale one when the probe learns nothing, because an endpoint being down must
    not cost a window we already had.

    `probe` is injected only so tests can drive it without a server.
    """
    if not model:
        # `state.db` locked, or the session row not yet written: `model_of` returns "" for
        # both. An empty model id can never match a real entry at `/models`, so probing for
        # it is a 5s round trip that is guaranteed useless — and unlike a probe that learns
        # nothing for a REAL model, this one would repeat on every single turn forever,
        # because `put` correctly refuses to cache a window against no model at all.
        return 0
    base, key = from_hermes_config()
    if not base:
        return 0
    known, fresh = windowcache.get(base, model)
    if fresh:
        return known
    slot = (base, model)
    if time.time() < _RETRY_AFTER.get(slot, 0.0):
        # A probe for this pair failed recently. Returning the known value — usually 0, and a
        # stale one when we have it — costs nothing and keeps the turn fast; the cascade reads
        # both the same way it would have anyway.
        return known
    call = probe or windowprobe.probe
    learned = call(base, key, model, timeout=PROBE_TIMEOUT_S)
    if learned > 0:
        windowcache.put(base, model, learned)
        _RETRY_AFTER.pop(slot, None)

        return learned

    _RETRY_AFTER[slot] = time.time() + RETRY_AFTER_S

    return known
