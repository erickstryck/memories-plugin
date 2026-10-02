"""How large is the context window of the model selected NOW? The host knows; we ask it.

THE HOST REPORTS FIRST. Only the host knows which model is selected at this moment and how
large its window is, and both hosts expose that to a process outside them in exactly one
place, measured on 2026-10-02:

  * claude-code hands `context_window.context_window_size` to the statusLine command, which
    runs when the session opens, after every assistant message and right after a `/model`
    (the hook payloads carry no model and no window);
  * hermes resolves the window itself (`agent.model_metadata.get_model_context_length`), and
    the memory provider runs inside hermes, in a hook called at the start of every turn.

Each of those publishes per session (`core.hostwindow`); this module only reads.

THE ORDER, and each step is consulted only when the one before it did not answer:

  1. The window the host reported for this session, unless it was a GUESS. hermes answers
     256,000 for a model it does not know, a number that by value cannot be told apart
     from a real 256K window, and one too small makes the guard refuse reads the real
     window has room for. So the publisher marks it, and it does not count.
  2. A window an endpoint reported, from the cache the hermes hooks fill. Used even when
     STALE: a day-old value is still a measured one.
  3. `context_window` in the config. The LAST resort, by the user's decision: a number
     typed once cannot follow a `/model`. It exists for a host that reports nothing, such
     as `claude -p`, which runs no statusLine.
  4. 0, and 0 is load-bearing: the caller must ALLOW the read when the window is unknown.
     Blocking on a guessed window is the one failure this guard must not produce.

NO TABLE OF MODEL NAMES, by the user's decision. A name says nothing reliable about the
window (the claude-code transcript records the bare `claude-opus-5-5` for a 1M session),
and a fixed list cannot follow the models the hosts add. The table this file held until
v1.2.0 was a list of CEILINGS for that reason, and `core.bigfile.decide` still carries the
other half of that lesson: `used >= window` refutes the number, and the read is allowed.

THIS FUNCTION NEVER REACHES THE NETWORK. It runs before every file read.
"""
from . import hostwindow


def _declared(cfg) -> int:
    declared = getattr(cfg, "context_window", 0)
    try:
        declared = int(declared)
    except (TypeError, ValueError):
        return 0

    return declared if declared > 0 else 0


def window_for(model: str, cfg, endpoint: str = "", session_id: str = "") -> int:
    """Tokens the context window of this session holds, or 0 when we do not know.

    `session_id` empty means "no host report to look for"; `endpoint` empty means "this
    host has no endpoint to offer" (claude-code), and skips the cache entirely.
    """
    reported = hostwindow.read(session_id) if session_id else None
    if reported is not None and not reported.guess:
        return reported.window

    if endpoint:
        from .windowcache import get as cached      # local: keeps this import off the
        window, _fresh = cached(endpoint, model)    # hot path for hosts that pass nothing
        if window > 0:
            return window

    return _declared(cfg)
