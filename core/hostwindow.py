"""The context window a host reported for a session, recorded for the guard to read.

WHY A RECORD AND NOT A QUESTION. The big-file guard needs the window of the model selected
NOW, and neither guard can ask for it. The claude-code hook payload carries no model and no
window (measured on 2.1.282: only the statusLine receives `context_window_size`), and the
hermes guard is a shell hook, a subprocess that does not import hermes. So the side of each
host that does know the window publishes it here, and the guard only reads:

  * claude-code: the statusLine command, which runs when the session opens, after every
    assistant message and right after a `/model`;
  * hermes: the memory provider, in its `pre_llm_call` hook, with hermes' own resolution.

ONE FILE PER SESSION, `window-<session>.json`, so two sessions on different models never
answer for each other, and the sweep that already removes dead per-session files removes
these too (`core.session_state.SESSION_FILE_PATTERNS`).

A GUESS IS RECORDED AS ONE. hermes answers 256,000 for a model it does not know, which by
value cannot be told apart from a real 256K window. The publisher says when it is the
fallback, and `core.windows` skips a guess: one too small would make the guard refuse reads
the real window has room for.

NOTHING HERE RAISES. A failed write is a `False`, an unreadable record is `None`, and
`None` means "the host said nothing", which the guard already knows how to handle.
"""
import json
import time
from typing import NamedTuple

from . import names, statefile
from .knobs import state_dir

#: The glob the per-session sweep uses; the names below must keep matching it.
PATTERN = "window-*.json"


class HostWindow(NamedTuple):
    model: str
    window: int
    source: str
    guess: bool
    at: float


def _path(session_id: str):
    return state_dir() / f"window-{names.safe(session_id)}.json"


def _positive_int(value) -> bool:
    # `bool` is an int subclass, and `True` is not a window.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def publish(session_id: str, model: str, window: int, source: str, guess: bool = False) -> bool:
    """Record what the host reported for this session. True when it landed."""
    if not session_id or not _positive_int(window):
        return False
    record = {"model": str(model or ""), "window": window, "source": str(source or ""),
              "guess": bool(guess), "at": time.time()}
    try:
        return statefile.write_json(_path(session_id), record)
    except Exception:  # noqa: BLE001 -- a publisher must never cost its host anything
        return False


def _parse(raw) -> HostWindow | None:
    if not isinstance(raw, dict) or not _positive_int(raw.get("window")):
        return None
    at = raw.get("at")
    if not isinstance(at, (int, float)) or isinstance(at, bool):
        return None

    return HostWindow(model=str(raw.get("model") or ""), window=raw["window"],
                      source=str(raw.get("source") or ""), guess=bool(raw.get("guess")),
                      at=float(at))


def read(session_id: str) -> HostWindow | None:
    """What the host last reported for this session, or None."""
    if not session_id:
        return None
    try:
        return _parse(json.loads(_path(session_id).read_text()))
    except Exception:  # noqa: BLE001 -- unreadable is the same answer as absent
        return None


def newest(source: str) -> tuple[str, HostWindow] | None:
    """(file stem, record) of the most recent record that `source` published, or None.

    For `qctx setup`, which reports what each host last said. The stem is the sanitised
    session id, enough to tell sessions apart in a report.
    """
    best = None
    try:
        paths = list(state_dir().glob(PATTERN))
    except OSError:
        return None
    for path in paths:
        try:
            record = _parse(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001
            continue
        if record is None or record.source != source:
            continue
        if best is None or record.at > best[1].at:
            best = (path.stem[len("window-"):], record)

    return best
