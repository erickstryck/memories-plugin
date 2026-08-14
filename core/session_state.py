"""Per-session state, and when the checkpoint is due.

Both hosts keep the same state and want the same cadence; only the SOURCE of the turn
number differs — the claude-code hook counts in a file because the host does not tell it,
hermes hands `turn_number` to `on_turn_start`. So the decision lives here and each adapter
supplies the number.

Nothing in this module raises. State is a convenience — it decides pointer-versus-full
reinjection and nothing more — so losing it must never cost a search that already
succeeded. An unwritable state directory once made the recall hook discard results it was
already holding and tell the model the search had not run: a safe direction with the wrong
message.
"""
import json
import time
from pathlib import Path

#: Rounds before a memory is reinjected in full instead of as a one-line pointer.
#: The context may have been compacted in between, so a pointer eventually stops being
#: enough to recover the content.
REINJECT_AFTER = 8


def load(path) -> dict:
    """Read the state, or a fresh one. Any failure is a fresh session, never an error."""
    try:
        state = json.loads(Path(path).read_text())
    except Exception:
        return {"round": 0, "seen": {}}
    if not isinstance(state, dict):
        return {"round": 0, "seen": {}}
    state.setdefault("round", 0)
    state.setdefault("seen", {})

    return state


def save(path, state: dict) -> None:
    """Persist the state. `None` means state was unavailable this round, which is not an error."""
    if path is None:
        return
    try:
        Path(path).write_text(json.dumps(state))
    except Exception:
        pass


def next_round(state: dict) -> int:
    """Advance the round counter and return it. A corrupted counter restarts at 1."""
    try:
        current = int(state.get("round", 0))
    except (TypeError, ValueError):
        current = 0
    state["round"] = current + 1

    return state["round"]


def prune(state: dict, reinject_after: int = REINJECT_AFTER) -> int:
    """Drop `seen` entries that can no longer change a decision.

    An entry only matters while `round - seen < reinject_after`: past that the memory comes
    back in full anyway, so keeping it just occupies space. Without pruning, a long session
    accumulates one entry per memory per round forever.
    """
    round_no = int(state.get("round", 0) or 0)
    seen = state.get("seen", {})
    stale = [mid for mid, r in seen.items()
             if not isinstance(r, int) or (round_no - r) >= reinject_after]
    for mid in stale:
        seen.pop(mid, None)

    return len(stale)


def purge_dead(state_dir, days: float = 7.0, pattern: str = "recall-*.json") -> int:
    """Delete state files untouched for `days`.

    Each session creates a file and nothing removed them: the directory grew forever. A
    session idle for a week is not coming back, and if it does the cost is starting with an
    empty `seen` — the worst effect is one memory reinjected once. The pattern is narrow on
    purpose: the log is not session state.
    """
    cutoff = time.time() - days * 86400
    removed = 0
    try:
        for path in Path(state_dir).glob(pattern):
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
    except Exception:
        pass

    return removed


def due(turn: int, interval: int) -> bool:
    """Whether the checkpoint is due on this turn. A non-positive interval disables it."""
    if interval <= 0:
        return False

    return turn % interval == 0
