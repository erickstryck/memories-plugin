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
    try:
        round_no = int(state.get("round", 0) or 0)
    except (TypeError, ValueError):
        round_no = 0
    seen = state.get("seen")
    seen = seen if isinstance(seen, dict) else {}
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


#: Rounds between dead-session sweeps. A cheap, occasional glob: once every 20 rounds is
#: enough to keep the directory from growing, and it does not pay for a `glob` on every
#: prompt. The number lives HERE, next to `purge_dead`, and not in an adapter: the cadence is
#: part of the job, and two hosts that each picked their own would be one more setting that
#: only looks shared.
PURGE_EVERY_ROUNDS = 20


def sweep_if_due(state_dir, round_no, days: float = 7.0,
                 every: int = PURGE_EVERY_ROUNDS) -> int:
    """`purge_dead` on the shared cadence. Returns how many files went; 0 when not due.

    Both hosts call this instead of testing `round_no % 20` themselves. The sweep used to be
    the claude-code hook's inline arithmetic, so when the purging moved into `core` for both
    hosts to share (spec §4) the hermes adapter inherited nothing and its state directory
    grew one file per session forever.

    Nothing here raises, like everything else in this module: an unswept file is a
    housekeeping cost, and it must never become the reason a recall failed.
    """
    try:
        round_no = int(round_no)
        every = int(every)
    except (TypeError, ValueError):
        return 0
    if every <= 0 or round_no <= 0 or round_no % every:
        return 0

    return purge_dead(state_dir, days=days)


def due(turn: int, interval: int) -> bool:
    """Whether the checkpoint is due on this turn. A non-positive interval disables it.

    `turn` and `interval` are coerced the same way `next_round` coerces `round`: a
    numeric string is usable, anything that is not never fires rather than raising —
    hermes hands this function a number we do not control.
    """
    try:
        turn = int(turn)
        interval = int(interval)
    except (TypeError, ValueError):
        return False
    if interval <= 0:
        return False

    return turn % interval == 0
