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
    """Read the state, or a fresh one. Any failure is a fresh session, never an error.

    A `seen` that is not a dict — a hand-edited file, an older format, a bad write — is
    REPLACED here, and that placement is the point. `core.blocks.split_by_budget` already
    refuses to crash on one, but it does so on a LOCAL substitute, deliberately: "the caller
    owns persistence", so it never reaches back into `state["seen"]`. Something therefore has
    to heal the persisted copy, or every later round loses the dedup memory again, forever —
    every recalled memory reinjected in full every turn, spending the whole char budget on
    repeats.

    That healing used to be a guard inside `hooks/recall.py`, which is exactly why the hermes
    adapter never had it: measured over 3 rounds against `{"round": 3, "seen": "corrupted"}`,
    claude-code's file came back a dict and hermes' did not. Here, it is the one function both
    hosts read state through, so a third host inherits it instead of having to remember it.
    """
    try:
        state = json.loads(Path(path).read_text())
    except Exception:
        return {"round": 0, "seen": {}}
    if not isinstance(state, dict):
        return {"round": 0, "seen": {}}
    state.setdefault("round", 0)
    if not isinstance(state.get("seen"), dict):
        state["seen"] = {}

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


#: Every per-session file this plugin writes, by glob. A session leaves more than one trace —
#: the recall state one hook writes, and the checkpoint counter the other does — and a sweep
#: that knew only about the first left the second accumulating forever, which is exactly the
#: growth `purge_dead` exists to stop. Measured on a real install: 61 `checkpoint-*.count`
#: against 29 `recall-*.json`, with the oldest counter three weeks older than the oldest
#: recall file. The list lives HERE, next to the sweep, so a host that starts writing a third
#: kind of per-session file adds it in one place rather than growing a directory in silence.
#: It stays narrow on purpose: the log is not session state.
SESSION_FILE_PATTERNS = ("recall-*.json", "checkpoint-*.count")


def purge_dead(state_dir, days: float = 7.0, pattern=None) -> int:
    """Delete state files untouched for `days`.

    Each session creates a file and nothing removed them: the directory grew forever. A
    session idle for a week is not coming back, and if it does the cost is starting with an
    empty `seen` — the worst effect is one memory reinjected once.

    `pattern` takes one glob or several; it defaults to every per-session file this plugin
    writes (see `SESSION_FILE_PATTERNS`).
    """
    if pattern is None:
        patterns = SESSION_FILE_PATTERNS
    elif isinstance(pattern, str):
        patterns = (pattern,)
    else:
        patterns = tuple(pattern)

    removed = 0
    try:
        # INSIDE the try, and coerced. It used to sit above it, which made this the one
        # function in a module whose docstring says nothing here raises: `days=None` or
        # `days="x"` raised TypeError before the guard it was standing next to could catch
        # it. `sweep_if_due` coerces `round_no` and `every` and forwarded `days` untouched,
        # so the tolerance stopped exactly one argument short. No caller passes a bad value
        # today; the contract is what callers rely on, and it has to hold without them
        # checking who calls it.
        cutoff = time.time() - float(days) * 86400
        for glob in patterns:
            for path in Path(state_dir).glob(glob):
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


def due_since(turn, last_fired, interval) -> bool:
    """Whether the checkpoint is OWED on this turn: due now, or due on a turn that was missed.

    WHY EXACT DIVISIBILITY IS NOT ENOUGH, on one host and not the other. The claude-code
    checkpoint is its own hook with its own counter, which advances once per event, so it can
    never step over a multiple. The hermes nudge rides inside `prefetch`, and the host gates
    `prefetch` behind a trivial-prompt filter while calling `on_turn_start` unconditionally —
    so a turn that is both due and trivial (`/status`, "thanks", "ok") lost the checkpoint
    entirely rather than deferring it, and the next one would not be a multiple either.

    Owed, not accumulated: one nudge covers every turn since the last one, because the
    procedure is the same however many turns were skipped, and repeating it per missed turn
    would be noise on exactly the sessions that already skipped it for being short.
    """
    try:
        turn = int(turn)
        last_fired = int(last_fired)
        interval = int(interval)
    except (TypeError, ValueError):
        return False
    if interval <= 0 or turn <= 0:
        return False

    # The most recent turn on which it was due. Zero means none has come round yet.
    latest_due = (turn // interval) * interval

    return latest_due > 0 and latest_due > last_fired
