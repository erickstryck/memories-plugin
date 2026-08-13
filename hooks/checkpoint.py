#!/usr/bin/env python3
"""CHECKPOINT hook: every N interactions, injects the writing procedure.

The write-side counterpart of `recall.py`. This hook stores nothing — it hands the
model the complete procedure at the moment there is accumulated conversation to
distil.

The text is deliberately SELF-SUFFICIENT. A one-line reminder ("save whatever is
durable") produces vague, duplicated, metadata-less memory, and the cost shows up
months later, when a search returns three contradictory versions of the same fact and
nobody knows which one holds. Whoever reads the block has to be able to act without
opening anything else.

Configuration:
    QCTX_CHECKPOINT_INTERVAL   interactions between checkpoints (default 5)
    QCTX_CHECKPOINT_DISABLED   "1" turns it off
    QCTX_STATE_DIR             where to keep the counter
"""
import json
import os
import sys
from pathlib import Path

INTERVAL = int(os.environ.get("QCTX_CHECKPOINT_INTERVAL")
                or os.environ.get("REMEMBER_INTERVAL") or "5")
STATE_DIR = Path(os.environ.get("QCTX_STATE_DIR") or (Path.home() / ".memories-plugin" / "state"))

PROCEDURE = """[memory checkpoint — writing to the long-term archive]
Interaction {count} of this conversation (every {interval}). Do the checkpoint NOW, in one
short pass, without straying from the task in progress. If nothing durable came up since
the last checkpoint, save nothing and say so in one line — an empty memory is better
than filler memory.

1. SWEEP the conversation since the last checkpoint and list the candidates. What
   qualifies, by type:
   - `user` — who the user is: role, expertise, stable preferences.
   - `feedback` — how they want you to work (corrections AND confirmed approaches).
     ALWAYS include the why; without the reason, the guidance gets reapplied out of
     context in the next session.
   - `project` — goals, constraints and in-flight decisions that do NOT follow from the
     code or the git history. Convert relative dates to absolute ones.
   - `reference` — external pointers (URL, dashboard, ticket) and MEASURED behaviour of a
     platform, SDK or library.
   WORTH MORE THAN ANYTHING ELSE: behaviour you had to MEASURE — a probe, a grep, a
   branch you ran. That is the expensive knowledge, and it is what keeps the next session
   from re-measuring. Record HOW it was measured, so it can be redone.
   DISCARD: passing conversation, one-off detail, volatile state ("we are on line 42"),
   and anything already in the repository, in git or in the project instructions.

2. DEDUPE before writing. One short search per candidate. A close match (high score,
   same fact) is an UPDATE on that id, NOT a new record.

3. FIX what is wrong, in the same pass. If a memory turned out to be wrong or obsolete —
   including one YOU wrote today, if a measurement or a review disproved it — update it
   saying what the old version claimed, what was measured and when. Never leave the wrong
   one standing with a new one beside it: two contradictory memories are worse than one
   corrected memory, because whoever reads them later has no way to know which wins.

4. WRITE one ATOMIC FACT per record. A whole paragraph as a single memory ruins semantic
   search, because the vector becomes the average of several subjects.
   Mandatory metadata: {{"type": "user|feedback|project|reference",
   "date": "YYYY-MM-DD", "source": "conversation"}}. Add `project`, `area`, `corrected`
   or `supersedes` when they help filter later.

5. CONFIRM in a short list: what was saved or updated, each item with its id, in the
   user's language.

The commands are in the memory skill; the essence of the procedure is here."""


def main() -> None:
    if os.environ.get("QCTX_CHECKPOINT_DISABLED") == "1":
        return

    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    session = "".join(c if c.isalnum() or c in "-_" else "_"
                     for c in str(data.get("session_id") or "default"))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    counter = STATE_DIR / f"checkpoint-{session}.count"

    try:
        n = int(counter.read_text().strip())
    except Exception:
        n = 0
    n += 1
    counter.write_text(str(n))

    if INTERVAL <= 0 or n % INTERVAL != 0:
        return  # silent on the intermediate interactions

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": PROCEDURE.format(count=n, interval=INTERVAL),
        }
    }))


if __name__ == "__main__":
    main()
