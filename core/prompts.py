"""The text this package injects into a host's context.

It lives in the core, not in an adapter, because every host injects the SAME words. When
these lived in hooks/ — the claude-code adapter — a second host could only get them by
copying, and a copy drifts on the first fix. This repo already paid that bill: three
copies of the two-stage pipeline, with the re-rank scale normalization present in one and
missing from the other.

Neither string is decoration. INSTRUCTIONS exists because a memory delivered without its
rules of use gets applied out of context — a stale `file:line` acted on as if current, a
vetoed design re-proposed. CHECKPOINT_PROCEDURE is deliberately self-sufficient: a
one-line "save what matters" reminder produces vague, duplicated, metadata-less records,
and the cost only appears months later when a search returns three contradictory versions
of the same fact.
"""

#: Rules that travel WITH the recalled memories, every time.
INSTRUCTIONS = """How to use this, without exception:
- A precedent or a veto from the user PREVAILS. Do not re-derive, do not re-propose \
what was vetoed; if you think it should change, say explicitly that it is a reversal.
- A memory that cites a file, a line, a flag or a version: VERIFY it against the \
current tree before acting. It reflects what was true when it was written.
- A memory that contradicts what you just measured: the measurement wins — and then \
FIX the memory, do not let the two coexist.
- A facet of the subject not covered below: run an explicit search from another angle."""

#: The write-side counterpart of INSTRUCTIONS: the procedure injected every N
#: interactions to distil the conversation into long-term memory. `{count}` and
#: `{interval}` are filled by `.format()`; the metadata example's braces are doubled
#: for the same reason.
CHECKPOINT_PROCEDURE = """[memory checkpoint — writing to the long-term archive]
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

The commands are in the memory skill or in the memory tools, whichever this host gives you;
the essence of the procedure is here."""
