---
name: memory
description: Long-term memory in Qdrant, via `qctx memory` — BOTH DIRECTIONS. Search first, then save. Use it when the user asks you to save/persist/remember something; AND — mandatory — before answering a non-trivial question, starting an investigation, asserting a fact about the codebase, an SDK, a platform or a past decision, proposing a design, or reversing an earlier conclusion. If precedent might exist, search before answering.
---

# memory

Long-term memory, across sessions, in the configured semantic archive. Two
directions, and reading is the one that gets forgotten:

- **SEARCH** — before asserting. Cheap, and it is what keeps you from re-deciding
  what has already been decided.
- **SAVE** — persist a durable fact, one atomic fact per record, deduplicated.

**Principle:** *search before asserting, save only what will still matter later.*

## Automatic recall already guarantees the floor — the depth is yours

The host runs a search before you see the text and injects the relevant memories — a
`UserPromptSubmit` hook in claude-code, `prefetch()` in hermes-agent, the same search and
the same block either way. It runs on most turns, not all: it deliberately skips text under
12 characters, bare acknowledgements ("ok, pode continuar") and bare slash commands, because
none of them names a subject. When it does run, you do not need to repeat that same
generic search.

Read the state of the injected block. There are FOUR, and the last one is the one people
forget because it looks like nothing:

| injected block | what it means |
|---|---|
| memories listed | the archive was consulted; treat each one by the table below |
| "no memory above the relevance cutoff" | the archive **was** consulted. If the same block also says the judgement was PARTIAL, that is not evidence of absence — search again with a different angle. Without that caveat, the search was complete: do not repeat it. |
| "automatic recall — UNAVAILABLE" | the search **did not run** (infra down, or the plugin is not configured). This is NOT evidence that no precedent exists. Never claim something is unprecedented on the strength of that turn; try `qctx memory recall` yourself. |
| **no block at all** | recall did not run: a short or trivial prompt, or it is disabled. Nothing was searched, and nothing claims otherwise. If the subject could have precedent, this is on you. |

**What automatic recall does not cover, and stays entirely yours:**

- **Mid-turn.** No new prompt arrives when you close out a step, open a sub-subject or
  reach a design decision 20 minutes into the task. Search again at each step. Memory
  written by a parallel session only appears to whoever re-reads.
- **Opening pointers.** Recall delivers the hit; opening the ids it cites is on you.
  An index is a pointer, not an answer.
- **Angles the prompt text does not contain.** The queries are built from the prompt's
  words. A facet the user did not name was not searched.
- **All of the writing.**
- **Other archives.** Collections belonging to other systems are never in automatic
  recall. Read one on request with `qctx memory search-collections "<topic>" --collections
  <name>`; it never writes.

## 1. SEARCH — before answering

```bash
qctx memory recall "<topic, in natural language>"     # two stages, with re-rank
qctx memory find "<topic>" --limit 8                  # pure dense, cheaper
```

**Triggers** — search when any of these holds:

- The user asks **how something works** in the codebase, an SDK, a platform or the
  infra.
- You are about to **assert a fact** about third-party behaviour (a library, an API).
- You are about to **propose a design or a decision** in an area with history.
- The user says something of the form *"didn't we already discuss this?"*, *"don't we
  have this already?"* — that is a search instruction, not a rhetorical question.
- You are **starting an investigation** — search before reading code.
- You are about to **reverse or contradict** something you said earlier.
- A **review or a subagent reports a finding** in an area with history.

Search by **topic**, not by exact symbol: the archive is semantic. Two or three
different angles beat one long query.

| situation | action |
|---|---|
| a precedent settles it | apply it, say it is precedent and cite the id. Do not re-derive. |
| the memory records a **decision or veto** by the user | it holds. Do not re-propose what was vetoed; if you think it should change, say explicitly that it is a reversal, with the new evidence. |
| the memory cites a file, a line, a flag or a version | **verify it against the current tree** before acting. |
| the memory contradicts what you measured | the measurement wins — and then **fix the memory** (§2.3). Never leave a knowingly wrong memory standing. |
| nothing relevant | carry on, and consider whether the answer you are about to produce deserves saving. |

### The failure modes this exists to prevent

Observed, repeatedly:

- **Re-proposing a vetoed design.** A decision the user had already made twice was
  reintroduced because the memory holding it was never opened.
- **Asserting platform behaviour a memory had already measured.** The user had to
  bring the correction.
- **Citing an index and never opening it.** The index was cited several times while the
  ids it points at went unread.

In all three, the cost was not the mistake — it was the user having to be the one who
noticed.

## 2. SAVE

### 2.1 What to save

- `user` — who the user is: role, expertise, stable preferences.
- `feedback` — how they want you to work (corrections and confirmed approaches);
  **include the why**, otherwise the rule gets reapplied out of context.
- `project` — goals, constraints and in-flight decisions that do not follow from the
  code or from git. Convert relative dates to absolute ones.
- `reference` — external pointers (URL, dashboard, ticket) and **measured** behaviour
  of a platform or an SDK.

**Especially:** behaviour you had to **measure** — a probe, a grep, a branch you ran.
That is the expensive knowledge, and what keeps the next session from re-measuring.
Record **how** it was measured.

**Discard:** passing conversation, one-off detail, volatile state, and anything already
in the repository, in git or in the project instructions.

### 2.2 Dedupe before writing

```bash
qctx memory find "<short query for the fact>"
```

A close match (high score, same fact) is an **update** on that id, not a new record.

### 2.3 Fix what is wrong

When a memory turns out to be wrong or obsolete, **update it in the same pass** — do
not write a new one competing with it. Say what the old version claimed, what was
measured and when. An archive with two contradictory memories is worse than one with a
corrected memory, because whoever reads it later does not know which holds. This
applies to a memory **you** wrote today too.

### 2.4 Write

```bash
qctx memory store "<atomic fact>" --type reference --project X --area Y
qctx memory update <id> --text "<corrected version>"
qctx memory delete <id>
```

**One atomic fact per record.** A whole paragraph as a single memory ruins the search,
because the vector becomes the average of several subjects. Metadata always with `type`
and a date.

### 2.5 Confirm

A short list of what was saved or updated, each item with its id, in the user's
language.

## Commands

| action | command |
|---|---|
| search with re-rank | `qctx memory recall "<topic>"` |
| dense search | `qctx memory find "<topic>" --limit N` |
| read by id | `qctx memory get <id>` |
| create | `qctx memory store "<text>" --type T` |
| update | `qctx memory update <id> --text "..."` |
| delete | `qctx memory delete <id>` |
| list | `qctx memory list --limit N` |
| read another system's archive | `qctx memory search-collections "<topic>" --collections <name>` |
| see config | `qctx config show` · `qctx collections list` |

`--json` on any command returns structured output.

Where the host exposes memory **tools** (`memory_recall`, `memory_find`, `memory_store`,
…, as hermes-agent does), they are the same operations over the same archive as the
commands above — use whichever the host gives you. Nothing in this skill changes.

## Common mistakes

**When reading:**
- Answering from what you think you know, when the archive holds the measurement.
- Treating *"didn't we already discuss…"* as rhetoric instead of a search instruction.
- Reading an index memory and never opening the ids it points at.
- Trusting a `file:line` from a memory without checking the current tree.

**When writing:**
- Storing a whole paragraph as one record → break it into atomic facts.
- Creating a near-duplicate → search first, then update.
- Saving volatile state ("we are on line 42").
- Forgetting metadata → always `type` and a date.
- Leaving a disproved memory standing with a new one beside it.
