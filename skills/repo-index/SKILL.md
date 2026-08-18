---
name: repo-index
description: Searches indexed code repositories semantically via `qctx repos` — across ONE repo or ALL of them at once, instead of grepping a tree or reading files to find where something lives. Use it when the question is "where is X handled", "does any project here do Y", "which repo mentions Z"; when the user asks to index a project for later; and before concluding a subject is absent from their codebases. Searching every repository at once is the thing grep on one tree cannot do.
---

# repo-index

Finding the four functions that handle retries, across six projects, should not cost
six `grep -r` sweeps and a dozen file reads. `qctx repos` slices each repository into
chunks **outside** your context and returns only the ones that answer — with the file
path and line range, so the read that follows is targeted.

The capability grep does not have is the cross-repository one: **one question, every
project, ranked together.**

## The scope is the working directory, and the refusal is deliberate

With neither `--repo` nor `--all`, the search resolves the repository from the git root
of the current directory. Outside an indexed repository it **refuses and names both
remedies** rather than silently searching everything — a silent widening is the noise the
scoped default exists to avoid.

```bash
qctx repos search "how are retries handled"              # this repository
qctx repos search "how are retries handled" --all        # every repository, grouped
qctx repos search "retry backoff" --repo other-project   # one named repository
qctx repos search "auth middleware" --all --limit 12     # more groups (see below)
```

`--limit` means two different things on purpose, and knowing which saves a confused
result: **scoped**, there is only one group, so it caps the hits you get back; **`--all`**,
it caps how many REPOSITORIES come back. When more matched than the limit allowed, the
output says so — that line is never absence, it is "the rest were never judged".

## Reading the output

Each hit is `score  path:start-end`, grouped by repository. A `[stale]` flag means the
file on disk changed after it was indexed: the chunk still points at the right place, but
read the file rather than trusting the text. **It has a repair**, and only the changed files
cost anything:

```bash
qctx repos refresh my-project     # reindex what changed; unchanged files are free
```

A file that was DELETED comes back as `missing` and its chunks are kept — deleting an archive
here is explicit and permanent (`repos drop`), so `missing` is a signal to decide, not an
automatic removal.

To stop doing it by hand, `qctx repos install-hook my-project` writes a git `post-commit` that
refreshes in the background after each commit. It never commits, stages or pushes anything, it
cannot fail a commit, and it refuses to overwrite a `post-commit` another tool already owns.

**When nothing matches, the output says so in a sentence.** An empty screen would read as
"there is nothing about this", which is the one conclusion this must never let you draw
by accident.

## Indexing

Registering is a separate, explicit step: `add` refuses a repository the registry does not
know, so a typo becomes a refusal instead of a second archive under a misspelled name.

```bash
qctx repos register my-project --label "the API server"   # declare it first
qctx repos add my-project src/**/*.py                     # index exactly these files
qctx repos list                                           # what exists, and its state
```

`add` takes **the files you name** — it does not walk a tree, so what gets indexed is
always a decision someone made. Re-indexing a file replaces its chunks rather than
accumulating them, so running `add` again after edits is the normal way to refresh.

The index is **permanent** and survives sessions and reboots. Nothing expires it; only
`qctx repos drop <name> --yes` removes it, and there is no undo.

## What `list` tells you beyond the names

It reports two kinds of divergence, and neither is decoration:

- **divergent** — chunks exist under a name the registry does not know. Such a repo cannot
  be dropped by name, so listing it is the only way it is visible at all.
- **emptied** — an entry claims chunks over an archive holding none, which is what an
  interrupted `drop` leaves behind. The repair is to re-index, not to drop.

A freshly registered repository that was never indexed is neither, and is silent.

## When to reach for this instead of grep

| Question | Tool |
|---|---|
| "where is X handled, in any of my projects" | `repos search --all` |
| "which repo mentions Z" | `repos search --all` |
| "where is X in THIS project, by meaning not by name" | `repos search` |
| an exact string, symbol, or identifier you know verbatim | `grep` |
| a file you can already name | read it |

Semantic search finds the code that *does* the thing when you cannot guess what it is
called. It is the wrong tool for a literal string you already know — grep is faster and
exact. Both are right, for different questions.

## On the other host

Every operation above exists as a tool on hermes-agent with the same names and the same
meaning: `repos_search` (`repo=`, `across=`, `limit=`), `repos_list`, `repos_register`,
`repos_add`, `repos_drop` (`confirmed=`). The refusals are identical, and they are decided
in one place for both hosts — see `core/repos.py`.
