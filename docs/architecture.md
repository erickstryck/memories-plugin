# Architecture and deep dives

The [README](../README.md) is the install path and [install.md](install.md) is what
each install step does. This file is the design behind them: how the search works,
the two hosts, the big-file read guard, the layout and the design decisions. Read
it when you are changing the code, or when you need to know why something is the
way it is.

## Three archives, three lifecycles

The separation is structural: distinct, configurable collections, not a convention:

| archive | what it holds | expires |
|---|---|---|
| **memory** | a curated atomic fact (decision, preference, measured behaviour) | no |
| **library** | a whole document kept for reference | no |
| **temporary** | a document opened for one task | yes, TTL |

Why not put everything in one collection: a long document becomes dozens of verbose
chunks. Mixed in with curated facts, they win on volume in every search and drown
precisely the archive that matters most. And the temporary archive is destroyable by
construction (there is a command that deletes the entire collection), so a permanent
archive cannot live there. The configuration refuses to point two roles at the same
collection.

## How the search works

Two stages, with distinct jobs:

1. **Dense** (`bge-m3` or another embedder): sweeps the whole archive by vector
   similarity. Cheap, approximate, and **practically indifferent to language**: a
   question in Portuguese finds an English document (measured: 0.460 against 0.475
   for the same question in the two languages).
2. **Cross-encoder** (`bge-reranker-v2-m3` or another): reads the question and the
   chunk in the SAME pass, with cross attention. It judges far better, and for that
   reason has no precomputable vector: it is one forward pass per pair, cost linear
   in the total token count.

Two measured findings shaped the design, and both are silent failures if ignored:

**The re-rank scale depends on the server.** The same model returns a sigmoid (0..1)
on one server and a raw logit on another. The same irrelevant document gave `1.6e-05`
and `-11.04`, the second being exactly `logit(1.6e-05)`. A cutoff calibrated on one
scale is inert on the other. The core detects it by range and normalizes, so the
calibrated number stays valid on any server.

**The cross-encoder collapses on a cross-lingual pair.** The same question about the
same English document: `0.2073` in English, `0.0004` in Portuguese, a 500x gap. It matches
language, not just semantics. Consequences in the design:

- In **document** search, the re-rank **orders but does not veto**: whoever asks has
  already chosen the document, and silence is worse than imperfect order. A collapse
  is detected (best score below `0.01`) and the dense order takes over, with a warning.
- In **memory** search, the re-rank keeps the veto: there precision matters more than
  reach, and a false positive pollutes the agent's context.

## Hosts

The same core serves two hosts, with the same operations and the same configuration.

| | claude-code | hermes-agent |
|---|---|---|
| adapter | `hooks/` | `hosts/hermes/` |
| install | `claude plugin marketplace add .` | a clone or a symlink at `$HERMES_HOME/plugins/memories` |
| recall | `UserPromptSubmit` hook | `prefetch()` |
| checkpoint | second `UserPromptSubmit` hook | rides along in `prefetch()` on the Nth turn |
| big-file guard | `PreToolUse` hook on `Read` | `pre_tool_call` shell hook, matcher `read_file` |
| operations | `qctx` CLI + 3 skills | 22 model-invokable tools, the same CLI, and the same 3 skills |
| what the model is told | the 3 skills | `system_prompt_block()`, from the same `core/prompts.py`, pointing at `memories:memory` |
| configuration | `~/.config/memories-plugin/config.json` | the same file |
| credentials | the environment | the environment, or `$HERMES_HOME/.env` |

Equivalence is not a claim in this table: `tests/test_host_equivalence.py` renders every
block state through both adapters and requires byte-identical output.

**The skills reach the two hosts by different routes, and that asymmetry is load-bearing.**
claude-code discovers `skills/` by itself. hermes does not: its loader hands the plugin a
context and registers only what `register()` explicitly asks for, so the adapter walks
`core/skills.py` and hands over each one. hermes also keeps plugin skills out of the
available-skills list it injects (explicit loads only), so registering a skill
makes it loadable but not findable. That is why `system_prompt_block()` names
`memories:memory` outright: without the pointer the model is never told the name it would
have to ask for. The skills were reachable on claude-code and unreachable on hermes for
exactly as long as the adapter registered the provider and nothing else.

The hermes install is a **symlink** into `$HERMES_HOME/plugins/memories`, one level deep and no
deeper: hermes' loader (`plugins/memory/__init__.py`) scans `$HERMES_HOME/plugins/<name>/`,
and a provider one directory further down is not discovered at all. That is measured
against the installed loader, not read off the documentation: `tests/test_hermes_provider.py`
drives it with a temp `HERMES_HOME` and requires the deeper layout to come back unfound.

hermes activates exactly ONE external memory provider, so installing this one **replaces**
whatever `memory.provider` names. The provider it replaces is disabled by configuration,
never deleted, and its own collection stays reachable read-only, outside automatic recall:

```bash
qctx memory search-collections "<topic>" --collections hermes_memory
```

Each host's install, step by step, with what each step costs and each gotcha that
was measured: [install.md](install.md).

### What the cutover script checks, and why

`is_available()` gates initialization, so a provider that reports unavailable is never
initialized and any diagnostic it might log from `initialize()` is unreachable. hermes
0.20.1 warns and appends the provider's `unavailable_reason()`
(`agent/agent_init.py`), but the failure it cannot describe at all is the one that matters
most here: **the two API keys live only in the environment.** `config set` refuses to write
a secret to `config.json`, and `is_available()` does not look at the keys; it checks the
Qdrant URL, the embedding endpoint and the collection. So a hermes started from a shell
without them reports a perfectly healthy provider and then fails every single search.

The script therefore checks the environment, not only the files: **every** spelling each key
accepts (three for the Qdrant key, `QCTX_QDRANT_API_KEY`, `QDRANT_SERVICE_API_KEY` and
`QDRANT_API_KEY`) in this shell **and** in `$HERMES_HOME/.env`, which hermes loads itself
(`hermes_cli/env_loader.py`, from `run_agent.py` and `cli.py`) and which is the only one of
the two a systemd/gateway hermes has. It prints the variable names, never their values.
Checking fewer aliases than the core accepts would be worse than not checking: the core
would resolve the key fine while the script told the operator to export what they already had.

And it asks the same question about the settings that are **not** secrets. The URLs and the
collection names can live in `config.json` (which both hosts read with no shell involved),
so the script resolves the configuration with the environment layer removed
(`core.load(env={})`) and reports what a hermes that inherits no shell would be missing,
pointing at `qctx config set qdrant-url …` rather than at `.env`. Without that second half
the only remedy on offer fixes the keys and leaves a gateway hermes memory-less on the URLs.

It also reports where the provider being replaced actually lives, whether the plugins
directory it is about to write to is the one the loader reads, and it **refuses** to
`--apply` from a git worktree (`--i-know-its-a-worktree` overrides): the symlink dies with the
worktree, and a dangling `plugins/memories` is skipped in silence: `load_memory_provider`
returns `None`, and hermes warns only when the provider is not `None`.

Every write it makes to `config.yaml` is verified by re-reading the key afterwards. The
rewriter exiting 0 says the file was replaced, not that `memory.provider` is now `memories`.

## The big-file read guard

Reading a large file into the context is the one mistake advice cannot prevent, because the
model does not know the size before it calls the tool. The incident that produced this
guard is the one in the spec: a 586 KB / 15,593-line JSON, read whole, **~171k tokens**,
where a five-chunk search over the indexed version costs ~6k.

So the read is refused before it happens, with a message that says what to do instead. Two
criteria, whichever fires first, both relative to what is LEFT of the window rather than to
the size of the file:

| | blocks when | default | knob |
|---|---|---|---|
| final remainder | the read would leave less than 20% of the window free | `0.20` | `QCTX_BIGFILE_FLOOR_PCT` |
| one file's share | the read costs more than 40% of what is free | `0.40` | `QCTX_BIGFILE_SHARE_PCT` |

**It does not index anything. It says what to index, and the model indexes**: a blocked
read must not silently fire off hundreds of embedding chunks nobody asked for. If the file
is already in an archive the message says so and points at `docs search` instead of `index`.
A file that cannot be indexed at all (a binary) is **allowed**: refusing it would leave no
way forward.

To read it anyway, put `--full` in your own message. It is scoped to that one turn and
evaporates with the next prompt, by construction; the hook reads the last user turn, so
there is nothing to switch back off and no way to leave the guard disabled by forgetting.
That is exactly why the escape is not an environment variable.

The marker itself is (`QCTX_BIGFILE_ESCAPE`, default `--full`), and configuring it does not
switch anything off: the escape still has to be typed, in that turn. Worth changing if
`--full` is a word your work contains on its own: on a CLI that has a `--full` flag, asking
about it would unlock the guard by accident, which is the false positive a literal marker is
there to avoid. A blank value falls back to the default; a marker of spaces would match
every message ever written.

**Declare `context_window`, or the guard mostly sleeps.** Neither host exposes the window
size where a hook can read it, and the model name does not settle it: the 1M and the 200k
variants of a model ship under the same bare name. `core/windows.py` therefore holds a
**ceiling** per name (the largest window any variant of that name can have) and treats
`used >= window` as its guess being refuted, which falls back to "unknown window", which
allows. Erring large only makes the guard sleep; erring small would make it block on a
guess, which is the one failure this design refuses to produce. The consequence is honest
and worth planning for: in a 200k session, or under any model the table does not know at
all, the guard is nearly inert until you say

```bash
qctx config set context-window 200000      # or export QCTX_CONTEXT_WINDOW
```

`./scripts/hermes_cutover.sh` reports whether that value is declared, and what ceiling it
would fall back to if not.

**Where the window comes from, and why it differs by host.** The guard decides by percentage
of what REMAINS, so it needs the window. It resolves in four steps, and each is consulted only
when the one before it did not answer:

1. `context_window` in your config; declaring it wins over everything.
2. A window the model's endpoint reported, cached. **hermes only**, because it is the only
   host that records which endpoint serves the model; the value is refreshed from `/models`
   by the hook that already talks to the network, never by the guard itself.
3. The **ceiling** table by model name, the LARGEST window any variant of that name can
   have, because the transcript records the bare name and a 200k variant is indistinguishable
   from a 1M one.
4. Zero, which ALLOWS: blocking on a window we are unsure of is the one failure this guard
   must not produce.

On claude-code, step 2 never fires: the host hands the window to its status line and not to
hooks, measured. So there the table decides, and it is right for a 1M variant and generous
for a 200k one. **If you run a 200k session, declare `context_window`**, or the guard will
believe there is five times more room than there is.

**The price is the price of the READ, not of the file.** One call loads at most 2,000 lines
(and on hermes at most 100,000 characters as well), so that is what it is charged. A 3.2 MB
file of 8,000 lines costs ~200k tokens, not ~800k. The honest consequence: the guard fires
LATER than "this file is 3 MB" would lead you to expect.

**Paging is not a hole in it**, and does not need fixing. Reading a file in slices drains
the window in cheap pieces, but `used` grows with every one of them, and the two criteria
start firing on their own as it does. It self-corrects.

**One known case errs toward blocking too much**, measured: a file whose first line is
enormous and whose remaining lines are short is priced by its TOTAL size, because the 8 KB
sample the estimator reads contains no line break at all and cannot tell that the rest of
the file has any. It is not a regression (it is what every file cost before the estimator
learned about line limits), but it is the only known case that errs in the dangerous
direction, and correcting it would cost a second read of the file on every tool call.

**The two hosts can decide differently on the same file.** They price a read identically
only while one read stays under 100,000 characters, roughly a file under ~100 KB, or
lines averaging under ~50 bytes. One read of a 400 KB file of 100-byte lines pulls 202,271
characters on claude-code against hermes' 100,000: **50,567 tokens against 25,000**, and on
a tight budget one blocks while the other allows. That is forced by the hosts (`tools/file_tools.py` truncates a `read_file` by
characters; claude-code's `Read` has no comparable readable ceiling), not chosen here, and
every such difference is listed in the divergence table in
`docs/superpowers/specs/2026-08-15-big-file-read-guard-design.md`, which
`tests/test_host_equivalence.py` derives from the two adapters and requires to be complete.
What both hosts DO guarantee is the decision itself: same budget and same cost, same
verdict.

Every failure of the guard (an unreadable transcript, a locked `state.db`, an unknown
window, a `stat` that fails, any unexpected exception) **allows** the read. A guard that
breaks has to get out of the way, never become a cage.

## Layout

```
core/       the portable core, no reference to a host or an agent
  config.py     configuration precedence, collection guards
  ports.py      the dependency contracts (Protocol)
  errors.py     the root of the error hierarchy
  http.py       JSON over HTTP, in one place
  qdrant.py     minimal client (stdlib)
  embedding.py  the embedder
  reranking.py  the cross-encoder, with scale normalization
  retrieval.py  the two-stage pipeline, shared
  chunk.py      slicing on structural boundaries
  query.py      preparing the question (angles, trivial-prompt filter)
  breaker.py    circuit breaker for a saturated GPU
  bigfile.py    the read guard's decision, pure: the budget and the cost, nothing else
  memory.py     memory CRUD + two-stage recall
  docs.py       document index, TTL, staleness
  inventory.py  which documents are already in an archive
  repos.py      repository archive: code chunks grouped by repo, plus its registry
  scan.py       which files of a repository go into the archive
  bindings.py   which repository this working copy is; declared, never derived
  indexer.py    what the daemon runs, kept apart so the daemon never knows Qdrant
  daemon.py     the background indexer: runs the queue, watches, ends with its hosts
  jobs.py       the queue, on disk: a file a command writes and the daemon reads
  quarantine.py files that could not be indexed, keyed by content so a repair releases them
  lease.py      who is still using the daemon; one note per host, pid and moment
  install.py    the wizard's checks, host-neutral, the same contract as setup.py
  setup.py      diagnostics and suggestions
  knobs.py      the tuning knobs from the environment, read at import
  names.py      turning a name into a filename; one owner of the expression
  windows.py    the ceiling per model name: the largest window any variant can have
  windowprobe.py  asking the serving endpoint how big its window is
  windowcache.py  the window an endpoint reported, remembered between processes
  blocks.py     the injected block, in all four of its states; one renderer, both hosts
  session_state.py  what was already injected, and when the checkpoint is due
  prompts.py    the instructions and the checkpoint procedure, shared by both hosts
  skills.py     which skills the package ships and where they are; one owner of the list
  version.py    the package version; the manifests copy it and a test holds them to it
cli/        the command-line interface over the core
hooks/      the claude-code adapter: recall.py, checkpoint.py, bigfile.py (the read guard), lease.py
hosts/
  hermes/       the hermes-agent adapter: the provider object, its 22 tools, and skill registration
skills/     memory, doc-index, repo-index
scripts/    install.sh (the wizard), cutover.sh (claude-code), hermes_cutover.sh (hermes-agent)
tests/      offline tests + integration tests
```

## Design

The core depends on CONTRACTS (`core/ports.py`, `typing.Protocol`) rather than
implementations: `VectorStore`, `EmbeddingModel`, `RerankModel`. Swapping Qdrant for
another vector store, or the embedding endpoint for a local library, is writing an
adapter, no rule file changes.

They are `Protocol`s and not abstract base classes on purpose: structural typing, no
inheritance and no runtime cost. The concrete gain is testing: the retrieval
pipeline, the most delicate logic in the package, runs with fakes in milliseconds.
Before, it could only be exercised against real infra, which means nobody ran it while
editing.

The two-stage pipeline lives in ONE place (`core/retrieval.py`) and the differences
between consumers are POLICY, not duplicated code:

| | memory | documents |
|---|---|---|
| may the re-rank eliminate? | yes, it vetoes | no, it only orders |
| does the order matter? | no, everything is injected together | yes, it is a list read top to bottom |
| why | a false positive pollutes the agent's context | whoever asks has already chosen the document; silence is worse than imperfect order |

There used to be three implementations of the same idea, and that already cost
something: the re-rank scale normalization existed in one consumer and not the other.

## Portability

`core/` does not know its caller. A new host is a thin adapter:

- **As a library:** `import core` and assemble with `build_memory(cfg)` /
  `build_docs(cfg)`.
- **As a process:** call `cli/qctx.py` and read the JSON from `--json`.

The slicing happens **inside** this process, so the document never passes through the
context of whoever is asking; that is what makes it viable to index a
30,000-character file in order to answer with three chunks.

## The hooks, and the same thing on the other host

`recall.py` runs before the model sees the text, on every prompt that names a subject.
It skips prompts under 12 characters, bare acknowledgements and bare slash commands. It
builds up to three angles on the question in a single embeddings call, fuses the results
by id keeping the highest score, applies both gates and injects the documents along with
the rules for using them. A memory injected recently comes back as a one-line pointer, and the freed
slot reveals more of the archive.

It fails silently for the **user** and never for the **model**: if the search does not
run, the prompt goes through as usual, but the injected block says explicitly that the
archive was not consulted. Without that warning, an absence of results is
indistinguishable from "there is no precedent", and that is how something gets called
unprecedented without anyone having looked.

`checkpoint.py` injects the complete writing procedure every N interactions. The text
is self-sufficient on purpose: a one-line reminder produces vague, duplicated,
metadata-less memory, and the cost shows up months later.

On hermes there is no hook to register: `prefetch()` is called with the upcoming turn's
text and returns the same block, from the same `core/blocks.py`, and the checkpoint rides
along in that same return value on the Nth turn rather than in a second call. Two hosts,
one renderer, which is what `tests/test_host_equivalence.py` is there to keep true.

`bigfile.py` is the exception to that last paragraph: it is the one hook hermes DOES have
to register, because a tool guard is not something a memory provider can offer. It runs
before every file read on both hosts (`PreToolUse` on `Read`, `pre_tool_call` with matcher
`read_file`) and it is described under [the big-file read guard](#the-big-file-read-guard).

## Language

Code, comments and user-facing messages are in English. Two things stay in Portuguese
on purpose, and both are data rather than prose:

- `TRIVIAL_WORDS` and `STOPWORDS` in `core/query.py`, matched against what the user
  types. Translating them would silently disable the trivial-prompt filter and the
  content angle.
- The stored memories themselves, and the checkpoint procedure's instruction to
  confirm in the user's language. The archive is written in whatever language the user
  writes; the dense stage is language-agnostic, and where the cross-encoder is not,
  the pipeline detects the collapse and falls back.

## Status

Done and tested: the core, the CLI, the three archives, the guided diagnostics, all
4 hooks, all 3 skills and the plugin manifest for claude-code; the provider, its 22
tools, the shared configuration wizard and the install script for hermes-agent; and the
big-file read guard on both hosts. Offline tests and integration tests against a real Qdrant
and real models.

Written against hermes-agent v0.20.1 as INSTALLED rather than as published, because the
two differ and the install is what runs. The adapter implements every method that version
declares, and the test that says so reads the surface off the install instead of a list
someone typed.