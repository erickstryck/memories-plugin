# memories-plugin

Long-term semantic memory and a document index on top of [Qdrant](https://qdrant.tech),
for agents. A pure-Python core (stdlib only), with thin per-host adapters.

The problem it solves is twofold:

- **Context that gets lost.** A decision made, a trap already paid for, behaviour
  already measured — all of it evaporates at the end of the session and is
  rediscovered from scratch in the next one, sometimes contradicting what was
  already known.
- **Context that does not fit.** Answering a question about 40 lines of an
  8,000-line file should not cost the whole file. Here the file is read, sliced and
  indexed **outside** the agent's context; the search returns only the chunks that
  answer.

## Three archives, three lifecycles

The separation is structural — distinct, configurable collections — not a convention:

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
   similarity. Cheap, approximate, and **practically indifferent to language** — a
   question in Portuguese finds an English document (measured: 0.460 against 0.475
   for the same question in the two languages).
2. **Cross-encoder** (`bge-reranker-v2-m3` or another): reads the question and the
   chunk in the SAME pass, with cross attention. It judges far better, and for that
   reason has no precomputable vector — it is one forward pass per pair, cost linear
   in the total token count.

Two measured findings shaped the design, and both are silent failures if ignored:

**The re-rank scale depends on the server.** The same model returns a sigmoid (0..1)
on one server and a raw logit on another — the same irrelevant document gave `1.6e-05`
and `-11.04`, the second being exactly `logit(1.6e-05)`. A cutoff calibrated on one
scale is inert on the other. The core detects it by range and normalizes, so the
calibrated number stays valid on any server.

**The cross-encoder collapses on a cross-lingual pair.** The same question about the
same English document: `0.2073` in English, `0.0004` in Portuguese — 500x. It matches
language, not just semantics. Consequences in the design:

- In **document** search, the re-rank **orders but does not veto**: whoever asks has
  already chosen the document, and silence is worse than imperfect order. A collapse
  is detected (best score below `0.01`) and the dense order takes over, with a warning.
- In **memory** search, the re-rank keeps the veto: there precision matters more than
  reach, and a false positive pollutes the agent's context.

## Installation

```bash
git clone git@github.com:erickstryck/memories-plugin.git
cd memories-plugin
python3 -m unittest discover -s tests    # 493 collected, 476 offline; no network, no deps
ln -s "$PWD/bin/qctx" ~/.local/bin/qctx  # so `qctx` works from anywhere
```

There is no `pip install`: the core uses only the standard library. That is
deliberate — this code runs inside hooks fired on every interaction, and a missing
dependency would turn an environment failure into a silent loss of functionality.

### As a Claude Code plugin

The repository is at once a plugin and a single-plugin marketplace:

```bash
claude plugin marketplace add ~/dev/memories-plugin
claude plugin install memories-plugin@memories-plugin
```

Enabling the plugin registers **two `UserPromptSubmit` hooks** (recall on every
prompt, checkpoint every N) and **two skills** (`memory`, `doc-index`). The hooks use
`${CLAUDE_PLUGIN_ROOT}`, so there is no hard-coded path to maintain.

**If you already had equivalent hooks registered by hand in `settings.json`, remove
them in the SAME pass.** Both sets fire together and recall gets injected twice into
the same prompt — which is worse than having none, because it doubles the context cost
without adding information. Check the log: one round per prompt, not two.

## Hosts

The same core serves two hosts, with the same operations and the same configuration.

| | claude-code | hermes-agent |
|---|---|---|
| adapter | `hooks/` | `hosts/hermes/` |
| install | `claude plugin marketplace add .` | a symlink into `$HERMES_HOME/plugins/memories` |
| recall | `UserPromptSubmit` hook | `prefetch()` |
| checkpoint | second `UserPromptSubmit` hook | rides along in `prefetch()` on the Nth turn |
| operations | `qctx` CLI + two skills | 15 model-invokable tools + the same CLI |
| what the model is told | the two skills | `system_prompt_block()`, from the same `core/prompts.py` |
| configuration | `~/.config/memories-plugin/config.json` | the same file |
| credentials | the environment | the environment, or `$HERMES_HOME/.env` |

Equivalence is not a claim in this table — `tests/test_host_equivalence.py` renders every
block state through both adapters and requires byte-identical output.

```bash
./scripts/hermes_cutover.sh            # check the install and report; writes nothing
./scripts/hermes_cutover.sh --apply    # install as the hermes memory provider
```

The install is a **symlink** into `$HERMES_HOME/plugins/memories`, one level deep and no
deeper: hermes' loader (`plugins/memory/__init__.py`) scans `$HERMES_HOME/plugins/<name>/`,
and a provider one directory further down is not discovered at all. That is measured
against the installed loader, not read off the documentation — `tests/test_hermes_provider.py`
drives it with a temp `HERMES_HOME` and requires the deeper layout to come back unfound.

hermes activates exactly ONE external memory provider, so installing this one **replaces**
whatever `memory.provider` names. The provider it replaces is disabled by configuration,
never deleted, and its own collection stays reachable read-only, outside automatic recall:

```bash
qctx memory search-collections "<topic>" --collections hermes_memory
```

### What the cutover script checks, and why

`is_available()` gates initialization, so a provider that reports unavailable is never
initialized and any diagnostic it might log from `initialize()` is unreachable. hermes
0.20.1 warns and appends the provider's `unavailable_reason()`
(`agent/agent_init.py`), but the failure it cannot describe at all is the one that matters
most here: **the two API keys live only in the environment.** `config set` refuses to write
a secret to `config.json`, and `is_available()` does not look at the keys — it checks the
Qdrant URL, the embedding endpoint and the collection. So a hermes started from a shell
without them reports a perfectly healthy provider and then fails every single search.

The script therefore checks the environment, not only the files: both accepted spellings of
each key, in this shell **and** in `$HERMES_HOME/.env` — which hermes loads itself
(`hermes_cli/env_loader.py`, from `run_agent.py` and `cli.py`) and which is the only one of
the two a systemd/gateway hermes has. It prints the variable names and never their values.

It also reports where the provider being replaced actually lives, whether the plugins
directory it is about to write to is the one the loader reads, and whether it is being run
from a git worktree — a symlink into a temporary checkout dies with it, leaving hermes with
a dangling directory that discovery silently skips.

## Configuration

Start with the guided diagnostics:

```bash
python3 cli/qctx.py setup
```

It checks Qdrant, the embedding endpoint (detecting the model's real dimension), the
re-rank endpoint (including which scale it answers in) and the three collections — and
prints, for each missing item, the exact command that fixes it. In an interactive
terminal it asks and writes; **with no TTY it never blocks**, it only reports. That is
deliberate: the command is also called by agents and by scripts, and a prompt waiting
for an answer that never comes would hang the call.

`--check` forces diagnose-only mode; `--json` returns the full picture for
consumption by a program.

Precedence: **environment variable > file > default**. The file lives at
`~/.config/memories-plugin/config.json`.

```bash
python3 cli/qctx.py collections list           # what exists in Qdrant, with dimensions
python3 cli/qctx.py config set memory-collection my_memories
python3 cli/qctx.py config show
```

`collections list` marks each collection as compatible or not with the configured
model's dimension. Writing into an archive of a different dimension is refused: it
would go through and degrade search with no error appearing.

Recognized variables (canonical first, legacy aliases accepted):

| config | environment |
|---|---|
| `qdrant_url` | `QCTX_QDRANT_URL`, `QDRANT_URL` |
| `qdrant_api_key` | `QCTX_QDRANT_API_KEY`, `QDRANT_SERVICE_API_KEY` |
| `api_base_url` | `QCTX_API_BASE_URL`, `SERVER_BASE_URL` |
| `api_key` | `QCTX_API_KEY`, `SERVER_API_KEY` |
| `embed_url` | `QCTX_EMBED_URL`, `RECALL_EMBED_URL` |
| `rerank_url` | `QCTX_RERANK_URL`, `RECALL_RERANK_URL` |
| `embed_model` | `QCTX_EMBED_MODEL`, `EMBEDDING_MODEL` |
| `rerank_model` | `QCTX_RERANK_MODEL`, `RECALL_RERANK_MODEL` |
| `memory_collection` | `QCTX_MEMORY_COLLECTION`, `COLLECTION_NAME` |
| `docs_collection` | `QCTX_DOCS_COLLECTION`, `DOCS_COLLECTION` |
| `library_collection` | `QCTX_LIBRARY_COLLECTION`, `LIBRARY_COLLECTION` |
| `vector_size` | `QCTX_VECTOR_SIZE`, `VECTOR_SIZE` |

The two API keys are the only settings that **cannot** go into the config file:
`config set` refuses them and points at the environment variable instead. A plaintext
secret ends up in backups and in dotfile sync.

`memory_collection` starts out **empty** on purpose: with no explicit choice the CLI
refuses to operate, so there is no accidental write path into the wrong archive.

## Usage

```bash
qctx() { python3 cli/qctx.py "$@"; }

# memory
qctx memory store "connector X's poll truncates at 100 items" --type reference
qctx memory find "poll pagination"            # dense, cheap
qctx memory recall "poll pagination"          # two stages, with re-rank
qctx memory update <id> --text "..." ; qctx memory delete <id>
qctx memory search-collections "poll pagination" --collections hermes_memory
                                              # read-only, in someone else's archive

# documents
qctx docs index ./huge-report.md --ttl 24h          # temporary
qctx docs keep ./api-manual.md                      # library, permanent
qctx docs search "how do I authenticate?" --scope all --limit 5
qctx docs list
qctx docs refresh --scope library                   # reindexes what changed on disk
qctx docs drop <doc-id> --scope library
qctx docs drop --purge-tmp                          # deletes only the temporary archive
```

For a text file, the search returns **`path:lines`** plus a short excerpt, rather than
the whole content: the consumer re-reads the exact region and works on the **current**
content, with no risk of operating on a stale snapshot. For a source that cannot be
re-read by region, it returns the text with the indexing date and a warning. In every
case, if the file changed since indexing, the result comes back marked.

## Layout

```
core/       the portable core — no reference to a host or an agent
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
  memory.py     memory CRUD + two-stage recall
  docs.py       document index, TTL, staleness
  setup.py      diagnostics and suggestions
  blocks.py     the injected block, in all four of its states — one renderer, both hosts
  session_state.py  what was already injected, and when the checkpoint is due
  prompts.py    the instructions and the checkpoint procedure, shared by both hosts
cli/        the command-line interface over the core
hooks/      the claude-code adapter: recall on every prompt, checkpoint every N
hosts/
  hermes/       the hermes-agent adapter: the provider object and its 15 tools
skills/     memory, doc-index
scripts/    cutover.sh (claude-code), hermes_cutover.sh (hermes-agent)
tests/      476 offline tests + 17 integration
```

## Design

The core depends on CONTRACTS (`core/ports.py`, `typing.Protocol`) rather than
implementations: `VectorStore`, `EmbeddingModel`, `RerankModel`. Swapping Qdrant for
another vector store, or the embedding endpoint for a local library, is writing an
adapter — no rule file changes.

They are `Protocol`s and not abstract base classes on purpose: structural typing, no
inheritance and no runtime cost. The concrete gain is testing — the retrieval
pipeline, the most delicate logic in the package, runs with fakes in milliseconds.
Before, it could only be exercised against real infra, which means nobody ran it while
editing.

The two-stage pipeline lives in ONE place (`core/retrieval.py`) and the differences
between consumers are POLICY, not duplicated code:

| | memory | documents |
|---|---|---|
| may the re-rank eliminate? | yes, it vetoes | no, it only orders |
| does the order matter? | no — everything is injected together | yes — it is a list read top to bottom |
| why | a false positive pollutes the agent's context | whoever asks has already chosen the document; silence is worse than imperfect order |

There used to be three implementations of the same idea, and that already cost
something: the re-rank scale normalization existed in one consumer and not the other.

## Portability

`core/` does not know its caller. A new host is a thin adapter:

- **As a library:** `import core` and assemble with `build_memory(cfg)` /
  `build_docs(cfg)`.
- **As a process:** call `cli/qctx.py` and read the JSON from `--json`.

The slicing happens **inside** this process, so the document never passes through the
context of whoever is asking — that is what makes it viable to index a
30,000-character file in order to answer with three chunks.

## The hooks, and the same thing on the other host

`recall.py` runs before the model sees the text, on every prompt that names a subject —
it skips prompts under 12 characters, bare acknowledgements and bare slash commands. It
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
one renderer — which is what `tests/test_host_equivalence.py` is there to keep true.

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

Done and tested: the core, the CLI, the three archives, the guided diagnostics, both
hooks, both skills and the plugin manifest for claude-code; the provider, its 15 tools,
the shared configuration wizard and the install script for hermes-agent. 476 offline tests
and 17 integration tests against a real Qdrant and real models.

Written against hermes-agent v0.20.1 as INSTALLED rather than as published, because the
two differ and the install is what runs. The adapter implements every method that version
declares, and the test that says so reads the surface off the install instead of a list
someone typed.
