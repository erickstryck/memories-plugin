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

## Contributing

`main` is protected. What that means depends on who you are:

- **Outside contributors** — fork, open a pull request. Direct pushes are refused, conversations
  on a PR must be resolved before it merges, and `main` can be neither force-pushed nor deleted by
  anyone at all, including the owner.
- **The owner** pushes to `main` directly. That is deliberate: this is a personal plugin with one
  maintainer, and a PR to oneself is a form with no reader. The rules exist to keep *someone else's*
  change reviewed, not to stage a review that never happens.

There is no CI yet, so a pull request is not gated on the tests. Run them before opening one — the
suite is offline, needs no network and no dependencies:

```bash
python3 -m unittest discover -s tests
```

## Installation

### The one command

On a machine where the plugin is already installed by its host, or on a fresh clone:

```bash
bash ~/.hermes/plugins/memories/scripts/install.sh          # installed by hermes
./scripts/install.sh                                        # cloned

# installed by claude-code. The cache holds one directory per commit and old ones stay
# behind, so a bare glob expands to all of them and the extra paths land on `qctx
# install` as unrecognised arguments. Take the newest, quoted, and pass one path.
bash "$(ls -dt ~/.claude/plugins/cache/memories-plugin/memories-plugin/*/scripts/install.sh | head -1)"
```

It puts `qctx` on PATH, asks for what is missing, offers to install into whichever host is
on this machine, and re-checks. Nothing it writes is silent, and what it asks before doing
differs by group: the launcher copy, the configuration and the credential writes are what
you just typed being saved, and are reported as they happen; the two that change a host —
installing the plugin into `claude`/`hermes`, and running that host's cutover — ask for a
`y` of their own first.

To verify a machine without changing it — which is also how you check a machine you set up
months ago:

```bash
qctx install --check      # reports; writes nothing
```

`--check` covers the plumbing, Qdrant, the embedding and re-rank endpoints, the
5 collections, whether a shell-less process would find the configuration, whether each of
the two keys is set and in which spelling and where (names and lengths only — never a
value), and each host's own cutover report.

### If you would rather do it by hand

**From zero to working, in order.** Each step is expanded below; nothing here is optional except
where it says so, and the two steps people skip are 3 and 5 — both fail silently.

| # | step | why it is not optional |
|---|---|---|
| 1 | a reachable Qdrant, and an embedding endpoint | there is nothing to search without them |
| 2 | `qctx config set` the **addresses** into the file | a process without your shell reads the file, and only the file |
| 3 | `export` the **two keys** — and for hermes, also `~/.hermes/.env` | keys never enter the config file; a shell-less hermes has none otherwise |
| 4 | `qctx config set memory-collection <name>` | it is empty on purpose, so nothing can write into the wrong archive |
| 5 | install on the host — and on hermes, run the cutover too | `plugins install` cannot register a shell hook, so the read guard is not installed by it |
| 6 | `qctx setup`, then the no-shell check | the only way to know steps 2 and 3 actually took |

```bash
git clone git@github.com:erickstryck/memories-plugin.git
cd memories-plugin
python3 -m unittest discover -s tests    # offline; no network, no deps
ln -s "$PWD/bin/qctx" ~/.local/bin/qctx  # so `qctx` works from anywhere
```

There is no `pip install`: the core uses only the standard library. That is
deliberate — this code runs inside hooks fired on every interaction, and a missing
dependency would turn an environment failure into a silent loss of functionality.

### Install on Claude Code

The repository is at once a plugin and a single-plugin marketplace, so two commands do it —
**from git**, with nothing cloned by hand:

```bash
claude plugin marketplace add erickstryck/memories-plugin
claude plugin install memories-plugin@memories-plugin
```

For **development**, add it from a path instead, and edits take effect with no reinstall:

```bash
claude plugin marketplace add ~/dev/memories-plugin
claude plugin install memories-plugin@memories-plugin
```

Then **open a new terminal** — the harness reads `settings.json` at start-up.

To pick up a newer commit later, the plugin must be named **with its marketplace**; the bare name
answers `Plugin "memories-plugin" not found`, which reads like a broken install and is not one:

```bash
claude plugin marketplace update memories-plugin
claude plugin update memories-plugin@memories-plugin      # name@marketplace, not just the name
```

The version it reports is the **commit SHA**, because the manifests declare no version on purpose
— a hand-maintained number goes stale and this one already had (0.3.0 declared, 0.2.0 installed).
`claude plugin details memories-plugin` lists what it found: 3 skills, 4 hooks.

That registers, with no path for you to maintain (the hooks resolve
`${CLAUDE_PLUGIN_ROOT}` themselves):

| what | when it runs |
|---|---|
| recall hook | every prompt, before the model sees it |
| checkpoint hook | every Nth prompt, to write memories down |
| big-file guard | before every `Read`, to refuse a read that would cost too much context |
| lease hook | at session start, to claim the indexing daemon and end it with the session |
| skills `memory` and `doc-index` | loaded on demand, when the model needs them |

To check it took: `claude plugin list` shows it enabled, and the recall log at
`~/.memories-plugin/state/recall.log` gets one round per prompt — **one, not two**.

**If you already had equivalent hooks registered by hand in `settings.json`, remove them
in the SAME pass** — `./scripts/cutover.sh --apply` does both at once. Two sets fire
together and recall lands twice in one prompt, which is worse than having none: it doubles
the context cost and adds no information.

### Install on hermes-agent

One command, from git — plus `--force`, and the reason matters:

```bash
hermes plugins install erickstryck/memories-plugin --enable --force
hermes config set memory.provider memories        # tell hermes to USE it
```

**Why `--force` is required, and what you are agreeing to.** hermes scans a cloned plugin before
installing it, and this tree scores `caution`: it ships a thousand tests that shell out to `git`
and `python3`, design documents full of example commands, and — the finding that actually matters
— **two scripts that edit host configuration**. That is the cutover's declared job, not a
surprise, and `--force` is you confirming it. Read the report before you type it; the plugin does
what it says, and the scanner is right to make you look.

Do **not** reach for `plugins.scan_on_install: false` to skip the report: that disables scanning
for every plugin from anywhere, which is a policy change rather than a decision about this one.

**Two things will stop the install before it starts.** Both were measured, and neither message
mentions the real cause:

- *"Invalid plugin name 'memories': resolves outside the plugins directory."* — you already have a
  **development symlink** at `$HERMES_HOME/plugins/memories`. The installer resolves the target
  path, follows the link out of the plugins directory, and refuses. `--force` does not help: the
  name is validated first. Remove the link, or keep it and skip the install (see below).
- A `dangerous` verdict, where `--force` does **not** override. One `critical` finding is enough,
  and a critical is not necessarily an action: a plugin that merely *writes out* the hermes config
  path in prose trips one. This tree keeps itself clear of those, and a test fails if a new one
  appears — see `tests/test_installable_from_git.py`.

**On your development machine, prefer the symlink and skip the install entirely.** A clone is a
copy: edits to your checkout change nothing until `hermes plugins update memories`. The symlink is
always at HEAD, with no update step, which is what you want where you are editing the code.

`--ref <40-char-sha>` pins an exact commit. The installer names any key missing from
`~/.hermes/.env`, because a hermes started by systemd or the gateway has no shell — and a key
that lives only in an interactive environment is one it will not have. **The keys never go in a
config file.**

Installing the SUBDIRECTORY (`erickstryck/memories-plugin/hosts/hermes`) is accepted syntax and
does not work: the adapter imports `core/` from the repository root, which a subdirectory install
does not bring. It fails silently — hermes' loader swallows a broken provider at debug level — so
install the whole repository, which is what the root `__init__.py` is for.

For **development**, keep the symlink install instead, so edits take effect immediately:

```bash
ln -s ~/dev/memories-plugin/hosts/hermes ~/.hermes/plugins/memories
```

**Installing is not the whole job on this host, and the missing half is silent.** A hermes plugin
manifest has no field for shell hooks — they live only in `$HERMES_HOME/config.yaml`, behind a
first-use consent allowlist — so `plugins install` gives you memory and the 22 tools, and the
big-file read guard is **not** registered. That is what the script below is for:

```bash
./scripts/hermes_cutover.sh            # reports what it would do; writes NOTHING
./scripts/hermes_cutover.sh --apply    # installs, with a dated backup of every file it edits
```

It reports the credentials, the URLs a shell-less hermes would find, the symlink, the provider
selection and the guard — and `--apply` writes only what is missing. It accepts either install
shape: a symlink at the repository root (what a git install produces) and one at `hosts/hermes`
both load, and it leaves whichever you have alone.

**Then approve the hook once.** After registration `hermes hooks list` shows it
`✗ not allowlisted`: the first file read of a new session asks at the TTY, and **a hermes with no
TTY skips the hook silently** until it has been approved once. Approving records it in
`~/.hermes/shell-hooks-allowlist.json`, which every later run — TTY or not — then honours.

```bash
hermes hooks list        # ✓ allowed, with the approval timestamp, once it is done
```

Do **not** reach for `hooks_auto_accept: true` to skip that step: it auto-approves every future
hook from anywhere, which is a policy change, not a fix for this one.

Run the first form first and read it. It checks the credentials, the collections, the
provider entry, the hook block and whether the context window is declared, and prints the
exact fix for each gap. The `--apply` form writes a `.bak-<timestamp>` beside anything it
edits and re-reads the result to confirm it took, rather than trusting that the write
returned zero.

Then restart hermes.

## Hosts

The same core serves two hosts, with the same operations and the same configuration.

| | claude-code | hermes-agent |
|---|---|---|
| adapter | `hooks/` | `hosts/hermes/` |
| install | `claude plugin marketplace add .` | a symlink into `$HERMES_HOME/plugins/memories` |
| recall | `UserPromptSubmit` hook | `prefetch()` |
| checkpoint | second `UserPromptSubmit` hook | rides along in `prefetch()` on the Nth turn |
| big-file guard | `PreToolUse` hook on `Read` | `pre_tool_call` shell hook, matcher `read_file` |
| operations | `qctx` CLI + 3 skills | 22 model-invokable tools + the same CLI |
| what the model is told | the 3 skills | `system_prompt_block()`, from the same `core/prompts.py` |
| configuration | `~/.config/memories-plugin/config.json` | the same file |
| credentials | the environment | the environment, or `$HERMES_HOME/.env` |

Equivalence is not a claim in this table — `tests/test_host_equivalence.py` renders every
block state through both adapters and requires byte-identical output.

The hermes install is a **symlink** into `$HERMES_HOME/plugins/memories`, one level deep and no
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

The script therefore checks the environment, not only the files: **every** spelling each key
accepts — three for the Qdrant key, `QCTX_QDRANT_API_KEY`, `QDRANT_SERVICE_API_KEY` and
`QDRANT_API_KEY` — in this shell **and** in `$HERMES_HOME/.env`, which hermes loads itself
(`hermes_cli/env_loader.py`, from `run_agent.py` and `cli.py`) and which is the only one of
the two a systemd/gateway hermes has. It prints the variable names, never their values.
Checking fewer aliases than the core accepts would be worse than not checking: the core
would resolve the key fine while the script told the operator to export what they already had.

And it asks the same question about the settings that are **not** secrets. The URLs and the
collection names can live in `config.json` — which both hosts read with no shell involved —
so the script resolves the configuration with the environment layer removed
(`core.load(env={})`) and reports what a hermes that inherits no shell would be missing,
pointing at `qctx config set qdrant-url …` rather than at `.env`. Without that second half
the only remedy on offer fixes the keys and leaves a gateway hermes memory-less on the URLs.

It also reports where the provider being replaced actually lives, whether the plugins
directory it is about to write to is the one the loader reads, and it **refuses** to
`--apply` from a git worktree (`--i-know-its-a-worktree` overrides): the symlink dies with the
worktree, and a dangling `plugins/memories` is skipped in silence — `load_memory_provider`
returns `None`, and hermes warns only when the provider is not `None`.

Every write it makes to `config.yaml` is verified by re-reading the key afterwards. The
rewriter exiting 0 says the file was replaced, not that `memory.provider` is now `memories`.

### The big-file read guard

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

**It does not index anything. It says what to index, and the model indexes** — a blocked
read must not silently fire off hundreds of embedding chunks nobody asked for. If the file
is already in an archive the message says so and points at `docs search` instead of `index`.
A file that cannot be indexed at all (a binary) is **allowed**: refusing it would leave no
way forward.

To read it anyway, put `--full` in your own message. It is scoped to that one turn and
evaporates with the next prompt, by construction — the hook reads the last user turn, so
there is nothing to switch back off and no way to leave the guard disabled by forgetting.
That is exactly why the escape is not an environment variable.

The marker itself is (`QCTX_BIGFILE_ESCAPE`, default `--full`), and configuring it does not
switch anything off — the escape still has to be typed, in that turn. Worth changing if
`--full` is a word your work contains on its own: on a CLI that has a `--full` flag, asking
about it would unlock the guard by accident, which is the false positive a literal marker is
there to avoid. A blank value falls back to the default; a marker of spaces would match
every message ever written.

**Declare `context_window`, or the guard mostly sleeps.** Neither host exposes the window
size where a hook can read it, and the model name does not settle it: the 1M and the 200k
variants of a model ship under the same bare name. `core/windows.py` therefore holds a
**ceiling** per name — the largest window any variant of that name can have — and treats
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

1. `context_window` in your config — declaring it wins over everything.
2. A window the model's endpoint reported, cached. **hermes only**, because it is the only
   host that records which endpoint serves the model; the value is refreshed from `/models`
   by the hook that already talks to the network, never by the guard itself.
3. The **ceiling** table by model name — the LARGEST window any variant of that name can
   have, because the transcript records the bare name and a 200k variant is indistinguishable
   from a 1M one.
4. Zero, which ALLOWS: blocking on a window we are unsure of is the one failure this guard
   must not produce.

On claude-code, step 2 never fires: the host hands the window to its status line and not to
hooks, measured. So there the table decides — and it is right for a 1M variant and generous
for a 200k one. **If you run a 200k session, declare `context_window`**, or the guard will
believe there is five times more room than there is.

**The price is the price of the READ, not of the file.** One call loads at most 2,000 lines
(and on hermes at most 100,000 characters as well), so that is what it is charged. A 3.2 MB
file of 8,000 lines costs ~200k tokens, not ~800k. The honest consequence: the guard fires
LATER than "this file is 3 MB" would lead you to expect.

**Paging is not a hole in it**, and does not need fixing. Reading a file in slices drains
the window in cheap pieces — but `used` grows with every one of them, and the two criteria
start firing on their own as it does. It self-corrects.

**One known case errs toward blocking too much**, measured: a file whose first line is
enormous and whose remaining lines are short is priced by its TOTAL size, because the 8 KB
sample the estimator reads contains no line break at all and cannot tell that the rest of
the file has any. It is not a regression — it is what every file cost before the estimator
learned about line limits — but it is the only known case that errs in the dangerous
direction, and correcting it would cost a second read of the file on every tool call.

**The two hosts can decide differently on the same file.** They price a read identically
only while one read stays under 100,000 characters — roughly, a file under ~100 KB, or
lines averaging under ~50 bytes. One read of a 400 KB file of 100-byte lines pulls 202,271
characters on claude-code against hermes' 100,000: **50,567 tokens against 25,000**, and on
a tight budget one blocks while the other allows. That is forced by the hosts (`tools/file_tools.py` truncates a `read_file` by
characters; claude-code's `Read` has no comparable readable ceiling), not chosen here, and
every such difference is listed in the divergence table in
`docs/superpowers/specs/2026-08-15-big-file-read-guard-design.md`, which
`tests/test_host_equivalence.py` derives from the two adapters and requires to be complete.
What both hosts DO guarantee is the decision itself: same budget and same cost, same
verdict.

Every failure of the guard — an unreadable transcript, a locked `state.db`, an unknown
window, a `stat` that fails, any unexpected exception — **allows** the read. A guard that
breaks has to get out of the way, never become a cage.

## Configuration

### The short version

Four things must be set before anything works: where Qdrant is, where the embedding
endpoint is, the two credentials, and which collection holds your memory. Everything
else has a working default.

```bash
# 1. the two secrets go in the environment, never in the file
export QCTX_QDRANT_API_KEY="..."      # your Qdrant key
export QCTX_API_KEY="..."             # your embedding/re-rank server key

# 2. the addresses go in the file, because they are not secret
qctx config set qdrant-url https://your-qdrant.example
qctx config set api-base-url https://your-llm-server.example/v1

# 3. name the archive your curated facts live in (it starts EMPTY on purpose)
qctx config set memory-collection my_memories

# 4. check everything, and let it tell you what is still missing
qctx setup
```

`qctx setup` is the one command to run when something is wrong: it probes Qdrant, the
embedding endpoint (detecting the model's real dimension), the re-rank endpoint and every
collection, and prints the exact command that fixes each gap.

### The case that breaks silently: a process with no shell

Everything above works for a program **you** started from **your** terminal, because it
inherits your environment. A hermes launched by systemd, by the gateway, or by cron inherits
nothing — and the symptom is not an error. It is an archive that looks simply **empty**.

Two habits hide this, and both were measured on a working machine:

- **Exporting the URLs instead of writing them.** `export QDRANT_URL=…` makes every command
  work in your shell while `config.json` stays empty. Step 2 above says the addresses go in the
  **file** for exactly this reason.
- **`qctx config show` MIXES the file and the environment.** It prints a complete-looking
  picture while the file holds empty strings. To see what a shell-less process would read, read
  the file: `cat ~/.config/memories-plugin/config.json`.

So, for hermes specifically, the two keys need a second home — the one hermes itself loads:

```bash
umask 077 && cat >> ~/.hermes/.env <<'ENV'
QDRANT_SERVICE_API_KEY=...
SERVER_API_KEY=...
ENV
chmod 600 ~/.hermes/.env
```

`~/.hermes/.env` is hermes' own credential file, and it is the only reason a shell-less hermes
has keys at all. It is **not** the plugin's config: the plugin still refuses secrets in its own
file, and `config set` will tell you so.

**Verify it the way that actually proves it** — an empty environment plus only that file, which
is what systemd gives you:

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin bash -c '
  set -a; . ~/.hermes/.env; set +a
  qctx setup'
```

Every line must be `[ok]`. If Qdrant or the embedding endpoint fails **there** while passing in
your own shell, the gap is one of the two above.

### The long version

Start with the guided diagnostics:

```bash
python3 cli/qctx.py setup
```

It checks Qdrant, the embedding endpoint (detecting the model's real dimension), the
re-rank endpoint (including which scale it answers in) and the 5 collections — and
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
| `qdrant_api_key` | `QCTX_QDRANT_API_KEY`, `QDRANT_SERVICE_API_KEY`, `QDRANT_API_KEY` |
| `api_base_url` | `QCTX_API_BASE_URL`, `SERVER_BASE_URL` |
| `api_key` | `QCTX_API_KEY`, `SERVER_API_KEY` |
| `embed_url` | `QCTX_EMBED_URL`, `RECALL_EMBED_URL` |
| `rerank_url` | `QCTX_RERANK_URL`, `RECALL_RERANK_URL` |
| `embed_model` | `QCTX_EMBED_MODEL`, `EMBEDDING_MODEL` |
| `rerank_model` | `QCTX_RERANK_MODEL`, `RECALL_RERANK_MODEL` |
| `memory_collection` | `QCTX_MEMORY_COLLECTION`, `COLLECTION_NAME` |
| `docs_collection` | `QCTX_DOCS_COLLECTION`, `DOCS_COLLECTION` |
| `library_collection` | `QCTX_LIBRARY_COLLECTION`, `LIBRARY_COLLECTION` |
| `repos_collection` | `QCTX_REPOS_COLLECTION`, `REPOS_COLLECTION` |
| `repos_registry_collection` | `QCTX_REPOS_REGISTRY_COLLECTION`, `REPOS_REGISTRY_COLLECTION` |
| `vector_size` | `QCTX_VECTOR_SIZE`, `VECTOR_SIZE` |
| `context_window` | `QCTX_CONTEXT_WINDOW` |

The two API keys are the only settings that **cannot** go into the config file:
`config set` refuses them and points at the environment variable instead. A plaintext
secret ends up in backups and in dotfile sync.

`memory_collection` starts out **empty** on purpose: with no explicit choice the CLI
refuses to operate, so there is no accidental write path into the wrong archive.

## Every command, at a glance

Four groups. `memory` is curated facts, `docs` is documents you point at, `repos` is whole
repositories, and the rest is configuration. Everything below exists on **both hosts** —
the CLI name is on the left, the hermes tool name on the right.

### Memory — facts worth keeping

| command | tool | what it does |
|---|---|---|
| `memory store` | `memory_store` | write one fact |
| `memory store-many` | `memory_store_many` | write a batch, all-or-nothing |
| `memory find` | `memory_find` | dense search, cheap, no re-rank |
| `memory recall` | `memory_recall` | two-stage search with re-rank — the accurate one |
| `memory get` | `memory_get` | read one by id |
| `memory list` | `memory_list` | list what is stored |
| `memory update` | `memory_update` | correct a fact in place |
| `memory delete` | `memory_delete` | remove one |
| `memory search-collections` | `memory_search_collections` | read-only search in someone else's archive |

### Docs — a document you point at

| command | tool | what it does |
|---|---|---|
| `docs index` | `docs_index` | index TEMPORARILY, with a TTL |
| `docs keep` | `docs_keep` | keep in the LIBRARY, no expiry |
| `docs search` | `docs_search` | search, returning `path:lines` and an excerpt |
| `docs list` | `docs_list` | what is indexed |
| `docs refresh` | `docs_refresh` | reindex what changed on disk |
| `docs drop` | `docs_drop` | delete one document, or the whole temporary archive |

### Repos — a whole repository, grouped

| command | tool | what it does |
|---|---|---|
| `repos register` | `repos_register` | declare a repository by name, before indexing anything |
| `repos add` | `repos_add` | index the given files under it |
| `repos search` | `repos_search` | search one repository, or `--all` to ask which ones mention it |
| `repos list` | `repos_list` | every repository, with counts and when it was last indexed |
| `repos refresh` | `repos_refresh` | reindex the files that changed on disk since indexing |
| `repos init` | `repos_init` | detect this working copy and offer to index it |
| `repos add-all` | — *(CLI only)* | index the whole repository, in the background |
| `repos status` | — *(CLI only)* | what is indexing, and whether the daemon is up |
| `repos cancel` | — *(CLI only)* | stop indexing; what is already indexed stays |
| `repos daemon` | — *(CLI only)* | start, stop, or run the background indexer |
| `repos drop` | `repos_drop` | delete a repository archive, permanently |

### Indexing a whole project

```bash
cd ~/dev/my-project
qctx repos init                  # detects the repo and offers a name
qctx repos add-all my-project    # queues it; the daemon does the work
qctx repos status                # progress, and whether the daemon is up
```

The daemon indexes in the background, so your terminal is free. It also **watches** the
repositories it indexed: a file you change is reindexed within a few seconds, including one you
have not committed. Cancelling keeps whatever was already indexed, and running `add-all` again
skips the files that did not change.

**It ends when you do.** Each session writes a lease with its host's pid; when the last claude or
hermes exits — cleanly or killed — the daemon notices within a cycle and stops. Nothing is left
running behind you.

### Configuration and diagnostics — CLI only

| command | what it does |
|---|---|
| `setup` | probe everything and print the exact fix for each gap |
| `config show` | the resolved configuration |
| `config set` | write one setting to the file |
| `config detect` | ask the embedding endpoint its real dimension and store it |
| `collections list` | what exists in Qdrant, and whether each matches your model |

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

# repositories
qctx repos register my-project                      # declare it first — the name IS the key
qctx repos add my-project $(git ls-files '*.py')    # index the files you hand it
qctx repos search "how is auth done" --limit 8      # this repository
qctx repos search "retry policy" --all              # which of my projects mention it?
qctx repos list                                     # counts, and when each was last indexed
qctx repos drop my-project --yes                    # permanent, and only ever manual
```

A repository name must be a **slug** — lowercase, digits and hyphens — because it is the
key everything filters on. `repos register "My Project"` is refused, naming `my-project`
as the remedy rather than silently rewriting what you typed.

`--all` is the one that pays for the rest: it groups **on the server**, one group per
repository, so a project with a single genuine mention still comes back next to one with
fifty. Grouping the top results on the client would answer a different question and answer
it confidently. When nothing clears the threshold it says so in words — it never returns
an empty result that reads as "no project mentions this".

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
  hermes/       the hermes-agent adapter: the provider object and its 22 tools
skills/     memory, doc-index, repo-index
scripts/    install.sh (the wizard), cutover.sh (claude-code), hermes_cutover.sh (hermes-agent)
tests/      offline tests + integration tests
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

`bigfile.py` is the exception to that last paragraph: it is the one hook hermes DOES have
to register, because a tool guard is not something a memory provider can offer. It runs
before every file read on both hosts — `PreToolUse` on `Read`, `pre_tool_call` with matcher
`read_file` — and it is described under [the big-file read guard](#the-big-file-read-guard).

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
