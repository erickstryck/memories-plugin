# Usage and configuration

Command reference for `qctx`, the full configuration picture, and the diagnostics
that tell you what is still missing. The [README](../README.md) covers installation
([install.md](install.md) covers what each installation step does);
[architecture](architecture.md) covers why things are the way they are.

## Every command, at a glance

Four groups became five with the local stack. `memory` is curated facts, `docs` is documents
you point at, `repos` is whole repositories, `stack` is the three endpoints running in
containers on this machine, and the rest is configuration. Everything below exists on
**both hosts**: the CLI name is on the left, the hermes tool name on the right.

### Memory: facts worth keeping

| command | tool | what it does |
|---|---|---|
| `memory store` | `memory_store` | write one fact |
| `memory store-many` | `memory_store_many` | write a batch, all-or-nothing |
| `memory find` | `memory_find` | dense search, cheap, no re-rank |
| `memory recall` | `memory_recall` | two-stage search with re-rank, the accurate one |
| `memory get` | `memory_get` | read one by id |
| `memory list` | `memory_list` | list what is stored, newest first |
| `memory update` | `memory_update` | correct a fact in place |
| `memory delete` | `memory_delete` | remove one |
| `memory search-collections` | `memory_search_collections` | read-only search in someone else's archive |

### Docs: a document you point at

| command | tool | what it does |
|---|---|---|
| `docs index` | `docs_index` | index TEMPORARILY, with a TTL |
| `docs keep` | `docs_keep` | keep in the LIBRARY, no expiry |
| `docs search` | `docs_search` | search, returning `path:lines` and an excerpt |
| `docs list` | `docs_list` | what is indexed |
| `docs refresh` | `docs_refresh` | reindex what changed on disk |
| `docs drop` | `docs_drop` | delete one document, or the whole temporary archive |

### Repos: a whole repository, grouped

| command | tool | what it does |
|---|---|---|
| `repos register` | `repos_register` | declare a repository by name, before indexing anything |
| `repos add` | `repos_add` | index the given files under it |
| `repos search` | `repos_search` | search one repository, or `--all` to ask which ones mention it |
| `repos list` | `repos_list` | every repository, with counts and when it was last indexed |
| `repos refresh` | `repos_refresh` | reindex the files that changed on disk since indexing |
| `repos init` | `repos_init` | detect this working copy and offer to index it |
| `repos add-all` | *(CLI only)* | index the whole repository, in the background |
| `repos status` | *(CLI only)* | what is indexing, whether the daemon is up, and what is held in quarantine |
| `repos cancel` | *(CLI only)* | stop indexing; what is already indexed stays |
| `repos quarantine clear` | *(CLI only)* | release files held as unindexable, so the daemon retries them |
| `repos daemon` | *(CLI only)* | start, stop, or run the background indexer |
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
hermes exits (cleanly or killed) the daemon notices within a cycle and stops. Nothing is left
running behind you.

**A file that cannot be indexed is held, not retried forever.** An empty file, or one whose
content the embedding endpoint refuses, would otherwise be queued on every cycle: the archive
never gets a chunk for it, so the next poll sees it missing all over again. Such a file goes into
a quarantine and `repos status` names it with the reason:

```
  my-project: 2 file(s) on record as unindexable
      /path/to/empty.json: nothing indexable (empty file, or whitespace only)
```

**It releases itself when the file changes.** The record is keyed by the file's content, not its
path, so editing it (or filling in a file that was empty) puts it back in the queue with no
command to run. An outage is deliberately *not* held: an unreachable endpoint fails the whole job
and is retried, because that is a fact about the minute, not about the file.

**When the reason went away instead of the file**, release it by hand. A server limit that was
raised, a model that was swapped, an endpoint that was fixed: nothing about the file changed, so
nothing would ever let go of it on its own.

```bash
qctx repos quarantine clear my-project                 # release every held file
qctx repos quarantine clear my-project /path/to/one.json  # or just these
```

It answers with how many it released, and `nothing held for 'my-project'` when there was nothing
to do, so a mistyped name does not read as success. Releasing is not an exemption: the next
attempt decides afresh, and a file that still cannot be indexed is simply held again.

### The local stack

The endpoints running in containers on this machine are a group of their own, stood up by
`qctx install` and managed by `qctx stack`.

| command | what it does |
|---|---|
| `qctx stack status` | the managed stack: phase, the health of the three endpoints, the pins, the config, the boot line |
| `qctx stack up` | start it again from `stack.json`, repeating exactly the recorded images |
| `qctx stack down` | stop it, keeping its volume and its models |
| `qctx stack remove` | delete it; with `--purge-models` also the downloaded models, with `--purge-data` the Qdrant volume (the archive), and `--yes` to skip the confirmation |

`qctx stack up` takes `--upgrade` to pull the catalogue's images, with `--image ROLE=REF`
to override one of them (the override is recorded in `stack.json` and repeated from then
on); without `--upgrade` it repeats exactly what `stack.json` holds, and refuses an
`--image`. A stack whose install did not finish (`stack status` says so) is resumed with
`qctx install`, not with `stack up`. `qctx stack remove` never touches the configuration: it names
the fields that still point at the stack it just deleted, and with no readable state it tries
a `down` on every runtime that answers so a corrupt install still cleans up.

The install flags that stand the stack up in the first place are on `qctx install`:
`--stack` is the profile (`auto` picks the best one the host serves, or `cpu`, `amd`, `intel`,
`nvidia`, `apple`), `--runtime` is the runtime that serves it when both answer (`docker` or
`podman`), and `--image ROLE=REF` overrides one image (it appends, so several may be given).
With `--stack`, the wizard runs the whole provisioning pass: what it downloads and costs, the
ports, and where the data lands are in [install.md, the local stack](install.md#the-local-stack).

### Configuration and diagnostics: CLI only

| command | what it does |
|---|---|
| `install` | the wizard: diagnose, then offer to fix, one group at a time; `--check` reports and never writes |
| `setup` | probe everything and print the exact fix for each gap |
| `config show` | the resolved configuration |
| `config set` | write one setting to the file |
| `config detect` | ask the embedding endpoint its real dimension and store it |
| `collections list` | what exists in Qdrant, and whether each matches your model |
| `stats` | what both hosts and the daemon recorded: recall rounds and latency per host, failures by dependency, the breaker, the daemon's jobs and errors |
| `statusline` | claude-code's status line: reads the payload claude-code sends, hands the context window to the big-file guard, prints `ctx 23% · 1M`. `statusline install [--apply]` adds it to `~/.claude/settings.json` |

`install --check` covers the plumbing, Qdrant, the embedding and re-rank endpoints, the
5 collections, whether a shell-less process would find the configuration, whether each of
the two keys is set and in which spelling and where (names and lengths only, never a
value), and each host's own cutover report. `install --yes` answers yes to every group
(the script case); `install --config-only` is the configuration pass only, touching no host.

`stats` reads two files in the state directory and nothing else, so it answers when Qdrant is
down: `recall.log` (one line per recall round, written by both hosts, each line naming its host)
and `daemon.log` (starts and stops with the version, each job with its duration and outcome,
what the watcher queued, and the watcher's failures). Both rotate to their newest half past
256 KB. `repos status` also shows the version the daemon runs, and the command to restart it
when that is not the version you just ran.

## Example session

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
qctx repos register my-project                      # declare it first; the name IS the key
qctx repos add my-project $(git ls-files '*.py')    # index the files you hand it
qctx repos search "how is auth done" --limit 8      # this repository
qctx repos search "retry policy" --all              # which of my projects mention it?
qctx repos list                                     # counts, and when each was last indexed
qctx repos drop my-project --yes                    # permanent, and only ever manual
```

A repository name must be a **slug** (lowercase, digits and hyphens) because it is the
key everything filters on. `repos register "My Project"` is refused, naming `my-project`
as the remedy rather than silently rewriting what you typed.

`--all` is the one that pays for the rest: it groups **on the server**, one group per
repository, so a project with a single genuine mention still comes back next to one with
fifty. Grouping the top results on the client would answer a different question and answer
it confidently. When nothing clears the threshold it says so in words; it never returns
an empty result that reads as "no project mentions this".

For a text file, the search returns **`path:lines`** plus a short excerpt, rather than
the whole content: the consumer re-reads the exact region and works on the **current**
content, with no risk of operating on a stale snapshot. For a source that cannot be
re-read by region, it returns the text with the indexing date and a warning. In every
case, if the file changed since indexing, the result comes back marked.

## Configuration

### The four things that must be set

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

Everything else has a working default. `qctx setup` is the one command to run when
something is wrong: it probes Qdrant, the embedding endpoint (detecting the model's real
dimension), the re-rank endpoint (including which scale it answers in) and the 5
collections, and prints, for each missing item, the exact command that fixes it. In an
interactive terminal it asks and writes; **with no TTY it never blocks**, it only
reports. That is deliberate: the command is also called by agents and by scripts, and a
prompt waiting for an answer that never comes would hang the call. `--check` forces
diagnose-only mode; `--json` returns the full picture for consumption by a program.

### Precedence, and where each thing lives

**Environment variable > file > default.** The file lives at
`$XDG_CONFIG_HOME/memories-plugin/config.json`, which is
`~/.config/memories-plugin/config.json` unless `XDG_CONFIG_HOME` is set, and `QCTX_CONFIG`
overrides both. `qctx config set` prints the path it wrote to, so the answer is never a guess.

```bash
qctx collections list           # what exists in Qdrant, with dimensions
qctx config show                # the resolved configuration (file AND environment mixed)
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
| `checkpoint_interval` | `QCTX_CHECKPOINT_INTERVAL`, `REMEMBER_INTERVAL` |
| `bigfile_floor_pct` | `QCTX_BIGFILE_FLOOR_PCT`, `BIGFILE_FLOOR_PCT` |
| `bigfile_share_pct` | `QCTX_BIGFILE_SHARE_PCT`, `BIGFILE_SHARE_PCT` |

The two API keys are the only settings that **cannot** go into the config file:
`config set` refuses them and points at the environment variable instead. A plaintext
secret ends up in backups and in dotfile sync.

`memory_collection` starts out **empty** on purpose: with no explicit choice the CLI
refuses to operate, so there is no accidental write path into the wrong archive.

### How often the model is asked to save

Every N turns both hosts hand the model the procedure for writing down what the
conversation produced (the "memory checkpoint"). N is `checkpoint_interval`, 5 by default:

```bash
qctx config set checkpoint-interval 10   # every 10 turns
qctx config set checkpoint-interval 0    # never
```

Both hosts read the same setting. The claude-code hook applies it from the next prompt;
hermes from its next session (`/new` and `/resume` included), when it reads the interval again.
Like every other setting, `QCTX_CHECKPOINT_INTERVAL` in the environment wins over the file,
and `QCTX_CHECKPOINT_DISABLED=1` turns the checkpoint off whatever the interval says. A
value that is not a number falls back to 5 and says so on stderr; `config set` refuses one
outright.

### How the guard learns the context window

The guard needs the window of the model each session is using. Both hosts report it on their
own, when a session starts and whenever the model changes, and the guard reads what the host
reported for that session. Your `context_window` setting is used only when no host reported
one.

- **claude-code** reports through its status line. Install it once:

  ```bash
  qctx statusline install           # shows what it would change
  qctx statusline install --apply   # adds it to ~/.claude/settings.json
  ```

  It shows `ctx 23% · 1M` at the bottom of claude-code. A status line you already have is left
  alone: have your own status line command hand its input to `qctx statusline` as well, or
  declare the window. If the `qctx` it runs moves, `qctx setup` says so and `install --apply`
  points it at the new one. `claude -p` runs no status line.

  Before going back to a version older than 1.3.0, remove the `statusLine` entry from
  `~/.claude/settings.json`: older versions have no `qctx statusline`, and claude-code would
  show the error in its place.
- **hermes** reports through the plugin itself, with nothing to install. For a custom endpoint
  (a `custom_providers` entry), hermes can only learn the window with that endpoint's key, so
  the plugin hands it the key the entry names (`key_env`, or `api_key` written as `${VAR}`);
  the key is never written anywhere. A `key_cmd` is never run, and a key that cannot be read
  leaves that session's window as a guess, which the guard skips (`qctx setup` warns).
  The first turn of a new session on a model other than your configured default is not
  covered; the next one is.
- **The last resort** is `qctx config set context-window <tokens>`. With no report and no
  declaration, the guard does not know the window and allows every read.

`qctx setup` shows where the window comes from on your machine and what each host last
reported.

### When the big-file guard refuses a read

The guard refuses a file read when the context left after it would be less than
`bigfile_floor_pct` of the window (0.20 by default), or when the read alone would take more
than `bigfile_share_pct` of what is free (0.40 by default). Both are fractions from 0 to 1:

```bash
qctx config set bigfile-floor-pct 0.15   # refuse only when less than 15% would be left
qctx config set bigfile-share-pct 0.5    # one read may take up to half of what is free
```

Both hosts read them from the config, on every read. A share of 1 refuses only a read that
does not fit at all; a share of 0 refuses every read that costs anything. A value outside
0 to 1 falls back to the default, and `qctx setup` and `qctx config show` name the value they
ignored; `config set` and the install wizard refuse it outright.

### The case that breaks silently: a process with no shell

Everything above works for a program **you** started from **your** terminal, because it
inherits your environment. A hermes launched by systemd, by the gateway, or by cron inherits
nothing, and the symptom is not an error. It is an archive that looks simply **empty**.

Two habits hide this, and both were measured on a working machine:

- **Exporting the URLs instead of writing them.** `export QDRANT_URL=…` makes every command
  work in your shell while `config.json` stays empty. The addresses go in the
  **file** for exactly this reason.
- **`qctx config show` MIXES the file and the environment.** It prints a complete-looking
  picture while the file holds empty strings. To see what a shell-less process would read, read
  the file, and read the one the code actually writes:
  `${XDG_CONFIG_HOME:-$HOME/.config}/memories-plugin/config.json`:
  `cat "${QCTX_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/memories-plugin/config.json}"`.

So, for hermes specifically, the two keys need a second home, the one hermes itself loads:

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

**Verify it the way that actually proves it**: an empty environment plus only that file, which
is what systemd gives you:

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin bash -c '
  set -a; . ~/.hermes/.env; set +a
  qctx setup'
```

Every line must be `[ok]`. If Qdrant or the embedding endpoint fails **there** while passing in
your own shell, the gap is one of the two above.