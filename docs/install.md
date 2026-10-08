# Installing, in the long form

The [README](../README.md) is the three-minute path; this file is what each of its
steps does, and what it costs: the per-host installs with their measured gotchas,
and the manual install from a clone. Read it when a line from the README does not
do what you expected, or when you install without the wizard.

Why the hosts are the way they are is in
[architecture.md](architecture.md); the commands themselves are in
[usage.md](usage.md).

## Install on Claude Code

The three-line fresh-machine sequence is in the [README](../README.md#install-by-os). This is
what that sequence does, and what it does not.

The wizard's line is the awkward one, because claude copies the plugin into a cache
directory named after the commit and the old directories stay behind: a bare glob expands
to all of them and the extra paths land on `qctx install` as unrecognised arguments. So
the README takes the newest, quoted, and passes one path.

The wizard does everything the hermes one does (keys, configuration, re-check) and offers
to run `scripts/cutover.sh --apply`, which on this host also removes hooks you registered
by hand in `settings.json` and any legacy qdrant-memory MCP server, with a dated backup
of both files. It ends with the one step it cannot do for you: **open a new terminal**;
the harness reads `settings.json` at start-up.

The repository is at once a plugin and a single-plugin marketplace, so two commands do it,
**from git**, with nothing cloned by hand. The wizard runs exactly these two when it finds
a `claude` binary and no plugin yet.

For **development**, add it from a path instead, and edits take effect with no reinstall:

```bash
claude plugin marketplace add ~/dev/mnemosine
claude plugin install mnemosine@mnemosine
```

Then **open a new terminal**; the harness reads `settings.json` at start-up.

To pick up a newer commit later, the plugin must be named **with its marketplace**; the bare name
answers `Plugin "mnemosine" not found`, which reads like a broken install and is not one:

```bash
claude plugin marketplace update mnemosine
claude plugin update mnemosine@mnemosine      # name@marketplace, not just the name
```

The version it reports is the one in the manifests, and `core/version.py` is where that number
is decided. Three manifests cannot read Python, so the string is written out four times and
`tests/test_installable_from_git.py::TestTheVersionIsONENumber` fails on the first disagreement:
the number went stale here once (0.3.0 declared, 0.2.0 installed, measured), and that drift is
now a red test instead of a silent lie. To pin an exact build, `--ref v1.0.0` (or a commit SHA)
still does it. `claude plugin details mnemosine` lists what it found: 3 skills, 4 hooks.

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
`~/.mnemosine/state/recall.log` gets one round per prompt, **one, not two**.

**If you already had equivalent hooks registered by hand in `settings.json`, remove them
in the SAME pass; `./scripts/cutover.sh --apply` does both at once. Two sets fire
together and recall lands twice in one prompt, which is worse than having none: it doubles
the context cost and adds no information.

## Install on hermes-agent

The three-line fresh hermes sequence is in the [README](../README.md#install-by-os): install
with `--enable --force`, set `memory.provider mnemosine`, run the wizard. What each of
the three costs:

**Why `--force` is required, and what you are agreeing to.** hermes scans a cloned plugin before
installing it, and this tree scores `caution`: it ships a thousand tests that shell out to `git`
and `python3`, design documents full of example commands, and the finding that actually matters,
**two scripts that edit host configuration**. That is the cutover's declared job, not a
surprise, and `--force` is you confirming it. Read the report before you type it; the plugin does
what it says, and the scanner is right to make you look.

Do **not** reach for `plugins.scan_on_install: false` to skip the report: that disables scanning
for every plugin from anywhere, which is a policy change rather than a decision about this one.

**Two things will stop the install before it starts.** Both were measured, and neither message
mentions the real cause:

- *"Invalid plugin name 'mnemosine': resolves outside the plugins directory."* You already have a
  **development symlink** at `$HERMES_HOME/plugins/mnemosine`. The installer resolves the target
  path, follows the link out of the plugins directory, and refuses. `--force` does not help: the
  name is validated first. Remove the link, or keep it and skip the install (see below).
- A `dangerous` verdict, where `--force` does **not** override. One `critical` finding is enough,
  and a critical is not necessarily an action: a plugin that merely *writes out* the hermes config
  path in prose trips one. This tree keeps itself clear of those, and a test fails if a new one
  appears; see `tests/test_installable_from_git.py`.

**On your development machine, prefer the symlink and skip the install entirely.** A clone is
a copy: edits to your checkout change nothing until `hermes plugins update mnemosine`. The symlink is
always at HEAD, with no update step, which is what you want where you are editing the code.

`--ref <40-char-sha>` pins an exact commit. The installer names any key missing from
`~/.hermes/.env`, because a hermes started by systemd or the gateway has no shell, and a key
that lives only in an interactive environment is one it will not have. **The keys never go in a
config file.**

Installing the SUBDIRECTORY (`erickstryck/mnemosine/hosts/hermes`) is accepted syntax and
does not work: the adapter imports `core/` from the repository root, which a subdirectory install
does not bring. It fails silently (hermes' loader swallows a broken provider at debug level), so
install the whole repository, which is what the root `__init__.py` is for.

For **development**, keep the symlink install instead, so edits take effect immediately:

```bash
ln -s ~/dev/mnemosine/hosts/hermes ~/.hermes/plugins/mnemosine
```

**Installing is not the whole job on this host, and the missing half is silent.** A hermes plugin
manifest has no field for shell hooks: they live only in `$HERMES_HOME/config.yaml`, behind a
first-use consent allowlist, so `plugins install` gives you memory and the 22 tools, and the
big-file read guard is **not** registered. That is what the script below is for:

```bash
./scripts/hermes_cutover.sh            # reports what it would do; writes NOTHING
./scripts/hermes_cutover.sh --apply    # installs, with a dated backup of every file it edits
```

It reports the credentials, the URLs a shell-less hermes would find, the symlink, the provider
selection and the guard, and `--apply` writes only what is missing. It accepts all three install
shapes and leaves whichever you have alone: the CLONE that `hermes plugins install` puts at
`$HERMES_HOME/plugins/mnemosine` (it copies the repository there, it does not link), a symlink to
the repository root, and a symlink to `hosts/hermes`. All three load, because the root
`__init__.py` re-exports the same provider the adapter directory does. The test is the inode,
not the text of the link, so a relative symlink counts as the same install as an absolute one.

**Then approve the hook once.** After registration `hermes hooks list` shows it
`✗ not allowlisted`: the first file read of a new session asks at the TTY, and **a hermes with no
TTY skips the hook silently** until it has been approved once. Approving records it in
`~/.hermes/shell-hooks-allowlist.json`, which every later run (TTY or not) then honours.

```bash
hermes hooks list        # ✓ allowed, with the approval timestamp, once it is done
```

Do **not** reach for `hooks_auto_accept: true` to skip that step: it auto-approves every future
hook from anywhere, which is a policy change, not a fix for this one.

**Then let the sandbox see the keys.** `execute_code` runs its Python in a child process that
scrubs every environment variable whose name contains `KEY`, `TOKEN`, `SECRET` or `AUTH`
(among others: `PASSWORD`, `CREDENTIAL`, `BEARER`, `APIKEY`, `WEBHOOK`, `DSN`), so
`qctx` works from the `terminal` tool and fails to authenticate from `execute_code`, on the same
machine, in the same session. Measured 2026-09-16: the terminal child had 163 variables including
all four credentials; the `execute_code` child had 54 and none of them.

The scrub is deliberate hardening, and the supported way through it is the opt-in allowlist:

```bash
hermes config set terminal.env_passthrough '["QCTX_QDRANT_API_KEY","QCTX_API_KEY"]'
```

Only the names listed pass; hermes' own provider credentials (`ANTHROPIC_API_KEY` and the rest)
are refused even if listed, which is the GHSA-rhgp-j443-p4rf blocklist and not something to work
around. Skip this and nothing breaks loudly: `qctx` simply rejects the credentials from one tool
and not from the other.

**List the spellings you actually set.** The allowlist matches variable NAMES, not the setting
behind them, so it does not follow the aliases `core/config.py` accepts. If your keys are in the
legacy spellings this manifest asks for, `QDRANT_SERVICE_API_KEY` and `SERVER_API_KEY`, list
those two instead: the line above allowlists only the canonical `QCTX_*` pair and would leave a
legacy install failing exactly as before, with the same silence.

Run the first form first and read it. It checks the credentials, the collections, the
provider entry, the hook block and whether the context window is declared, and prints the
exact fix for each gap. The `--apply` form writes a `.bak-<timestamp>` beside anything it
edits and re-reads the result to confirm it took, rather than trusting that the write
returned zero.

Then restart hermes.

## Installing by hand, from a clone

**From zero to working, in order.** Each step is expanded here or in the README;
nothing here is optional except where it says so, and the two steps people skip are 3 and
5, and both fail silently.

| # | step | why it is not optional |
|---|---|---|
| 1 | a reachable Qdrant, and an embedding endpoint | there is nothing to search without them |
| 2 | `qctx config set` the **addresses** into the file | a process without your shell reads the file, and only the file |
| 3 | `export` the **two keys**, and for hermes also `~/.hermes/.env` | keys never enter the config file; a shell-less hermes has none otherwise |
| 4 | `qctx config set memory-collection <name>` | it is empty on purpose, so nothing can write into the wrong archive |
| 5 | install on the host, and on hermes run the cutover too | `plugins install` cannot register a shell hook, so the read guard is not installed by it |
| 6 | `qctx setup`, then the no-shell check | the only way to know steps 2 and 3 actually took |

```bash
git clone git@github.com:erickstryck/mnemosine.git
cd mnemosine
python3 -m unittest discover -s tests    # offline; no network, no deps
ln -s "$PWD/bin/qctx" ~/.local/bin/qctx  # so `qctx` works from anywhere
```

There is no `pip install`: the core uses only the standard library. That is
deliberate. This code runs inside hooks fired on every interaction, and a missing
dependency would turn an environment failure into a silent loss of functionality.

## The local stack

The other way to reach the three endpoints: let the wizard stand up the
infrastructure itself. With Docker or Podman already installed, run:

```bash
qctx install --stack auto
```

and the wizard does the whole local setup in one pass: it detects the runtimes
that answer, proves which GPU profile (if any) the host serves, pulls the two
images, downloads the two models, renders the compose file, brings the three
containers up, verifies that the endpoints answer and that the calibration is
within budget, and then points the configuration at the stack it just built.
`auto` picks the best profile the host serves; the other choices are `cpu`,
`amd`, `intel`, `nvidia` and `apple`, and `--runtime docker` or `--runtime
podman` decides when both runtimes answer.

**What it costs, measured.** The download is the llama.cpp server image, 294 MiB
for the amd64 build and 290 MiB for the arm64 build, the Qdrant image, about
71 MiB, and the two models, 836 MiB together (bge-m3 at 437,778,496 bytes and
bge-reranker-v2-m3 at 438,376,864 bytes, both Q4_K_M). The resident memory the
two llama-servers hold depends on the batch flags they run with
(`-c -b -ub 8192`) and the thread count, so the calibration reports the figure
for your machine at install time; on this machine, measured with the stack's
own configuration, the cpu embed server holds about 2.3 GB and the cpu rerank
server about 5 GB, and on a GPU the embed server drops to about 190 MB while
the rerank server still holds about 4.7 GB. The disk gate keeps 400 MiB of
headroom beyond the models, because the Qdrant volume grows in it.

**The ports.** Everything publishes on 127.0.0.1 only, so nothing leaves the
machine: Qdrant on 6333, the embed server on 8003, the rerank server on 8004.
A port that is already busy moves to the first free one at `port + 10000`, and
the move is reported.

**Where the data lives.** One directory holds the state, the compose file and
the models: `$QCTX_STACK_DIR` when set, otherwise
`${XDG_DATA_HOME:-~/.local/share}/mnemosine/stack`, and inside it
`stack.json`, `compose.yaml` and `models/`. Qdrant's own data lives in a named
volume the engine prefixes with the project:
`mnemosine_mnemosine-qdrant` by default. `qctx stack down` stops
the containers and keeps both; `qctx stack remove` deletes the files, and only
deletes the models (`--purge-models`) and the volume (`--purge-data`) when
asked.

**After a reboot.** The containers carry `restart: always`, and how that
survives depends on the runtime. On Docker the daemon has to start at boot, and
that brings the containers back with it. On Podman on Linux, enable
`podman-restart.service` for the session and linger, with
`systemctl --user enable --now podman-restart.service` and
`loginctl enable-linger $USER`. On the other Podman case (macOS), `podman
machine start` brings the VM back and the containers come back with it.

**Where each option runs.** The profiles and their runtimes, one row per
platform:

| platform | option | Docker | Podman | label | still needs |
|---|---|---|---|---|---|
| linux | cpu | yes | yes | Docker and Podman | nothing |
| linux | amd | yes | yes | Docker and Podman | a DRI render node for that vendor |
| linux | intel | yes | yes | Docker and Podman | a DRI render node for that vendor |
| linux | nvidia | yes | yes | Docker and Podman | an NVIDIA driver that lists the card, plus the CDI spec |
| linux | apple | - | - | not available here | not in phase 1 |
| macos | cpu | yes | yes | Docker and Podman | nothing |
| macos | amd | - | - | not available here | not in phase 1 |
| macos | intel | - | - | not available here | not in phase 1 |
| macos | nvidia | - | - | not available here | not in phase 1 |
| macos | apple | - | yes | Podman only | an Apple GPU |

The "still needs" column is the hardware prerequisite in plain words: the
driver and node the container has to reach, or nothing for the cpu profile,
which runs everywhere. The rows marked "not available here" are where the
phase-1 matrix is the spec: the Apple profile is experimental and macOS-only,
and the DRI and NVIDIA profiles are Linux-only. The menu still lists a profile
the platform cannot run, marked unavailable with its reason (on Linux the Apple
line says it runs on macOS only), so you can see why it is not offered; picking
it repeats the reason and the menu comes back.

**Windows.** The local stack is not offered on Windows (WSL included) in phase
1; it arrives in phase 3. Until then the step says so in one line and points at
the manual path in [Local models, step by step](../README.md#local-models).

**The environment trap.** The configuration resolves as environment variable,
then file, then default, so a variable already exported in the shell wins over
what the wizard just wrote to the file. The stack points the configuration at
its own URLs, but a `QCTX_QDRANT_URL` or a legacy `QDRANT_URL` (and the same
for the embedding, rerank and collection names) that still points elsewhere
keeps winning every time the shell reasserts itself. The installer names each
one of these, with its value, and tells you to remove its export from your
shell rc; it never edits the rc for you, because the rc is yours.

