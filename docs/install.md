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
claude plugin marketplace add ~/dev/memories-plugin
claude plugin install memories-plugin@memories-plugin
```

Then **open a new terminal**; the harness reads `settings.json` at start-up.

To pick up a newer commit later, the plugin must be named **with its marketplace**; the bare name
answers `Plugin "memories-plugin" not found`, which reads like a broken install and is not one:

```bash
claude plugin marketplace update memories-plugin
claude plugin update memories-plugin@memories-plugin      # name@marketplace, not just the name
```

The version it reports is the **commit SHA**, because the manifests declare no version on purpose
A hand-maintained number goes stale and this one already had (0.3.0 declared, 0.2.0 installed).
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
`~/.memories-plugin/state/recall.log` gets one round per prompt, **one, not two**.

**If you already had equivalent hooks registered by hand in `settings.json`, remove them
in the SAME pass; `./scripts/cutover.sh --apply` does both at once. Two sets fire
together and recall lands twice in one prompt, which is worse than having none: it doubles
the context cost and adds no information.

## Install on hermes-agent

The three-line fresh hermes sequence is in the [README](../README.md#install-by-os): install
with `--enable --force`, set `memory.provider memories`, run the wizard. What each of
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

- *"Invalid plugin name 'memories': resolves outside the plugins directory."* You already have a
  **development symlink** at `$HERMES_HOME/plugins/memories`. The installer resolves the target
  path, follows the link out of the plugins directory, and refuses. `--force` does not help: the
  name is validated first. Remove the link, or keep it and skip the install (see below).
- A `dangerous` verdict, where `--force` does **not** override. One `critical` finding is enough,
  and a critical is not necessarily an action: a plugin that merely *writes out* the hermes config
  path in prose trips one. This tree keeps itself clear of those, and a test fails if a new one
  appears; see `tests/test_installable_from_git.py`.

**On your development machine, prefer the symlink and skip the install entirely.** A clone is
a copy: edits to your checkout change nothing until `hermes plugins update memories`. The symlink is
always at HEAD, with no update step, which is what you want where you are editing the code.

`--ref <40-char-sha>` pins an exact commit. The installer names any key missing from
`~/.hermes/.env`, because a hermes started by systemd or the gateway has no shell, and a key
that lives only in an interactive environment is one it will not have. **The keys never go in a
config file.**

Installing the SUBDIRECTORY (`erickstryck/memories-plugin/hosts/hermes`) is accepted syntax and
does not work: the adapter imports `core/` from the repository root, which a subdirectory install
does not bring. It fails silently (hermes' loader swallows a broken provider at debug level), so
install the whole repository, which is what the root `__init__.py` is for.

For **development**, keep the symlink install instead, so edits take effect immediately:

```bash
ln -s ~/dev/memories-plugin/hosts/hermes ~/.hermes/plugins/memories
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
`$HERMES_HOME/plugins/memories` (it copies the repository there, it does not link), a symlink to
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
git clone git@github.com:erickstryck/memories-plugin.git
cd memories-plugin
python3 -m unittest discover -s tests    # offline; no network, no deps
ln -s "$PWD/bin/qctx" ~/.local/bin/qctx  # so `qctx` works from anywhere
```

There is no `pip install`: the core uses only the standard library. That is
deliberate. This code runs inside hooks fired on every interaction, and a missing
dependency would turn an environment failure into a silent loss of functionality.

