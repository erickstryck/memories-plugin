# memories-plugin

Long-term semantic memory and a document index on top of [Qdrant](https://qdrant.tech),
for agents. A pure-Python core (stdlib only), with thin adapters for the two hosts it
ships with: claude-code and hermes-agent.

It keeps what a session would otherwise forget (a decision made, a trap already paid
for, behaviour already measured), and it keeps large files out of the agent's context:
the file is read, sliced and indexed **outside** the agent, and the search returns only
the chunks that answer.

## What you need

Three reachable endpoints, nothing else. Each can live locally or in the cloud, in any
mix, and the same configuration covers all of them:

| endpoint | local | cloud |
|---|---|---|
| a vector database | Qdrant in a container, at `http://127.0.0.1:6333` | any Qdrant deployment, with its API key |
| an embedding server | `llama-server` with `bge-m3` (1024-dim), at `http://127.0.0.1:8003` | any OpenAI-compatible `/v1/embeddings` |
| a rerank server | `llama-server` with `bge-reranker-v2-m3`, at `http://127.0.0.1:8004` | any Jina-style `/v1/rerank` (or a HuggingFace endpoint ending in `/score`) |

The two local models are the plugin's defaults:

| role | model | GGUF |
|---|---|---|
| embedding | `bge-m3` (1024-dim) | [gpustack/bge-m3-GGUF](https://huggingface.co/gpustack/bge-m3-GGUF) |
| rerank | `bge-reranker-v2-m3` | [gpustack/bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF) |

The local stack runs on CPU alone, on x86 and on ARM: a MacBook or a laptop can be the
whole backend. [Local models, step by step](#local-models).

## Install, by OS

Three pieces, in order, on every OS:

1. **The infrastructure**: the three endpoints, if you do not have them somewhere
   already. Skip the lines if you do.
2. **The host**: install the plugin into the agent you are using (two lines).
3. **The wizard**: one command that writes the configuration and the credentials,
   runs the host's cutover, and re-checks the whole stack.

The wizard is the same on every OS: it is `bash` plus `python3`, nothing OS-specific.
On Windows you run it under WSL or git-bash.

**Linux**

```bash
# 1. infrastructure, all local: Qdrant on :6333, llama.cpp on :8003 and :8004
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
llama-server --hf-repo gpustack/bge-m3-GGUF --hf-file bge-m3-Q4_K_M.gguf --embedding --port 8003 &
llama-server --hf-repo gpustack/bge-reranker-v2-m3-GGUF --hf-file bge-reranker-v2-m3-Q4_K_M.gguf --reranking -c 8192 -b 8192 -ub 8192 --port 8004 &

# 2. the host: pick the agent you use
hermes plugins install erickstryck/memories-plugin --enable --force && hermes config set memory.provider memories
# or: claude plugin marketplace add erickstryck/memories-plugin && claude plugin install memories-plugin@memories-plugin

# 3. the wizard: configuration, credentials, cutover, re-check
bash ~/.hermes/plugins/memories/scripts/install.sh
# or, if the host is claude-code:
bash "$(ls -dt ~/.claude/plugins/cache/memories-plugin/memories-plugin/*/scripts/install.sh | head -1)"
```

**macOS**

Same three steps; the only extra line is installing llama.cpp once:

```bash
brew install llama.cpp
```

Then the infrastructure, host and wizard lines exactly as on Linux.

**Windows**

```powershell
# 1. infrastructure; one terminal per llama-server
winget install llama.cpp      # once; open a new terminal after
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
llama-server --hf-repo gpustack/bge-m3-GGUF --hf-file bge-m3-Q4_K_M.gguf --embedding --port 8003
llama-server --hf-repo gpustack/bge-reranker-v2-m3-GGUF --hf-file bge-reranker-v2-m3-Q4_K_M.gguf --reranking -c 8192 -b 8192 -ub 8192 --port 8004
```

Then the host lines, and the wizard in a terminal with `bash` (WSL or git-bash). The
one Windows-specific gotcha: `~/.local/bin` is not on PATH by default. Either add it
(Settings, system, environment variables) or call the wizard by its full path; the
wizard copies `qctx` there, so once the directory is on PATH every later step is just
`qctx ...`.

### The wizard

The three ways to reach it, depending on how the plugin got onto the machine:

```bash
bash ~/.hermes/plugins/memories/scripts/install.sh          # installed by hermes
./scripts/install.sh                                        # cloned
bash "$(ls -dt ~/.claude/plugins/cache/memories-plugin/memories-plugin/*/scripts/install.sh | head -1)"
```

The last line is the claude-code case, and it is the awkward one: claude copies the
plugin into a cache directory named after the commit, and the old directories stay
behind, so a bare glob expands to all of them. Take the newest, quoted, and pass one
path.

It puts `qctx` on PATH, asks for what is missing, offers to install into whichever
host is on this machine, and re-checks. Nothing it writes is silent, and what it asks
before doing differs by group: the launcher copy, the configuration and the credential
writes are reported as they happen; the two that change a host (installing the plugin,
running that host's cutover) ask for a `y` of their own first. On a fully local stack,
answer the two key prompts by pressing Enter.

It ends with the one manual step per host: hermes approves the read guard once at a
TTY; claude-code opens a new terminal (the harness reads `settings.json` at start-up).

Flags:

```bash
qctx install --check      # reports; writes nothing
qctx install --yes        # answers yes to every group; the script case
qctx install --config-only  # the configuration pass only; touches no host
```

## Local models

The fastest local setup for the three endpoints, step by step. If your models are
already in the cloud, skip to [the cloud paragraph](#or-use-models-in-the-cloud).

### 1. Install llama.cpp

| OS | command |
|---|---|
| macOS or Linux | `brew install llama.cpp` |
| Windows | `winget install llama.cpp` |
| any, via conda | `conda install -c conda-forge llama.cpp` |

Each lands a `llama-server` binary on PATH. Prefer a prebuilt from the
[releases page](https://github.com/ggml-org/llama.cpp/releases) if you want a pinned
version: `llama-b*-bin-macos-arm64.tar.gz` (and `-macos-x64`),
`llama-b*-bin-ubuntu-x64.tar.gz` (and `-ubuntu-arm64`), or
`llama-b*-bin-win-cpu-x64.zip` (and `-win-cpu-arm64`). Unpack and put the binary on PATH.

### 2. Start Qdrant

Local, no auth, data in a named volume:

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

The same line works with `podman run`. Qdrant now answers at `http://127.0.0.1:6333`.
A local Qdrant has no API key.

### 3. Serve the two models

One model per `llama-server` process, so two processes on two ports. `--hf-repo` and
`--hf-file` download the GGUF to a cache on the first run and reuse it after. The GGUFs
carry their own pooling and an 8192-token context, so there is no `--pooling` flag to set.

```bash
# embedding, on :8003
llama-server --hf-repo gpustack/bge-m3-GGUF --hf-file bge-m3-Q4_K_M.gguf \
  --embedding --host 127.0.0.1 --port 8003

# rerank, on :8004
llama-server --hf-repo gpustack/bge-reranker-v2-m3-GGUF --hf-file bge-reranker-v2-m3-Q4_K_M.gguf \
  --reranking -c 8192 -b 8192 -ub 8192 --host 127.0.0.1 --port 8004
```

Three things here are not optional, and each fails in its own way if skipped:

- **`--reranking` is what turns on the rerank route.** A server started without it answers
  `/v1/rerank` with a 501 that says so. `--embedding` is the equivalent switch for the embed route.
- **The rerank batch flags (`-c 8192 -b 8192 -ub 8192`) are required.** A query plus a document
  runs to a few thousand tokens, and llama.cpp refuses input larger than the batch. At the 512
  default the first real rerank fails.
- **Quantization is your call on memory.** `Q4_K_M` is about 420 MiB per model and is the
  default in the commands above. `Q8_0` is about 605 MiB, `FP16` about 1.1 GiB. All three,
  plus Qdrant, fit in RAM on any of the machines in step 1.

The served filename can be anything. A single-model server ignores the `model` field of the
request, so you do not need to rename the GGUF to match the plugin's configured model name.
The two servers are unauthenticated on `127.0.0.1`. If you expose them past localhost, put
them behind something that adds a key, and set that key in the plugin's environment.

### 4. Point the plugin at them

```bash
qctx config set qdrant-url http://127.0.0.1:6333
qctx config set api-base-url http://127.0.0.1:8003
qctx config set rerank-url http://127.0.0.1:8004/rerank
qctx config set memory-collection my_memories
qctx config detect
qctx setup
```

Two of those lines look redundant and are not. `api-base-url` is the embedding server:
the plugin builds the embedding route as `{base}/embeddings`, so `http://127.0.0.1:8003`
serves embedding with no further setting. It would also build the rerank route as
`{base}/rerank`, which is `http://127.0.0.1:8003/rerank`, the wrong port. Setting
`rerank-url` explicitly is what points rerank at `:8004`. `config detect` asks the embedding
endpoint for its real vector size (1024) and stores it, so the dimension is read from the
model rather than typed.

Leave both API keys blank. A fully local stack has no auth, and the plugin sends no auth
header when a key is empty. If you later point at a remote Qdrant or a keyed model server,
export those keys in the environment (`QCTX_QDRANT_API_KEY`, `QCTX_API_KEY`), never in the
config file, which refuses them.

The wizard writes the same settings, then re-checks the whole stack, including the path a
process with no shell would read.

### Or use models in the cloud

Skip steps 1 and 3 and point the plugin at what you already have:

```bash
qctx config set qdrant-url https://your-qdrant.example
qctx config set api-base-url https://your-embed-endpoint.example/v1
qctx config set rerank-url https://your-rerank-endpoint.example/v1/rerank
qctx config set memory-collection my_memories
qctx setup
```

For the embedding endpoint, any OpenAI-compatible server works (OpenAI, OpenRouter, a
self-hosted vLLM, and the like all answer `/v1/embeddings`). For rerank, the Jina-style
body (model, query, documents) is the common shape, and a HuggingFace Inference endpoint
ending in `/score` is recognized too. If your embedding server also serves rerank on the
same base, drop the explicit `rerank-url` line and the plugin builds it from
`api-base-url`. Export the two keys in the environment before running `qctx setup`, and
set the model names if they differ from the defaults:

```bash
export QCTX_QDRANT_API_KEY="..."
export QCTX_API_KEY="..."
qctx config set embed-model text-embedding-3-small   # only if it differs from bge-m3
```

## Configuration

Four things must be set before anything works, and the wizard sets all four: where
Qdrant is, where the model servers are, the two credentials, and which collection holds
your memory. Everything else has a working default.

```bash
# the two secrets go in the environment, never in the file
export QCTX_QDRANT_API_KEY="..."
export QCTX_API_KEY="..."

# the addresses go in the file, because they are not secret
qctx config set qdrant-url https://your-qdrant.example
qctx config set api-base-url https://your-llm-server.example/v1

# name the archive your curated facts live in (it starts EMPTY on purpose)
qctx config set memory-collection my_memories

# check everything, and let it tell you what is still missing
qctx setup
```

Precedence is **environment variable > file > default**; the file lives at
`~/.config/memories-plugin/config.json`. `qctx setup` is the one command to run when
something is wrong: it probes the three endpoints and the collections, and prints the
exact command that fixes each gap.

**One case breaks silently: a process with no shell.** A hermes launched by systemd, the
gateway or cron inherits nothing from your terminal, so the two keys need a second home,
the one hermes itself loads: `~/.hermes/.env` (the wizard writes it). The addresses must
be in the config file, not only exported, for the same reason. The full picture,
including the verification that actually proves it, is in
[usage.md, the no-shell case](docs/usage.md#the-case-that-breaks-silently-a-process-with-no-shell).

## Contributing

`main` is protected. What that means depends on who you are:

- **Outside contributors**: fork, open a pull request. Direct pushes are refused, conversations
  on a PR must be resolved before it merges, and `main` can be neither force-pushed nor deleted by
  anyone at all, including the owner.
- **The owner** pushes to `main` directly. That is deliberate: this is a personal plugin with one
  maintainer, and a PR to oneself is a form with no reader. The rules exist to keep *someone else's*
  change reviewed, not to stage a review that never happens.

There is no CI yet, so a pull request is not gated on the tests. Run them before opening one; the
suite is offline, needs no network and no dependencies:

```bash
python3 -m unittest discover -s tests
```

## More

- [docs/usage.md](docs/usage.md): the full command reference (memory, docs, repos, the
  background indexer, configuration and diagnostics), with an example session.
- [docs/architecture.md](docs/architecture.md): how the two-stage search works, the two
  hosts and their install gotchas, the big-file read guard, the manual-from-clone install,
  the layout and the design decisions.