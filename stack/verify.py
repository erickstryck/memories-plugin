"""What the stack runs with, and whether it is fast enough for each host.

`stack/health.py` owns "is it up"; this module owns the two questions that come
after the endpoints answer: "does it behave like the configuration the plugin
will actually use" (the `functional` check, reusing `core.setup.diagnose` and
filtering to the three services this step stood up) and "is it fast enough"
(`calibrate`, one measurement per host against the budgets the CLI passes in).
The two stay apart because they answer different failures: a misconfigured
stack fails the first, a too-slow one passes it and degrades at recall time.

The configuration is built WITHOUT the user's file or environment: `load` with
`NO_FILE` and an empty `env` resolves defaults only, and the URLs the stack
just stood up are patched on top. That is deliberate — an old `embed_url` in
the user's file or a `QDRANT_URL` in the environment pointing elsewhere would
otherwise make the check probe a server the installer never started, and the
plan names exactly that trap ("Config e ambiente que já apontam para outro
lugar"). `env_overrides` is the other half of it: after the check, the caller
reports which environment variables still point elsewhere, so the user can
unset them or let the installer rewrite the file.

The calibration measures the WARM regime: the first call to a freshly loaded
llama-server costs 3 to 7 times the next one (M7), and recall lives in the
warm regime, so a warm-up call goes to each server before anything is
timed, with the same inputs that are then timed (a smaller warm-up leaves the
first call of the measured shape, and its GPU setup cost, inside the number).
The samples are the plugin's own working set — one embed of
`HARD_MAX_CHARS`, the ceiling a chunk can reach, and one rerank of
`CALIBRATION_RERANK_DOCS` documents of `TARGET_CHARS`, the `TOP_K` the recall
hook ranks. An over-budget measurement is a WARNING named after the host,
never a raise: the stack works, it just misses that host's deadline, and the
fix is a faster profile, not a red install.
"""
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from core import build_embedder, build_reranker
from core import config as core_config
from core import chunk as core_chunk
from core import setup as core_setup
from core.config import Config
from core.setup import Check

from . import catalog

#: The bge-m3 dimension, re-exported so `stack_config`'s default reads beside
#: the catalogue that owns it; the Qdrant collection is sized from it.
EMBED_DIM = catalog.EMBED_DIM

#: The three checks that belong to the services this step stood up. `diagnose`
#: runs the whole picture (context windows, collections, the user's settings);
#: the stack verification keeps only its own three, in the order the installer
#: reports them.
FUNCTIONAL_CHECKS = ("Qdrant", "Embedding", "Re-rank")

#: The calibration sends the same pool the recall hook sends: its `TOP_K`
#: documents. The reranker client then judges only its `max_docs` (12) of them
#: (`core/reranking.py`), so the measurement is the 20-sent / 12-judged call
#: recall actually makes. `tests/test_stack_verify.py` reads the sent count out
#: of `hooks/recall.py` by AST so the two numbers cannot drift apart.
CALIBRATION_RERANK_DOCS = 20


@dataclass(frozen=True)
class Budget:
    """The latency budget of one host: the seconds its recall may spend.

    The values come from the CLI, which names the hosts (`stack` must not
    know them); this dataclass is the shape they take in here.
    """
    host: str
    embed_s: float
    rerank_s: float


def stack_urls(ports: Mapping[str, int]) -> dict[str, str]:
    """The config fields the stack writes, for the ports it stood up on.

    M2: llama-server b11382 answers a bare `POST /embeddings` with a raw JSON
    list and `core/embedding.py` crashes on it; only the `/v1` path has the
    OpenAI shape. So `api_base_url` ends in `/v1` and `embed_url` stays empty
    (the plugin reaches `/v1/embeddings` through it), and the rerank URL keeps
    its own path because the reranker's contract lives on `/v1/rerank`.
    """
    return {
        "qdrant_url": f"http://127.0.0.1:{ports['qdrant']}",
        "api_base_url": f"http://127.0.0.1:{ports['embed']}/v1",
        "embed_url": "",
        "rerank_url": f"http://127.0.0.1:{ports['rerank']}/v1/rerank",
    }


def config_url_pairs(eff: Config, ports: Mapping[str, int]) -> list[tuple[str, str, str]]:
    """The three (field, the value the config carries, the URL the stack owns)
    pairs, for the fields that name a host -- `embed_url` kept out because the
    stack always clears it to `""`. This is the one home of the pairing; a caller
    that asks "does the config still point at this stack" (the `status` line,
    the `remove` warning) computes it from these instead of re-listing the
    fields. The stack URLs come from `stack_urls`, so the two cannot drift."""
    urls = stack_urls(ports)
    return [("qdrant_url", eff.qdrant_url, urls["qdrant_url"]),
            ("api_base_url", eff.api_base_url, urls["api_base_url"]),
            ("rerank_url", eff.rerank_url, urls["rerank_url"])]


def stack_config(ports: Mapping[str, int], *, vector_size: int = EMBED_DIM) -> Config:
    """The `Config` the verification runs with: defaults plus the stack's URLs.

    `load` with `NO_FILE` and an empty `env` resolves WITHOUT the user's file
    and WITHOUT their environment (see the module docstring for the trap), so
    what is verified is exactly what the installer is about to write.
    """
    cfg = core_config.load(core_config.NO_FILE, env={})
    urls = stack_urls(ports)
    return Config(
        qdrant_url=urls["qdrant_url"],
        qdrant_api_key=cfg.qdrant_api_key,
        api_base_url=urls["api_base_url"],
        api_key=cfg.api_key,
        embed_url=urls["embed_url"],
        rerank_url=urls["rerank_url"],
        embed_model=cfg.embed_model,
        rerank_model=cfg.rerank_model,
        memory_collection=cfg.memory_collection,
        docs_collection=cfg.docs_collection,
        library_collection=cfg.library_collection,
        repos_collection=cfg.repos_collection,
        repos_registry_collection=cfg.repos_registry_collection,
        vector_size=vector_size,
        context_window=cfg.context_window,
        checkpoint_interval=cfg.checkpoint_interval,
        bigfile_floor_pct=cfg.bigfile_floor_pct,
        bigfile_share_pct=cfg.bigfile_share_pct,
    )


def functional(cfg: Config, *, diagnose: Callable[[Config], dict] = core_setup.diagnose
               ) -> tuple[list[Check], int | None]:
    """The stack's three checks, out of the full `diagnose` picture.

    The spec names this as a reuse: the diagnostic runs whole (its `Embedding`
    check already compares the dimension to `vector_size`, its `Re-rank`
    check already probes with Paris in first) and the verification keeps only
    the three services it stood up, in the order `diagnose` reports them,
    which is the `FUNCTIONAL_CHECKS` order. The `detected_dim` comes back for
    the caller, which writes `vector_size` from it.
    """
    picture = diagnose(cfg)
    rows = [row for row in picture["checks"] if row["name"] in FUNCTIONAL_CHECKS]
    checks = [Check(**row) for row in rows]
    return checks, picture.get("detected_dim")


def functional_ok(checks: list[Check]) -> bool:
    """Whether the stack passes its functional check.

    Every check must be `ok`. In `diagnose` a failed Re-rank is a WARNING —
    optional, the search works without it — but on a stack the reranker is one
    of the three services the step just stood up, so here a warning counts as
    a failure: a stack whose reranker does not rank is not a working stack.
    """
    return all(check.ok for check in checks)


def calibrate(cfg: Config, budgets: list[Budget], *,
              embedder=None, reranker=None,
              clock: Callable[[], float] = time.monotonic,
              memory: Callable[[], dict[str, int]] | None = None
              ) -> tuple[list[Check], dict]:
    """Time the warm embed and rerank once, and compare each host's budget.

    One measurement per kind, shared by every host (the servers are the same
    for all of them): a warm-up embed and a warm-up rerank first (M7), on the
    same inputs that are then timed: one `HARD_MAX_CHARS` text and a rerank of
    `CALIBRATION_RERANK_DOCS` `TARGET_CHARS` documents — the same 20-sent /
    12-judged call recall makes, because the client judges its `max_docs`.
    `memory`, when given, is the runtime's `stats` for the stack's containers,
    read once per host so the check carries the containers' footprint beside
    the timing.

    The answer is one `Check` per host, named after it, plus an info dict the
    installer reports. An over-budget measurement sets `warning=True` and
    names the host, the kind, the measured value and the budget; it NEVER
    raises — the stack works, it just misses the deadline, and the fix is a
    faster profile. A rerank whose server FAILED (the client's `ok=False`
    answer, e.g. the 15 s timeout on a slow CPU) is the same kind of warning:
    `rerank_s` is `None` (no rerank was measured), and recall on that host
    degrades to the first stage until the server answers.
    """
    if embedder is None:
        embedder = build_embedder(cfg)
    if reranker is None:
        reranker = build_reranker(cfg)
    query = "what is the capital of France?"
    embed_text = "x" * (core_chunk.HARD_MAX_CHARS - 1) + " "
    rerank_pool = ["y" * core_chunk.TARGET_CHARS] * CALIBRATION_RERANK_DOCS

    # M7: the first call to a freshly loaded server costs 3 to 7 times the
    # next one, and recall lives in the warm regime, so warm up both before
    # any clock starts -- with the SAME inputs that are then timed. On the GPU
    # the first call of a new shape pays the backend's setup again, so a small
    # warm-up left that cost inside the measurement (measured 2026-10-08, Intel
    # Vulkan2, fresh server: rerank 5.30 s after a one-document warm-up, 1.98 s
    # after a same-shape one), and every GPU install warned of a budget miss
    # that recall never sees.
    embedder.embed([embed_text])
    if reranker is not None:
        reranker.rank(query, rerank_pool)

    embed_start = clock()
    embedder.embed([embed_text])
    embed_s = clock() - embed_start

    rerank_s: float | None = None
    rerank_error: str | None = None
    if reranker is not None:
        rerank_start = clock()
        _pairs, rerank_info = reranker.rank(query, rerank_pool)
        if rerank_info.get("ok"):
            rerank_s = clock() - rerank_start
        else:
            # `rank` NEVER raises: a failure (the client's 15 s timeout on a slow
            # CPU, a refused connection) comes back as `ok=False`. Reading the
            # clock around it would measure the failure, not a rerank, and a host
            # whose recall cannot rerank must not read "within budget".
            rerank_error = rerank_info.get("error") or "no usable hits"

    checks: list[Check] = []
    info: dict = {}
    # A rerank the host does not have at all (no `rerank_url`) is not a failure:
    # it is `ok` and simply not measured. Only a rerank whose server FAILED
    # (the client's `ok=False`, e.g. the 15 s timeout on a slow CPU) is the
    # warning below, because that host's recall degrades to the first stage.
    no_rerank = reranker is None
    for budget in budgets:
        over = []
        if embed_s > budget.embed_s:
            over.append(f"embed {embed_s:.1f} s over its {budget.embed_s:.1f} s budget")
        if no_rerank:
            detail = (f"{budget.host}: embed {embed_s:.1f} s (budget {budget.embed_s:.1f} s), "
                      "rerank not measured (no rerank configured)")
            suffix = ""
            fix = None
        elif rerank_s is not None:
            if rerank_s > budget.rerank_s:
                over.append(f"rerank {rerank_s:.1f} s over its {budget.rerank_s:.1f} s budget")
            detail = (f"{budget.host}: embed {embed_s:.1f} s (budget {budget.embed_s:.1f} s), "
                      f"rerank {rerank_s:.1f} s (budget {budget.rerank_s:.1f} s)")
            suffix = " — recall will miss its deadline on this host"
            fix = "a GPU profile is faster here; reinstall the stack with one"
        else:
            reason = rerank_error or "no usable hits"
            over.append(f"rerank failed: {reason}")
            detail = (f"{budget.host}: embed {embed_s:.1f} s (budget {budget.embed_s:.1f} s), "
                      f"rerank not measured ({reason})")
            suffix = (" — recall on this host degrades to the first stage "
                      "until the rerank server answers")
            fix = ("check the rerank server's log (qctx stack up shows it); "
                   "the stack is up and the config will be written")
        if over:
            detail = (f"{budget.host}: " + "; ".join(over) + suffix)
            check = Check(budget.host, False, detail, fix, warning=True)
        else:
            check = Check(budget.host, True, detail)
        checks.append(check)
        info[budget.host] = {
            "embed_s": embed_s,
            "rerank_s": rerank_s,
            "memory": memory() if memory is not None else None,
        }
    return checks, info


def env_overrides(env: Mapping[str, str], wanted: dict[str, str]) -> list[tuple[str, str]]:
    """The environment variables that still point elsewhere, `(name, value)`.

    For each field in `wanted` (the config the installer is about to write),
    the FIRST non-blank alias of `ENV_ALIASES[field]` whose value differs from
    the wanted one is named — the canonical `QCTX_*` beats the legacy spell,
    and a blank value is ignored the way `load` ignores it. The installer
    reports these before it touches the file: an override the user set is
    theirs to clear, and a file that loses to it would look broken the moment
    the environment reasserts itself.
    """
    out: list[tuple[str, str]] = []
    for field, want in wanted.items():
        for name in core_config.ENV_ALIASES.get(field, ()):
            value = (env.get(name) or "").strip()
            if not value:
                continue
            if value != want:
                out.append((name, env[name]))
            break
    return out
