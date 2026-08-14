"""hermes-agent host adapter.

A thin shell over `core`, like `hooks/` is for claude-code. Everything that decides
anything — which queries to build, the two-stage retrieval, the four block states, the
context budget, the cadence — lives in `core` and is shared. What lives here is the shape
hermes expects: a provider object with methods it calls, returning strings it injects.

WHY register(ctx) AND NOT INHERITANCE. The loader tries `register(ctx)` first and only
falls back to scanning for a `MemoryProvider` subclass, and the register path does no
issubclass check. That matters because this repo's test suite runs WITHOUT hermes on the
path — it lives in its own venv — so a hard `from agent.memory_provider import
MemoryProvider` at module level would make the adapter untestable here. The ABC is
imported when available, purely to inherit its defaults, and `object` otherwise.

WHY realpath AND NOT abspath. The plugin is installed as a symlink into
`$HERMES_HOME/plugins/memories`. Measured: through that symlink `abspath(__file__)` is
`$HERMES_HOME/plugins/memories/__init__.py`, so walking up gives `$HERMES_HOME/plugins`
and `core` is never found. Only `realpath` resolves to the repo.

VERSION REALITY. Written against hermes as INSTALLED, not as published, because the two
differ and the install is what runs. v0.20.0 had no `RecallStatus`, no `recall_status()`
and no `unavailable_reason()`; both methods were implemented anyway, inert there and
working if hermes was upgraded — and v0.20.1, measured on this machine, does declare all
three and does call `unavailable_reason()` when a selected provider reports unavailable
(`agent/agent_init.py::_warn_memory_provider_unavailable`). That is exactly the payoff for
writing them while nothing called them. `RecallStatus` is still imported lazily inside
`recall_status()`, with its own `except ImportError: return None`: a hermes old enough to
lack it must not make this adapter fail to import at all.
"""
import copy
import os
import sys
from pathlib import Path

#: `realpath` first — see the module docstring. This is the one line that must not be
#: "simplified" to abspath. Three `dirname()` calls, not two: this file lives at
#: `hosts/hermes/__init__.py`, two directory levels below the repo root (unlike
#: `hooks/recall.py`, which is only one level below) — measured, not assumed, by the
#: symlink-install test below.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import core  # noqa: E402
from core import blocks, query, session_state  # noqa: E402
from core.breaker import Breaker  # noqa: E402
from core.prompts import CHECKPOINT_PROCEDURE, INSTRUCTIONS  # noqa: E402

try:  # hermes is present in production and absent in this repo's test run
    from agent.memory_provider import MemoryProvider as _Base
except ImportError:
    _Base = object


#: hermes runs an external provider's prefetch in a thread with this ceiling
#: (`_EXTERNAL_PREFETCH_TIMEOUT_S` in agent/memory_manager.py, measured 8.0). The recall
#: measures 0.5-1.7s against the real archive, so blocking is fine and a background queue
#: would be complexity without a reason. The dependency timeouts below are derived from
#: this the same way the claude-code hook derives its own: divided among the calls that
#: will actually be made, never repeated per call.
HERMES_PREFETCH_BUDGET_S = 8.0

#: `query.angles()` returns AT MOST 3 angles — the raw text, content words, and the
#: longest sentence; see its own docstring, this is not a guess. `_ensure_store` builds
#: the store's timeouts ONCE per session and reuses them for every later prefetch, so
#: they have to be sized for the WORST angle count any prompt could ever ask for, not
#: whatever the first prompt happened to have — a 1-angle first prompt would otherwise
#: bake in a per-call qdrant share that a later 3-angle prompt then multiplies past the
#: budget, landing exactly at HERMES_PREFETCH_BUDGET_S with the reserved headroom gone.
MAX_ANGLES = 3


def _env(name: str, legacy: str, default: str) -> str:
    return os.environ.get(name) or os.environ.get(legacy) or default


def _env_num(name: str, legacy: str, default: str, kind=float):
    """Read a number from the environment without letting a typo kill the provider.

    Same tolerance the claude-code hook needed and for the same reason: this runs at import
    time, before any guard, so `QCTX_RECALL_MAX_CHARS=14k` would otherwise take the whole
    provider down instead of falling back.
    """
    raw = _env(name, legacy, default)
    try:
        return kind(raw)
    except (TypeError, ValueError):
        return kind(default)


def _safe(session_id: str) -> str:
    """A session id that is safe as a filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_id or "default"))


class MemoriesProvider(_Base):
    """memories-plugin as a hermes memory provider."""

    #: Must equal the install directory name — that is what `memory.provider` selects.
    name = "memories"

    # -- tuning, read from the environment with the SAME names the claude-code hook uses --
    #
    # Identical names are not a nicety: the equivalence test extracts every QCTX_RECALL_*
    # name from both adapters and requires the two sets to match, because a setting that
    # moves one host and not the other is a configuration that only looks shared.
    STRICT_FLOOR = _env_num("QCTX_RECALL_STRICT_FLOOR", "RECALL_MIN_SCORE", "0.58")
    DENSE_FLOOR = _env_num("QCTX_RECALL_DENSE_FLOOR", "RECALL_DENSE_FLOOR", "0.45")
    MIN_SCORE = _env_num("QCTX_RECALL_MIN_SCORE", "RECALL_RERANK_MIN_SCORE", "0.10")
    MAX_MEMORIES = _env_num("QCTX_RECALL_MAX_MEMORIES", "RECALL_MAX_MEMORIES", "6", int)
    MAX_CHARS = _env_num("QCTX_RECALL_MAX_CHARS", "RECALL_MAX_CHARS", "14000", int)
    MAX_PER_MEM = _env_num("QCTX_RECALL_MAX_PER_MEM", "RECALL_MAX_PER_MEM", "4500", int)
    BREAKER_SECONDS = _env_num("QCTX_RECALL_BREAKER", "RECALL_RERANK_BREAKER", "300")
    QDRANT_BUDGET = _env_num("QCTX_RECALL_QDRANT_BUDGET", "RECALL_QDRANT_BUDGET", "5.0")
    #: Read through `_env_num` like every other numeric knob above, and NOT with a bare
    #: `int(_env(...))` — measured: `QCTX_RECALL_TOP_K=8x` raised ValueError while this class
    #: body was executing, hermes' loader swallowed it at `logger.debug`, and because the
    #: provider came back None `agent/agent_init.py` warned about nothing at all. The user
    #: lost recall, the checkpoint and all 15 tools, silently. The same value on claude-code
    #: imports fine and degrades to a visible unavailability block.
    TOP_K = _env_num("QCTX_RECALL_TOP_K", "RECALL_TOP_K", "20", int)

    #: Turns between checkpoint nudges. Same env var as the claude-code hook
    #: (QCTX_CHECKPOINT_INTERVAL, legacy REMEMBER_INTERVAL) — the equivalence test in
    #: test_host_equivalence.py extracts both names from this file and requires them to
    #: match hooks/checkpoint.py, because a setting that moves one host and not the other
    #: is a configuration that only looks shared.
    #:
    #: Read tolerantly, through the same `_env_num` every other numeric knob above uses:
    #: this is a class attribute computed at IMPORT time, before any guard runs, and on
    #: this host it feeds `prefetch` — the same call that produces the recall block. A
    #: bad value here is worse than in the hook, where it can only cost itself; here it
    #: would take recall down with it if it raised.
    CHECKPOINT_INTERVAL = _env_num("QCTX_CHECKPOINT_INTERVAL", "REMEMBER_INTERVAL", "5", int)

    BUDGET = blocks.Budget(max_memories=MAX_MEMORIES, max_chars=MAX_CHARS,
                           max_per_mem=MAX_PER_MEM,
                           reinject_after=session_state.REINJECT_AFTER)

    def __init__(self):
        self._cfg = None
        self._store = None
        self._reason = ""
        self._session_id = ""
        self._state_dir = None
        self._last_count = 0
        #: 0 means "on_turn_start was never called" — see `_with_checkpoint`, which
        #: treats that as never due rather than asking `session_state.due(0, ...)`,
        #: which would say yes for any positive interval.
        self._turn = 0

    # -- availability ---------------------------------------------------------

    def is_available(self) -> bool:
        """Whether the plugin is configured. NO network calls — the contract forbids them,
        and `is_available` gates initialization, so a slow probe here delays every start."""
        try:
            cfg = core.load()
            cfg.require_qdrant()
            cfg.resolved_embed_url()
            cfg.require_memory_collection()
        except core.ConfigError as exc:
            self._reason = str(exc)

            return False
        self._cfg = cfg

        return True

    def unavailable_reason(self) -> str:
        """Actionable reason, since `initialize` is never reached when unavailable.

        Not called by v0.20.0; v0.20.1 appends it to its "provider unavailable" warning at
        session start. It is a log line either way, so `scripts/hermes_cutover.sh` asks the
        provider the same question before the switch is flipped, where the operator is
        looking.
        """
        return self._reason

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or "default"
        # The contract says profile-scoped storage belongs under hermes_home, not at a
        # path we hardcode. Absent the key (offline tests, an older hermes), `_state_path`
        # falls back to the plugin's own directory below.
        hermes_home = kwargs.get("hermes_home")
        if hermes_home:
            self._state_dir = Path(hermes_home) / "memories-state"

    def shutdown(self) -> None:
        self._store = None

    def get_tool_schemas(self) -> list:
        """The 15 operations the model may invoke. See tools.py for the five left out.

        A deep copy, not the module constant: hermes normalizes and wraps what it gets
        here, and an edit to a shared nested dict would outlive the call and reach the next
        session with it.
        """
        return copy.deepcopy(tools.SCHEMAS)

    def system_prompt_block(self) -> str:
        """STATIC provider info. Recall goes through prefetch, never here."""
        return (
            "Long-term memory is available and searched automatically before each turn.\n"
            "To search or write it yourself, use the memory tools, or the CLI:\n"
            "  qctx memory recall \"<topic>\"   ·   qctx memory store \"<atomic fact>\"\n"
            + INSTRUCTIONS
        )

    def prefetch(self, query_text: str, *, session_id: str = "") -> str:
        """Recall for the upcoming turn. NEVER raises, ALWAYS tells the model the truth.

        Failure is degradation, not an exception: the turn must proceed. But an absent
        result is indistinguishable from "there is no precedent" unless we say so, and a
        model that reads silence goes on to call something unprecedented when nobody
        looked. So every failure path returns the unavailability block rather than "".
        """
        try:
            return self._prefetch(query_text, session_id or self._session_id)
        except core.CoreError as exc:
            return blocks.unavailable_block(type(exc).__name__, str(exc)[:200])
        except BaseException as exc:  # noqa: BLE001 — see docstring
            return blocks.unavailable_block("the memory provider",
                                            f"{type(exc).__name__}: {exc}"[:200])

    def _prefetch(self, query_text: str, session_id: str) -> str:
        # Read fresh on every call, not cached at import time like the class-level tuning
        # constants above: unlike a hook, this provider is one long-lived object for the
        # whole process, so a value fixed at import would never see a later change. Same
        # name and same silent short-circuit as the hook — a user who disabled recall
        # expects it disabled in both hosts.
        if (os.environ.get("QCTX_RECALL_DISABLED") == "1"
                or os.environ.get("RECALL_DISABLED") == "1"):
            return self._with_checkpoint("")

        skip = query.skip_reason(query_text)
        if skip:
            return self._with_checkpoint("")

        angles = query.angles(query_text)
        store = self._ensure_store()
        breaker = Breaker(self._state_path("rerank-breaker"), self.BREAKER_SECONDS)
        idle = breaker.is_open()
        suppressed = None
        if idle is not None:
            store.reranker = None
            suppressed = f"circuit breaker: the re-rank failed {idle:.0f}s ago"

        policy = core.Policy(dense_floor=self.DENSE_FLOOR, strict_floor=self.STRICT_FLOOR,
                             min_score=self.MIN_SCORE, max_results=self.BUDGET.max_memories,
                             veto=True, order_matters=False)
        hits, outcome = store.recall(angles, policy, self.TOP_K, suppressed=suppressed)

        if outcome is not None and outcome.rerank_error:
            breaker.arm()
        elif outcome is not None and outcome.by_rerank:
            breaker.clear()

        path = self._state_path(f"recall-{_safe(session_id)}.json")
        state = session_state.load(path)
        round_no = session_state.next_round(state)
        seen = state.setdefault("seen", {})

        if not hits:
            session_state.prune(state)
            session_state.save(path, state)
            self._last_count = 0

            return self._with_checkpoint(blocks.empty_block(outcome, len(angles)))

        full, pointers = blocks.split_by_budget(hits, seen, round_no, self.BUDGET)
        session_state.prune(state)
        session_state.save(path, state)
        self._last_count = len(full)

        return self._with_checkpoint(
            blocks.recall_block(full, pointers, len(angles), outcome, self.BUDGET))

    def _with_checkpoint(self, block: str) -> str:
        """Append the write procedure when the cadence says so.

        The recall comes first and the nudge second: the memories and their rules of use
        are what the turn needs to answer, the nudge is what the session needs to
        remember.

        Two things must never happen here:
        - A provider nobody has told a turn number about (`_turn` still 0 — true of every
          `prefetch` call driven directly in a test, and of the very first read of a real
          session before `on_turn_start` runs first) must not be "due". `session_state.due`
          treats 0 as divisible by anything, so without this guard a caller who never
          calls `on_turn_start` would get a checkpoint the claude-code side never renders,
          which is exactly what test_host_equivalence.py exists to catch.
        - Any failure in composing the nudge must cost at most the nudge, never the
          `block` already built by `_prefetch`. The checkpoint and the recall now share
          one return value; a raise here must not turn a successful search into an
          UNAVAILABLE block.
        """
        try:
            if os.environ.get("QCTX_CHECKPOINT_DISABLED") == "1":
                return block
            turn = getattr(self, "_turn", 0)
            if not turn or not session_state.due(turn, self.CHECKPOINT_INTERVAL):
                return block
            nudge = CHECKPOINT_PROCEDURE.format(count=turn, interval=self.CHECKPOINT_INTERVAL)

            return f"{block}\n\n{nudge}" if block else nudge
        except BaseException:  # noqa: BLE001 — a skipped checkpoint beats a lost recall
            return block

    def recall_status(self):
        """Deterministic "recalled N" indicator. Absent from v0.20.0, present in v0.20.1 —
        which is the upgrade this was written for, and it cost nothing to be ready."""
        if not getattr(self, "_last_count", 0):
            return None
        try:
            from agent.memory_provider import RecallStatus
        except ImportError:
            return None

        return RecallStatus(provider_label="memories", count=self._last_count)

    def _state_path(self, name: str):
        """State lives under HERMES_HOME when hermes gives us one, and falls back to the
        plugin's own directory — the same one the claude-code hook uses — otherwise. The
        `initialize` contract says to use hermes_home for profile-scoped storage instead of
        hardcoding a path."""
        base = getattr(self, "_state_dir", None)
        if base is None:
            base = Path(os.environ.get("QCTX_STATE_DIR")
                        or (Path.home() / ".memories-plugin" / "state"))
            self._state_dir = base
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None          # state is a convenience; losing it must not cost a search

        return base / name

    def _ensure_store(self):
        """Build the memory store once per session, with timeouts DERIVED from the host's
        prefetch ceiling rather than picked per call.

        hermes runs an external provider's prefetch in a thread capped at
        HERMES_PREFETCH_BUDGET_S. `recall` issues one vector search PER ANGLE, so a
        per-call timeout multiplies by the angle count — that is exactly how the
        claude-code hook's budget came to exceed its own host deadline. Dividing a total
        among the calls that will actually be made does not multiply.

        Sized for MAX_ANGLES, not this call's actual angle count: the store — and the
        timeouts baked into it — is built ONCE and reused for the rest of the session, so
        it must be safe for whichever prompt asks the most angles, not just whichever
        prompt happened to build the store first. Taking `n_angles` as a parameter here
        would invite sizing it from the current call, which is exactly the mistake this
        comment exists to head off.

        `QDRANT_BUDGET` is honoured the same way the hook honours it — a ceiling the
        deployer can TIGHTEN — but capped by this host's own derived share. The hook's
        default (5.0s) was sized for its own multi-second host deadline; taking it
        literally here, on an 8s ceiling shared with embed and rerank, would blow past it
        on the default alone. min() lets the knob lower the timeout; it cannot raise it
        past what this host can afford.
        """
        if self._store is not None:
            return self._store
        qdrant_calls = MAX_ANGLES + 1              # sized for the worst case, not THIS call
        share = HERMES_PREFETCH_BUDGET_S / 4.0     # embed + qdrant total + rerank + headroom
        qdrant_total = min(self.QDRANT_BUDGET, share)
        self._store = core.build_memory(
            self._cfg or core.load(),
            timeouts={"embed": share,
                      "qdrant": qdrant_total / qdrant_calls,
                      "rerank": share},
        )

        return self._store

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Record the turn. It cannot inject — it returns None, by contract — so
        `prefetch` carries the checkpoint when this number says it is due.

        Tolerant like everything else here: a malformed `turn_number` degrades to "no
        turn recorded" (0, never due) rather than raising out of a method the host does
        not expect a return value from in the first place.

        WHAT TO RE-CHECK WHEN THE PINNED HERMES VERSION MOVES: treating turn 0 as never
        due means the checkpoint depends on this method being called before `prefetch`.
        That holds in v0.20.0, and it was verified by reading the host rather than by
        trusting the ABC's prose — `agent/turn_context.py` calls `on_turn_start` (:1252)
        and then `prefetch_all` (:1260) in one synchronous function, and it is the only
        call site pairing them. If a later hermes reaches `prefetch` from somewhere that
        never announced a turn, the write side goes quiet with nothing to show for it:
        recall keeps working, so the failure looks like a model that stopped bothering to
        save rather than a feature that is dead. Assert the pairing, don't assume it.
        """
        try:
            self._turn = int(turn_number or 0)
        except (TypeError, ValueError):
            self._turn = 0

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Run one of this provider's tools. Returns a JSON STRING, always.

        The dispatcher is what guarantees that: an unknown name, a missing argument or a
        wrong-typed one comes back as a JSON error the model can read and retry from, not
        as an exception, which the host would surface as a crashed turn.
        """
        return tools.dispatch(tool_name, args, cfg=self._config())

    def _config(self):
        """The configuration a tool call runs against.

        `is_available()` caches it, and hermes calls that before initializing. Falling back
        to a fresh `core.load()` covers a host that dispatches a tool without having asked
        — better than handing `None` to a handler and answering with an AttributeError,
        which tells the model nothing it can act on. An unconfigured install returns None,
        and the dispatcher turns that into a message naming what the operator has to do.
        """
        if self._cfg is None:
            try:
                self._cfg = core.load()
            except core.CoreError as exc:
                self._reason = str(exc)

                return None

        return self._cfg

    def get_config_schema(self) -> list:
        """Fields `hermes memory setup` walks. The same settings `qctx setup` collects.

        Derived from core.config rather than restated, so a field added there cannot become
        a setting the hermes wizard silently cannot reach.
        """
        from dataclasses import fields as dc_fields

        from core.config import DEFAULTS, ENV_ALIASES, SECRET_FIELDS, Config

        described = {
            "qdrant_url": "Qdrant base URL, e.g. https://host/qdrant",
            "qdrant_api_key": "Qdrant API key",
            "api_base_url": "OpenAI-compatible base URL serving the models",
            "api_key": "API key for that endpoint",
            "embed_url": "Full /embeddings URL (optional if api_base_url is set)",
            "rerank_url": "Full /rerank URL (optional; without it search still works)",
            "embed_model": "Embedding model name",
            "rerank_model": "Cross-encoder model name",
            "memory_collection": "Collection holding curated facts. Point it at the one "
                                 "claude-code uses to share the archive.",
            "docs_collection": "Collection for temporary document chunks (TTL)",
            "library_collection": "Collection for permanently kept documents",
            "vector_size": "Embedding dimension; `qctx config detect` measures it",
        }
        out = []
        for f in dc_fields(Config):
            secret = f.name in SECRET_FIELDS
            field = {
                "key": f.name,
                "description": described[f.name],
                "secret": secret,
                "required": f.name in ("qdrant_url", "memory_collection"),
                "type": "integer" if f.name == "vector_size" else "text",
            }
            default = DEFAULTS.get(f.name)
            if default not in ("", None):
                field["default"] = default
            if secret:
                field["env_var"] = ENV_ALIASES[f.name][0]
            out.append(field)

        return out

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Write to the plugin's native location — the SAME file claude-code reads.

        That is what makes the two hosts share one configuration instead of two that drift.
        Secrets are dropped here, not merely unsaved: `core.save` refuses them, and routing
        around that refusal would put a key in a text file that ends up in backups and in
        dotfile sync.
        """
        from core.config import SECRET_FIELDS

        patch = {k: v for k, v in (values or {}).items()
                 if k not in SECRET_FIELDS and v not in ("", None)}
        if patch:
            core.save(patch)

    # -- explicit no-ops, with the reason each one is a no-op -----------------
    #
    # Defined rather than inherited. The ABC would supply these as defaults in production,
    # but this suite runs without hermes on the path, and a default also silently absorbs a
    # method a hermes upgrade renames — a rename would become a feature that quietly stops
    # working instead of a test that fails.

    def queue_prefetch(self, query_text: str, *, session_id: str = "") -> None:
        """No background prefetch: the recall measures 0.5-1.7s against an 8s ceiling, so
        blocking in `prefetch` is simpler and costs nothing worth reclaiming."""

    def on_session_end(self, messages: list) -> None:
        """No end-of-session extraction, deliberately. claude-code has no session-end hook,
        and implementing extraction only here would make the two hosts behave differently —
        the asymmetry this whole adapter exists to avoid. See the spec, §3.4."""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages=None) -> None:
        """Writing is the checkpoint procedure's job, and the model performs it through the
        memory tools. Capturing turns here would store raw conversation, which the
        procedure explicitly discards as filler."""

    # -- explicit no-ops for the REAL ABC's other default hooks ---------------
    #
    # `dir(MemoryProvider)` off the v0.20.0 install has 19 non-underscore names, not the 16
    # the skeleton above covers (v0.20.1 has 21: it adds `unavailable_reason` and
    # `recall_status`, both already above — these five are NOT new there). They are optional
    # hooks the ABC itself defaults to no-op: in production they would be inherited for free
    # from `_Base`. But this suite
    # builds the provider with `_Base = object` (hermes off the path), so nothing here would
    # answer to them without an explicit definition, and the contract test that reads the
    # REAL surface (test_hermes_provider.py) catches exactly that gap.

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs) -> None:
        """No per-session cache to rotate: `_session_id` is set once in `initialize()` and
        nothing here keys off it yet. Revisit once a later task adds cached per-session
        state that would need to move with a `/resume`, `/branch` or compression event."""

    def on_pre_compress(self, messages: list) -> str:
        """No compression-time extraction. Empty string is the ABC's own default; defined
        here explicitly so the offline stub answers to it too, not only the real ABC."""
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "",
                      **kwargs) -> None:
        """No parent-side delegation observation implemented."""

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: dict = None) -> None:
        """No mirroring of hermes' built-in memory writes into this store."""

    def backup_paths(self) -> list:
        """Nothing to declare: Qdrant is a remote service, not a local path under
        HERMES_HOME that `hermes backup` would otherwise miss."""
        return []


def _load_tools():
    """Import the sibling `tools` module, in either of the two load modes.

    `from . import tools` is the normal path: the loader creates a synthetic namespace
    parent and registers the plugin module in `sys.modules` BEFORE exec, which is what
    makes a relative import inside the plugin resolvable at all.

    DO NOT cite `hermes_cli/plugins.py::_load_local_module` here, as an earlier version of
    this docstring did. That is the WRONG loader and naming it is how this adapter once
    shipped broken: `plugin.yaml` declares `kind: exclusive`, and that loader deliberately
    SKIPS exclusive plugins. Memory providers load through
    `plugins/memory/__init__.py::_load_provider_from_dir` (v0.20.0, measured), which
    pre-execs every sibling `*.py` BEFORE this file and swallows the failure at
    `logger.debug`. That is why `tools.py` bootstraps `sys.path` itself instead of relying
    on this module having run first — and why the `except ImportError` below cannot be
    trusted as the safety net it looks like: the pre-exec leaves a broken shell in
    `sys.modules`, so the relative import SUCCEEDS and hands back a module with nothing in
    it. See the comment at the top of `tools.py`.

    A host that instead execs this file by path WITHOUT registering it cannot resolve one —
    measured: `ModuleNotFoundError: No module named '<synthetic parent>'`, and this repo's
    own symlink-install test loads the adapter exactly that way. So the fallback loads the
    sibling by path, under a name of its own.

    What it must NOT fall back to is a bare `import tools`: hermes ships a top-level
    `tools` package, so that would import the HOST's module and leave the model with no
    memory tools while everything appeared to load.
    """
    try:
        from . import tools as module

        return module
    except ImportError:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tools.py")
        spec = importlib.util.spec_from_file_location("memories_plugin_hermes_tools", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return module


#: The tool surface, wired in rather than merely present: `get_tool_schemas` and
#: `handle_tool_call` above route through this module, so a schema that is not declared
#: here is not offered and a tool that is not routed here cannot be called.
tools = _load_tools()

#: `memory_recall` as a tool runs with the same floors automatic recall uses on this host.
#: The QCTX_RECALL_* reads have to stay in THIS file — test_host_equivalence.py extracts
#: those names from here and requires them to match the claude-code hook's — so the
#: provider hands its class to the tools module instead of the tools module reading the
#: environment a second time.
tools.bind_tuning(MemoriesProvider)


def register(ctx) -> None:
    """Entry point the loader prefers. Also the string discovery greps for."""
    ctx.register_memory_provider(MemoriesProvider())
