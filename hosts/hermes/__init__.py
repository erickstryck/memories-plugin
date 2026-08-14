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

VERSION REALITY. Written against hermes v0.20.0 as INSTALLED, not as published: the
install has no `RecallStatus`, no `recall_status()` and no `unavailable_reason()`. Both
methods are implemented anyway — inert where nothing calls them, working if hermes is
upgraded. `RecallStatus` is imported lazily inside `recall_status()`, with its own
`except ImportError: return None`, because importing it at module load would make the
adapter fail to import at all against the installed v0.20.0.
"""
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
from core.prompts import INSTRUCTIONS  # noqa: E402

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
    TOP_K = int(_env("QCTX_RECALL_TOP_K", "RECALL_TOP_K", "20"))

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

        Not called by v0.20.0; kept so an upgraded hermes surfaces it for free.
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
        return []          # Task 8 fills this

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
        skip = query.skip_reason(query_text)
        if skip:
            return ""

        angles = query.angles(query_text)
        store = self._ensure_store(len(angles))
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

            return blocks.empty_block(outcome, len(angles))

        full, pointers = blocks.split_by_budget(hits, seen, round_no, self.BUDGET)
        session_state.prune(state)
        session_state.save(path, state)
        self._last_count = len(full)

        return blocks.recall_block(full, pointers, len(angles), outcome, self.BUDGET)

    def recall_status(self):
        """Deterministic "recalled N" indicator. Absent from v0.20.0; free if upgraded."""
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

    def _ensure_store(self, n_angles: int):
        """Build the memory store once per session, with timeouts DERIVED from the host's
        prefetch ceiling rather than picked per call.

        hermes runs an external provider's prefetch in a thread capped at
        HERMES_PREFETCH_BUDGET_S. `recall` issues one vector search PER ANGLE, so a
        per-call timeout multiplies by the angle count — that is exactly how the
        claude-code hook's budget came to exceed its own host deadline. Dividing a total
        among the calls that will actually be made does not multiply.
        """
        if self._store is not None:
            return self._store
        qdrant_calls = max(1, n_angles + 1)      # one existence check, then one per angle
        share = HERMES_PREFETCH_BUDGET_S / 4.0   # embed + qdrant total + rerank + headroom
        self._store = core.build_memory(
            self._cfg or core.load(),
            timeouts={"embed": share,
                      "qdrant": share / qdrant_calls,
                      "rerank": share},
        )

        return self._store

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass               # Task 7 fills this

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        return "{}"        # Task 8 fills this

    def get_config_schema(self) -> list:
        return []          # Task 9 fills this

    def save_config(self, values: dict, hermes_home: str) -> None:
        pass               # Task 9 fills this

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
    # the skeleton above covers. These five are optional hooks the ABC itself defaults to
    # no-op — in production they would be inherited for free from `_Base`. But this suite
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


def register(ctx) -> None:
    """Entry point the loader prefers. Also the string discovery greps for."""
    ctx.register_memory_provider(MemoriesProvider())
