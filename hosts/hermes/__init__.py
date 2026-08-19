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


def _env_num(name: str, legacy: str, default: str, kind=float, minimum=None):
    """Read a number from the environment without letting a typo kill the provider.

    Same tolerance the claude-code hook needed and for the same reason: this runs at import
    time, before any guard, so `QCTX_RECALL_MAX_CHARS=14k` would otherwise take the whole
    provider down instead of falling back.

    `minimum` CLAMPS, and it does not refuse. A knob whose value would leave nothing to
    return is worse than a knob that ignores an absurd value: `retrieval` applies
    `max_memories` as a slice, so measured against three stored memories that all match,
    `QCTX_RECALL_MAX_MEMORIES=6` gave 3 hits, `=1` gave 1, and `=0` gave 0 — an empty block
    reading "There is no recorded precedent on this subject", on every prompt, from an
    archive that answered. `=-1` silently dropped the lowest hit, and `QCTX_RECALL_TOP_K=0`
    asked Qdrant for nothing at all. Since `0` meaning "unlimited" is a common deployer
    convention, that lie was one plausible typo away. Refusing to start would trade a silent
    false claim for a loud dead host; clamping keeps the archive answering and says so.

    The note goes to stderr because that is the one channel neither host reads as data:
    stdout carries the hook protocol on claude-code and the block itself here.
    """
    raw = _env(name, legacy, default)
    try:
        value = kind(raw)
    except (TypeError, ValueError):
        value = kind(default)
    if minimum is not None and value < minimum:
        print(f"memories: {name}={raw!r} would leave nothing to return — using {minimum}",
              file=sys.stderr)

        return kind(minimum)

    return value


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
    #
    # WHICH KNOBS CARRY A FLOOR, and why the others must not. `minimum=1` goes on every knob
    # that can zero the RESULT SET — the four below — because a zero there produces a false
    # claim that the archive holds nothing (see `_env_num`). It deliberately does NOT go on:
    #   - CHECKPOINT_INTERVAL: 0 DISABLES the nudge, a documented and tested feature.
    #   - BREAKER_SECONDS: 0 disables the breaker (`core.breaker.Breaker`: `cooldown <= 0`),
    #     likewise deliberate.
    #   - the three floors: they are thresholds, so 0 lets everything through rather than
    #     nothing — the opposite failure, and not a claim of absence.
    #   - QDRANT_BUDGET: a 0 timeout makes every search fail LOUDLY, and both hosts turn
    #     that into an explicit unavailability block. Degraded, but never a lie.
    STRICT_FLOOR = _env_num("QCTX_RECALL_STRICT_FLOOR", "RECALL_MIN_SCORE", "0.58")
    DENSE_FLOOR = _env_num("QCTX_RECALL_DENSE_FLOOR", "RECALL_DENSE_FLOOR", "0.45")
    MIN_SCORE = _env_num("QCTX_RECALL_MIN_SCORE", "RECALL_RERANK_MIN_SCORE", "0.10")
    MAX_MEMORIES = _env_num("QCTX_RECALL_MAX_MEMORIES", "RECALL_MAX_MEMORIES", "6", int,
                            minimum=1)
    MAX_CHARS = _env_num("QCTX_RECALL_MAX_CHARS", "RECALL_MAX_CHARS", "14000", int,
                         minimum=1)
    MAX_PER_MEM = _env_num("QCTX_RECALL_MAX_PER_MEM", "RECALL_MAX_PER_MEM", "4500", int,
                           minimum=1)
    BREAKER_SECONDS = _env_num("QCTX_RECALL_BREAKER", "RECALL_RERANK_BREAKER", "300")
    QDRANT_BUDGET = _env_num("QCTX_RECALL_QDRANT_BUDGET", "RECALL_QDRANT_BUDGET", "5.0")
    #: Hits per angle asked of Qdrant. TWO defaults for ONE knob, and the pair is not a
    #: hermes invention: `hooks/recall.py` reads the same variable with the default
    #: `"20" if store.reranker else "8"`, chosen AFTER breaker suppression. With no second
    #: stage to filter the candidates there is no reason to pull 2.5x the payload from the
    #: same archive — least of all in the degraded states the breaker exists to shed load
    #: in, and on this host's tighter 8s ceiling. `_prefetch` picks between them from the
    #: store's reranker, exactly as the hook does; a value the deployer sets explicitly wins
    #: in both states, because both read the same variable.
    #:
    #: Read through `_env_num` like every other numeric knob above, and NOT with a bare
    #: `int(_env(...))` — measured: `QCTX_RECALL_TOP_K=8x` raised ValueError while this class
    #: body was executing, hermes' loader swallowed it at `logger.debug`, and because the
    #: provider came back None `agent/agent_init.py` warned about nothing at all. The user
    #: lost recall, the checkpoint and all 15 tools, silently. The same value on claude-code
    #: imports fine and degrades to a visible unavailability block.
    TOP_K = _env_num("QCTX_RECALL_TOP_K", "RECALL_TOP_K", "20", int, minimum=1)
    TOP_K_STRICT = _env_num("QCTX_RECALL_TOP_K", "RECALL_TOP_K", "8", int, minimum=1)

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
        #: The store's REAL cross-encoder, remembered so a cooled breaker can hand it
        #: back — see `_reranker_for_this_turn`. Filled on the first prefetch, not here:
        #: the store does not exist yet.
        self._reranker = None
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
        # The provider runs INSIDE hermes, so this process IS the host: no tree to walk.
        try:
            from core import daemon, lease

            lease.write(self._session_id, "hermes")
            # The lease alone only records that hermes is alive; without also starting the
            # daemon here, watching would only ever last for a session where someone happened
            # to type a `repos` command — which the user's "se o projeto sofrer atualizações"
            # promise does not survive. `daemon.start()` is idempotent ("already" when one is
            # alive), so this is a call, not a race, and `start()` itself is responsible for
            # never leaving a claim behind that this long-lived process would then be stuck
            # holding forever if the spawn could not be confirmed — see its docstring.
            #
            # QCTX_DAEMON_AUTOSTART_DISABLED="1" skips it — same escape hatch, same naming
            # convention, as the one `hooks/lease.py` honours for the other host, so a test
            # that calls `initialize()` in-process never spawns a real detached daemon.
            if os.environ.get("QCTX_DAEMON_AUTOSTART_DISABLED") != "1":
                daemon.start()
        except Exception:                           # noqa: BLE001
            pass                                    # a missing lease or daemon costs watching
                                                    # that starts later, never a broken session

    def shutdown(self) -> None:
        self._store = None
        # With the store goes the cross-encoder remembered off it: the next `_ensure_store`
        # builds a fresh one, and handing the new store the old store's reranker would keep
        # a dependency of a connection nobody holds any more.
        self._reranker = None

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
            suppressed = f"circuit breaker: the re-rank failed {idle:.0f}s ago"
        # Decided per TURN and in BOTH directions — see `_reranker_for_this_turn`.
        self._reranker_for_this_turn(store, suppressed)

        policy = core.Policy(dense_floor=self.DENSE_FLOOR, strict_floor=self.STRICT_FLOOR,
                             min_score=self.MIN_SCORE, max_results=self.BUDGET.max_memories,
                             veto=True, order_matters=False)
        # Conditional on the second stage, and read AFTER the suppression above so a
        # breaker-disabled reranker counts as an absent one — the same order, from the same
        # source of truth, as hooks/recall.py. tests/test_host_equivalence.py compares the
        # number that arrives HERE against the number that arrives at the hook's own
        # `store.recall`, in both states.
        # `getattr` and not `store.reranker`: a store object that does not declare the
        # attribute at all must cost at most a stricter top_k, never the whole recall — the
        # same rule the rest of this file follows, since an AttributeError here would be
        # caught by `prefetch` and turned into an UNAVAILABLE block for an archive that was
        # perfectly reachable. `core.build_memory` always sets it (to None when no
        # cross-encoder is configured), so in production this reads the real value.
        top_k = self.TOP_K if getattr(store, "reranker", None) else self.TOP_K_STRICT
        try:
            hits, outcome = store.recall(angles, policy, top_k, suppressed=suppressed)
        except Exception:                           # noqa: BLE001
            # The archive is unreachable; the MODEL endpoint is a different server and says
            # nothing about it, so this failure must not cost us the window. `except Exception`
            # and NOT `finally`: a `finally` also runs while a BaseException propagates, so
            # Ctrl-C used to buy the user a multi-second probe before their interrupt was
            # allowed through — and a BaseException from the probe would have replaced the
            # KeyboardInterrupt they asked for. Being cancelled is not a failure of anything.
            self._refresh_window(session_id)

            raise

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
            self._refresh_window(session_id)

            return self._with_checkpoint(blocks.empty_block(outcome, len(angles)))

        full, pointers = blocks.split_by_budget(hits, seen, round_no, self.BUDGET)
        session_state.prune(state)
        session_state.save(path, state)
        self._sweep_dead_state(round_no)
        self._last_count = len(full)
        self._refresh_window(session_id)

        return self._with_checkpoint(
            blocks.recall_block(full, pointers, len(angles), outcome, self.BUDGET))

    def _refresh_window(self, session_id: str) -> None:
        """Teaches the window cache what this session's model endpoint reports, if anything.

        WHY IT LIVES HERE AND NOT AT ONE `return`. The window is knowable on this host, and
        this method is called from every path of `_prefetch` that has ALREADY paid for a
        network round trip — including the one with no hits. A fresh hermes install, or an
        empty memory collection, returns no hits on every single prefetch; a refresh that only
        ran on the with-hits path would never run at all there, the cache would stay empty
        forever, step 2 of the cascade would never fire, and for a model absent from the
        ceiling table the window would resolve to 0. The guard would then allow every read,
        permanently, with no error anywhere. It is deliberately NOT called from the two genuine
        pre-network short-circuits (recall disabled, a trivial prompt): neither has paid for
        anything, and putting a network call behind a user's explicit "recall off" would be a
        worse bug than the one this closes.

        WHY IT GOES LAST. It talks to a network with a multi-second timeout, which makes it the
        most interruptible thing in the method. Every caller runs it AFTER the breaker and
        `session_state.save`, so a turn killed mid-probe still keeps the round number and the
        seen-set it had already earned.

        Never raises: a window we did not learn is simply the next step of the cascade, and it
        must never turn a working recall into an unavailable block.
        """
        try:
            from . import bigfile
            from .endpoint import refresh_window

            refresh_window(bigfile.model_of(bigfile.state_db_path(), session_id))
        except Exception:                           # noqa: BLE001 — see the docstring
            pass

    def _reranker_for_this_turn(self, store, suppressed: str | None) -> None:
        """Point the CACHED store's reranker at what the breaker says right now.

        The claude-code hook can write `store.reranker = None` and forget about it: it is a
        fresh process per prompt, so the next prompt rebuilds everything. This host is one
        long-lived object whose store — and the connection its 8s budget depends on — is
        built once per session, so a suppression written into that store OUTLIVES the turn
        that wrote it. Measured, three prefetches with the breaker armed only for the first:

            turn 1 (breaker OPEN)  top_k=8 reranker=False suppressed='circuit breaker: …'
            turn 2 (breaker cold)  top_k=8 reranker=False suppressed=None
            turn 3 (breaker cold)  top_k=8 reranker=False suppressed=None

        One rerank failure — the event the breaker exists FOR, because the shared GPU
        saturates for minutes — degraded every remaining turn of the session to a
        single-stage pipeline at the strict floor with the stricter top_k. Worse than the
        lost precision: `suppressed` went back to None once the cooldown passed, so the
        degradation note disappeared while the degradation itself did not, and an empty
        result printed "There is no recorded precedent on this subject" — a flat claim of
        absence produced by a pipeline that was still crippled. That is the one statement
        this whole package exists to prevent.

        So the decision is re-applied every turn, in BOTH directions, from the breaker's
        CURRENT state: the real cross-encoder is remembered here rather than thrown away,
        and a cooled breaker restores two-stage retrieval exactly as the hook's next
        process would. Rebuilding the store per turn would fix the symptom by discarding
        the cached connection the budget is sized around — the wrong half to give up.

        The assignment is guarded for the same reason `top_k` is read with `getattr`: a
        store object that refuses it must cost at most a stricter top_k, never a recall
        turned into an UNAVAILABLE block by `prefetch`'s catch-all while the archive was
        perfectly reachable.
        """
        if self._reranker is None:
            self._reranker = getattr(store, "reranker", None)
        try:
            store.reranker = None if suppressed else self._reranker
        except Exception:  # noqa: BLE001 — a stricter top_k beats a lost recall
            pass

    def _sweep_dead_state(self, round_no: int) -> None:
        """Delete the state of sessions that are not coming back.

        The cadence is the claude-code hook's, read from it and not invented here: the same
        `core.session_state.sweep_if_due`, called at the same point — after the state save,
        on a round that found memories. Without this call the directory grew one file per
        session forever, which is verbatim what `purge_dead`'s own docstring says it exists
        to prevent; the purging had already moved into `core` so both hosts would share it,
        and only one host called it.

        It must never cost a recall. `sweep_if_due` already swallows everything, and this
        second guard is here because the alternative — a housekeeping failure turning a
        successful search into an UNAVAILABLE block via `prefetch`'s catch-all — is worth
        strictly less than one unswept file.
        """
        try:
            base = getattr(self, "_state_dir", None)
            if base is not None:
                session_state.sweep_if_due(base, round_no)
        except BaseException:  # noqa: BLE001 — an unswept file beats a lost recall
            pass

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
            "repos_collection": "Collection holding repository chunks, grouped by repo",
            "repos_registry_collection": "Collection holding one entry per indexed repository",
            "vector_size": "Embedding dimension; `qctx config detect` measures it",
            "context_window": "Model's context window in tokens; overrides the built-in "
                               "table when the bare model name is ambiguous (e.g. a 1M "
                               "variant). 0 means unknown/use the table.",
        }
        out = []
        for f in dc_fields(Config):
            secret = f.name in SECRET_FIELDS
            field = {
                "key": f.name,
                "description": described[f.name],
                "secret": secret,
                "required": f.name in ("qdrant_url", "memory_collection"),
                "type": "integer" if f.name in ("vector_size", "context_window") else "text",
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
