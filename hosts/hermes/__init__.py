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
upgraded — and `RecallStatus` is imported with a local fallback.
"""
import os
import sys

#: `realpath` first — see the module docstring. This is the one line that must not be
#: "simplified" to abspath. Three `dirname()` calls, not two: this file lives at
#: `hosts/hermes/__init__.py`, two directory levels below the repo root (unlike
#: `hooks/recall.py`, which is only one level below) — measured, not assumed, by the
#: symlink-install test below.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import core  # noqa: E402

try:  # hermes is present in production and absent in this repo's test run
    from agent.memory_provider import MemoryProvider as _Base
except ImportError:
    _Base = object


class MemoriesProvider(_Base):
    """memories-plugin as a hermes memory provider."""

    #: Must equal the install directory name — that is what `memory.provider` selects.
    name = "memories"

    def __init__(self):
        self._cfg = None
        self._store = None
        self._reason = ""
        self._session_id = ""

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

    def shutdown(self) -> None:
        self._store = None

    def get_tool_schemas(self) -> list:
        return []          # Task 8 fills this

    def system_prompt_block(self) -> str:
        return ""          # Task 5 fills this

    def prefetch(self, query_text: str, *, session_id: str = "") -> str:
        return ""          # Task 5 fills this

    def recall_status(self):
        return None        # Task 5 fills this

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
