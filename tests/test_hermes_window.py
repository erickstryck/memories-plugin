"""The hermes side of the context window the big-file guard needs.

The hermes guard is a shell hook, a subprocess that does not import hermes, so it cannot
ask hermes for the window. The memory provider runs INSIDE hermes, and hermes calls its
`pre_llm_call` hook at the start of every turn, before any tool call, with the session id
and the model already switched (read on 2026-10-02; no other provider hook carries the
model, and none carries the window). So the provider asks hermes the question hermes asks
itself, `agent.model_metadata.get_model_context_length`, for the session's own route, and
publishes the answer per session (`core.hostwindow`). Only when the route changed: the
first turn of a session, and the turn after a `/model`.

THE ROUTE is the one hermes records for the session, read with hermes' own reader
(`SessionDB.session_gateway_runtime`). The `billing_*` columns are not it: hermes writes
them once, at the first accounted call, and a `/model` to another provider leaves them as
they were (read on 2026-10-02).

THE KEY goes to custom routes only, by the user's decision of 2026-10-02. Measured on the
Eukrio route (Qwen3.8-27B): without the key hermes answers 131,072 from its own table of
model names, going to the network on every call; with the key it answers 524,288, what the
endpoint's /models reports. A known provider (Anthropic) answers from hermes' catalogue
without a key and without the network, and is asked without one.

hermes is not importable here, so these tests stand in for its modules. What they pin is
what the provider hands hermes and what it does with the answer; `TestTheFakesMatchHermes`
holds the two copied hermes functions to the real ones.
"""
import ast
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import hostwindow  # noqa: E402
from hosts.hermes import window  # noqa: E402

FALLBACK = 256_000
KEY_ENV = "QCTX_TEST_EUKRIO_KEY"
KEY = "sk-test-not-a-real-key-7f3a"
EUKRIO = {"name": "Eukrio", "base_url": "https://ai.example/api/v1", "key_env": KEY_ENV,
          "model": "Qwen3.8-27B"}
OPEN_LOCAL = {"name": "Local vLLM", "base_url": "http://127.0.0.1:8000/v1"}
BY_REFERENCE = {"name": "Other", "base_url": "https://other.example/v1",
                "api_key": "${QCTX_TEST_OTHER_KEY}"}
NO_SWITCH = json.dumps({"provider": None, "base_url": None, "api_mode": None,
                        "gateway_runtime": None})


def custom_provider_aliases(display_name, provider_key=""):
    """`hermes_cli.providers.custom_provider_aliases`, copied (held to the real one below)."""
    aliases = set()
    for value in (display_name, provider_key):
        raw = str(value or "").strip().lower()
        if not raw:
            continue
        normalized = raw.replace(" ", "-")
        aliases.update({raw, normalized,
                        normalized if normalized.startswith("custom:") else f"custom:{normalized}"})
        if normalized.startswith("custom:"):
            suffix = normalized.split(":", 1)[1]
            if suffix:
                aliases.update({suffix, f"custom:{normalized}"})

    return frozenset(aliases)


def session_gateway_runtime(session_meta):
    """`hermes_state.SessionDB.session_gateway_runtime`, copied (held to the real one below)."""
    raw = (session_meta or {}).get("model_config")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    runtime = raw.get("gateway_runtime")
    if isinstance(runtime, dict) and runtime.get("provider"):
        return {k: v for k, v in runtime.items() if v is not None}
    top_level = {key: raw.get(key) for key in ("provider", "base_url", "api_mode") if raw.get(key)}
    billing_provider = str((session_meta or {}).get("billing_provider") or "").strip()
    if billing_provider and billing_provider.lower() not in {"auto", "custom"}:
        top_level.setdefault("provider", billing_provider)
    if top_level:
        return top_level
    if not isinstance(runtime, dict):
        return {}

    return {k: v for k, v in runtime.items() if v is not None}


class FakeHermes:
    """The hermes modules the provider imports, as far as it uses them."""

    def __init__(self, answer=1_000_000, pin=None, config=None, raises=None, custom=None,
                 secrets=None):
        self.calls = []
        self.answer, self.pin, self.raises, self.secrets = answer, pin, raises, secrets
        self.config = config if config is not None else {
            "model": {"default": "claude-opus-5-5", "provider": "anthropic"}}
        self.custom_providers = custom if custom is not None else [EUKRIO, OPEN_LOCAL, BY_REFERENCE]

    def get_model_context_length(self, model, base_url="", api_key="",
                                 config_context_length=None, provider="", custom_providers=None):
        self.calls.append({"model": model, "base_url": base_url, "api_key": api_key,
                           "config_context_length": config_context_length,
                           "provider": provider, "custom_providers": custom_providers})
        if self.raises:
            raise self.raises
        if isinstance(config_context_length, int) and config_context_length > 0:
            return config_context_length

        return self.answer

    def modules(self) -> dict:
        metadata = types.ModuleType("agent.model_metadata")
        metadata.get_model_context_length = self.get_model_context_length
        metadata.DEFAULT_FALLBACK_CONTEXT = FALLBACK
        init = types.ModuleType("agent.agent_init")
        init.config_context_length_for_runtime = lambda runtime, cfg=None: self.pin
        agent = types.ModuleType("agent")
        agent.model_metadata, agent.agent_init = metadata, init
        config = types.ModuleType("hermes_cli.config")
        config.load_config = lambda: self.config
        config.get_compatible_custom_providers = lambda cfg=None: self.custom_providers
        config.get_custom_provider_context_length = lambda *a, **kw: None
        providers = types.ModuleType("hermes_cli.providers")
        providers.custom_provider_aliases = custom_provider_aliases
        cli = types.ModuleType("hermes_cli")
        cli.config, cli.providers = config, providers
        state = types.ModuleType("hermes_state")
        state.SessionDB = type("SessionDB", (), {
            "session_gateway_runtime": staticmethod(session_gateway_runtime)})

        modules = {"agent": agent, "agent.model_metadata": metadata, "agent.agent_init": init,
                   "hermes_cli": cli, "hermes_cli.config": config,
                   "hermes_cli.providers": providers, "hermes_state": state}
        if self.secrets is not None:
            scope = types.ModuleType("agent.secret_scope")

            def get_secret_str(name, default=""):
                if isinstance(self.secrets, Exception):
                    raise self.secrets
                return self.secrets.get(name, default)

            scope.get_secret_str = get_secret_str
            agent.secret_scope = modules["agent.secret_scope"] = scope

        return modules


def a_state_db(rows) -> str:
    """`sessions` with the columns the provider reads, shaped like the live table."""
    path = os.path.join(tempfile.mkdtemp(), "state.db")
    con = sqlite3.connect(path)
    con.execute("create table sessions (id text primary key, model text, model_config text, "
                "billing_provider text, billing_base_url text)")
    con.executemany("insert into sessions values (?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()

    return path


class PublisherCase(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp()
        self.db = a_state_db([("s1", "claude-opus-5-5", NO_SWITCH, "anthropic",
                               "https://api.anthropic.com")])
        patcher = mock.patch.dict(os.environ, {"QCTX_STATE_DIR": self.state,
                                               "QCTX_HERMES_STATE_DB": self.db, KEY_ENV: KEY})
        patcher.start()
        self.addCleanup(patcher.stop)
        window.forget()
        self.addCleanup(window.forget)

    def turn(self, hermes, session="s1", model="claude-opus-5-5"):
        with mock.patch.dict(sys.modules, hermes.modules()):
            return window.on_pre_llm_call(session_id=session, model=model, is_first_turn=True,
                                          platform="cli", turn_id="t1")

    def set_row(self, session, model_config=NO_SWITCH, billing_provider=None, model=None):
        con = sqlite3.connect(self.db)
        con.execute("insert or replace into sessions values (?, ?, ?, ?, ?)",
                    (session, model, model_config, billing_provider, None))
        con.commit()
        con.close()


class TestTheRouteIsTheOneHermesRecordsForTheSession(PublisherCase):
    def test_a_session_without_a_switch_is_asked_on_its_billing_provider(self):
        hermes = FakeHermes()
        self.turn(hermes)
        (call,) = hermes.calls
        self.assertEqual({k: call[k] for k in ("model", "provider", "base_url", "api_key")},
                         {"model": "claude-opus-5-5", "provider": "anthropic", "base_url": "",
                          "api_key": ""})
        self.assertIs(call["custom_providers"], hermes.custom_providers)

    def test_a_switch_to_another_provider_is_followed_although_billing_is_not(self):
        """The defect the billing columns would have shipped: hermes leaves them on the
        first provider, while the switch writes the route into `model_config`."""
        route = {"provider": "custom:eukrio", "base_url": "https://ai.example/api/v1",
                 "api_mode": "chat_completions"}
        self.set_row("s1", json.dumps({**route, "gateway_runtime": route}), "anthropic")
        hermes = FakeHermes(answer=524_288)
        self.turn(hermes, model="Qwen3.8-27B")
        self.assertEqual((hermes.calls[0]["provider"], hermes.calls[0]["base_url"]),
                         ("custom:eukrio", "https://ai.example/api/v1"))
        self.assertEqual(hostwindow.read("s1").window, 524_288)

    def test_the_tui_shape_with_top_level_keys_is_followed(self):
        self.set_row("s1", json.dumps({"provider": "custom:eukrio",
                                       "base_url": "https://ai.example/api/v1"}), "anthropic")
        hermes = FakeHermes()
        self.turn(hermes, model="Qwen3.8-27B")
        self.assertEqual(hermes.calls[0]["provider"], "custom:eukrio")

    def test_a_new_session_on_the_default_model_is_asked_on_the_configured_route(self):
        hermes = FakeHermes(config={"model": {"default": "claude-opus-5-5",
                                              "provider": "anthropic",
                                              "base_url": "https://api.anthropic.com"}})
        self.turn(hermes, session="fresh")
        self.assertEqual((hermes.calls[0]["provider"], hermes.calls[0]["base_url"]),
                         ("anthropic", "https://api.anthropic.com"))
        self.assertEqual(hostwindow.read("fresh").window, 1_000_000)

    def test_a_new_session_on_another_model_waits_until_hermes_records_its_route(self):
        """The configured route belongs to the configured model; asking it about another
        model would publish an answer for the wrong endpoint."""
        hermes = FakeHermes(answer=524_288)
        self.turn(hermes, session="fresh", model="Qwen3.8-27B")
        self.assertEqual(hermes.calls, [])
        self.assertIsNone(hostwindow.read("fresh"))
        self.set_row("fresh", NO_SWITCH, "custom:eukrio", "Qwen3.8-27B")
        self.turn(hermes, session="fresh", model="Qwen3.8-27B")
        self.assertEqual(hostwindow.read("fresh").window, 524_288)


class TestTheKeyGoesOnlyToCustomRoutes(PublisherCase):
    def eukrio_session(self, hermes=None):
        self.set_row("s1", NO_SWITCH, "custom:eukrio", "Qwen3.8-27B")
        hermes = hermes or FakeHermes(answer=524_288)
        self.turn(hermes, model="Qwen3.8-27B")

        return hermes

    def test_a_custom_route_is_asked_with_the_key_its_entry_names(self):
        hermes = self.eukrio_session()
        self.assertEqual((hermes.calls[0]["api_key"], hermes.calls[0]["base_url"]),
                         (KEY, "https://ai.example/api/v1"))
        got = hostwindow.read("s1")
        self.assertEqual((got.window, got.guess), (524_288, False))

    def test_a_known_provider_is_never_asked_with_a_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-should-not-travel"}):
            hermes = FakeHermes()
            self.turn(hermes)
        self.assertEqual(hermes.calls[0]["api_key"], "")

    def test_a_custom_entry_that_declares_no_key_is_asked_without_one(self):
        self.set_row("s1", NO_SWITCH, "custom:local-vllm", "qwen-local")
        hermes = FakeHermes(answer=32_768)
        self.turn(hermes, model="qwen-local")
        self.assertEqual((hermes.calls[0]["api_key"], hermes.calls[0]["base_url"]),
                         ("", "http://127.0.0.1:8000/v1"))
        self.assertFalse(hostwindow.read("s1").guess)

    def test_a_key_given_as_a_variable_reference_is_read_from_the_environment(self):
        self.set_row("s1", NO_SWITCH, "custom:other", "m")
        hermes = FakeHermes()
        with mock.patch.dict(os.environ, {"QCTX_TEST_OTHER_KEY": "sk-other"}):
            self.turn(hermes, model="m")
        self.assertEqual(hermes.calls[0]["api_key"], "sk-other")

    def test_a_custom_route_whose_key_is_missing_publishes_a_guess(self):
        """The endpoint cannot answer without its key, so what hermes answers comes from
        somewhere else (its table of names, measured): published, and skipped."""
        with mock.patch.dict(os.environ):
            del os.environ[KEY_ENV]
            hermes = self.eukrio_session(FakeHermes(answer=131_072))
        self.assertEqual(hermes.calls[0]["api_key"], "")
        got = hostwindow.read("s1")
        self.assertEqual((got.window, got.guess), (131_072, True))

    def test_a_pinned_window_needs_no_key(self):
        with mock.patch.dict(os.environ):
            del os.environ[KEY_ENV]
            self.eukrio_session(FakeHermes(pin=500_000))
        got = hostwindow.read("s1")
        self.assertEqual((got.window, got.guess), (500_000, False))

    def test_the_key_never_reaches_the_record(self):
        self.eukrio_session()
        written = b"".join(p.read_bytes() for p in Path(self.state).rglob("*") if p.is_file())
        self.assertTrue(written, "nothing was published")
        self.assertNotIn(KEY.encode(), written)

    def entry_session(self, entry, env=None):
        self.set_row("s1", NO_SWITCH, f"custom:{entry['name'].lower()}", "m")
        hermes = FakeHermes(custom=[entry], answer=131_072)
        with mock.patch.dict(os.environ, env or {}):
            self.turn(hermes, model="m")

        return hermes

    def test_an_env_prefixed_reference_is_read_from_the_environment(self):
        hermes = self.entry_session({"name": "Prefixed", "base_url": "https://p.example/v1",
                                     "api_key": "${env:QCTX_TEST_P_KEY}"},
                                    {"QCTX_TEST_P_KEY": "sk-p"})
        self.assertEqual(hermes.calls[0]["api_key"], "sk-p")

    def test_a_reference_this_cannot_read_is_never_sent_and_publishes_a_guess(self):
        hermes = self.entry_session({"name": "Odd", "base_url": "https://o.example/v1",
                                     "api_key": "${vault:secret/key}"})
        self.assertEqual(hermes.calls[0]["api_key"], "")
        self.assertTrue(hostwindow.read("s1").guess)

    def test_api_key_env_names_the_variable_like_key_env(self):
        hermes = self.entry_session({"name": "Aliased", "base_url": "https://a.example/v1",
                                     "api_key_env": "QCTX_TEST_A_KEY"},
                                    {"QCTX_TEST_A_KEY": "sk-a"})
        self.assertEqual(hermes.calls[0]["api_key"], "sk-a")

    def test_the_variable_is_read_through_hermes_secret_scope(self):
        """A multiplexed gateway serves several profiles in one process, and hermes reads
        a key_env through the profile's scope (#84079): the process environment holds
        another profile's value."""
        hermes = self.eukrio_session(FakeHermes(answer=524_288, secrets={KEY_ENV: "sk-scoped"}))
        self.assertEqual(hermes.calls[0]["api_key"], "sk-scoped")

    def test_hermes_refusing_the_read_leaves_no_key(self):
        """Multiplexing with no profile scope, hermes fails closed (`UnscopedSecretError`):
        the process environment would hold another profile's value."""
        hermes = self.eukrio_session(FakeHermes(answer=131_072,
                                                secrets=RuntimeError("no secret scope")))
        self.assertEqual(hermes.calls[0]["api_key"], "")
        self.assertTrue(hostwindow.read("s1").guess)

    def test_a_key_command_is_never_run_and_publishes_a_guess(self):
        hermes = self.entry_session({"name": "Cmd", "base_url": "https://c.example/v1",
                                     "key_cmd": "pass show eukrio"})
        self.assertEqual(hermes.calls[0]["api_key"], "")
        self.assertTrue(hostwindow.read("s1").guess)


class TheModelBlocksKeyStaysWithTheModelBlocksEndpoint(PublisherCase):
    """A `model:` block that is itself a custom endpoint carries its own key. hermes hands
    that key over only when the URL IS `model.base_url` (runtime_provider_custom.py,
    #67453); a session recorded on another URL, a direct alias for instance, must not
    receive it."""

    CONFIG = {"model": {"default": "local-model", "provider": "custom",
                        "base_url": "https://mine.example/v1", "key_env": KEY_ENV}}

    def session_on(self, recorded_url):
        if recorded_url:
            route = {"provider": "custom", "base_url": recorded_url}
            self.set_row("s1", json.dumps({**route, "gateway_runtime": route}), None,
                         "local-model")
        else:
            self.set_row("s1", NO_SWITCH, None, "local-model")
        hermes = FakeHermes(config=self.CONFIG, answer=65_536)
        self.turn(hermes, model="local-model")

        return hermes.calls[0]

    def test_the_configured_endpoint_gets_its_key(self):
        call = self.session_on(None)
        self.assertEqual((call["base_url"], call["api_key"]), ("https://mine.example/v1", KEY))
        self.assertFalse(hostwindow.read("s1").guess)

    def test_a_recorded_route_without_a_url_is_the_configured_endpoint(self):
        route = {"provider": "custom"}
        self.set_row("s1", json.dumps({**route, "gateway_runtime": route}), None, "local-model")
        hermes = FakeHermes(config=self.CONFIG, answer=65_536)
        self.turn(hermes, model="local-model")
        self.assertEqual((hermes.calls[0]["base_url"], hermes.calls[0]["api_key"]),
                         ("https://mine.example/v1", KEY))

    def test_the_same_endpoint_recorded_with_a_trailing_slash_gets_its_key(self):
        self.assertEqual(self.session_on("https://mine.example/v1/")["api_key"], KEY)

    def test_another_endpoint_never_gets_the_model_blocks_key(self):
        call = self.session_on("https://api.other.example/v1")
        self.assertEqual((call["base_url"], call["api_key"]), ("https://api.other.example/v1", ""))
        self.assertTrue(hostwindow.read("s1").guess)


class LoadedByPath(unittest.TestCase):
    def test_a_sibling_is_loaded_once(self):
        """Loaded by path (no package), the guard module used to be executed again on every
        turn."""
        path = REPO / "hosts" / "hermes" / "window.py"
        spec = importlib.util.spec_from_file_location("qctx_test_window_by_path", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ("bigfile", "endpoint"):
            with self.subTest(name=name):
                self.assertIs(module._sibling(name), module._sibling(name))


class TestTheProviderPublishesWhatHermesResolved(PublisherCase):
    def test_the_first_turn_publishes_hermes_answer(self):
        self.turn(FakeHermes(answer=1_000_000))
        got = hostwindow.read("s1")
        self.assertEqual((got.model, got.window, got.source, got.guess),
                         ("claude-opus-5-5", 1_000_000, "hermes", False))

    def test_a_pinned_context_length_is_handed_to_hermes(self):
        hermes = FakeHermes(pin=180_000)
        self.turn(hermes)
        self.assertEqual(hermes.calls[0]["config_context_length"], 180_000)
        self.assertEqual(hostwindow.read("s1").window, 180_000)

    def test_the_same_route_again_does_not_ask_twice(self):
        hermes = FakeHermes()
        self.turn(hermes)
        self.turn(hermes)
        self.assertEqual(len(hermes.calls), 1)

    def test_a_model_switch_asks_again_and_republishes(self):
        self.turn(FakeHermes(answer=1_000_000))
        hermes = FakeHermes(answer=200_000)
        self.turn(hermes, model="claude-haiku-4-5-20251001")
        self.assertEqual(len(hermes.calls), 1)
        got = hostwindow.read("s1")
        self.assertEqual((got.model, got.window), ("claude-haiku-4-5-20251001", 200_000))


class TestAGuessIsPublishedAsOne(PublisherCase):
    def test_hermes_fallback_is_a_guess(self):
        self.turn(FakeHermes(answer=FALLBACK))
        self.assertTrue(hostwindow.read("s1").guess)

    def test_a_pinned_value_equal_to_the_fallback_is_not_a_guess(self):
        self.turn(FakeHermes(answer=1_000_000, pin=FALLBACK))
        got = hostwindow.read("s1")
        self.assertEqual((got.window, got.guess), (FALLBACK, False))


class TestThePublisherNeverCostsHermesATurn(PublisherCase):
    def test_it_returns_none_because_hermes_injects_what_it_returns(self):
        self.assertIsNone(self.turn(FakeHermes()))
        self.assertIsNone(self.turn(FakeHermes(raises=RuntimeError("boom")), session="s2"))

    def test_a_failing_resolution_publishes_nothing_and_raises_nothing(self):
        self.turn(FakeHermes(raises=RuntimeError("boom")))
        self.assertIsNone(hostwindow.read("s1"))

    def test_a_failure_is_retried_on_the_next_turn(self):
        self.turn(FakeHermes(raises=RuntimeError("boom")))
        self.turn(FakeHermes(answer=1_000_000))
        self.assertEqual(hostwindow.read("s1").window, 1_000_000)

    def test_without_hermes_nothing_happens(self):
        absent = dict.fromkeys(FakeHermes().modules())
        with mock.patch.dict(sys.modules, absent):
            self.assertIsNone(window.on_pre_llm_call(session_id="s1", model="m"))
        self.assertIsNone(hostwindow.read("s1"))

    def test_no_session_or_no_model_is_ignored(self):
        hermes = FakeHermes()
        self.turn(hermes, session="")
        self.turn(hermes, model="")
        self.assertEqual(hermes.calls, [])


class TestTheHookIsRegistered(unittest.TestCase):
    def test_register_hooks_pre_llm_call(self):
        from hosts.hermes import register

        class Ctx:
            def __init__(self):
                self.hooks = []

            def register_memory_provider(self, provider):
                self.provider = provider

            def register_hook(self, name, callback):
                self.hooks.append((name, callback))

        ctx = Ctx()
        register(ctx)
        self.assertEqual([name for name, _ in ctx.hooks], ["pre_llm_call"])
        self.assertEqual(ctx.hooks[0][1].__name__, "on_pre_llm_call")

    def test_a_host_without_register_hook_still_gets_the_provider(self):
        from hosts.hermes import MemoriesProvider, register

        class Ctx:
            provider = None

            def register_memory_provider(self, provider):
                self.provider = provider

        ctx = Ctx()
        register(ctx)
        self.assertIsInstance(ctx.provider, MemoriesProvider)

    def test_a_register_hook_that_raises_does_not_cost_the_provider(self):
        from hosts.hermes import MemoriesProvider, register

        class Ctx:
            provider = None

            def register_memory_provider(self, provider):
                self.provider = provider

            def register_hook(self, name, callback):
                raise ValueError("unknown hook")

        ctx = Ctx()
        with mock.patch("os.write"):
            register(ctx)
        self.assertIsInstance(ctx.provider, MemoriesProvider)


class TestTheGuardReadsWhatTheProviderPublished(PublisherCase):
    """The real publisher in-process, the real guard as the subprocess hermes starts."""

    def test_the_published_window_decides(self):
        from tests.test_bigfile_hermes import a_file_of, a_session_using, guard_env, run_guard
        guard_db = a_session_using(4 * 60_000)
        env = guard_env(guard_db, QCTX_CONTEXT_WINDOW="0")
        with mock.patch.dict(os.environ, {"QCTX_STATE_DIR": env["QCTX_STATE_DIR"]}):
            self.turn(FakeHermes(answer=100_000))
        out, code = run_guard(a_file_of(400_000), guard_db, env=env)
        self.assertTrue(out.strip(), "a 100k window did not refuse a read the guard should")
        with mock.patch.dict(os.environ, {"QCTX_STATE_DIR": env["QCTX_STATE_DIR"]}):
            window.forget()
            self.turn(FakeHermes(answer=1_000_000), model="claude-opus-5-5[1m]")
        out, code = run_guard(a_file_of(400_000), guard_db, env=env)
        self.assertEqual((out, code), ("", 0))


HERMES_AGENT = Path(os.environ.get("QCTX_HERMES_AGENT_DIR")
                    or Path.home() / ".hermes" / "hermes-agent")
HERMES_SOURCES = [HERMES_AGENT / name for name in
                  ("hermes_state_gateway.py", "hermes_state.py", "hermes_cli/providers.py")]


def hermes_functions(source: Path, *names: str, **globals_) -> dict:
    """Compile the named functions from hermes' SOURCE, without importing hermes.

    Importing `hermes_cli` runs hermes' launcher bootstrap (`hermes_bootstrap` calls
    `venv_sync.prepare_launch`), which republishes hermes' launchers for the CURRENT HOME.
    Under a throwaway HOME it points the user's real `hermes` command at a runtime under
    /tmp: measured on 2026-10-02, when a probe did exactly that. Compiling the functions from
    the source runs their behaviour and nothing else.
    """
    tree = ast.parse(source.read_text())
    found = {node.name: node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in names}
    missing = set(names) - set(found)
    if missing:
        raise AssertionError(f"{sorted(missing)} not found in {source}")
    namespace = dict(globals_)
    for name in names:
        node = found[name]
        node.decorator_list, node.returns = [], None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

    return namespace


def hermes_constant(source: Path, name: str):
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name
                                                for t in node.targets):
            return eval(compile(ast.Expression(node.value), str(source), "eval"),
                        {"__builtins__": {}, "frozenset": frozenset, "set": set})
    raise AssertionError(f"{name} not found in {source}")


@unittest.skipUnless(os.environ.get("QCTX_INTEGRATION") == "1"
                     and all(path.exists() for path in HERMES_SOURCES),
                     "needs QCTX_INTEGRATION=1 and the hermes-agent sources")
class TestTheFakesMatchHermes(unittest.TestCase):
    """The two hermes functions copied above, against hermes' own code, over one table,
    without starting hermes or importing a single hermes module."""

    ROWS = [
        {"model_config": NO_SWITCH, "billing_provider": "anthropic"},
        {"model_config": json.dumps({"gateway_runtime": {"provider": "custom:eukrio",
                                                         "base_url": "https://x/v1",
                                                         "api_mode": None}}),
         "billing_provider": "anthropic"},
        {"model_config": json.dumps({"provider": "custom:eukrio", "base_url": "https://x/v1"}),
         "billing_provider": None},
        {"model_config": None, "billing_provider": "custom"},
        {"model_config": "not json", "billing_provider": "auto"},
        {"model_config": json.dumps({"gateway_runtime": {"provider": None,
                                                         "base_url": "https://y"}}),
         "billing_provider": None},
    ]
    NAMES = [["Eukrio", ""], ["Local vLLM", ""], ["custom:Mine", ""], ["Display", "my-key"]]

    def test_the_copies_answer_what_hermes_answers(self):
        state, gateway, providers = (HERMES_AGENT / "hermes_state.py",
                                     HERMES_AGENT / "hermes_state_gateway.py",
                                     HERMES_AGENT / "hermes_cli" / "providers.py")
        real_runtime = hermes_functions(gateway, "session_gateway_runtime",
                                        json=json)["session_gateway_runtime"]
        real_aliases = hermes_functions(providers, "custom_provider_slug",
                                        "custom_provider_aliases")["custom_provider_aliases"]
        stub = types.ModuleType("hermes_state")
        stub._BARE_BILLING_PROVIDERS = hermes_constant(state, "_BARE_BILLING_PROVIDERS")
        with mock.patch.dict(sys.modules, {"hermes_state": stub}), \
                mock.patch("subprocess.run", side_effect=AssertionError("started a process")):
            routes = [real_runtime(dict(row)) for row in self.ROWS]
            aliases = [sorted(real_aliases(*n)) for n in self.NAMES]
        self.assertEqual(routes, [session_gateway_runtime(r) for r in self.ROWS])
        self.assertEqual(aliases, [sorted(custom_provider_aliases(*n)) for n in self.NAMES])


if __name__ == "__main__":
    unittest.main(verbosity=2)
