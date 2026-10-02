"""The context window hermes resolves for a session, published for the big-file guard.

WHY THE PROVIDER AND NOT THE GUARD. The hermes guard (`hosts/hermes/bigfile.py`) is a shell
hook: a subprocess per read that does not import hermes, so it cannot ask hermes anything.
The memory provider runs INSIDE hermes, and `register(ctx)` can attach a hook. hermes calls
`pre_llm_call` at the start of every turn, before any tool call of that turn, with the
session id and the model already switched; it is called without the `has_hook` gate that
`pre_api_request` sits behind, so registering it costs hermes nothing per request (read on
2026-10-02). No other provider hook carries the model, and none carries the window.

WHAT IS ASKED. The question hermes asks itself for the compressor:
`agent.model_metadata.get_model_context_length`, with hermes' own `custom_providers` and
its `model.context_length` pin, for the session's own route.

THE ROUTE is the one hermes records for the session, read with hermes' own reader,
`SessionDB.session_gateway_runtime`: the `/model` paths write it into `model_config`. It is
NOT the `billing_provider`/`billing_base_url` pair: hermes writes those once, at the first
accounted call, and a switch to another provider leaves them as they were (read on
2026-10-02). While hermes has recorded no route (the first turn of a new session), the
configured route stands in, but only for the configured model. It says nothing about
another model, so that turn publishes nothing and the next one tries again.

THE KEY GOES TO CUSTOM ROUTES ONLY, by the user's decision of 2026-10-02. A custom endpoint
answers /models only with its key; without it hermes falls back to its own table of model
names. Measured on the Eukrio route, Qwen3.8-27B: 131,072 without the key, going to the
network on every call; 524,288 with it, which is what the endpoint reports, in 258 ms once
and 0.8 ms after that. The key is read the way `hosts/hermes/endpoint.py` reads it, from the
variable the entry names (`key_env`, or `api_key: ${VAR}`; a literal `api_key` as it is),
and it is never stored, logged or put in a message. A known provider is asked without a
key: with an Anthropic API key hermes makes an uncached HTTP request on every call, and
without one it answers from its catalogue with no network (1,000,000 for claude-opus-5-5
and 200,000 for claude-haiku-4-5-20251001, measured).

WHAT COUNTS AS A GUESS, published and then skipped by the guard: hermes' own fallback
(`DEFAULT_FALLBACK_CONTEXT`), and any answer for a custom route that declares a key the
environment does not hold, since the endpoint could not have given it. A pinned value is
never a guess.

ONLY WHEN THE ROUTE CHANGED: the first turn of a session, and the turn after a `/model`.
Every other turn costs one read-only query and a dict lookup.

NEVER COSTS HERMES A TURN. Every failure is swallowed, and the hook returns None: hermes
injects into the prompt whatever a `pre_llm_call` callback returns.
"""
import os
import re
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import hostwindow  # noqa: E402

#: How the guard tells this host's reports apart from claude-code's.
SOURCE = "hermes"

#: Session id -> (model, provider, base_url) last published. Bounded: a gateway process
#: serves sessions for days, and an entry costs a re-resolution at most. It holds no key.
_LAST: dict = {}
_LAST_MAX = 512

_ROW_SQL = "select model_config, billing_provider from sessions where id=? limit 1"

#: `api_key: ${VAR}`, the form a hermes config uses to point at a secret.
_REFERENCE = re.compile(r"^\$\{(\w+)\}$")


def forget() -> None:
    """Drop what was published in this process (for tests, and harmless otherwise)."""
    _LAST.clear()


def _hermes_config() -> dict:
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:  # noqa: BLE001
        return {}

    return config if isinstance(config, dict) else {}


def _custom_providers(config: dict) -> list:
    try:
        from hermes_cli.config import get_compatible_custom_providers
        found = get_compatible_custom_providers(config)
    except Exception:  # noqa: BLE001
        return []

    return found if isinstance(found, list) else []


def _bigfile():
    """The guard module, for its read-only, short-timeout query of hermes' database.

    Relative first, the way the loader makes siblings importable; by path otherwise, never
    as `hosts.hermes.bigfile`, which would execute the provider package a second time
    under another name."""
    try:
        from . import bigfile

        return bigfile
    except ImportError:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bigfile.py")
        spec = importlib.util.spec_from_file_location("memories_plugin_hermes_bigfile", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return module


def _recorded_route(session_id: str) -> dict:
    """The route hermes records for this session, read with hermes' own reader; {} while
    the session has no row."""
    bigfile = _bigfile()
    rows = bigfile._rows(bigfile.state_db_path(), _ROW_SQL, (session_id,))
    if not rows:
        return {}
    from hermes_state import SessionDB
    route = SessionDB.session_gateway_runtime(
        {"model_config": rows[0][0], "billing_provider": rows[0][1]})

    return route if isinstance(route, dict) else {}


def _custom_entry(provider: str, custom_providers: list):
    """The `custom_providers` entry this provider names, matched by hermes' own aliases."""
    from hermes_cli.providers import custom_provider_aliases

    wanted = provider.strip().lower()
    for entry in custom_providers:
        if isinstance(entry, dict) and wanted in custom_provider_aliases(
                str(entry.get("name") or ""), str(entry.get("provider_key") or "")):
            return entry

    return None


def _key_of(fields: dict) -> tuple[bool, str]:
    """(declares a key, the key) of a custom entry or of the `model:` block."""
    name = str(fields.get("key_env") or "").strip()
    if name:
        return True, os.environ.get(name, "")
    raw = str(fields.get("api_key") or "").strip()
    if not raw:
        return False, ""
    reference = _REFERENCE.match(raw)

    return True, (os.environ.get(reference.group(1), "") if reference else raw)


def route_of(session_id: str, model: str, config: dict, custom_providers: list):
    """(provider, base_url, key, declares_key) to ask hermes about, or None while hermes has
    recorded no route and the model is not the configured one."""
    section = config.get("model")
    section = section if isinstance(section, dict) else {}
    recorded = _recorded_route(session_id)
    if recorded.get("provider"):
        provider, base_url = str(recorded["provider"]), str(recorded.get("base_url") or "")
    elif model == str(section.get("default") or ""):
        provider, base_url = str(section.get("provider") or ""), str(section.get("base_url") or "")
    else:
        return None
    if provider.strip().lower() == "custom":
        declares, key = _key_of(section)              # the `model:` block's own endpoint

        return provider, base_url, key, declares
    entry = _custom_entry(provider, custom_providers)
    if entry is None:
        return provider, base_url, "", False          # a known provider: never a key
    declares, key = _key_of(entry)
    url = base_url or str(entry.get("base_url") or entry.get("url") or entry.get("api") or "")

    return provider, url, key, declares


def _pin(model: str, provider: str, base_url: str, config: dict, custom_providers):
    """The `context_length` hermes would hand its compressor for this route, or None."""
    runtime = SimpleNamespace(model=model, provider=provider, base_url=base_url)
    try:
        from agent.agent_init import config_context_length_for_runtime
        pin = config_context_length_for_runtime(runtime, config)
        if pin is not None:
            return pin
    except Exception:  # noqa: BLE001
        pass
    try:
        from hermes_cli.config import get_custom_provider_context_length
        return get_custom_provider_context_length(model=model, base_url=base_url,
                                                  custom_providers=custom_providers)
    except Exception:  # noqa: BLE001
        return None


def resolve(model: str, route: tuple, config: dict, custom_providers: list) -> tuple[int, bool]:
    """(window, guess) as hermes resolves this route. Raises when hermes cannot be asked."""
    from agent import model_metadata

    provider, base_url, key, declares_key = route
    pin = _pin(model, provider, base_url, config, custom_providers)
    window = int(model_metadata.get_model_context_length(
        model, base_url=base_url, api_key=key, config_context_length=pin, provider=provider,
        custom_providers=custom_providers) or 0)
    if pin is not None and pin == window:
        return window, False
    fallback = getattr(model_metadata, "DEFAULT_FALLBACK_CONTEXT", None)

    return window, (window == fallback or (declares_key and not key))


def on_pre_llm_call(session_id: str = "", model: str = "", **_) -> None:
    """Publish this session's window when its route changed. Returns None, always."""
    try:
        if not session_id or not model:
            return None
        config = _hermes_config()
        custom_providers = _custom_providers(config)
        route = route_of(session_id, model, config, custom_providers)
        if route is None:
            return None
        seen = (model, route[0], route[1])
        if _LAST.get(session_id) == seen:
            return None
        window, guess = resolve(model, route, config, custom_providers)
        if hostwindow.publish(session_id, model, window, SOURCE, guess=guess):
            if len(_LAST) >= _LAST_MAX:
                _LAST.clear()
            _LAST[session_id] = seen
    except Exception:  # noqa: BLE001 -- see the module docstring
        pass

    return None
