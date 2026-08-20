"""A process environment built from nothing, for the tests that shell out.

Two of the wizard's tests reached the developer's real machine, and one of them took
over 100 seconds doing it. Both did the same thing: they started from `dict(os.environ)`
and overrode a few names.

That is not isolation, for two reasons this module exists to remove:

- `HOME` and `HERMES_HOME` decide where credentials are WRITTEN. `_store_secrets` writes
  `$HERMES_HOME/.env` whenever that directory exists, so an exported `HERMES_HOME` turns
  a test that types `qkey` into a rewrite of the developer's live credential file. It was
  unset on this machine — luck, not isolation.
- `core.config` accepts LEGACY spellings beside the canonical `QCTX_*` names:
  `QDRANT_URL`, `QDRANT_SERVICE_API_KEY`, `QDRANT_API_KEY`, `SERVER_BASE_URL`,
  `SERVER_API_KEY` and the rest of `ENV_ALIASES`. Clearing `QCTX_QDRANT_URL` therefore
  clears nothing: the alias still resolves, `diagnose` still dials the real Qdrant, and
  the "offline" test waits out its timeouts.

So the environment is ASSEMBLED, never inherited. Anything a test needs, it names.
"""
import os

#: The bare minimum for a Python subprocess to start. `PATH` is deliberately the system
#: one and nothing else: `~/.local/bin` on the real PATH is how `claude` and `hermes`
#: were found, which is how a check-only test ended up running both host cutovers.
BASE = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}


def hermetic_env(home, **overrides) -> dict:
    """The environment for a subprocess that must not see this machine.

    `home` is the temporary HOME. Everything else is opt-in through `overrides`.
    """
    env = dict(BASE)
    env["HOME"] = str(home)
    env.update({k: str(v) for k, v in overrides.items()})

    return env


def assert_hermetic(case, env: dict, allowed=()) -> None:
    """Fails when an assembled environment carries anything of this machine in it.

    The check is a WHITELIST, not a blacklist of known-bad names. A blacklist only fires
    when the offending variable happens to be exported while the suite runs, which is the
    same coin toss that let `HERMES_HOME` sit unnoticed. Here, any name the test did not
    name is a leak, and so is any value under the real HOME.
    """
    real_home = os.path.expanduser("~")
    extra = set(env) - set(BASE) - {"HOME"} - set(allowed)
    case.assertEqual(extra, set(),
                     f"inherited environment reaches the subprocess: {sorted(extra)}")
    for name, value in env.items():
        case.assertFalse(value == real_home or str(value).startswith(real_home + os.sep),
                         f"{name}={value} points at the real HOME")
