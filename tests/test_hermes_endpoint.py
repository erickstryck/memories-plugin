# tests/test_hermes_endpoint.py
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hosts.hermes import endpoint  # noqa: E402

CONFIG = """\
model:
  provider: custom
  base_url: https://server.example/api/v1
  key_env: MY_KEY_VAR
memory:
  provider: memories
"""


def a_hermes_home(text: str = CONFIG) -> str:
    home = tempfile.mkdtemp()
    with open(os.path.join(home, "config.yaml"), "w") as fh:
        fh.write(text)

    return home


def _set_env(case: unittest.TestCase, **values) -> None:
    """Sets environment variables for the duration of one test, restoring exactly what was
    there before it — absent, or whatever value it held — once the test ends, via
    `addCleanup` so restoration runs even when the test fails.

    `HERMES_HOME` locates the HOST: a value this file leaves behind can leak into another
    test module and mask a test that would otherwise have reached the real hermes home. Every
    variable this file sets goes through here rather than a bare `os.environ[...] = ...`.
    """
    for key, value in values.items():
        previous = os.environ.get(key)
        os.environ[key] = value
        if previous is None:
            case.addCleanup(os.environ.pop, key, None)
        else:
            case.addCleanup(os.environ.__setitem__, key, previous)


class TestReadingTheHermesConfig(unittest.TestCase):
    def setUp(self):
        _set_env(self, QCTX_STATE_DIR=tempfile.mkdtemp(), MY_KEY_VAR="secret-value")

    def test_it_reads_the_base_url_and_resolves_the_key_from_the_environment(self):
        """`key_env` names a VARIABLE, not a secret. The config holds the name; the value
        lives in the environment the hermes process already has."""
        base, key = endpoint.from_hermes_config(a_hermes_home())
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "secret-value")

    def test_a_missing_config_yields_nothing_rather_than_raising(self):
        self.assertEqual(endpoint.from_hermes_config(tempfile.mkdtemp()), ("", ""))

    def test_a_config_without_a_base_url_yields_nothing(self):
        self.assertEqual(endpoint.from_hermes_config(a_hermes_home("memory:\n  provider: x\n")),
                         ("", ""))

    def test_a_key_variable_absent_from_the_environment_still_yields_the_url(self):
        """An endpoint that needs no key is a real case, and refusing the URL because the
        variable is unset would turn a working setup into no setup."""
        os.environ.pop("MY_KEY_VAR", None)
        base, key = endpoint.from_hermes_config(a_hermes_home())
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "")

    def test_a_config_that_is_not_readable_yields_nothing(self):
        home = a_hermes_home()
        os.chmod(os.path.join(home, "config.yaml"), 0)
        try:
            self.assertEqual(endpoint.from_hermes_config(home), ("", ""))
        finally:
            os.chmod(os.path.join(home, "config.yaml"), 0o644)

    def test_a_crlf_config_still_finds_the_model_block(self):
        """A CRLF-saved config leaves a trailing `\\r` before every `\\n`, including right
        after `model:`, and this is a defense in `_model_block` itself, called directly with
        an in-memory string that still carries `\\r`: `open()` in `from_hermes_config` reads
        in the default text mode, whose universal-newline translation already collapses
        `\\r\\n` to `\\n` before this function ever runs — measured, so no end-to-end fixture
        through `a_hermes_home` can make the pre-fix regex fail at all. This still matters
        because `_model_block` is not guaranteed to always be fed a freshly-`open()`ed
        string."""
        text = ("model:\r\n"
                "  provider: custom\r\n"
                "  base_url: https://server.example/api/v1\r\n"
                "  key_env: MY_KEY_VAR\r\n")
        self.assertNotEqual(endpoint._model_block(text), "",
                            "the model: block must be found even with CRLF line endings")


# --- Fixtures shaped after a REAL ~/.hermes/config.yaml -----------------------------------
# The original parser was designed against a config shape nobody had verified against the
# real file. Measured on the actual file: the ACTIVE selection lives in the top-level
# `model:` block, using `api_key: ${VAR}` interpolation — `key_env:` never appears there at
# all. `key_env:` shows up only inside a `custom_providers:` CATALOGUE further down, listing
# servers hermes is not currently using. A whole-file search that does not distinguish the
# two returns an empty key on the real file, silently: the cascade then falls to the ceiling
# table, which is exactly the outcome this feature exists to prevent.

CONFIG_INTERPOLATED = """\
model:
  provider: custom
  base_url: https://server.example/api/v1
  api_key: ${MY_KEY_VAR}
memory:
  provider: memories
"""

CONFIG_CATALOGUE_FIRST = """\
custom_providers:
  - name: Old-Server
    base_url: https://catalogue.example/api/v1
    key_env: OLD_KEY_VAR
    model: Old-Model
model:
  provider: custom
  base_url: https://server.example/api/v1
  key_env: MY_KEY_VAR
memory:
  provider: memories
"""

CONFIG_CATALOGUE_OWN_KEY = """\
model:
  provider: custom
  base_url: https://server.example/api/v1
  api_key: ${MY_KEY_VAR}
custom_providers:
  - name: Other-Server
    base_url: https://server.example/api/v1
    key_env: CATALOGUE_KEY_VAR
    model: Other-Model
"""


class TestTheActiveBlockIsWhatAnswers(unittest.TestCase):
    """A `custom_providers:` catalogue elsewhere in the file lists servers hermes is NOT
    currently using. It must never be able to answer for the endpoint actually in use,
    regardless of where in the file it sits."""

    def setUp(self):
        _set_env(self, QCTX_STATE_DIR=tempfile.mkdtemp(), MY_KEY_VAR="secret-value")

    def test_the_active_blocks_interpolated_api_key_resolves_from_the_environment(self):
        """The real config credits the active model with `api_key: ${VAR}` — `${...}`
        interpolation — and `key_env:` never appears in that block at all. A parser that
        only knew `key_env:` would read the URL and silently miss the key."""
        base, key = endpoint.from_hermes_config(a_hermes_home(CONFIG_INTERPOLATED))
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "secret-value")

    def test_a_catalogue_before_the_active_block_does_not_supply_the_base_url(self):
        """This is the fixture that PINS the scoping: the catalogue's base_url comes first
        in the text, on purpose. A fix that merely changed which occurrence `re.search`
        prefers, rather than actually scoping to the `model:` block, would still pass a
        fixture where the catalogue came second — this one would not."""
        _set_env(self, OLD_KEY_VAR="should-never-be-read")
        base, key = endpoint.from_hermes_config(a_hermes_home(CONFIG_CATALOGUE_FIRST))
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "secret-value")

    def test_a_catalogues_key_env_does_not_supply_the_key_when_the_active_block_has_its_own(
            self):
        """The active block's own `api_key: ${VAR}` must win over a `key_env:` that belongs
        to a DIFFERENT entry in the catalogue, even one naming the same base_url."""
        _set_env(self, CATALOGUE_KEY_VAR="should-never-be-read")
        base, key = endpoint.from_hermes_config(a_hermes_home(CONFIG_CATALOGUE_OWN_KEY))
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "secret-value")

    def test_a_quoted_base_url_is_returned_without_the_quotes(self):
        """`base_url: "https://x"` is YAML-legal. A regex that only looks for non-whitespace
        captures the quotes along with the URL, which is a broken URL."""
        for quote in ('"', "'"):
            with self.subTest(quote=quote):
                text = ("model:\n  provider: custom\n"
                       f"  base_url: {quote}https://server.example/api/v1{quote}\n"
                       "  key_env: MY_KEY_VAR\n")
                base, _ = endpoint.from_hermes_config(a_hermes_home(text))
                self.assertEqual(base, "https://server.example/api/v1")


# --- Comment lines in a heavily-commented real config --------------------------------------
# Measured against the real ~/.hermes/config.yaml: 36 flush-left comment lines, none inside
# the `model:` block today. A block-end lookahead that treated ANY flush-left character as
# "the next key" would close the block the moment one landed there — one hermes upgrade or
# one user note away — and it fails with no error anywhere: base_url vanishes from the
# captured block, the cascade falls to the ceiling table, and the guard sleeps.

CONFIG_COMMENT_BEFORE_KEYS = """\
model:
# a flush-left note the user left here
  provider: custom
  base_url: https://server.example/api/v1
  key_env: MY_KEY_VAR
memory:
  provider: memories
"""

CONFIG_COMMENT_AFTER_BLOCK = """\
model:
  provider: custom
  base_url: https://server.example/api/v1
  api_key: ${MY_KEY_VAR}
# a flush-left note between sections
custom_providers:
  - name: Should-Not-Answer
    base_url: https://should-not-be-read.example/api/v1
    key_env: SHOULD_NOT_BE_READ_KEY
"""

CONFIG_COMMENT_INDENTED = """\
model:
  provider: custom
  # an indented note
  base_url: https://server.example/api/v1
  key_env: MY_KEY_VAR
memory:
  provider: memories
"""


class TestAFlushLeftCommentDoesNotCloseTheBlock(unittest.TestCase):
    def setUp(self):
        _set_env(self, QCTX_STATE_DIR=tempfile.mkdtemp(), MY_KEY_VAR="secret-value")

    def test_a_flush_left_comment_between_model_and_its_first_key_does_not_hide_base_url(self):
        """This is the fixture that reproduces the risk: a comment landing right after
        `model:` and before `provider:`/`base_url:` must not make the rest of the block
        invisible to the parser."""
        base, key = endpoint.from_hermes_config(a_hermes_home(CONFIG_COMMENT_BEFORE_KEYS))
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "secret-value")

    def test_a_flush_left_comment_after_the_block_does_not_swallow_the_next_section(self):
        """The block must stop at the next REAL key, not merely skip the comment that
        precedes it: if it swallowed the catalogue below, that catalogue's OWN key_env —
        absent from the active block entirely — would suddenly outrank the active block's
        api_key, exactly the failure the scoping fix exists to prevent."""
        _set_env(self, SHOULD_NOT_BE_READ_KEY="should-never-be-read")
        base, key = endpoint.from_hermes_config(a_hermes_home(CONFIG_COMMENT_AFTER_BLOCK))
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "secret-value")

    def test_an_indented_comment_inside_the_block_is_harmless(self):
        """An indented `#` was never flush-left and never threatened the boundary; this
        pins that the fix did not accidentally make indentation matter for comments."""
        base, key = endpoint.from_hermes_config(a_hermes_home(CONFIG_COMMENT_INDENTED))
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "secret-value")


class TestRefreshing(unittest.TestCase):
    def setUp(self):
        _set_env(self, QCTX_STATE_DIR=tempfile.mkdtemp(), HERMES_HOME=a_hermes_home(),
                  MY_KEY_VAR="secret-value")
        # The back-off is process-global by design, so it outlives a test the way an env var
        # would. A test that probed and failed would otherwise deny the NEXT test its probe —
        # which is exactly how this line came to be written.
        endpoint.forget_failures()
        self.addCleanup(endpoint.forget_failures)

    def test_a_fresh_cache_is_NOT_probed_again(self):
        from core import windowcache
        windowcache.put("https://server.example/api/v1", "m", 524288)
        calls = []
        got = endpoint.refresh_window("m", probe=lambda *a, **k: calls.append(1) or 999)
        self.assertEqual(got, 524288)
        self.assertEqual(calls, [], "a fresh cache was refreshed anyway")

    def test_an_empty_cache_is_probed_and_the_answer_stored(self):
        from core import windowcache
        got = endpoint.refresh_window("m", probe=lambda *a, **k: 524288)
        self.assertEqual(got, 524288)
        self.assertEqual(windowcache.get("https://server.example/api/v1", "m")[0], 524288)

    def test_a_probe_that_learns_nothing_stores_nothing(self):
        from core import windowcache
        self.assertEqual(endpoint.refresh_window("m", probe=lambda *a, **k: 0), 0)
        self.assertEqual(windowcache.get("https://server.example/api/v1", "m"), (0, False))

    def test_a_failed_probe_keeps_the_STALE_value(self):
        """The endpoint being down must not cost a window we already knew."""
        from core import windowcache
        windowcache.put("https://server.example/api/v1", "m", 204800, ttl=-1)
        got = endpoint.refresh_window("m", probe=lambda *a, **k: 0)
        self.assertEqual(got, 204800)

    def test_with_no_endpoint_configured_it_does_nothing_quietly(self):
        _set_env(self, HERMES_HOME=tempfile.mkdtemp())
        self.assertEqual(endpoint.refresh_window("m", probe=lambda *a, **k: 999), 0)

    def test_an_empty_model_id_is_never_probed(self):
        """`model_of` returns "" when `state.db` is locked or the session row is not yet
        written. An empty id can never match a real `/models` entry, so probing for it is a
        5s round trip that would repeat on every single turn, forever — unlike a real model
        whose probe merely failed once."""
        calls = []
        got = endpoint.refresh_window("", probe=lambda *a, **k: calls.append(1) or 999)
        self.assertEqual(got, 0)
        self.assertEqual(calls, [], "an empty model id was probed anyway")


class TestAProbeThatLearnedNothingBacksOff(unittest.TestCase):
    """`put` refuses to cache a zero, deliberately — absence and "answered zero" have to stay
    different, or the cascade would stop at a step that knows nothing. The cost of that
    correctness is that a probe which learns nothing leaves NOTHING behind, so the next turn
    probes again, and the turn after that, forever. With the archive reachable and the model
    endpoint down, that is PROBE_TIMEOUT_S added to every single turn.

    The back-off lives in this process and not in the cache file on purpose: it is a statement
    about a request that just failed, not a fact about the model, and it must not outlive the
    process or be read by the other host."""

    def setUp(self):
        _set_env(self, QCTX_STATE_DIR=tempfile.mkdtemp(), HERMES_HOME=a_hermes_home(),
                 MY_KEY_VAR="secret-value")
        endpoint.forget_failures()
        self.addCleanup(endpoint.forget_failures)

    def test_a_second_call_after_a_failed_probe_does_NOT_probe_again(self):
        calls = []
        for _ in range(3):
            endpoint.refresh_window("m", probe=lambda *a, **k: calls.append(1) or 0)
        self.assertEqual(len(calls), 1, f"the failed probe was retried {len(calls)} times")

    def test_the_back_off_EXPIRES_so_a_server_that_came_back_is_found(self):
        """A back-off that never expired would be worse than the repetition it replaces: the
        endpoint returning is the case we most want to notice."""
        endpoint.refresh_window("m", probe=lambda *a, **k: 0)
        endpoint.forget_failures()          # what the passage of RETRY_AFTER_S does
        got = endpoint.refresh_window("m", probe=lambda *a, **k: 524288)
        self.assertEqual(got, 524288)

    def test_the_back_off_is_PER_MODEL_and_per_endpoint(self):
        """One model being unavailable says nothing about another on the same server."""
        endpoint.refresh_window("m", probe=lambda *a, **k: 0)
        calls = []
        endpoint.refresh_window("other", probe=lambda *a, **k: calls.append(1) or 524288)
        self.assertEqual(len(calls), 1, "a different model was denied its first probe")

    def test_a_SUCCESSFUL_probe_leaves_no_back_off_behind(self):
        endpoint.refresh_window("m", probe=lambda *a, **k: 524288)
        from core import windowcache
        windowcache.put("https://server.example/api/v1", "m", 524288, ttl=-1)   # ficou velho
        calls = []
        endpoint.refresh_window("m", probe=lambda *a, **k: calls.append(1) or 262144)
        self.assertEqual(len(calls), 1, "a stale entry was not refreshed after an earlier win")


if __name__ == "__main__":
    unittest.main(verbosity=2)
