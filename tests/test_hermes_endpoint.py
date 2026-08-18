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


class TestReadingTheHermesConfig(unittest.TestCase):
    def setUp(self):
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        os.environ["MY_KEY_VAR"] = "secret-value"

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
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        os.environ["MY_KEY_VAR"] = "secret-value"

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
        os.environ["OLD_KEY_VAR"] = "should-never-be-read"
        base, key = endpoint.from_hermes_config(a_hermes_home(CONFIG_CATALOGUE_FIRST))
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "secret-value")

    def test_a_catalogues_key_env_does_not_supply_the_key_when_the_active_block_has_its_own(
            self):
        """The active block's own `api_key: ${VAR}` must win over a `key_env:` that belongs
        to a DIFFERENT entry in the catalogue, even one naming the same base_url."""
        os.environ["CATALOGUE_KEY_VAR"] = "should-never-be-read"
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


class TestRefreshing(unittest.TestCase):
    def setUp(self):
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        os.environ["HERMES_HOME"] = a_hermes_home()
        os.environ["MY_KEY_VAR"] = "secret-value"

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
        os.environ["HERMES_HOME"] = tempfile.mkdtemp()
        self.assertEqual(endpoint.refresh_window("m", probe=lambda *a, **k: 999), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
