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
