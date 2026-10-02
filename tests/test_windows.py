# tests/test_windows.py
"""The order the big-file guard learns the context window in, and its one owner.

The host's own report comes first, because only the host knows which model is selected NOW
and how large its window is: the claude-code statusLine and the hermes provider publish it
per session (`core.hostwindow`). An endpoint's report, from the cache the hermes hooks fill,
comes next. The `context_window` in the config is the LAST resort, for a host that reports
nothing (`claude -p` runs no statusLine). With none of them the window is unknown, 0, and
the guard lets the read through.

There is no table of model names any more, by the user's decision: a name alone resolves to
0. And a host's GUESS (hermes' 256,000 fallback) does not count as a report.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import hostwindow, windowcache, windows  # noqa: E402
from tests.test_hermes_tools import a_config  # noqa: E402


class StateDirCase(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("QCTX_STATE_DIR")
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        self.addCleanup(self._restore)

    def _restore(self):
        if self._old is None:
            os.environ.pop("QCTX_STATE_DIR", None)
        else:
            os.environ["QCTX_STATE_DIR"] = self._old


class TestTheOrder(StateDirCase):
    """Each fixture satisfies exactly the sources it names, so a test can only pass through
    the step it is about. A fixture that satisfies two proves neither."""

    def test_the_host_record_beats_the_config(self):
        hostwindow.publish("s1", "claude-opus-5-5[1m]", 1_000_000, "claude-code")
        cfg = a_config(context_window=200_000)
        self.assertEqual(windows.window_for("claude-opus-5-5", cfg, session_id="s1"), 1_000_000)

    def test_the_host_record_beats_a_smaller_config_too(self):
        """Not only the direction that makes the guard sleep: a session that switched to a
        200k model is reported as 200k even with a 1M number declared by hand."""
        hostwindow.publish("s1", "claude-haiku-4-5-20251001", 200_000, "claude-code")
        cfg = a_config(context_window=1_000_000)
        self.assertEqual(windows.window_for("claude-haiku-4-5", cfg, session_id="s1"), 200_000)

    def test_the_host_record_beats_the_endpoint_cache(self):
        windowcache.put("http://x/v1", "m", 204_800)
        hostwindow.publish("s1", "m", 524_288, "hermes")
        self.assertEqual(windows.window_for("m", a_config(), "http://x/v1", session_id="s1"),
                         524_288)

    def test_a_guess_is_skipped_for_the_cache_then_the_config_then_zero(self):
        hostwindow.publish("s1", "m", 256_000, "hermes", guess=True)
        windowcache.put("http://x/v1", "m", 204_800)
        self.assertEqual(windows.window_for("m", a_config(), "http://x/v1", session_id="s1"),
                         204_800)
        self.assertEqual(windows.window_for("m", a_config(context_window=300_000), "",
                                            session_id="s1"), 300_000)
        self.assertEqual(windows.window_for("m", a_config(), "", session_id="s1"), 0)

    def test_the_endpoint_cache_beats_the_config(self):
        """The config is the last resort now; v1.2.0 let it override everything."""
        windowcache.put("http://x/v1", "m", 204_800)
        self.assertEqual(windows.window_for("m", a_config(context_window=333_000),
                                            "http://x/v1"), 204_800)

    def test_the_config_answers_when_no_host_reported_anything(self):
        self.assertEqual(windows.window_for("whatever", a_config(context_window=333_000),
                                            session_id="s1"), 333_000)

    def test_a_record_of_another_session_is_not_used(self):
        hostwindow.publish("other", "m", 1_000_000, "claude-code")
        self.assertEqual(windows.window_for("m", a_config(), session_id="s1"), 0)

    def test_no_session_id_means_no_record_is_consulted(self):
        hostwindow.publish("default", "m", 1_000_000, "claude-code")
        self.assertEqual(windows.window_for("m", a_config()), 0)

    def test_a_model_name_alone_resolves_to_zero(self):
        """No table: the name a transcript records says nothing about the window."""
        for model in ("claude-opus-5", "claude-opus-5-5", "claude-sonnet-5", "claude-haiku-4-5"):
            with self.subTest(model=model):
                self.assertEqual(windows.window_for(model, a_config()), 0)

    def test_a_nonsense_config_value_is_no_declaration(self):
        self.assertEqual(windows.window_for("m", a_config(context_window=-5)), 0)

    def test_a_cached_value_is_used_even_when_STALE(self):
        windowcache.put("http://x/v1", "MiniMax-M2.7", 204_800, ttl=-1)
        self.assertEqual(windows.window_for("MiniMax-M2.7", a_config(), "http://x/v1"), 204_800)

    def test_with_NO_endpoint_the_cache_is_not_consulted_at_all(self):
        windowcache.put("", "m", 42)
        self.assertEqual(windows.window_for("m", a_config()), 0)


class TestTheResolverNeverReachesTheNetwork(StateDirCase):
    """The guard calls this before EVERY file read. A probe here would be a network call on
    the hot path, and the reason the cache exists at all."""

    def test_resolving_with_the_socket_broken_still_answers(self):
        import socket
        hostwindow.publish("s1", "m", 1_000_000, "claude-code")
        original = socket.socket.connect
        socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("window_for reached the network"))
        try:
            self.assertEqual(windows.window_for("m", a_config(), "http://x/v1", session_id="s1"),
                             1_000_000)
            self.assertEqual(windows.window_for("m", a_config(), "http://x/v1"), 0)
        finally:
            socket.socket.connect = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
