# tests/test_windows.py
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import windows
from tests.test_hermes_tools import a_config


class TestWindowFor(unittest.TestCase):
    def test_a_known_model_resolves_from_the_table(self):
        self.assertGreater(windows.window_for("claude-opus-5", a_config()), 0)

    def test_an_unknown_model_is_zero_not_a_guess(self):
        """Zero means "unknown", and the caller must then ALLOW the read. A plausible
        default here would make the guard block on a guess, which is what the design
        forbids: the window is not derivable from disk on either host."""
        self.assertEqual(windows.window_for("some-model-nobody-shipped-yet", a_config()), 0)

    def test_the_config_beats_the_table(self):
        """The table is a ceiling, so the config is the only way to state a SMALLER window
        — which is what someone genuinely on a 200k variant has to do to get the guard to
        fire at all."""
        cfg = a_config(context_window=200_000)
        self.assertEqual(windows.window_for("claude-opus-5", cfg), 200_000)

    def test_the_config_beats_the_table_for_an_unknown_model_too(self):
        cfg = a_config(context_window=333_000)
        self.assertEqual(windows.window_for("whatever", cfg), 333_000)

    def test_a_nonsense_config_value_falls_back_to_the_table(self):
        """The config is read tolerantly everywhere else in this plugin; a typo must not
        turn the guard into a blocker calibrated on garbage."""
        self.assertGreater(windows.window_for("claude-opus-5", a_config(context_window=-5)), 0)


class TestTheTableHoldsCeilingsAndNotNominalWindows(unittest.TestCase):
    """The 2026-08-16 amendment, pinned so it cannot be "fixed" back down.

    The transcript records the BARE name — `claude-opus-5`, 82 times in the session that
    found this, with `[1m]` appearing nowhere — so a 200k variant and a 1M variant are
    indistinguishable at this layer. Erring large makes the guard SLEEP; erring small made
    the real hook deny a 4 KB file. Only one of those is a failure mode this design accepts.
    """

    def test_a_name_with_a_1m_variant_carries_the_1m_ceiling(self):
        """`claude-opus-5[1m]` and `sonnet-5[1m]` are both shipped model ids in the claude
        v2.1.233 binary, so 1M is the largest window either bare name can mean."""
        for model in ("claude-opus-5", "claude-sonnet-5"):
            with self.subTest(model=model):
                self.assertEqual(windows.window_for(model, a_config()), 1_000_000)

    def test_a_name_that_can_never_be_1m_keeps_its_own_ceiling(self):
        """Not symmetry-breaking for its own sake: the binary's own predicate for "cannot
        ever be 1M" enumerates `claude-haiku-4-5` by name, which is what makes 200k a
        ceiling here rather than another guess."""
        self.assertEqual(windows.window_for("claude-haiku-4-5", a_config()), 200_000)

    def test_an_unestablished_name_is_absent_rather_than_guessed(self):
        """A ceiling nobody can establish is not written down at all — absent resolves to
        0, and 0 allows."""
        self.assertEqual(windows.window_for("claude-opus-6", a_config()), 0)


class TestTheCascade(unittest.TestCase):
    """Four steps, and each fixture satisfies EXACTLY ONE of them. A fixture that satisfies
    two proves neither — the lesson this repo paid for twice."""

    def setUp(self):
        import tempfile
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()

    def test_declared_beats_everything_including_a_cached_probe(self):
        from core import windowcache
        windowcache.put("http://x/v1", "claude-opus-5", 111_111)
        cfg = type("C", (), {"context_window": 333_000})()
        self.assertEqual(windows.window_for("claude-opus-5", cfg, "http://x/v1"), 333_000)

    def test_a_cached_probe_beats_the_ceiling_table(self):
        """The table says 1,000,000 for this name. A real endpoint saying 204,800 is closer
        to the truth than a ceiling, and the whole point of probing."""
        from core import windowcache
        windowcache.put("http://x/v1", "claude-opus-5", 204_800)
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("claude-opus-5", cfg, "http://x/v1"), 204_800)

    def test_the_table_answers_when_nothing_was_probed(self):
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("claude-opus-5", cfg, "http://x/v1"), 1_000_000)

    def test_an_unknown_model_with_no_probe_is_still_zero(self):
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("MiniMax-M2.7", cfg, "http://x/v1"), 0)

    def test_with_NO_endpoint_the_cache_is_not_consulted_at_all(self):
        """claude-code has no endpoint to offer, and passing none must behave exactly as it
        did before this cascade existed."""
        from core import windowcache
        windowcache.put("", "claude-opus-5", 42)
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("claude-opus-5", cfg), 1_000_000)

    def test_a_cached_value_is_used_even_when_STALE(self):
        from core import windowcache
        windowcache.put("http://x/v1", "MiniMax-M2.7", 204_800, ttl=-1)
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("MiniMax-M2.7", cfg, "http://x/v1"), 204_800)


class TestTheResolverNeverReachesTheNetwork(unittest.TestCase):
    """The guard calls this before EVERY file read. A probe here would be a network call on
    the hot path, and the reason the cache exists at all."""

    def test_resolving_with_the_socket_broken_still_answers(self):
        import socket
        import tempfile
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        original = socket.socket.connect
        socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("window_for reached the network"))
        try:
            cfg = type("C", (), {"context_window": 0})()
            self.assertEqual(windows.window_for("claude-opus-5", cfg, "http://x/v1"), 1_000_000)
        finally:
            socket.socket.connect = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
