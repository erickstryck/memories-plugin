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
        """Measured trap that motivated this: the session where this was designed ran a
        1M variant of claude-opus-5, whose bare name would otherwise map to 200k — a 5x
        silent error. The operator has to be able to state the truth."""
        cfg = a_config(context_window=1_000_000)
        self.assertEqual(windows.window_for("claude-opus-5", cfg), 1_000_000)

    def test_the_config_beats_the_table_for_an_unknown_model_too(self):
        cfg = a_config(context_window=333_000)
        self.assertEqual(windows.window_for("whatever", cfg), 333_000)

    def test_a_nonsense_config_value_falls_back_to_the_table(self):
        """The config is read tolerantly everywhere else in this plugin; a typo must not
        turn the guard into a blocker calibrated on garbage."""
        self.assertGreater(windows.window_for("claude-opus-5", a_config(context_window=-5)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
