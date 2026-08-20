"""`context_window` is the one setting nothing checked.

Measured on 2026-08-17: the big-file guard resolves the window per model name, and a model
outside `core/windows.py::MODEL_WINDOWS` resolves to 0 — and 0 allows every read. The
hermes model of the day (`MiniMax-M2.7`) is exactly such a model. Until now only
`scripts/hermes_cutover.sh` reported it, so on a claude-only machine nothing said a word.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import config, setup  # noqa: E402


def cfg_with(**over):
    values = dict(config.DEFAULTS)
    values.update(over)
    return config.Config(**values)


class ContextWindowCheck(unittest.TestCase):
    def test_absent_is_a_warning_not_a_blocker(self):
        check = setup._check_context_window(cfg_with(context_window=0))
        self.assertFalse(check.ok)
        self.assertTrue(check.warning)
        self.assertIn("config set context-window", check.fix_hint)

    def test_declared_is_ok_and_says_the_value(self):
        check = setup._check_context_window(cfg_with(context_window=200000))
        self.assertTrue(check.ok)
        self.assertIn("200000", check.detail)

    def test_a_negative_declaration_is_not_a_declaration(self):
        """The guard in `core/windows.py` gates on `declared > 0`, so a negative value
        is resolved per model exactly like 0. The check gated on truthiness instead, so
        it reported "declared" over a number the guard ignores — the two disagreeing
        about the same field is the whole failure mode this check exists to catch."""
        check = setup._check_context_window(cfg_with(context_window=-1))
        self.assertFalse(check.ok)
        self.assertTrue(check.warning)

    def test_diagnose_includes_it(self):
        """Reachability is irrelevant here: the check must be present even with Qdrant
        down, which is what an offline suite gives us."""
        names = [c["name"] for c in setup.diagnose(cfg_with())["checks"]]
        self.assertIn("Context window", names)


if __name__ == "__main__":
    unittest.main()
