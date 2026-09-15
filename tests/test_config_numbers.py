"""A malformed number in the config file must not take the plugin down.

`core/knobs.py` exists because a value a user can type is a value a user can mistype, and its
own docstring says a missing tolerance inside a hook turns an environment problem into silent
loss of functionality. Two fields were coerced with bare `int()` six lines below that helper,
and a single typo then killed EVERY command — including the one that repairs the file — and,
in the hermes host, made the memory provider disappear with one debug line, because ValueError
is not a CoreError and the loader swallows what it cannot classify.
"""
import dataclasses
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.config as config  # noqa: E402


def a_config_file(**values) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(values, fh)

    return Path(path)


class TestANumericFieldSurvivesRubbish(unittest.TestCase):
    def setUp(self):
        self.env = {}

    def test_a_non_numeric_context_window_falls_back_instead_of_raising(self):
        path = a_config_file(context_window="abc")
        cfg = config.load(path, env=self.env)
        self.assertEqual(cfg.context_window, int(config.DEFAULTS["context_window"]))

    def test_a_non_numeric_vector_size_falls_back_instead_of_raising(self):
        path = a_config_file(vector_size="not-a-number")
        cfg = config.load(path, env=self.env)
        self.assertEqual(cfg.vector_size, int(config.DEFAULTS["vector_size"]))

    def test_rubbish_in_the_ENVIRONMENT_is_tolerated_too(self):
        """The environment wins over the file, so it is a second door to the same failure."""
        cfg = config.load(a_config_file(), env={"QCTX_CONTEXT_WINDOW": "huge"})
        self.assertEqual(cfg.context_window, int(config.DEFAULTS["context_window"]))

    def test_a_good_value_is_still_read(self):
        """The guard must not swallow the setting it exists to protect."""
        cfg = config.load(a_config_file(context_window="120000"), env=self.env)
        self.assertEqual(cfg.context_window, 120000)


class TestEveryNumericFieldIsCovered(unittest.TestCase):
    """Derived from the dataclass, not from a hand-kept list: a numeric field added later gets
    this tolerance without anyone remembering to extend a literal. The sibling failure this
    project already measured was nine knobs reading through a tolerant helper and ONE using
    bare int() — invisible when reading the file, because all ten look alike."""

    def test_no_numeric_field_raises_on_rubbish(self):
        numeric = [f.name for f in dataclasses.fields(config.Config)
                   if f.type in (int, "int")]
        self.assertTrue(numeric, "guard on the guard: the derivation found no numeric field")
        for name in numeric:
            with self.subTest(field=name):
                cfg = config.load(a_config_file(**{name: "rubbish"}), env={})
                self.assertEqual(getattr(cfg, name), int(config.DEFAULTS[name]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
