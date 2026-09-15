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
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.config as config  # noqa: E402
import cli.qctx as qctx  # noqa: E402
import core  # noqa: E402


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


class TestTheCLIRefusesWhatTheLoaderWouldHaveToTolerate(unittest.TestCase):
    """`load` degrading and `config set` refusing are the two halves of one decision: be
    tolerant where a bad value is already on disk and you must still start, be strict at the
    only door that puts it there. The loader's tolerance got tests; the refusal had none, so
    deleting it left the suite green while `qctx config set context-window 200k` wrote a value
    that silently became the default on the next read."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = Path(self.dir) / "config.json"
        with open(self.path, "w") as fh:
            json.dump({"qdrant_url": "http://127.0.0.1:6333"}, fh)
        # `core.save` resolves DEFAULT_CONFIG_PATH at import time, so the env var alone does
        # not redirect it. Patched at the module the handler actually calls.
        self._real = config.DEFAULT_CONFIG_PATH
        config.DEFAULT_CONFIG_PATH = self.path
        self.addCleanup(setattr, config, "DEFAULT_CONFIG_PATH", self._real)

    def _set(self, key, value):
        """Runs the real handler, returning what landed on disk. `core.save` owns the path."""
        args = SimpleNamespace(key=key, value=value, json=False)
        qctx.cmd_config_set(args, config.load(self.path, env={}))
        with open(self.path) as fh:
            return json.load(fh)

    def test_a_numeric_field_refuses_a_value_that_is_not_a_number(self):
        # `ConfigError` and not a return code: `main()` turns a CoreError into a diagnostic
        # and exit 1, which is the same channel every other refusal in this CLI uses.
        with self.assertRaises(core.ConfigError):
            self._set("context-window", "200k")

        with open(self.path) as fh:
            self.assertNotIn("context_window", json.load(fh), "the bad value reached the file")

    def test_a_numeric_field_still_accepts_a_number(self):
        """The refusal must not cost the command its job."""
        self.assertEqual(self._set("context-window", "180000")["context_window"], 180000)

    def test_a_free_text_field_is_untouched_by_the_numeric_rule(self):
        written = self._set("qdrant-url", "http://127.0.0.1:9999")

        self.assertEqual(written["qdrant_url"], "http://127.0.0.1:9999")


if __name__ == "__main__":
    unittest.main(verbosity=2)
