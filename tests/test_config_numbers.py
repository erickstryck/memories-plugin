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


class TestTheCheckpointIntervalIsAConfigSetting(unittest.TestCase):
    """How often the write procedure is handed to the model. It was an environment variable
    only, read by each host on its own, so `qctx config set` could not reach it and the two
    reads were two copies of one rule. It is a `Config` field now, resolved like every other:
    environment, then file, then default."""

    def test_the_file_sets_it(self):
        cfg = config.load(a_config_file(checkpoint_interval=2), env={})
        self.assertEqual(cfg.checkpoint_interval, 2)

    def test_the_default_is_five(self):
        self.assertEqual(config.load(a_config_file(), env={}).checkpoint_interval, 5)

    def test_the_environment_wins_over_the_file_and_the_legacy_name_still_counts(self):
        path = a_config_file(checkpoint_interval=2)
        self.assertEqual(config.load(path, env={"QCTX_CHECKPOINT_INTERVAL": "3"})
                         .checkpoint_interval, 3)
        self.assertEqual(config.load(path, env={"REMEMBER_INTERVAL": "4"})
                         .checkpoint_interval, 4)

    def test_a_malformed_value_falls_back_and_SAYS_so(self):
        """Falling back silently hides the typo: the hook's own test demands the note."""
        notes = []
        cfg = config.load(a_config_file(), env={"QCTX_CHECKPOINT_INTERVAL": "5x"},
                          note=notes.append)
        self.assertEqual(cfg.checkpoint_interval, 5)
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("QCTX_CHECKPOINT_INTERVAL", notes[0])
        self.assertIn("not a number", notes[0])

    def test_a_valid_configuration_says_nothing(self):
        notes = []
        config.load(a_config_file(checkpoint_interval=7), env={}, note=notes.append)
        self.assertEqual(notes, [])

    def test_a_blank_environment_value_is_unset_and_the_legacy_name_counts(self):
        """As `core.knobs.env` reads a knob, and as 1.1.0 read this one. Measured by review:
        a blank QCTX_CHECKPOINT_INTERVAL beside REMEMBER_INTERVAL=4 fired every 4 turns on
        1.1.0, and every 5 with a note on every prompt before this."""
        notes = []
        cfg = config.load(a_config_file(), env={"QCTX_CHECKPOINT_INTERVAL": "  ",
                                                "REMEMBER_INTERVAL": "4"}, note=notes.append)
        self.assertEqual(cfg.checkpoint_interval, 4)
        self.assertEqual(notes, [])

    def test_the_note_for_a_value_in_the_file_names_the_file(self):
        path = a_config_file(checkpoint_interval="5x")
        notes = []
        config.load(path, env={}, note=notes.append)
        self.assertEqual(len(notes), 1, notes)
        self.assertIn(str(path), notes[0])

    def test_config_set_through_the_real_entry_point(self):
        """The wiring, not the handler: `config set` then `config show`, as typed."""
        import subprocess
        import sys
        cli = Path(__file__).resolve().parent.parent / "cli" / "qctx.py"
        path = Path(tempfile.mkdtemp()) / "config.json"
        env = {k: v for k, v in os.environ.items()
               if k not in ("QCTX_CHECKPOINT_INTERVAL", "REMEMBER_INTERVAL")}
        env["QCTX_CONFIG"] = str(path)

        def qctx(*args):
            return subprocess.run([sys.executable, str(cli), *args], capture_output=True,
                                  text=True, env=env, timeout=60)

        done = qctx("config", "set", "checkpoint-interval", "10")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(path.read_text())["checkpoint_interval"], 10)
        shown = json.loads(qctx("--json", "config", "show").stdout)
        self.assertEqual(shown["checkpoint_interval"], 10)
        refused = qctx("config", "set", "checkpoint-interval", "10x")
        self.assertNotEqual(refused.returncode, 0)
        self.assertEqual(json.loads(path.read_text())["checkpoint_interval"], 10,
                         "a refused value reached the file")


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

    def test_a_fraction_field_refuses_a_value_that_is_not_a_number(self):
        with self.assertRaisesRegex(core.ConfigError, "between 0 and 1"):
            self._set("bigfile-floor-pct", "banana")

        with open(self.path) as fh:
            self.assertNotIn("bigfile_floor_pct", json.load(fh), "the bad value reached the file")

    def test_a_fraction_field_refuses_a_value_outside_zero_to_one(self):
        for value in ("1.5", "-0.1", "20"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(core.ConfigError, "between 0 and 1"):
                    self._set("bigfile-share-pct", value)

        with open(self.path) as fh:
            self.assertNotIn("bigfile_share_pct", json.load(fh), "the bad value reached the file")

    def test_a_fraction_field_accepts_a_fraction_and_writes_a_number(self):
        written = self._set("bigfile-floor-pct", "0.15")

        self.assertEqual(written["bigfile_floor_pct"], 0.15)
        self.assertIsInstance(written["bigfile_floor_pct"], float)


class TestTheGuardThresholdsAreConfigSettings(unittest.TestCase):
    """The two numbers the big-file guard decides with: block when the free context is at or
    below `bigfile_floor_pct` of the window, or when one read would take more than
    `bigfile_share_pct` of what is free. They were environment variables only, read by each
    adapter at import, so `qctx config set` could not reach them. They are `Config` fields
    now, resolved like every other: environment, then file, then default."""

    def test_the_defaults_are_the_ones_the_guard_had(self):
        from core import bigfile
        cfg = config.load(a_config_file(), env={})
        self.assertEqual(cfg.bigfile_floor_pct, bigfile.FLOOR_PCT)
        self.assertEqual(cfg.bigfile_share_pct, bigfile.SHARE_PCT)
        self.assertEqual((cfg.bigfile_floor_pct, cfg.bigfile_share_pct), (0.20, 0.40))

    def test_the_file_sets_them(self):
        cfg = config.load(a_config_file(bigfile_floor_pct=0.15, bigfile_share_pct=0.5), env={})
        self.assertEqual((cfg.bigfile_floor_pct, cfg.bigfile_share_pct), (0.15, 0.5))

    def test_the_environment_beats_the_file(self):
        path = a_config_file(bigfile_floor_pct=0.15)
        cfg = config.load(path, env={"QCTX_BIGFILE_FLOOR_PCT": "0.05"})
        self.assertEqual(cfg.bigfile_floor_pct, 0.05)

    def test_the_legacy_names_still_work(self):
        cfg = config.load(a_config_file(), env={"BIGFILE_FLOOR_PCT": "0.1",
                                                "BIGFILE_SHARE_PCT": "0.9"})
        self.assertEqual((cfg.bigfile_floor_pct, cfg.bigfile_share_pct), (0.1, 0.9))

    def test_a_blank_canonical_name_falls_through_to_the_legacy_one(self):
        cfg = config.load(a_config_file(), env={"QCTX_BIGFILE_SHARE_PCT": " ",
                                                "BIGFILE_SHARE_PCT": "0.3"})
        self.assertEqual(cfg.bigfile_share_pct, 0.3)

    def test_a_non_number_falls_back_and_says_where(self):
        notes = []
        cfg = config.load(a_config_file(), env={"QCTX_BIGFILE_FLOOR_PCT": "20%"},
                          note=notes.append)
        self.assertEqual(cfg.bigfile_floor_pct, 0.20)
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("QCTX_BIGFILE_FLOOR_PCT", notes[0])
        self.assertIn("between 0 and 1", notes[0])

    def test_outside_zero_to_one_falls_back_and_says_so(self):
        for value in ("1.5", "-0.1", "nan", "inf"):
            with self.subTest(value=value):
                notes = []
                path = a_config_file(bigfile_share_pct=value)
                cfg = config.load(path, env={}, note=notes.append)
                self.assertEqual(cfg.bigfile_share_pct, 0.40)
                self.assertEqual(len(notes), 1, notes)
                self.assertIn(str(path), notes[0])

    def test_the_bounds_zero_and_one_are_kept(self):
        """Both ends mean something in `core.bigfile._blocks`: a floor of 0 refuses only a
        read that would overflow the window, a share of 1 refuses only a read that does not
        fit in what is free, and a share of 0 refuses every read that costs anything."""
        cfg = config.load(a_config_file(bigfile_floor_pct=0, bigfile_share_pct="1"), env={})
        self.assertEqual((cfg.bigfile_floor_pct, cfg.bigfile_share_pct), (0.0, 1.0))

    def test_a_valid_configuration_says_nothing(self):
        notes = []
        config.load(a_config_file(bigfile_floor_pct=0.3), env={}, note=notes.append)
        self.assertEqual(notes, [])

    def test_they_are_fractions_not_whole_numbers(self):
        self.assertEqual(set(config.fraction_fields()),
                         {"bigfile_floor_pct", "bigfile_share_pct"})
        self.assertFalse(set(config.fraction_fields()) & set(config.numeric_fields()))

    def test_every_fraction_field_survives_rubbish(self):
        """Derived from the dataclass, like the whole-number fields above."""
        fractions = [f.name for f in dataclasses.fields(config.Config)
                     if f.type in (float, "float")]
        self.assertTrue(fractions, "guard on the guard: the derivation found no fraction field")
        for name in fractions:
            with self.subTest(field=name):
                cfg = config.load(a_config_file(**{name: "rubbish"}), env={})
                self.assertEqual(getattr(cfg, name), config.DEFAULTS[name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
