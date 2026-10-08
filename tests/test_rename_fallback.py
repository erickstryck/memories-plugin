"""The 2.0.0 rename (memories-plugin -> mnemosine) must not orphan an existing
install's data: the config path (which holds the Qdrant pointer and the memory
collection) and the state dirs keep honouring the pre-rename locations when the
new ones do not exist. Fresh installs and env overrides are unaffected.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import config, knobs  # noqa: E402


class TestTheConfigPathFallsBackToThePreRenameLocation(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="mnemosine-config-"))
        self.addCleanup(self._rm)

    def _rm(self):
        import shutil
        shutil.rmtree(self.home, ignore_errors=True)

    def test_a_legacy_config_is_found_when_the_new_path_is_empty(self):
        # The file holding the Qdrant pointer sits at the pre-rename location; the
        # default path must resolve to IT, not to an empty new file.
        legacy = self.home / ".config" / "memories-plugin"
        legacy.mkdir(parents=True)
        (legacy / "config.json").write_text(json.dumps(
            {"memory_collection": "claude_memory",
             "qdrant_url": "https://qdrant.example"}))
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.home / ".config"),
                                          "HOME": str(self.home)}, clear=False):
            # Drop the machine's own Qdrant env so the FILE value is what is tested
            # (environment beats file, and the real env is exactly what we must not
            # leak into the assertion).
            for var in ("QCTX_CONFIG", "QCTX_QDRANT_URL", "QDRANT_URL"):
                os.environ.pop(var, None)
            # `_default_config_path()` is what runs at import in a fresh process; it
            # reads env and the disk at call time, so the patched env applies.
            resolved = config._default_config_path()
            self.assertEqual(resolved, legacy / "config.json")
            with mock.patch.object(config, "DEFAULT_CONFIG_PATH", resolved):
                cfg = config.load()
        self.assertEqual(cfg.memory_collection, "claude_memory")
        self.assertEqual(cfg.qdrant_url, "https://qdrant.example")

    def test_a_fresh_config_wins_over_the_legacy_one(self):
        for d in ("memories-plugin", "mnemosine"):
            (self.home / ".config" / d).mkdir(parents=True)
        (self.home / ".config" / "memories-plugin" / "config.json").write_text(
            json.dumps({"memory_collection": "old"}))
        (self.home / ".config" / "mnemosine" / "config.json").write_text(
            json.dumps({"memory_collection": "new"}))
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.home / ".config"),
                                          "HOME": str(self.home)}, clear=False):
            os.environ.pop("QCTX_CONFIG", None)
            self.assertEqual(config._default_config_path(),
                             self.home / ".config" / "mnemosine" / "config.json")

    def test_a_fresh_install_gets_the_new_path(self):
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.home / ".config"),
                                          "HOME": str(self.home)}, clear=False):
            os.environ.pop("QCTX_CONFIG", None)
            self.assertEqual(config._default_config_path(),
                             self.home / ".config" / "mnemosine" / "config.json")

    def test_save_writes_to_the_legacy_file_it_found(self):
        # A config that was READ from the legacy location must be written back to
        # it, in place: an update must not fork the file into an empty new path
        # and lose the Qdrant pointer.
        legacy = self.home / ".config" / "memories-plugin"
        legacy.mkdir(parents=True)
        (legacy / "config.json").write_text(json.dumps({"memory_collection": "claude_memory"}))
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.home / ".config"),
                                          "HOME": str(self.home)}, clear=False):
            os.environ.pop("QCTX_CONFIG", None)
            resolved = config._default_config_path()
            self.assertEqual(resolved, legacy / "config.json")
            with mock.patch.object(config, "DEFAULT_CONFIG_PATH", resolved):
                config.save({"checkpoint_interval": 3})
        data = json.loads((legacy / "config.json").read_text())
        self.assertEqual(data["memory_collection"], "claude_memory")
        self.assertEqual(data["checkpoint_interval"], 3)
        self.assertFalse((self.home / ".config" / "mnemosine").exists(),
                         "the legacy file is honoured in place, never forked")

    def test_qctx_config_still_overrides(self):
        target = self.home / "elsewhere.json"
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.home / ".config"),
                                          "QCTX_CONFIG": str(target),
                                          "HOME": str(self.home)}, clear=False):
            self.assertEqual(config._default_config_path(), target)


class TestTheStateDirFallsBackToThePreRenameLocation(unittest.TestCase):
    def test_a_legacy_state_dir_is_found_when_the_new_one_is_empty(self):
        home = Path(tempfile.mkdtemp(prefix="mnemosine-state-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(home, ignore_errors=True))
        legacy = home / ".memories-plugin" / "state"
        legacy.mkdir(parents=True)
        with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=False):
            os.environ.pop("QCTX_STATE_DIR", None)
            self.assertEqual(knobs._state_dir_default(), legacy)
            self.assertEqual(knobs.state_dir(), legacy)

    def test_a_fresh_state_dir_wins(self):
        home = Path(tempfile.mkdtemp(prefix="mnemosine-state-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(home, ignore_errors=True))
        for d in (".memories-plugin", ".mnemosine"):
            (home / d / "state").mkdir(parents=True)
        with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=False):
            os.environ.pop("QCTX_STATE_DIR", None)
            self.assertEqual(knobs._state_dir_default(), home / ".mnemosine" / "state")

    def test_a_fresh_install_gets_the_new_dir(self):
        home = Path(tempfile.mkdtemp(prefix="mnemosine-state-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(home, ignore_errors=True))
        with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=False):
            os.environ.pop("QCTX_STATE_DIR", None)
            self.assertEqual(knobs._state_dir_default(), home / ".mnemosine" / "state")


class TestTheHermesProviderStateDirFallsBack(unittest.TestCase):
    def test_a_legacy_provider_state_dir_is_found(self):
        from hosts.hermes import provider_state_dir
        home = Path(tempfile.mkdtemp(prefix="mnemosine-hermes-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(home, ignore_errors=True))
        legacy = home / "memories-state"
        legacy.mkdir()
        self.assertEqual(provider_state_dir(home), legacy)

    def test_a_fresh_provider_state_dir_wins(self):
        from hosts.hermes import provider_state_dir
        home = Path(tempfile.mkdtemp(prefix="mnemosine-hermes-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(home, ignore_errors=True))
        fresh, legacy = home / "mnemosine-state", home / "memories-state"
        fresh.mkdir(); legacy.mkdir()
        self.assertEqual(provider_state_dir(home), fresh)

    def test_a_fresh_install_gets_the_new_dir(self):
        from hosts.hermes import provider_state_dir
        home = Path(tempfile.mkdtemp(prefix="mnemosine-hermes-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(home, ignore_errors=True))
        self.assertEqual(provider_state_dir(home), home / "mnemosine-state")


if __name__ == "__main__":
    unittest.main()
