"""The hermes-agent host adapter.

It is verified WITHOUT hermes importable, because this repo's suite has to run offline and
hermes lives in its own venv. What the adapter promises is therefore checked structurally:
the method set hermes v0.20.0 actually calls, measured from the install rather than assumed
from the published source — the two differ, and the installed one is what runs.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hosts.hermes import MemoriesProvider, register

#: Where hermes is installed on this machine. The ABC IS importable from there (measured);
#: only `RecallStatus` is absent in v0.20.0. Used by the test below, skipped elsewhere so
#: the suite stays portable.
HERMES_INSTALL = Path.home() / ".hermes" / "hermes-agent"

#: What the adapter implements EXPLICITLY — no reliance on the ABC's defaults, because the
#: suite runs with `_Base = object` and because a default silently absorbs a method hermes
#: renames. Everything here must exist whether or not hermes is importable.
IMPLEMENTED = {
    "name", "is_available", "unavailable_reason", "initialize", "shutdown",
    "prefetch", "queue_prefetch", "recall_status", "system_prompt_block",
    "on_turn_start", "on_session_end", "sync_turn",
    "get_tool_schemas", "handle_tool_call", "get_config_schema", "save_config",
}
HERMES_ABSTRACTS = {"get_tool_schemas", "initialize", "is_available", "name"}


class TestTheContract(unittest.TestCase):
    def test_it_implements_every_abstract(self):
        p = MemoriesProvider()
        for method in HERMES_ABSTRACTS:
            self.assertTrue(hasattr(p, method), f"missing abstract {method}")

    def test_it_defines_its_surface_without_leaning_on_inherited_defaults(self):
        """Explicit, not inherited. Two reasons: this suite runs without hermes, so the
        defaults are not there to lean on; and a default silently absorbs a method a hermes
        upgrade renames, turning a rename into a feature that quietly stops working."""
        p = MemoriesProvider()
        missing = {m for m in IMPLEMENTED if m not in vars(type(p))
                   and not any(m in vars(k) for k in type(p).__mro__[:-1])}
        self.assertEqual(missing, set(), f"declared as implemented but absent: {missing}")

    @unittest.skipUnless(HERMES_INSTALL.exists(), "hermes-agent not installed here")
    def test_it_answers_every_method_the_REAL_installed_abc_declares(self):
        """The honest version of the contract check: read the surface off the install
        instead of trusting a list I typed. Skipped where hermes is absent, so a machine
        without it still runs the rest."""
        import subprocess
        script = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "from agent.memory_provider import MemoryProvider as M\n"
            "print(json.dumps([m for m in dir(M) if not m.startswith('_')]))\n"
        ) % str(HERMES_INSTALL)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        surface = set(json.loads(out.stdout))
        p = MemoriesProvider()
        unanswered = {m for m in surface if not hasattr(p, m)}
        self.assertEqual(unanswered, set(),
                         f"hermes calls these and the provider has no answer: {unanswered}")

    def test_the_name_is_the_install_directory_name(self):
        self.assertEqual(MemoriesProvider().name, "memories")

    def test_register_hands_the_provider_to_the_collector(self):
        class Collector:
            provider = None

            def register_memory_provider(self, provider):
                self.provider = provider

        c = Collector()
        register(c)
        self.assertIsInstance(c.provider, MemoriesProvider)


class TestAvailability(unittest.TestCase):
    """is_available gates initialization, so a provider that reports unavailable is never
    initialized — any diagnostic it would log from initialize() is unreachable. The reason
    has to be actionable and it has to come from here."""

    def _clean_env(self):
        keep = {"PATH", "HOME", "LANG", "PYTHONPATH"}

        return {k: v for k, v in os.environ.items() if k in keep}

    def test_unconfigured_is_unavailable_with_an_actionable_reason(self):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "p = MemoriesProvider()\n"
            "print(int(p.is_available())); print(p.unavailable_reason())\n"
        ) % str(REPO)
        env = self._clean_env()
        env["QCTX_CONFIG"] = "/nonexistent/config.json"
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        available, reason = out.stdout.strip().splitlines()[:2]
        self.assertEqual(available, "0")
        self.assertIn("QCTX_", reason, "the reason must say WHICH variable to set")

    def test_is_available_makes_no_network_call(self):
        """Contract: 'Should not make network calls — just check config and installed deps.'
        Pointed at an unroutable address it must still answer fast."""
        import time
        env = dict(os.environ, QCTX_QDRANT_URL="http://10.255.255.1:9/qdrant")
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "MemoriesProvider().is_available()\n"
        ) % str(REPO)
        t0 = time.monotonic()
        subprocess.run([sys.executable, "-c", script], capture_output=True, env=env)
        self.assertLess(time.monotonic() - t0, 3.0, "is_available reached the network")


class TestSymlinkInstall(unittest.TestCase):
    def test_it_finds_the_repo_when_installed_through_a_symlink(self):
        """Measured trap: through a symlink, abspath(__file__) resolves to the SYMLINK's
        directory ($HERMES_HOME/plugins), not the repo. The path pattern hooks/recall.py
        uses would fail here; only realpath finds core/."""
        home = Path(tempfile.mkdtemp()) / "plugins"
        home.mkdir(parents=True)
        link = home / "memories"
        link.symlink_to(REPO / "hosts" / "hermes")

        script = (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('u.memories', %r,\n"
            "    submodule_search_locations=[%r])\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(m.REPO_ROOT)\n"
            "print(m.MemoriesProvider().name)\n"
        ) % (str(link / "__init__.py"), str(link))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        root, name = out.stdout.strip().splitlines()[:2]
        self.assertEqual(Path(root).resolve(), REPO.resolve())
        self.assertEqual(name, "memories")


class TestManifest(unittest.TestCase):
    def test_the_yaml_declares_what_the_loader_reads(self):
        text = (REPO / "hosts" / "hermes" / "plugin.yaml").read_text()
        for key in ("name: memories", "category: memory", "kind: exclusive"):
            self.assertIn(key, text)

    def test_the_init_contains_the_string_discovery_greps_for(self):
        """_is_memory_provider_dir is a cheap text scan of the first 8192 bytes for
        'register_memory_provider' or 'MemoryProvider'. Without one of those the directory
        is not even considered a provider."""
        head = (REPO / "hosts" / "hermes" / "__init__.py").read_text()[:8192]
        self.assertTrue("register_memory_provider" in head or "MemoryProvider" in head)
