"""Both hosts must be installable from a git clone, and each proves it against the REAL host.

A one-command install is a promise about a layout, and the two hosts read that layout very
differently. hermes imports `$HERMES_HOME/plugins/<name>/__init__.py` by path and swallows any
failure at debug level — a broken install is INVISIBLE, which is how this plugin once shipped
loading nothing on that host. claude-code reads `.claude-plugin/` manifests and has a validator.

WHY THESE TESTS USE `git archive` AND NOT A DIRECTORY COPY. A clone delivers exactly the TRACKED
tree. Copying the working directory would let an untracked file — a scratch module, a stale
`.pyc`, a config someone forgot to commit — make the test pass while a real install failed. The
archive is the same set of bytes a user gets.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HERMES_SRC = Path.home() / ".hermes" / "hermes-agent"
HERMES_PY = HERMES_SRC / "venv" / "bin" / "python"
CLAUDE = shutil.which("claude")


def top_level_keys(path: Path) -> set:
    """Top-level mapping keys of a YAML file, read without pyyaml.

    Column 0 and a colon — enough for a flat manifest, and it keeps this suite stdlib-only.
    Reading KEYS and not raw text is the point: the first version of the manifest test searched
    the whole file for "provides_tools" and matched the COMMENT explaining why it is absent.
    """
    keys = set()
    for line in path.read_text().splitlines():
        if line[:1].isalpha() and ":" in line:
            keys.add(line.split(":", 1)[0].strip())

    return keys


def tracked_tree_into(destination: Path) -> None:
    """The tracked tree at HEAD, extracted into `destination` — what a clone would deliver."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "archive", "HEAD"], cwd=REPO, capture_output=True, timeout=120)
    assert archive.returncode == 0, archive.stderr.decode()[:400]
    extract = subprocess.run(["tar", "-x", "-C", str(destination)], input=archive.stdout,
                             capture_output=True, timeout=120)
    assert extract.returncode == 0, extract.stderr.decode()[:400]


class TestTheRepositoryRootIsAWorkingHermesPlugin(unittest.TestCase):
    """`hermes plugins install owner/repo` clones into `$HERMES_HOME/plugins/<name>/`, so the
    thing hermes imports is whatever sits at the repository ROOT. That is the only reason
    `__init__.py` exists there."""

    def test_the_root_reexports_what_the_loader_looks_for(self):
        """Offline half — no hermes needed. The loader wants a `register(ctx)` function or a
        class extending its MemoryProvider; the names are read from the file as TEXT because
        importing it here would prove nothing about a fresh process."""
        source = (REPO / "__init__.py").read_text()
        self.assertIn("MemoriesProvider", source)
        self.assertIn("register", source)

    def test_the_root_carries_the_marker_the_loaders_TEXT_SCAN_needs(self):
        """Before importing anything, hermes greps the first 8192 bytes for one of two strings
        (`_is_memory_provider_dir`). A file that imports correctly but carries neither is never
        even considered, and the failure reads as "the provider does not exist"."""
        head = (REPO / "__init__.py").read_bytes()[:8192]
        self.assertTrue(b"register_memory_provider" in head or b"MemoryProvider" in head,
                        "the root __init__.py would not survive hermes' text scan")

    def test_it_OPENS_NO_SOCKET_at_import_time(self):
        """`plugins list` and a doctor run import this file, and neither asked for network.

        Asserted by BREAKING the socket and importing anyway, not by grepping the source for
        suspicious words — the first version of this test did the latter and failed on the word
        "connection" inside its own docstring. A substring in prose is not a behaviour."""
        probe = (
            "import socket, sys\n"
            "socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw("
            "AssertionError('the import opened a socket'))\n"
            f"sys.path.insert(0, {str(REPO.parent)!r})\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            "import importlib.util, pathlib\n"
            f"spec = importlib.util.spec_from_file_location('probe_root', {str(REPO / '__init__.py')!r})\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['probe_root'] = m\n"
            "spec.loader.exec_module(m)\n"
            "print('IMPORTED', hasattr(m, 'MemoriesProvider'), hasattr(m, 'register'))\n"
        )
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                             cwd=tempfile.mkdtemp(), timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        self.assertEqual(out.stdout.strip(), "IMPORTED True True")

    @unittest.skipUnless(HERMES_PY.exists(), f"hermes is not installed at {HERMES_SRC}")
    def test_the_REAL_loader_returns_a_provider_from_a_clone(self):
        """The half a fake cannot prove. Builds the exact layout an install produces — the
        tracked tree at `$HERMES_HOME/plugins/memories/` — and asks hermes' own loader for a
        provider. Everything is temporary: `HERMES_HOME` points at the box, so the real
        `~/.hermes` is neither read nor written."""
        box = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, box, True)
        tracked_tree_into(box / "plugins" / "memories")
        probe = (
            "import os, sys, pathlib\n"
            f"os.environ['HERMES_HOME'] = {str(box)!r}\n"
            f"sys.path.insert(0, {str(HERMES_SRC)!r})\n"
            "from plugins.memory import _load_provider_from_dir\n"
            f"p = _load_provider_from_dir(pathlib.Path({str(box / 'plugins' / 'memories')!r}))\n"
            "print(type(p).__name__ if p else 'NONE')\n"
            "print(len(p.get_tool_schemas()) if p else 0)\n"
        )
        out = subprocess.run([str(HERMES_PY), "-c", probe], capture_output=True, text=True,
                             cwd="/tmp", timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        lines = out.stdout.strip().splitlines()
        self.assertEqual(lines[0], "MemoriesProvider",
                         f"hermes' own loader refused the clone (it fails SILENTLY): {out.stderr[-400:]}")
        self.assertGreater(int(lines[1]), 0, "the provider loaded but exposes no tools")


class TestTheHermesManifestDescribesWhatSHIPS(unittest.TestCase):
    def test_it_declares_only_fields_the_installer_reads(self):
        """`provides_tools` is deliberately absent: the installer resolves those names against
        hermes' TOOLSET registry, and this plugin's tools are exposed through the memory
        provider. Declaring them would make `plugins show` answer wrongly, which is worse than
        answering less."""
        keys = top_level_keys(REPO / "plugin.yaml")
        self.assertNotIn("provides_tools", keys)
        for required in ("manifest_version", "name", "description"):
            with self.subTest(field=required):
                self.assertIn(required, keys)

    def test_it_names_the_two_keys_that_a_shell_less_hermes_will_not_have(self):
        """The installer checks `requires_env` against `~/.hermes/.env` and NAMES what is
        missing. That is the whole reason to declare them: a systemd or gateway hermes has no
        shell, so a key that lives only in an interactive environment is one it will not have —
        and the symptom is an archive that looks simply empty."""
        text = (REPO / "plugin.yaml").read_text()
        self.assertIn("QDRANT_SERVICE_API_KEY", text)
        self.assertIn("SERVER_API_KEY", text)

    def test_it_declares_NO_python_dependencies(self):
        """Stdlib only is a standing constraint here, and the manifest is where an installer
        would try to fix a violation by installing something."""
        self.assertRegex((REPO / "plugin.yaml").read_text(), r"python_dependencies:\s*\[\]")


class TestTheClaudeManifestsStayTRUE(unittest.TestCase):
    """claude-code has shipped installable for a while; what it lacked was anything holding the
    manifests honest as the plugin grew."""

    def manifests(self):
        root = REPO / ".claude-plugin"

        return (json.loads((root / "plugin.json").read_text()),
                json.loads((root / "marketplace.json").read_text()))

    def test_every_hook_command_points_at_a_file_that_EXISTS(self):
        """A hook naming a moved script fails at a moment nobody is watching — the harness runs
        it, it errors, and the prompt continues without recall. Paths are plugin-relative
        through ${CLAUDE_PLUGIN_ROOT}, which is exactly what makes them easy to break silently
        by renaming a file."""
        hooks = json.loads((REPO / "hooks" / "hooks.json").read_text())["hooks"]
        seen = 0
        for event, entries in hooks.items():
            for entry in entries:
                for hook in entry.get("hooks", []):
                    command = hook.get("command", "")
                    self.assertIn("${CLAUDE_PLUGIN_ROOT}", command,
                                  f"{event}: a relative path resolves from HOME, not the plugin")
                    tail = command.split("${CLAUDE_PLUGIN_ROOT}/", 1)[1].strip('"').strip()
                    with self.subTest(event=event, script=tail):
                        self.assertTrue((REPO / tail).is_file(), f"{tail} does not exist")
                    seen += 1
        self.assertGreaterEqual(seen, 3, "the three hooks this plugin ships are not all declared")

    def test_NO_manifest_declares_a_version(self):
        """The commit is the version, on both hosts, and this test is the decision.

        A hand-maintained version string had already gone stale here: the manifests said 0.3.0
        while `claude plugin list` reported 0.2.0 installed — measured. claude-code auto-versions
        from the commit SHA when the field is absent, so every push is deliverable rather than
        waiting on someone remembering to bump; hermes reads the field for display only and its
        `plugins update` is a git pull. `--ref <sha>` is how a specific commit gets pinned.

        The claude validator WARNS about the absent field and still passes. The warning is the
        price of the field never lying, which is the trade being made on purpose."""
        plugin, marketplace = self.manifests()
        self.assertNotIn("version", plugin, "plugin.json declares a version to go stale")
        self.assertNotIn("version", top_level_keys(REPO / "plugin.yaml"),
                         "plugin.yaml declares a version to go stale")
        for entry in marketplace["plugins"]:
            with self.subTest(plugin=entry["name"]):
                self.assertNotIn("version", entry)

    def test_the_marketplace_points_at_this_repository(self):
        """`source: ./` is what makes one repository both the plugin and its marketplace, which
        is what allows `marketplace add <git-url>` to work with no second repo to maintain."""
        _plugin, marketplace = self.manifests()
        self.assertEqual([e["source"] for e in marketplace["plugins"]], ["./"])

    def test_every_skill_directory_is_a_skill(self):
        """Skills are auto-discovered from `skills/`, so a directory without a SKILL.md is a
        silent no-op rather than an error."""
        for directory in sorted((REPO / "skills").iterdir()):
            if directory.is_dir():
                with self.subTest(skill=directory.name):
                    self.assertTrue((directory / "SKILL.md").is_file())

    @unittest.skipUnless(CLAUDE, "the claude CLI is not on PATH")
    def test_the_REAL_validator_accepts_the_manifests(self):
        """The other host's equivalent of running hermes' loader: its own validator, over the
        tracked tree, so an uncommitted fix cannot make this pass."""
        box = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, box, True)
        tracked_tree_into(box)
        out = subprocess.run([CLAUDE, "plugin", "validate", str(box)],
                             capture_output=True, text=True, timeout=300,
                             env={**os.environ, "CLAUDE_CONFIG_DIR": str(box / "config")})
        self.assertEqual(out.returncode, 0, (out.stdout + out.stderr)[-800:])
        self.assertIn("Validation passed", out.stdout + out.stderr)


class TestTheREADMEStatesWhatFAILSSILENTLY(unittest.TestCase):
    """The README is the only place a new machine gets told what breaks without saying so.

    Three failures on this plugin produce no error at all — a shell-less process with no keys, a
    config file left empty while the shell has the values, and a read guard that `plugins install`
    cannot register. Each was measured on a working installation, and each looked like "the archive
    is just empty". What is pinned here is that the README NAMES them; the wording is free.
    """

    def readme(self) -> str:
        return (REPO / "README.md").read_text()

    def test_it_says_the_keys_need_a_second_home_for_a_shell_less_hermes(self):
        text = self.readme()
        self.assertIn(".hermes/.env", text)
        self.assertRegex(text, r"(?i)systemd|no shell|shell-less")

    def test_it_warns_that_config_show_MIXES_file_and_environment(self):
        """The trap that hid this for weeks: the diagnostic looked complete while the file was
        empty, because it reads both and prints one picture."""
        self.assertRegex(self.readme(), r"(?i)config show.{0,80}(mix|both)")

    def test_it_says_plugins_install_does_NOT_register_the_guard(self):
        """A reader who assumes the install did everything gets a guard that never fires and
        nothing anywhere saying why."""
        text = self.readme()
        self.assertRegex(text, r"(?i)(no field for shell hooks|not.{0,20}registered)")
        self.assertIn("hermes_cutover.sh", text)

    def test_it_says_the_hook_needs_approving_ONCE(self):
        """`✗ not allowlisted` is the state right after a correct install, and a hermes with no
        TTY skips the hook until it has been approved."""
        text = self.readme()
        self.assertRegex(text, r"(?i)allowlist")
        self.assertRegex(text, r"(?i)TTY")

    def test_it_gives_the_no_shell_VERIFICATION_and_not_only_the_warning(self):
        """A warning without a check is advice. `env -i` is the recipe that proves it."""
        self.assertIn("env -i", self.readme())

    def test_it_names_the_marketplace_form_of_the_update_command(self):
        """`claude plugin update <bare name>` answers "not found", which reads like a broken
        install. The form that works carries the marketplace."""
        self.assertIn("memories-plugin@memories-plugin", self.readme())


if __name__ == "__main__":
    unittest.main(verbosity=2)
