"""The wiring of the CLI to the stack: the `qctx stack` group and the step in `qctx install`.

Two families of tests, the split the brief asks for:

- the SUBPROCESS tests run `cli/qctx.py` in a hermetic environment (a temporary HOME, a
  temporary `QCTX_STACK_DIR`) with a PATH whose only container binaries are FAKE
  `docker`/`podman` scripts. Those scripts are fixtures: they only record their invocation
  in a file and answer enough for `stack.runtimes.discover` to find a runtime. No real
  engine exists here (that is Task 14's job), so the tests assert on the recording file, not
  on engine output.

- the IN-PROCESS and SUBPROCESS-PROBE tests assert the pure parts: that building the parser
  never imports the heavy stack modules (the load-bearing latency constraint for
  `qctx statusline`), that the `STACK_BUDGETS` tuples stay in lockstep with the timeout
  constants the recall hook and the hermes host read (by AST, so they cannot drift), and
  that the interrupted-install case dispatches to `provision` again.

Nothing in this file touches a live engine, the network or the real HOME.
"""
import ast
import io
import json
import os
import stat
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
CLI = REPO / "cli" / "qctx.py"

from tests.isolation import hermetic_env  # noqa: E402


def _exec(path: Path, script: str) -> Path:
    """Write `script` to `path` and make it executable (a fixture on the PATH)."""
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


# The fake runtime scripts. They are FIXTURES: the only job is to record each invocation
# (the tests assert on the recording file, not on engine output) and to answer the two
# commands `stack.runtimes.discover` runs -- `<runtime> info` and `<runtime> compose
# version` -- so the binary is found and the runtime's engine answers. No real engine is
# started; a real one is Task 14.
FAKE_DOCKER = """#!/usr/bin/env bash
# fixture: record the invocation, then answer enough for discover to find the runtime
if [ -n "${STACK_CLI_LOG:-}" ]; then printf 'docker %s\\n' "$*" >> "$STACK_CLI_LOG"; fi
if [ "$1" = "info" ]; then
  printf '{"ServerVersion": "27.0.0", "OSType": "linux", "Architecture": "x86_64", "ServerErrors": []}'
  exit 0
fi
if [ "$1" = "compose" ] && [ "$2" = "version" ]; then
  printf 'Docker Compose version v5.2.0\\n'
  exit 0
fi
exit 0
"""

FAKE_PODMAN = """#!/usr/bin/env bash
# fixture: record the invocation, then answer enough for discover to find the runtime
if [ -n "${STACK_CLI_LOG:-}" ]; then printf 'podman %s\\n' "$*" >> "$STACK_CLI_LOG"; fi
if [ "$1" = "info" ]; then
  printf '{"version": {"Version": "5.7.0"}, "host": {"os": "linux", "arch": "amd64", "kernel": "7.0.0-34-generic"}}'
  exit 0
fi
if [ "$1" = "compose" ] && [ "$2" = "version" ]; then
  printf 'Docker Compose version v5.2.0\\n'
  exit 0
fi
exit 0
"""


def load_cli():
    """Imports cli/qctx.py as a module (it is a script, not a package member) -- the same
    loader `tests/test_cli_install.py` uses."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("qctx_cli_for_stack", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_stack_dir(tmp: Path, *, phase: str = "compose", runtime: str = "docker") -> Path:
    """A recorded install the way a real `qctx install` leaves `stack.json`: the shape
    `state.load` reads. The tests that need a state in a particular phase set `phase` (an
    interrupted install is the one the resume test provisions again)."""
    import stack.catalog as catalog
    import stack.state as state
    directory = tmp / "stack"
    directory.mkdir(parents=True, exist_ok=True)
    state.save(directory, state.StackState(
        role="local", listen="127.0.0.1", platform="linux", runtime=runtime,
        provider=["docker compose"], profile="cpu", device=None, gpu_index=None,
        ports={"qdrant": 6333, "embed": 8003, "rerank": 8004},
        images={"llama": catalog.LLAMA_IMAGE, "qdrant": catalog.QDRANT_IMAGE},
        models={"bge-m3-Q4_K_M.gguf": "x", "bge-reranker-v2-m3-Q4_K_M.gguf": "y"},
        qdrant_version="1.19.2", selinux=False, phase=phase,
        created_at="2026-10-06T00:00:00Z", updated_at="2026-10-06T00:00:00Z",
        project=catalog.PROJECT, schema=1))
    return directory


class StackGroupSubprocess(unittest.TestCase):
    """The `qctx stack` group and the `qctx install` flags, run as subprocesses."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        _exec(self.bin / "docker", FAKE_DOCKER)
        _exec(self.bin / "podman", FAKE_PODMAN)
        self.empty = self.root / "empty"
        self.empty.mkdir()
        self.log = self.root / "invocations"
        self.addCleanup(self.tmp.cleanup)

    def env(self, **overrides):
        return hermetic_env(self.home, PATH=str(self.bin), STACK_CLI_LOG=self.log,
                            **overrides)

    def run_cli(self, *argv, **env_over):
        return subprocess.run([sys.executable, str(CLI), *argv],
                              capture_output=True, text=True, env=self.env(**env_over),
                              timeout=180, stdin=subprocess.DEVNULL)

    def test_stack_status_without_a_stack_exits_0(self):
        """No `stack.json` in the directory: the command says so and exits clean. The
        hermetic home has no `QCTX_STACK_DIR` and no XDG override, so the default
        (`$home/.local/share/...`) does not exist yet."""
        done = self.run_cli("stack", "status")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("no managed stack", done.stdout)

    def test_stack_status_json(self):
        """`--json` answers in the channel a program reads: the structured shape."""
        stack_dir = self.root / "stack"
        stack_dir.mkdir()
        done = self.run_cli("stack", "status", "--json", QCTX_STACK_DIR=stack_dir)
        self.assertEqual(done.returncode, 0, done.stderr)
        payload = json.loads(done.stdout)
        self.assertFalse(payload["managed"])

    def test_stack_help_lists_the_four_commands(self):
        done = self.run_cli("stack", "--help")
        for verb in ("status", "up", "down", "remove"):
            self.assertIn(verb, done.stdout)
        self.assertEqual(done.returncode, 0)

    def test_install_help_shows_the_three_flags(self):
        done = self.run_cli("install", "--help")
        for flag in ("--stack", "--runtime", "--image"):
            self.assertIn(flag, done.stdout)
        self.assertEqual(done.returncode, 0)

    def test_install_yes_alone_prints_the_auto_line_and_runs_no_runtime(self):
        """`--yes` without `--stack` does NOT provision: it prints the line that would
        provision and touches no runtime (the recording file stays empty/absent)."""
        env = self.env(QCTX_STACK_DIR=self.root / "stack")
        done = subprocess.run([sys.executable, str(CLI), "install", "--yes"],
                              capture_output=True, text=True, env=env,
                              timeout=180, stdin=subprocess.DEVNULL)
        self.assertIn("--stack auto", done.stdout,
                      "the --yes line did not name the flag that would provision")
        self.assertFalse(self.log.exists(),
                         "a runtime was invoked although --yes alone must not provision")

    def test_install_check_json_carries_the_stack_key(self):
        done = self.run_cli("install", "--check", "--json",
                            QCTX_STACK_DIR=self.root / "stack")
        (self.root / "stack").mkdir(exist_ok=True)
        payload = json.loads(done.stdout)
        self.assertIn("stack", payload)
        self.assertFalse(payload["stack"]["managed"])

    def test_install_check_writes_nothing_in_the_stack_dir(self):
        stack_dir = make_stack_dir(self.root)
        before = sorted(p.name for p in stack_dir.iterdir())
        done = self.run_cli("install", "--check", QCTX_STACK_DIR=stack_dir)
        after = sorted(p.name for p in stack_dir.iterdir())
        self.assertEqual(before, after, "--check wrote into the stack directory")
        self.assertNotIn("Traceback", done.stderr)

    def test_install_stack_with_no_runtime_exits_1_naming_the_fix(self):
        """An empty PATH (no container binary) and `--stack auto`: the step cannot run,
        and it exits 1 naming what to install rather than a bare failure."""
        env = hermetic_env(self.home, PATH=str(self.empty),
                           QCTX_STACK_DIR=self.root / "stack")
        (self.root / "stack").mkdir(exist_ok=True)
        done = subprocess.run([sys.executable, str(CLI), "install", "--stack", "auto",
                               "--yes"],
                              capture_output=True, text=True, env=env,
                              timeout=180, stdin=subprocess.DEVNULL)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertNotIn("Traceback", done.stderr)
        self.assertIn("install", (done.stderr + done.stdout).lower(),
                      "the fix did not name installing a runtime")


class TheInstallStep(unittest.TestCase):
    """The `install_step` dispatcher, driven in-process with fakes.

    `install_step` is the small decision table over `provision` / `status` /
    `check_section`. It does not re-implement the flow: when it provisions, the whole
    twelve-step flow runs through `installer.provision`, so a test swaps `provision` and
    asserts it was called (an interrupted install resumes by provisioning again)."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def args(self, **over):
        base = dict(check=False, json=False, yes=True, config_only=False,
                    host=None, stack=None, runtime=None, image=None)
        base.update(over)
        return SimpleNamespace(**base)

    def test_an_interrupted_install_provisions_again(self):
        """A `stack.json` stuck in the `compose` phase means the last install was
        interrupted: the step resumes by provisioning again (it is not healthy, so it is
        not a stop-and-offer case). `provision` is replaced so the test asserts the
        DISPATCH decision without standing up a real runtime."""
        import stack.cli as stack_cli
        import stack.facts as facts
        import stack.installer as installer
        from stack.verify import Budget

        called = []

        def fake_provision(request, deps):
            called.append(request)
            return None

        stack_dir = make_stack_dir(self.root, phase="compose")
        linux = facts.HostFacts(system="linux", arch="amd64", wsl=False,
                                ram_bytes=None, disk_free_bytes=None, gpus=(),
                                render_nodes=(), selinux=False)
        budgets = [Budget("claude-code", 8.0, 6.0), Budget("hermes", 2.0, 2.0)]
        with mock.patch.dict(os.environ, {"QCTX_STACK_DIR": str(stack_dir)}), \
                mock.patch.object(stack_cli, "_discover",
                                  return_value=[SimpleNamespace(name="docker")]), \
                mock.patch.object(facts, "collect", return_value=linux), \
                mock.patch.object(installer, "provision", fake_provision), \
                redirect_stdout(io.StringIO()):
            stack_cli.install_step(
                self.args(stack="auto", yes=True),
                {"blockers": [], "ready": True, "checks": []},
                budgets=budgets,
                ask=lambda prompt: "",
                interactive=True)
        self.assertEqual(len(called), 1, "an interrupted install must provision again")
        self.assertEqual(called[0].profile, "auto")


class ParserIsLight(unittest.TestCase):
    """The load-bearing constraint: building the parser must not import the heavy stack
    modules, because `qctx statusline` builds the parser on every assistant message."""

    def test_the_parser_does_not_import_the_heavy_modules(self):
        """A SUBPROCESS (a fresh interpreter) imports `cli/qctx.py`, builds the parser,
        and asserts the four heavy modules are NOT in `sys.modules`. Running it in a
        subprocess is what catches a heavy import creeping back into the module level:
        an in-process test would share this suite's already-imported `stack.*`."""
        script = (
            "import sys\n"
            "sys.path.insert(0, {repo!r})\n"
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('qctx_probe', {cli!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "mod.build_parser()\n"
            "heavy = ['stack.installer', 'stack.lifecycle', 'stack.runtimes', "
            "'stack.process']\n"
            "imported = [name for name in heavy if name in sys.modules]\n"
            "assert not imported, 'parser imported a heavy stack module: ' + repr(imported)\n"
        ).format(repo=str(REPO), cli=str(CLI))
        done = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True,
                              env=hermetic_env(REPO / "nonexistent-home"),
                              timeout=180)
        self.assertEqual(done.returncode, 0,
                         "the parser pulled in a heavy stack module:\n"
                         + done.stderr + done.stdout)


class BudgetsMatchTheHosts(unittest.TestCase):
    """`STACK_BUDGETS` must stay in lockstep with the timeouts the recall hook and the
    hermes host read, or a budget change would desync the recall timing. Read BY AST so
    the constants cannot drift without this test failing."""

    @staticmethod
    def _top_level_value(path: Path, name: str):
        """The value of a top-level `NAME = <literal>` assignment, or None when absent."""
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
                    and isinstance(node.value, ast.Constant)):
                return node.value.value
        return None

    @staticmethod
    def _hermes_divisor(path: Path):
        """The divisor on the `share = ... / <N>` assignment in the hermes host."""
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "share" for t in node.targets)
                    and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.Div)
                    and isinstance(node.value.right, ast.Constant)):
                return float(node.value.right.value)
        return None

    def test_stack_budgets_match_the_hosts(self):
        budgets = {host: (embed, rerank) for host, embed, rerank
                   in load_cli().STACK_BUDGETS}
        self.assertEqual(set(budgets), {"claude-code", "hermes"})

        recall = REPO / "hooks" / "recall.py"
        embed_t = self._top_level_value(recall, "EMBED_TIMEOUT_S")
        rerank_t = self._top_level_value(recall, "RERANK_TIMEOUT_S")
        self.assertIsNotNone(embed_t, "EMBED_TIMEOUT_S = <number> in hooks/recall.py")
        self.assertIsNotNone(rerank_t, "RERANK_TIMEOUT_S = <number> in hooks/recall.py")
        self.assertEqual(budgets["claude-code"], (float(embed_t), float(rerank_t)),
                         "the claude-code budget drifted from the recall hook's timeouts")

        hermes = REPO / "hosts" / "hermes" / "__init__.py"
        budget = self._top_level_value(hermes, "HERMES_PREFETCH_BUDGET_S")
        divisor = self._hermes_divisor(hermes)
        self.assertIsNotNone(budget,
                             "HERMES_PREFETCH_BUDGET_S in hosts/hermes/__init__.py")
        self.assertIsNotNone(divisor,
                             "the `share = ... / <N>` assignment in the hermes host")
        self.assertEqual(budgets["hermes"],
                         (float(budget) / float(divisor), float(budget) / float(divisor)),
                         "the hermes budget drifted from HERMES_PREFETCH_BUDGET_S / divisor")


if __name__ == "__main__":
    unittest.main()
