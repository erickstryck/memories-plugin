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
from stack import StackError  # noqa: E402


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
        images={"llama": catalog.LLAMA_IMAGE, "qdrant": catalog.QDRANT_IMAGE,
                "llama-dzn": catalog.LLAMA_DZN_IMAGE},
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

    def test_section_lines_renders_the_stack_status_shape(self):
        """`section_lines` is fed by TWO producers with different `services` shapes:
        `check_section` (the install report) maps a service to its int status, but
        `lifecycle.status` (the `qctx stack status` verb) maps it to
        `{"url": …, "status": …}`. A section that is healthy must render every
        200 service as up and a non-200 one as down in BOTH shapes -- the dict
        shape used to fall through `status == 200` as a dict, so a running stack
        printed 'ok endpoints: … down, … down, … down' (measured 2026-10-07, T15)."""
        from stack.cli import section_lines
        base = {"managed": True, "phase": "running", "runtime": "podman",
                "profile": "intel", "ports": {"qdrant": 6333, "embed": 8003,
                                              "rerank": 8004}}
        # the lifecycle.status shape: a dict per service
        lifecycle_shape = dict(base, healthy=True, services={
            "qdrant": {"url": "http://127.0.0.1:6333", "status": 200},
            "embed": {"url": "http://127.0.0.1:8003/v1", "status": 200},
            "rerank": {"url": "http://127.0.0.1:8004/v1/rerank", "status": 200}})
        lines = section_lines(lifecycle_shape)
        endpoints = next(l for l in lines if "endpoints:" in l)
        self.assertIn("qdrant up", endpoints)
        self.assertIn("embed up", endpoints)
        self.assertIn("rerank up", endpoints)
        self.assertNotIn("down", endpoints)
        # a non-200 (the model still loading) renders down, not up
        partial = dict(base, healthy=False, services={
            "qdrant": {"url": "http://127.0.0.1:6333", "status": 200},
            "embed": {"url": "http://127.0.0.1:8003/v1", "status": 503},
            "rerank": {"url": "http://127.0.0.1:8004/v1/rerank", "status": None}})
        line = next(l for l in section_lines(partial) if "endpoints:" in l)
        self.assertIn("qdrant up", line)
        self.assertIn("embed down", line)
        self.assertIn("rerank down", line)
        # the check_section shape (int per service) still renders the same way
        int_shape = dict(base, healthy=True,
                         services={"qdrant": 200, "embed": 200, "rerank": 200})
        endpoints = next(l for l in section_lines(int_shape) if "endpoints:" in l)
        self.assertIn("qdrant up", endpoints)
        self.assertNotIn("down", endpoints)

    def test_section_lines_names_the_stop_points_at_up_and_renders_boot_and_config(self):
        """`qctx stack status` is the verb that must name a broken install and
        say how to fix it (spec: it shows the boot line and whether the config
        still points at the stack, and in EVERY case it accuses a stopped stack
        and points at `qctx stack up`). The boot and config fields only exist in
        the `lifecycle.status` shape, so a section without them (the install
        report) still renders, just without those two lines."""
        from stack.cli import section_lines
        base = {"managed": True, "phase": "stopped", "runtime": "podman",
                "profile": "cpu", "ports": {"qdrant": 6333, "embed": 8003,
                                            "rerank": 8004}}
        # a stopped stack: the endpoints are down, and the section carries the
        # boot line and the config verdict (the lifecycle.status shape)
        down = dict(base)
        down.update(healthy=False, services={
            "qdrant": {"url": "http://127.0.0.1:6333", "status": None},
            "embed": {"url": "http://127.0.0.1:8003/v1", "status": None},
            "rerank": {"url": "http://127.0.0.1:8004/v1/rerank", "status": None}},
            boot="podman-restart.service is disabled; linger is no",
            config_points_here=False)
        lines = section_lines(down)
        joined = "\n".join(lines)
        self.assertIn("qctx stack up", joined, "a stopped stack must name the fix")
        self.assertIn("podman-restart.service is disabled", joined, "the boot line")
        self.assertIn("does not point", joined, "the config does not point at the stack")
        # a healthy stack does NOT point at `stack up`
        up = dict(base)
        up.update(phase="running", healthy=True, services={
            "qdrant": {"url": "http://127.0.0.1:6333", "status": 200},
            "embed": {"url": "http://127.0.0.1:8003/v1", "status": 200},
            "rerank": {"url": "http://127.0.0.1:8004/v1/rerank", "status": 200}},
            boot="the docker daemon must start at boot; restart: always "
                 "brings the containers back",
            config_points_here=True)
        up_joined = "\n".join(section_lines(up))
        self.assertNotIn("qctx stack up", up_joined)
        self.assertIn("points at this stack", up_joined)
        # the install-report shape (no boot / no config fields) still renders,
        # with neither of those two lines
        minimal = dict(base)
        minimal.update(healthy=False,
                       services={"qdrant": None, "embed": None, "rerank": None})
        minimal_lines = "\n".join(section_lines(minimal))
        self.assertTrue(minimal_lines, "the minimal shape must not be empty")
        self.assertNotIn("boot:", minimal_lines)
        self.assertNotIn("config:", minimal_lines)
        # an unfinished install (phase compose) is resumed by the install, never
        # by `stack up`, which verifies nothing and writes no config
        unfinished = dict(base)
        unfinished.update(phase="compose", healthy=True, services={
            "qdrant": 200, "embed": 200, "rerank": 200})
        unfinished_lines = "\n".join(section_lines(unfinished))
        self.assertIn("resume it with: qctx install", unfinished_lines)
        self.assertNotIn("qctx stack up", unfinished_lines)

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

    def test_stack_up_refuses_an_image_without_upgrade(self):
        """Without --upgrade `up` repeats exactly what stack.json holds, so an
        `--image` there was accepted and silently dropped. It is refused, with the
        command that does take it."""
        import stack.cli as stack_cli
        from stack import StackError
        with mock.patch.object(stack_cli, "_life_deps",
                               side_effect=AssertionError("must refuse before any runtime")), \
                self.assertRaises(StackError) as ctx:
            stack_cli._cmd_up(SimpleNamespace(image=["llama=example.org/llama:x"],
                                              upgrade=False))
        self.assertIn("--upgrade", ctx.exception.fix)

    def test_the_stack_flag_offers_exactly_the_catalogue_profiles(self):
        """`--stack` lists its choices by hand (the parser must not import the
        heavy `stack.backends`); this holds the list to the registry, so a new
        backend cannot be forgotten in the flag."""
        import argparse
        import stack.cli as stack_cli
        from stack.backends import BACKENDS
        parser = argparse.ArgumentParser()
        stack_cli.add_install_flags(parser)
        action = next(a for a in parser._actions if a.dest == "stack")
        self.assertEqual(tuple(action.choices), ("auto", *BACKENDS))

    def test_the_lifecycle_verbs_get_the_real_environment(self):
        """`qctx stack up --upgrade` resolves the QCTX_STACK_IMAGE_* overrides from
        the `env` the CLI hands the lifecycle. The unit tests drive `lifecycle.up`
        with an env they build, so only this test proves the CLI passes the real
        one (a `_life_deps` that dropped it brought the bug back unnoticed)."""
        import stack.cli as stack_cli
        with TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"QCTX_STACK_DIR": tmp,
                                             "QCTX_STACK_IMAGE_LLAMA": "example.org/llama:env"}), \
                mock.patch.object(stack_cli, "_discover", return_value=[]):
            deps = stack_cli._life_deps(SimpleNamespace(), None)
        self.assertEqual(deps.env.get("QCTX_STACK_IMAGE_LLAMA"), "example.org/llama:env")

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

    def test_a_stopped_stack_restarts_not_provisions(self):
        """A `stack.json` in the `stopped` phase (only `qctx stack down` writes it)
        restarts with the CHEAP `lifecycle.up` (re-render + `compose up -d`, the
        images `stack.json` already holds), not a twelve-step re-provision. The
        post-reboot case -- phase still `running`, the endpoints answering
        nothing -- is the next test. This is the round-7 fix: the stopped and
        the interrupted branches used to both call `_provision`."""
        import stack.cli as stack_cli
        import stack.facts as facts
        import stack.installer as installer
        import stack.lifecycle as lifecycle

        up_calls = []
        provision_calls = []

        def fake_up(deps, *, upgrade=False, images=None):
            up_calls.append((upgrade, images))
            return None

        def fake_provision(request, deps):
            provision_calls.append(request)
            return None

        stack_dir = make_stack_dir(self.root, phase="stopped")
        with mock.patch.dict(os.environ, {"QCTX_STACK_DIR": str(stack_dir)}), \
                mock.patch.object(stack_cli, "_discover",
                                  return_value=[SimpleNamespace(name="docker")]), \
                mock.patch.object(facts, "is_native_windows", return_value=False), \
                mock.patch.object(lifecycle, "up", fake_up), \
                mock.patch.object(installer, "provision", fake_provision), \
                redirect_stdout(io.StringIO()):
            stack_cli.install_step(
                self.args(stack=None, yes=True),
                {"blockers": [], "ready": True, "checks": []},
                budgets=[], ask=lambda prompt: "", interactive=True)
        self.assertEqual(len(up_calls), 1, "a stopped stack restarts via lifecycle.up")
        self.assertEqual(up_calls[0], (False, None),
                         "the restart repeats the recorded images (no upgrade)")
        self.assertEqual(len(provision_calls), 0,
                         "a stopped stack must NOT re-provision (re-pull, re-proof)")

    def test_a_healthy_running_stack_prints_status_and_offers_nothing(self):
        """The guard for the reboot test above: a `running` stack whose endpoints
        DO answer is healthy, so the step prints the status and goes on -- it must
        NOT offer a restart (that would re-`up` a stack that is already up). The
        pins hint still appears when the catalogue moved."""
        import stack.cli as stack_cli
        import stack.facts as facts
        import stack.installer as installer
        import stack.lifecycle as lifecycle

        up_calls = []
        provision_calls = []

        def fake_up(deps, *, upgrade=False, images=None):
            up_calls.append((upgrade, images))
            return None

        def fake_provision(request, deps):
            provision_calls.append(request)
            return None

        stack_dir = make_stack_dir(self.root, phase="running")
        healthy = {"managed": True, "phase": "running", "runtime": "docker",
                   "profile": "cpu", "ports": {"qdrant": 6333, "embed": 8003,
                                               "rerank": 8004},
                   "services": {"qdrant": {"url": "http://127.0.0.1:6333", "status": 200},
                                "embed": {"url": "http://127.0.0.1:8003/v1", "status": 200},
                                "rerank": {"url": "http://127.0.0.1:8004/v1/rerank",
                                           "status": 200}},
                   "healthy": True, "outdated_pins": []}
        with mock.patch.dict(os.environ, {"QCTX_STACK_DIR": str(stack_dir)}), \
                mock.patch.object(stack_cli, "_discover",
                                  return_value=[SimpleNamespace(name="docker")]), \
                mock.patch.object(facts, "is_native_windows", return_value=False), \
                mock.patch.object(lifecycle, "up", fake_up), \
                mock.patch.object(lifecycle, "status", return_value=(healthy, 0)), \
                mock.patch.object(installer, "provision", fake_provision):
            buf = io.StringIO()
            with redirect_stdout(buf):
                stack_cli.install_step(
                    self.args(stack=None, yes=True),
                    {"blockers": [], "ready": True, "checks": []},
                    budgets=[], ask=lambda prompt: "y", interactive=True)
        said = buf.getvalue()
        self.assertEqual(len(up_calls), 0,
                         "a healthy stack must NOT be restarted")
        self.assertEqual(len(provision_calls), 0)
        self.assertNotIn("restart the stack now", said)
        self.assertIn("managed stack: running", said, "the status is printed")

    def test_a_rebooted_stack_with_the_restart_declined_says_it_once(self):
        """Interactive, restart declined: the install report above already printed
        the stack block (with its `qctx stack up` pointer); the step adds its one
        line and the question, and does not print the block a second time."""
        import stack.cli as stack_cli
        import stack.facts as facts
        import stack.lifecycle as lifecycle

        stack_dir = make_stack_dir(self.root, phase="running")
        unhealthy = {"managed": True, "phase": "running", "runtime": "docker",
                     "profile": "cpu", "ports": {"qdrant": 6333, "embed": 8003,
                                                 "rerank": 8004},
                     "services": {"qdrant": None, "embed": None, "rerank": None},
                     "healthy": False, "outdated_pins": []}
        with mock.patch.dict(os.environ, {"QCTX_STACK_DIR": str(stack_dir)}), \
                mock.patch.object(facts, "is_native_windows", return_value=False), \
                mock.patch.object(lifecycle, "status", return_value=(unhealthy, 1)), \
                mock.patch.object(lifecycle, "up") as up:
            buf = io.StringIO()
            with redirect_stdout(buf):
                stack_cli.install_step(
                    self.args(stack=None, yes=False),
                    {"blockers": [], "ready": True, "checks": []},
                    budgets=[], ask=lambda prompt: "n", interactive=True)
        said = buf.getvalue()
        up.assert_not_called()
        self.assertNotIn("managed stack:", said, "the block is the report's, not repeated")
        self.assertEqual(said.count("qctx stack up"), 1, said)

    def test_a_rebooted_stack_running_with_dead_endpoints_offers_to_restart(self):
        """The common post-reboot case: `down` is the only verb that writes
        `stopped`, so a reboot leaves the phase `running` while every endpoint
        answers nothing. The step must treat that as a stopped stack -- offer
        the cheap `lifecycle.up` (with `--yes`, take it) -- and not just print
        the status and go on (spec: a stopped stack offers to be restarted).
        A healthy running stack does NOT get the offer (the next branch
        proves it keeps printing status only)."""
        import stack.cli as stack_cli
        import stack.facts as facts
        import stack.installer as installer
        import stack.lifecycle as lifecycle

        up_calls = []
        provision_calls = []

        def fake_up(deps, *, upgrade=False, images=None):
            up_calls.append((upgrade, images))
            return None

        def fake_provision(request, deps):
            provision_calls.append(request)
            return None

        stack_dir = make_stack_dir(self.root, phase="running")
        # The section `_step_managed` dispatches on: the phase says running
        # (a reboot never rewrites it to stopped) but every endpoint answers
        # nothing, so `healthy` is False. This is the reboot case.
        unhealthy = {"managed": True, "phase": "running", "runtime": "docker",
                     "profile": "cpu", "ports": {"qdrant": 6333, "embed": 8003,
                                                 "rerank": 8004},
                     "services": {"qdrant": None, "embed": None, "rerank": None},
                     "healthy": False, "outdated_pins": []}
        with mock.patch.dict(os.environ, {"QCTX_STACK_DIR": str(stack_dir)}), \
                mock.patch.object(stack_cli, "_discover",
                                  return_value=[SimpleNamespace(name="docker")]), \
                mock.patch.object(facts, "is_native_windows", return_value=False), \
                mock.patch.object(lifecycle, "up", fake_up), \
                mock.patch.object(lifecycle, "status",
                                  return_value=(unhealthy, 1)), \
                mock.patch.object(installer, "provision", fake_provision):
            buf = io.StringIO()
            with redirect_stdout(buf):
                stack_cli.install_step(
                    self.args(stack=None, yes=True),
                    {"blockers": [], "ready": True, "checks": []},
                    budgets=[], ask=lambda prompt: "y", interactive=True)
        said = buf.getvalue()
        self.assertEqual(len(up_calls), 1,
                         "a rebooted stack must be restarted via lifecycle.up")
        self.assertEqual(up_calls[0], (False, None))
        self.assertEqual(len(provision_calls), 0)
        self.assertIn("qctx stack up", said, "the offer names the fix")

    def test_native_windows_is_not_offered_it_says_the_line_and_returns(self):
        """The plan's first `install_step` branch, now native-only: on bare
        Windows (the system name says Windows, WSL or not) the step is not
        offered -- it says the one line, points at the README's `## Local
        models`, and returns. It must not offer and then die in
        `installer._check_platform` with `StackError(step="platform")`. WSL2
        does NOT take this branch: it reports the system name Linux and is
        classified by the runtime discovery instead (the next two tests)."""
        import stack.cli as stack_cli
        import stack.facts as facts
        import stack.installer as installer
        import stack.lifecycle as lifecycle

        provision_calls = []
        up_calls = []

        def fake_provision(request, deps):
            provision_calls.append(request)
            return None

        def fake_up(deps, *, upgrade=False, images=None):
            up_calls.append(deps)
            return None

        stack_dir = self.root / "stack"
        stack_dir.mkdir(exist_ok=True)
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"QCTX_STACK_DIR": str(stack_dir)}), \
                mock.patch.object(facts, "is_native_windows", return_value=True), \
                mock.patch.object(installer, "provision", fake_provision), \
                mock.patch.object(lifecycle, "up", fake_up), \
                redirect_stdout(buf):
            stack_cli.install_step(
                self.args(stack="auto", yes=True),
                {"blockers": [{"name": "Qdrant"}], "ready": False, "checks": []},
                budgets=[], ask=lambda prompt: "", interactive=True)
        out = buf.getvalue()
        self.assertEqual(len(provision_calls), 0, "native Windows never provisions")
        self.assertEqual(len(up_calls), 0, "native Windows never restarts")
        self.assertIn("Local models", out, "the line must point at the README section")
        self.assertIn("not available", out, "the line must say it is not available")

    def test_wsl2_with_a_runtime_proceeds_to_provision(self):
        """WSL2 is no longer the one-line refusal: the gate reads the native
        system name, a WSL2 distro reports `Linux`, and with a runtime that
        answers the step goes on and provisions (the platform it is then
        classified as, `windows`, is accepted by `_check_platform`)."""
        import stack.cli as stack_cli
        import stack.facts as facts
        import stack.installer as installer

        provision_calls = []

        def fake_provision(request, deps):
            provision_calls.append(request)
            return None

        stack_dir = self.root / "stack"
        stack_dir.mkdir(exist_ok=True)
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"QCTX_STACK_DIR": str(stack_dir)}), \
                mock.patch.object(stack_cli, "_discover",
                                  return_value=[SimpleNamespace(name="docker")]), \
                mock.patch.object(facts, "is_native_windows", return_value=False), \
                mock.patch.object(facts, "collect",
                                  return_value=facts.HostFacts(
                                      system="linux", arch="amd64", wsl=True,
                                      ram_bytes=None, disk_free_bytes=None,
                                      gpus=(), render_nodes=(), selinux=False)), \
                mock.patch.object(installer, "provision", fake_provision), \
                redirect_stdout(buf):
            stack_cli.install_step(
                self.args(stack="auto", yes=True),
                {"blockers": [{"name": "Qdrant"}], "ready": False, "checks": []},
                budgets=[], ask=lambda prompt: "", interactive=True)
        out = buf.getvalue()
        self.assertEqual(len(provision_calls), 1,
                         "WSL2 with an answering runtime provisions")
        self.assertEqual(provision_calls[0].profile, "auto")
        self.assertEqual(provision_calls[0].yes, True)
        self.assertNotIn("Local models", out,
                         "the refusal line must not be said on WSL2")
        self.assertNotIn("not available", out,
                         "the refusal line must not be said on WSL2")

    def test_wsl2_with_no_runtime_aborts_with_step_runtime(self):
        """WSL2 with no runtime answering: the abort is the runtime one,
        naming the install, NOT the one-line platform refusal (which only
        covers native Windows). The refusal is `installer._choose_runtime`'s,
        and it rises unchanged from `provision`."""
        import stack.cli as stack_cli
        import stack.facts as facts
        import stack.installer as installer

        def fake_provision(request, deps):
            raise StackError("no container runtime with a compose provider answers",
                             step="runtime", fix="install Docker or Podman")

        stack_dir = self.root / "stack"
        stack_dir.mkdir(exist_ok=True)
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"QCTX_STACK_DIR": str(stack_dir)}), \
                mock.patch.object(stack_cli, "_discover", return_value=[]), \
                mock.patch.object(facts, "is_native_windows", return_value=False), \
                mock.patch.object(facts, "collect",
                                  return_value=facts.HostFacts(
                                      system="linux", arch="amd64", wsl=True,
                                      ram_bytes=None, disk_free_bytes=None,
                                      gpus=(), render_nodes=(), selinux=False)), \
                mock.patch.object(installer, "provision", fake_provision), \
                redirect_stdout(buf):
            with self.assertRaises(StackError) as ctx:
                stack_cli.install_step(
                    self.args(stack="auto", yes=True),
                    {"blockers": [{"name": "Qdrant"}], "ready": False, "checks": []},
                    budgets=[], ask=lambda prompt: "", interactive=True)
        self.assertEqual(ctx.exception.step, "runtime")
        self.assertTrue(ctx.exception.fix, "the runtime abort must name a fix")
        self.assertIn("install", ctx.exception.fix.lower(),
                      "the runtime abort must name the install")
        out = buf.getvalue()
        self.assertNotIn("Local models", out,
                         "no platform refusal: the abort names the runtime, not the platform")

    def test_the_offer_explains_what_it_would_do_before_it_asks(self):
        """The spec's "Quando aparece": a no-stack blocker explains what it would do --
        what it downloads, how much, the ports, where the data lives -- BEFORE it asks
        `y/N`. The numbers come from the catalogue (never a literal), so they cannot
        drift from the download. Declining still ends the step without provisioning."""
        import stack.cli as stack_cli
        import stack.facts as facts
        import stack.installer as installer
        import stack.catalog as catalog

        provision_calls = []
        asked = []

        def fake_provision(request, deps):
            provision_calls.append(request)
            return None

        def fake_ask(prompt):
            asked.append(prompt)
            return "n"  # decline

        stack_dir = self.root / "stack"
        stack_dir.mkdir(exist_ok=True)
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"QCTX_STACK_DIR": str(stack_dir)}), \
                mock.patch.object(facts, "is_native_windows", return_value=False), \
                mock.patch.object(installer, "provision", fake_provision), \
                redirect_stdout(buf):
            stack_cli.install_step(
                self.args(stack=None, yes=False),
                {"blockers": [{"name": "Qdrant"}, {"name": "Embedding"}],
                 "ready": False, "checks": []},
                budgets=[], ask=fake_ask, interactive=True)
        out = buf.getvalue()
        # the question was asked
        self.assertTrue(any("y/N" in p for p in asked), "the offer must ask y/N")
        # the disclosure is present, and its numbers are the catalogue's
        self.assertIn(f"{catalog.MODELS_BYTES / 2 ** 20:.0f} MiB", out,
                      "the offer must name the download size from the catalogue")
        self.assertIn("qdrant on 6333", out, "the offer must name the ports")
        self.assertIn(str(stack_dir), out, "the offer must name the data directory")
        self.assertEqual(len(provision_calls), 0, "declining must not provision")


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
