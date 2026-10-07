"""Docker and Podman behind one runtime contract: the engine, the compose provider, the stats.

The real command outputs below were measured on this machine on 2026-10-06 (spec, "Detecção"):
a `podman info --format json` reduced to the fields that are read, a `podman compose version`
whose docker-compose banner lands in stderr, and a `docker info --format json`. What is pinned
here is behaviour, not the bytes: an engine that reports a DIFFERENT but well-formed output
still passes, and the M3 lie (`remoteSocket.exists: true` while the socket file is gone) is
pinned by `socket_alive`, which is the only verdict the code trusts.
"""
import json
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError  # noqa: E402
from stack import runtimes  # noqa: E402
from stack.runtimes import (  # noqa: E402
    Completed,
    Docker,
    Podman,
    SubprocessRunner,
    discover,
    normalize_arch,
    parse_size,
    socket_alive,
)
from tests.stack_fakes import FakeRunner, FakeRuntime  # noqa: E402

# -- measured on this machine, 2026-10-06 -----------------------------------

#: The fields of the real `podman info --format json` that `engine()` reads.
PODMAN_INFO = {
    "version": {"Version": "5.7.0"},
    "host": {
        "os": "linux",
        "arch": "amd64",
        "kernel": "7.0.0-34-generic",
        "security": {"rootless": True},
        "remoteSocket": {
            "path": "/run/user/1000/podman/podman.sock",
            "exists": True,  # the lie measured as M3: the file is not there
        },
    },
}

#: The real `podman compose version` here: the banner announces the external
#: docker-compose provider in stderr.
PODMAN_COMPOSE_VERSION = Completed(
    0, "Docker Compose version v5.2.0",
    '>>>> Executing external compose provider "/usr/local/bin/docker-compose". '
    'Please see podman-compose(1) for how to disable this message. <<<<')
PODMAN_COMPOSE_VERSION_FAILED = Completed(1, "", "no such command: podman compose")

#: The real `podman-compose version` here (podman 5.7.0, podman-compose 1.6.0): the podman
#: line comes FIRST, so the provider's version is the one on the line that names compose.
PODMAN_COMPOSE_VERSION_STANDALONE = Completed(
    0, "podman version 5.7.0\npodman-compose version 1.6.0\n")

#: The fields of `docker info --format '{{json .}}'` that `engine()` reads, under the
#: Docker API's `/info` names. There is no Docker engine on this machine, so these are an
#: example, not a measurement. `OSType` is the engine's OS; `OperatingSystem` is a label
#: (a distro, or "Docker Desktop").
DOCKER_INFO = {
    "ServerVersion": "28.3.2",
    "OSType": "linux",
    "OperatingSystem": "Ubuntu 24.04.3 LTS",
    "KernelVersion": "6.8.0-45-generic",
    "Architecture": "x86_64",
    "SecurityOptions": ["name=seccomp,profile=default", "name=rootless"],
}
DOCKER_INFO_ROOTFUL = {**DOCKER_INFO, "SecurityOptions": ["name=seccomp,profile=default"]}
#: Docker Desktop with the WSL2 backend: the engine kernel is the WSL one.
DOCKER_INFO_WSL = {**DOCKER_INFO, "OperatingSystem": "Docker Desktop",
                   "KernelVersion": "5.15.167.4-microsoft-standard-WSL2"}
#: `--format '{{json .}}'` works on every Docker CLI. The `--format json` shorthand exists
#: only since Docker 23; an older CLI prints the literal word `json` instead.
DOCKER_JSON = "{{json .}}"

#: A Mac. `podman info` answers from INSIDE the VM, so its socket is the VM's path; the
#: one `podman compose` hands docker-compose is the host side one, from the machine's
#: ConnectionInfo. Shapes read in the podman source at v5.7.0 and v6.0.0
#: (pkg/machine/config.go InspectInfo, pkg/domain/entities/machine.go MachineHostInfo,
#: cmd/podman/compose.go composeDockerHost): `podman machine inspect` has NO VMType.
MACOS_PODMAN_INFO = {
    "version": {"Version": "5.7.0"},
    "host": {
        "os": "linux",
        "arch": "arm64",
        "kernel": "6.17.7-200.fc43.aarch64",
        "security": {"rootless": True},
        "remoteSocket": {"path": "/run/user/501/podman/podman.sock", "exists": True},
    },
}
MACOS_HOST_SOCKET = "/var/folders/zz/T/podman/podman-machine-default-api.sock"
MACOS_RESPONSES = {
    ("podman", "info", "--format", "json"): Completed(0, json.dumps(MACOS_PODMAN_INFO)),
    ("podman", "machine", "info", "--format", "json"): Completed(0, json.dumps(
        {"Host": {"VMType": "libkrun", "CurrentMachine": "podman-machine-default",
                  "MachineState": "Running"},
         "Version": {"Version": "5.7.0"}})),
    ("podman", "machine", "inspect", "podman-machine-default"): Completed(0, json.dumps(
        [{"Name": "podman-machine-default", "State": "running",
          "ConnectionInfo": {"PodmanSocket": {"Path": MACOS_HOST_SOCKET},
                             "PodmanPipe": None}}])),
    ("podman", "compose", "version"): PODMAN_COMPOSE_VERSION,
}
DOCKER_COMPOSE_VERSION = Completed(0, "Docker Compose version v2.35.0")
DOCKER_COMPOSE_VERSION_FAILED = Completed(1, "", "docker: 'compose' is not a docker command.")
DOCKER_COMPOSE_BINARY_VERSION = Completed(0, "docker-compose version 2.35.0")

#: The real `podman stats --no-stream --format json`: a JSON LIST of objects.
PODMAN_STATS = json.dumps([
    {"name": "memories-plugin-qdrant", "mem_usage": "190.7MB / 132.5GB"},
    {"name": "memories-plugin-embed", "mem_usage": "2.0GiB / 132.5GB"},
])

#: The Docker shape: one JSON object per line.
DOCKER_STATS = (
    '{"Name": "memories-plugin-qdrant", "MemUsage": "1.8GiB / 125GiB"}\n'
    '{"Name": "memories-plugin-embed", "MemUsage": "2.0GiB / 125GiB"}\n')

PROBE_FILE = Path("/x/probe/intel.yaml")


def _info(exists: bool):
    info = json.loads(json.dumps(PODMAN_INFO))
    info["host"]["remoteSocket"]["exists"] = exists
    return info


def _podman_provider(info, which, alive, compose_version, standalone=None):
    responses = {
        ("podman", "info", "--format", "json"): Completed(0, json.dumps(info)),
        ("podman", "compose", "version"): compose_version,
    }
    if standalone is not None:
        responses[("podman-compose", "version")] = standalone
    return Podman(FakeRunner(responses), which=which,
                  host_system="linux", socket_alive=alive)


class TestTheComposeContract(unittest.TestCase):
    def test_contract_every_runtime_builds_the_same_compose_argv(self):
        provider = runtimes.Provider(("podman", "compose"), "wrapper", "5.2.0")
        expected = [*provider.argv, "-p", "p", "-f", "/x/compose.yaml", "up", "-d"]
        # Every argv the three runtimes build starts with the provider's ("podman",
        # "compose"), so one prefix answer lets each `compose` return; the assertion
        # then reads the argv that was actually handed to the runner.
        docker_runner = FakeRunner({("podman",): Completed(0)})
        podman_runner = FakeRunner({("podman",): Completed(0)})
        fake = FakeRuntime("fake", engine=None, provider_info=None, list_devices={})
        # The argv is observed through the runner's recorded call, because the real
        # `compose` returns a `Completed` and hands the argv to the runner.
        for runtime, runner in ((Docker(docker_runner), docker_runner),
                                (Podman(podman_runner), podman_runner),
                                (fake, fake)):
            with self.subTest(runtime=runtime.name):
                runtime.compose(provider, "p", Path("/x/compose.yaml"), "up", "-d",
                                timeout=600.0)
                self.assertEqual(runner.calls[-1][0], expected)


class TestTheEngine(unittest.TestCase):
    def test_podman_engine_reads_version_os_arch_rootless_and_socket(self):
        podman = _podman_provider(_info(True), which={"podman": "/usr/bin/podman"},
                                  alive=lambda path: False,
                                  compose_version=PODMAN_COMPOSE_VERSION)
        engine = podman.engine()

        self.assertEqual(engine.name, "podman")
        self.assertEqual(engine.version, "5.7.0")
        self.assertEqual(engine.os, "linux")
        self.assertEqual(engine.arch, "amd64")
        self.assertTrue(engine.rootless)
        self.assertEqual(engine.kernel, "7.0.0-34-generic")
        self.assertEqual(engine.socket, "/run/user/1000/podman/podman.sock")
        self.assertIsNone(engine.vm)

    def test_podman_engine_is_none_when_the_binary_or_info_is_gone(self):
        missing = Podman(FakeRunner({}), which={}, host_system="linux")
        self.assertIsNone(missing.engine())
        no_info = Podman(FakeRunner({("podman", "info", "--format", "json"):
                                     Completed(1, "", "cannot connect")}),
                         which={"podman": "/usr/bin/podman"}, host_system="linux")
        self.assertIsNone(no_info.engine())

    def test_docker_engine_reads_rootless_from_security_options(self):
        docker = Docker(FakeRunner({("docker", "info", "--format", DOCKER_JSON):
                                    Completed(0, json.dumps(DOCKER_INFO))}),
                        which={"docker": "/usr/bin/docker"})
        engine = docker.engine()

        self.assertEqual(engine.name, "docker")
        self.assertEqual(engine.version, "28.3.2")
        self.assertEqual(engine.os, "linux")  # OSType, not the distro label
        self.assertEqual(engine.kernel, "6.8.0-45-generic")
        self.assertEqual(engine.arch, "amd64")  # x86_64 normalized
        self.assertTrue(engine.rootless)

        rootful = Docker(FakeRunner({("docker", "info", "--format", DOCKER_JSON):
                                     Completed(0, json.dumps(DOCKER_INFO_ROOTFUL))}),
                         which={"docker": "/usr/bin/docker"})
        self.assertFalse(rootful.engine().rootless)

    def test_docker_desktop_on_wsl_carries_the_wsl_kernel(self):
        # platform_of reads `microsoft` in the engine kernel to send WSL to the phase-3
        # refusal; without the kernel, a Docker engine under WSL would pass for linux.
        docker = Docker(FakeRunner({("docker", "info", "--format", DOCKER_JSON):
                                    Completed(0, json.dumps(DOCKER_INFO_WSL))}),
                        which={"docker": "/usr/bin/docker"})
        self.assertIn("microsoft", docker.engine().kernel)

    def test_an_engine_whose_info_is_not_json_is_absent(self):
        # What a Docker CLI older than 23 prints for `--format json`: the literal word.
        # The engine counts as not answering (discover skips it), never a traceback.
        docker = Docker(FakeRunner({("docker", "info"): Completed(0, "json\n")}),
                        which={"docker": "/usr/bin/docker"})
        self.assertIsNone(docker.engine())
        podman = Podman(FakeRunner({("podman", "info"): Completed(0, "not json")}),
                        which={"podman": "/usr/bin/podman"}, host_system="linux")
        self.assertIsNone(podman.engine())

    def test_on_macos_the_vm_type_and_the_socket_come_from_the_machine(self):
        podman = Podman(FakeRunner(MACOS_RESPONSES),
                        which={"podman": "/opt/podman/bin/podman"}, host_system="macos")
        engine = podman.engine()

        self.assertEqual(engine.vm, "libkrun")
        self.assertEqual(engine.socket, MACOS_HOST_SOCKET)
        self.assertEqual(engine.arch, "arm64")

    def test_on_macos_a_machine_that_does_not_answer_leaves_vm_and_socket_unknown(self):
        responses = {**MACOS_RESPONSES,
                     ("podman", "machine", "info", "--format", "json"):
                         Completed(125, "", "Error: no machine")}
        podman = Podman(FakeRunner(responses),
                        which={"podman": "/opt/podman/bin/podman"}, host_system="macos")
        engine = podman.engine()

        self.assertIsNone(engine.vm)
        self.assertIsNone(engine.socket)

    def test_normalize_arch_folds_machine_names(self):
        self.assertEqual(normalize_arch("x86_64"), "amd64")
        self.assertEqual(normalize_arch("amd64"), "amd64")
        self.assertEqual(normalize_arch("aarch64"), "arm64")
        self.assertEqual(normalize_arch("arm64"), "arm64")
        self.assertEqual(normalize_arch("ARM64"), "arm64")
        self.assertEqual(normalize_arch("riscv64"), "riscv64")


class TestTheComposeProvider(unittest.TestCase):
    def test_podman_compose_behind_podman_needs_a_live_socket(self):
        runtime = _podman_provider(
            _info(False),
            which={"podman": "/usr/bin/podman"},  # no podman-compose on PATH
            alive=lambda path: False,
            compose_version=PODMAN_COMPOSE_VERSION)
        info = runtime.compose_provider()

        self.assertIsNone(info.provider)
        self.assertIsNotNone(info.problem)
        self.assertIn("/run/user/1000/podman/podman.sock", info.problem)
        self.assertEqual(info.fix, "systemctl --user enable --now podman.socket")

    def test_a_dead_socket_falls_back_to_podman_compose(self):
        runtime = _podman_provider(
            _info(False),
            which={"podman": "/usr/bin/podman",
                   "podman-compose": "/usr/bin/podman-compose"},
            alive=lambda path: False,
            compose_version=PODMAN_COMPOSE_VERSION,
            standalone=PODMAN_COMPOSE_VERSION_STANDALONE)
        info = runtime.compose_provider()

        self.assertEqual(info.provider.argv, ("podman-compose",))
        self.assertEqual(info.provider.name, "podman-compose")
        self.assertEqual(info.provider.version, "1.6.0")
        self.assertIsNone(info.problem)
        self.assertIsNotNone(info.note)

    def test_podman_info_saying_the_socket_exists_is_not_believed(self):
        # M3: the verdict is `socket_alive`, never the `exists` field. The lie
        # (`exists: true` on a dead socket) must come out identical to the honest one.
        lying = _podman_provider(
            _info(True), which={"podman": "/usr/bin/podman"},
            alive=lambda path: False, compose_version=PODMAN_COMPOSE_VERSION)
        honest = _podman_provider(
            _info(False), which={"podman": "/usr/bin/podman"},
            alive=lambda path: False, compose_version=PODMAN_COMPOSE_VERSION)

        self.assertEqual(lying.compose_provider().problem,
                         honest.compose_provider().problem)
        self.assertEqual(lying.compose_provider().fix, honest.compose_provider().fix)
        self.assertIsNone(lying.compose_provider().provider)
        # The engine really does carry the lie, so the divergence is real:
        self.assertIsNotNone(lying.engine().socket)

    def test_a_live_socket_keeps_podman_compose_wrapper(self):
        runtime = _podman_provider(
            _info(True),
            which={"podman": "/usr/bin/podman"},
            alive=lambda path: True,
            compose_version=PODMAN_COMPOSE_VERSION)
        info = runtime.compose_provider()

        self.assertEqual(info.provider.argv, ("podman", "compose"))
        self.assertEqual(info.provider.version, "v5.2.0")
        self.assertIsNone(info.problem)
        self.assertIsNone(info.note)

    def test_on_macos_the_liveness_check_asks_the_host_side_socket(self):
        asked = []
        podman = Podman(FakeRunner(MACOS_RESPONSES),
                        which={"podman": "/opt/podman/bin/podman"}, host_system="macos",
                        socket_alive=lambda path: asked.append(path) or True)
        info = podman.compose_provider()

        self.assertEqual(info.provider.argv, ("podman", "compose"))
        self.assertEqual(asked, [MACOS_HOST_SOCKET])

    def test_a_native_podman_compose_needs_no_socket(self):
        # The banner names something that is not docker-compose: `podman compose` talks to
        # the engine in-process, so a dead socket is no obstacle.
        runtime = _podman_provider(
            _info(False),
            which={"podman": "/usr/bin/podman"},
            alive=lambda path: False,
            compose_version=Completed(
                0, "podman-compose version 1.6.0",
                '>>>> Executing external compose provider "/usr/local/bin/podman-compose".'))
        info = runtime.compose_provider()

        self.assertEqual(info.provider.argv, ("podman", "compose"))

    def test_when_the_wrapper_fails_the_standalone_binary_is_tried(self):
        runtime = _podman_provider(
            _info(True),
            which={"podman": "/usr/bin/podman",
                   "podman-compose": "/usr/bin/podman-compose"},
            alive=lambda path: True,
            compose_version=PODMAN_COMPOSE_VERSION_FAILED,
            standalone=PODMAN_COMPOSE_VERSION_STANDALONE)
        info = runtime.compose_provider()

        self.assertEqual(info.provider.argv, ("podman-compose",))
        self.assertEqual(info.provider.version, "1.6.0")

    def test_docker_prefers_the_plugin_then_the_standalone_binary(self):
        plugin = Docker(FakeRunner({("docker", "compose", "version"):
                                    DOCKER_COMPOSE_VERSION}),
                        which={"docker": "/usr/bin/docker",
                               "docker-compose": "/usr/local/bin/docker-compose"})
        info = plugin.compose_provider()
        self.assertEqual(info.provider.argv, ("docker", "compose"))
        self.assertEqual(info.provider.version, "v2.35.0")

        binary = Docker(FakeRunner({("docker", "compose", "version"):
                                    DOCKER_COMPOSE_VERSION_FAILED,
                                    ("docker-compose", "version"):
                                    DOCKER_COMPOSE_BINARY_VERSION}),
                        which={"docker": "/usr/bin/docker",
                               "docker-compose": "/usr/local/bin/docker-compose"})
        info = binary.compose_provider()
        self.assertEqual(info.provider.argv, ("docker-compose",))
        self.assertEqual(info.provider.version, "2.35.0")

        none = Docker(FakeRunner({("docker", "compose", "version"):
                                  DOCKER_COMPOSE_VERSION_FAILED}),
                      which={"docker": "/usr/bin/docker"})
        info = none.compose_provider()
        self.assertIsNone(info.provider)
        self.assertIsNotNone(info.problem)


class TestTheSocketVerdict(unittest.TestCase):
    def test_socket_alive_connects_for_real(self):
        # The one test that does real I/O: a listener on loopback AF_UNIX is the only way
        # to prove the connect is what decides.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "podman.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(path))
                server.listen(1)
                self.assertTrue(socket_alive(str(path)))
            finally:
                server.close()

        self.assertFalse(socket_alive(str(Path(tmp) / "gone.sock")))
        self.assertFalse(socket_alive(None))


class TestTheSizeAndStats(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(parse_size("1.831GB"), 1831000000)
        self.assertEqual(parse_size("190.7MB"), 190700000)
        self.assertEqual(parse_size("1.8GiB"), int(1.8 * 2**30))
        self.assertEqual(parse_size("512KiB"), 524288)
        self.assertEqual(parse_size("0B"), 0)
        self.assertIsNone(parse_size("--"))

    def test_stats_reads_both_formats(self):
        podman = Podman(FakeRunner({("podman", "stats", "--no-stream", "--format", "json"):
                                    Completed(0, PODMAN_STATS)}),
                        which={"podman": "/usr/bin/podman"})
        self.assertEqual(podman.stats(["a", "b"]),
                         {"memories-plugin-qdrant": 190700000,
                          "memories-plugin-embed": 2147483648})

        docker = Docker(FakeRunner({("docker", "stats", "--no-stream", "--format", DOCKER_JSON):
                                    Completed(0, DOCKER_STATS)}),
                        which={"docker": "/usr/bin/docker"})
        self.assertEqual(docker.stats(["a", "b"]),
                         {"memories-plugin-qdrant": 1932735283,
                          "memories-plugin-embed": 2147483648})

        self.assertEqual(podman.stats([]), {})
        self.assertEqual(Docker(FakeRunner({}), which={}).stats([]), {})


class TestTheSubprocessRunner(unittest.TestCase):
    def test_a_missing_binary_is_127(self):
        completed = SubprocessRunner().run(["/definitely/not/a/binary"], timeout=5.0)

        self.assertEqual(completed.returncode, 127)
        self.assertEqual(completed.stdout, "")
        self.assertIn("not/a/binary: not found", completed.stderr)
        self.assertFalse(completed.ok)

    def test_the_subprocess_runner_turns_a_timeout_into_a_stack_error(self):
        with self.assertRaises(StackError) as ctx:
            SubprocessRunner().run(["sleep", "30"], timeout=0.1)

        self.assertEqual(ctx.exception.step, "runtime")
        self.assertIn("sleep", str(ctx.exception))
        self.assertIn("0.1", str(ctx.exception))

    def test_a_streamed_command_that_hangs_is_a_stack_error_too(self):
        with self.assertRaises(StackError) as ctx:
            SubprocessRunner().run(["sleep", "30"], timeout=0.1, stream=True)

        self.assertEqual(ctx.exception.step, "runtime")

    def test_the_runner_captures_when_not_streaming(self):
        completed = SubprocessRunner().run(["echo", "hi"], timeout=5.0)

        self.assertTrue(completed.ok)
        self.assertEqual(completed.stdout.strip(), "hi")

    def test_a_timeout_kills_the_whole_process_group(self):
        # R2 item m2: a compose provider runs its backend as a CHILD of the
        # command. A plain timeout kills only the direct child and leaves the
        # backend alive. `start_new_session` makes the command a session leader
        # and the kill goes to the whole group, so the grandchild dies too.
        pid_file = Path(tempfile.gettempdir()) / "qctx-r2-grandchild"
        # the direct child is the shell; it spawns `sleep 30` (the grandchild)
        # and records its pid, then waits. The command hangs on the `wait`.
        script = f"sleep 30 & echo $! > {pid_file}; wait"
        with self.assertRaises(StackError):
            SubprocessRunner().run(["sh", "-c", script], timeout=1.0)
        if not pid_file.exists():
            self.fail("the command was killed before it recorded the grandchild pid")
        grandchild = int(pid_file.read_text().strip())
        pid_file.unlink()
        # poll: the grandchild must be reaped by the group kill, not linger
        for _ in range(20):
            try:
                os.kill(grandchild, 0)  # signal 0: does the process exist?
            except OSError:
                return  # gone: the group kill reached it
            time.sleep(0.1)
        self.fail(f"grandchild {grandchild} survived the timeout kill")


class TestDiscover(unittest.TestCase):
    def test_discover_skips_a_runtime_whose_engine_does_not_answer(self):
        # docker: binary present, `docker info` does not answer -> skipped.
        # podman: binary present, `podman info` answers -> kept.
        responses = {
            ("docker", "info", "--format", DOCKER_JSON): Completed(1, "", "permission denied"),
            ("podman", "info", "--format", "json"): Completed(0, json.dumps(_info(True))),
        }
        runtimes_found = discover(FakeRunner(responses),
                                  which={"docker": "/usr/bin/docker",
                                         "podman": "/usr/bin/podman"})

        self.assertEqual([r.name for r in runtimes_found], ["podman"])

    def test_discover_lists_docker_before_podman(self):
        responses = {
            ("docker", "info", "--format", DOCKER_JSON): Completed(0, json.dumps(DOCKER_INFO)),
            ("podman", "info", "--format", "json"): Completed(0, json.dumps(_info(True))),
        }
        runtimes_found = discover(FakeRunner(responses),
                                  which={"docker": "/usr/bin/docker",
                                         "podman": "/usr/bin/podman"})

        self.assertEqual([r.name for r in runtimes_found], ["docker", "podman"])


class TestTheFakes(unittest.TestCase):
    def test_the_fake_runner_matches_by_longest_prefix_and_records_calls(self):
        fake = FakeRunner({
            ("podman", "info"): Completed(0, "short"),
            ("podman", "info", "--format", "json"): Completed(0, "long"),
        })
        self.assertEqual(fake.run(["podman", "info", "--format", "json"],
                                  timeout=30.0).stdout, "long")
        self.assertEqual(fake.run(["podman", "info", "extra"],
                                  timeout=30.0).stdout, "short")
        self.assertEqual(fake.calls, [
            (["podman", "info", "--format", "json"], 30.0),
            (["podman", "info", "extra"], 30.0)])

    def test_the_fake_runner_refuses_a_call_no_prefix_answers(self):
        fake = FakeRunner({("podman", "info"): Completed(0, "x")})
        with self.assertRaises(KeyError):
            fake.run(["docker", "info"], timeout=30.0)

    def test_the_fake_runtime_answers_list_devices_by_the_probe_file_stem(self):
        fake = FakeRuntime("podman", engine=None, provider_info=None,
                           list_devices={"intel": "Vulkan0: Intel (32656 MiB)"})
        # The probe returns a Completed (as the real runtimes do), stdout carrying the device list.
        self.assertEqual(
            fake.compose(runtimes.Provider(("podman", "compose"), "w", "1"), "p",
                         PROBE_FILE, "run", "-T", "probe", "--list-devices",
                         timeout=180.0),
            Completed(0, "Vulkan0: Intel (32656 MiB)"))
        self.assertEqual(fake.calls[0][2], ("run", "-T", "probe", "--list-devices"))

        # A probe file without a registered answer gives the no-device shape.
        self.assertEqual(
            fake.compose(runtimes.Provider(("podman", "compose"), "w", "1"), "p",
                         Path("/x/probe/nvidia-0.yaml"), "run", "-T", "probe",
                         "--list-devices", timeout=180.0),
            Completed(0, ""))

    def test_the_fake_runtime_returns_registered_failures_before_anything_else(self):
        fake = FakeRuntime("podman", engine=None, provider_info=None, list_devices={},
                           fail={"up": Completed(1, "", "pull failed")})
        self.assertEqual(
            fake.compose(runtimes.Provider(("podman", "compose"), "w", "1"), "p",
                         PROBE_FILE, "up", "-d", timeout=600.0),
            Completed(1, "", "pull failed"))
        # A plain non-probe `up` with no registered failure is a success.
        self.assertEqual(
            FakeRuntime("podman", engine=None, provider_info=None, list_devices={}).compose(
                runtimes.Provider(("podman", "compose"), "w", "1"), "p", PROBE_FILE,
                "up", "-d", timeout=600.0), Completed(0))


# -- measured for review round R2, 2026-10-06 ---------------------------------
# The fields of `docker info --format '{{json .}}'` when the Docker CLI is present
# but the DAEMON does not answer. Measured 2026-10-06 with the real Docker CLI
# 27.5.1 (downloaded to the scratch dir) run against a missing socket:
#   DOCKER_HOST=unix:///nonexistent.sock docker info --format '{{json .}}'; echo rc=$?
# The CLI EXITS 0 and prints the client part with zero-valued server fields and
# "SecurityOptions": null (R2 items C1 and I3): iterating SecurityOptions then
# raises TypeError, and discover crashes although Podman works. The guard is
# therefore the JSON (a non-empty ServerVersion and no ServerErrors), not the
# exit code. Trimmed here to the keys `engine()` reads, with ServerErrors.
DOCKER_INFO_DEAD_DAEMON = {
    "ServerVersion": "",
    "OSType": "",
    "Architecture": "",
    "KernelVersion": "",
    "SecurityOptions": None,
    "ServerErrors": ["Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
                     "Is the docker daemon running?"],
    "MemTotal": 0,
}
#: A `docker` that is the podman-docker shim (`docker` = a script doing
#: `exec podman "$@"`, podman v5.7.0 docker/docker.in): `docker info` prints
#: PODMAN's info, which has no ServerVersion, so the engine() guard must reject
#: it and let Podman be discovered on its own (R2 item I3).
DOCKER_SHIM_TO_PODMAN = json.dumps(PODMAN_INFO)


class TestReviewRoundR2(unittest.TestCase):
    # ---- C1 / I3: the engine is answered by the JSON, not the exit code ----
    def test_a_dead_daemon_that_exits_zero_is_not_an_engine(self):
        # Docker CLI 23.0 to 28.0 exits 0 with zero-valued fields on a dead daemon.
        # The old code iterated SecurityOptions (None) and raised TypeError.
        docker = Docker(FakeRunner({("docker", "info", "--format", DOCKER_JSON):
                                    Completed(0, json.dumps(DOCKER_INFO_DEAD_DAEMON))}),
                        which={"docker": "/usr/bin/docker"})
        self.assertIsNone(docker.engine())

    def test_docker_info_with_server_errors_is_not_an_engine(self):
        # a non-empty ServerErrors means the daemon did not fully answer
        info = {**DOCKER_INFO, "ServerErrors": ["some daemon error"]}
        docker = Docker(FakeRunner({("docker", "info", "--format", DOCKER_JSON):
                                    Completed(0, json.dumps(info))}),
                        which={"docker": "/usr/bin/docker"})
        self.assertIsNone(docker.engine())

    def test_a_docker_that_is_a_podman_shim_is_not_a_docker_engine(self):
        # the shim prints podman's info: no ServerVersion, so it is rejected.
        docker = Docker(FakeRunner({("docker", "info", "--format", DOCKER_JSON):
                                    Completed(0, DOCKER_SHIM_TO_PODMAN)}),
                        which={"docker": "/usr/bin/docker"})
        self.assertIsNone(docker.engine())

    def test_discover_skips_docker_with_a_dead_daemon_and_keeps_podman(self):
        # the C1 regression end to end: a dead-daemon Docker (exit 0) must not
        # crash discover and must not shadow the working Podman.
        responses = {
            ("docker", "info", "--format", DOCKER_JSON):
                Completed(0, json.dumps(DOCKER_INFO_DEAD_DAEMON)),
            ("podman", "info", "--format", "json"): Completed(0, json.dumps(_info(True))),
        }
        found = discover(FakeRunner(responses),
                         which={"docker": "/usr/bin/docker",
                                "podman": "/usr/bin/podman"})
        self.assertEqual([r.name for r in found], ["podman"])

    def test_discover_skips_an_engine_whose_info_times_out(self):
        # R2 item m1: a HUNG `docker info` (the runner raises StackError on
        # timeout) must not stop Podman from being tried.
        class HangingRunner:
            def run(self, argv, *, timeout, stream=False):
                if argv[:2] == ["docker", "info"]:
                    raise StackError("timed out after 30.0s: docker info", step="runtime")
                if argv[:2] == ["podman", "info"]:
                    return Completed(0, json.dumps(_info(True)))
                raise KeyError(argv)

        found = discover(HangingRunner(),
                         which={"docker": "/usr/bin/docker",
                                "podman": "/usr/bin/podman"})
        self.assertEqual([r.name for r in found], ["podman"])

    def test_the_engine_carries_the_memory_the_containers_get(self):
        # R2 item I6: the engine's MemTotal (docker) / host.memTotal (podman) is
        # what the containers get, not the Mac's hw.memsize.
        docker = Docker(FakeRunner({("docker", "info", "--format", DOCKER_JSON):
                                    Completed(0, json.dumps(DOCKER_INFO))}),
                        which={"docker": "/usr/bin/docker"})
        self.assertIsNone(docker.engine().memory_bytes)  # DOCKER_INFO has no MemTotal

        podman = _podman_provider(_info(True), which={"podman": "/usr/bin/podman"},
                                  alive=lambda path: True,
                                  compose_version=PODMAN_COMPOSE_VERSION)
        # PODMAN_INFO has no memTotal: the field is None, not a guess
        self.assertIsNone(podman.engine().memory_bytes)
        with_mem = json.loads(json.dumps(PODMAN_INFO))
        with_mem["host"]["memTotal"] = 2147483648
        podman2 = _podman_provider(with_mem, which={"podman": "/usr/bin/podman"},
                                   alive=lambda path: True,
                                   compose_version=PODMAN_COMPOSE_VERSION)
        self.assertEqual(podman2.engine().memory_bytes, 2147483648)

    # ---- I1: docker-compose is recognised by the banner OR the stdout ----
    def test_banner_off_still_detects_docker_compose_and_needs_the_socket(self):
        # the banner can be switched off (PODMAN_COMPOSE_WARNING_LOGS=false);
        # stdout still says `Docker Compose version`. Measured 2026-10-06.
        runtime = _podman_provider(
            _info(False),
            which={"podman": "/usr/bin/podman"},  # no podman-compose on PATH
            alive=lambda path: False,
            compose_version=Completed(0, "Docker Compose version v5.2.0", ""))
        info = runtime.compose_provider()
        self.assertIsNone(info.provider)
        self.assertEqual(info.fix, "systemctl --user enable --now podman.socket")

    def test_banner_off_with_a_live_socket_keeps_the_wrapper(self):
        runtime = _podman_provider(
            _info(True),
            which={"podman": "/usr/bin/podman"},
            alive=lambda path: True,
            compose_version=Completed(0, "Docker Compose version v5.2.0", ""))
        info = runtime.compose_provider()
        self.assertEqual(info.provider.argv, ("podman", "compose"))
        self.assertEqual(info.provider.version, "v5.2.0")

    # ---- I2: a failing `podman compose version` means no provider ----
    def test_a_failed_version_is_a_missing_provider_not_the_socket(self):
        # the real stderr, measured 2026-10-06 with both providers hidden on
        # PATH (env PATH=/usr/bin:/bin podman compose version; rc=125). The
        # socket is NOT the cause: `version` never touches it.
        real_stderr = ("Error: looking up compose provider failed\n"
                       "7 errors occurred:\n"
                       "\t* exec: \"/home/me/.docker/cli-plugins/docker-compose\": "
                       "stat /home/me/.docker/cli-plugins/docker-compose: no such file "
                       "or directory\n"
                       "\t* exec: \"docker-compose\": executable file not found in $PATH\n"
                       "\t* exec: \"podman-compose\": executable file not found in $PATH")
        runtime = _podman_provider(
            _info(True),  # a LIVE socket: the old code blamed the socket anyway
            which={"podman": "/usr/bin/podman"},  # no podman-compose on PATH
            alive=lambda path: True,
            compose_version=Completed(125, "", real_stderr))
        info = runtime.compose_provider()
        self.assertIsNone(info.provider)
        self.assertIn("looking up compose provider failed", info.problem)
        self.assertEqual(info.fix, "install podman-compose (or docker-compose)")

    def test_a_failed_version_falls_back_to_podman_compose(self):
        runtime = _podman_provider(
            _info(True),
            which={"podman": "/usr/bin/podman",
                   "podman-compose": "/usr/bin/podman-compose"},
            alive=lambda path: True,
            compose_version=Completed(125, "", "Error: looking up compose provider failed"),
            standalone=PODMAN_COMPOSE_VERSION_STANDALONE)
        info = runtime.compose_provider()
        self.assertEqual(info.provider.argv, ("podman-compose",))
        self.assertEqual(info.provider.version, "1.6.0")

    # ---- m5: a failed stats carries the tool's own first stderr line ----
    def test_a_failed_stats_carries_the_tools_first_stderr_line(self):
        # measured: podman stats on a missing container exits 125 and names it
        podman = Podman(FakeRunner({("podman", "stats", "--no-stream", "--format", "json"):
                                    Completed(125, "", "time=\"now\" level=warning msg=\"no "
                                                       "such container\"\nError: no such "
                                                       "container: gone")}),
                        which={"podman": "/usr/bin/podman"})
        with self.assertRaises(StackError) as ctx:
            podman.stats(["gone"])
        self.assertEqual(ctx.exception.step, "runtime")
        self.assertIn("podman stats failed", str(ctx.exception))
        self.assertIn("no such container", str(ctx.exception))

    def test_a_failed_docker_stats_is_a_stack_error_not_a_traceback(self):
        docker = Docker(FakeRunner({("docker", "stats", "--no-stream", "--format",
                                     DOCKER_JSON): Completed(1, "", "Cannot connect to "
                                                                    "the Docker daemon")}),
                        which={"docker": "/usr/bin/docker"})
        with self.assertRaises(StackError) as ctx:
            docker.stats(["gone"])
        self.assertEqual(ctx.exception.step, "runtime")
        self.assertIn("docker stats failed", str(ctx.exception))

    # ---- m7: the compose version token strips a trailing comma ----
    def test_the_compose_version_strips_a_trailing_comma(self):
        # v1 docker-compose prints `docker-compose version 1.29.2, build 5becea4c`:
        # the token after `version` must drop the comma.
        self.assertEqual(_compose_version("docker-compose version 1.29.2, build 5becea4c"),
                         "1.29.2")
        self.assertEqual(_compose_version("Docker Compose version v5.2.0"), "v5.2.0")
        self.assertEqual(_compose_version("podman-compose version 1.6.0"), "1.6.0")


# keep the name importable for the class above
from stack.runtimes import _compose_version  # noqa: E402


if __name__ == "__main__":
    unittest.main(verbosity=2)
