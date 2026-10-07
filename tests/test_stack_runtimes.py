"""Docker and Podman behind one runtime contract: the engine, the compose provider, the stats,
and the process groups of the runner.

Each fixture of tool output below says, beside it, how it was obtained: measured on this
machine (the command and the date), read in the tool's source at a named tag, or labelled as an
example (there is no Docker engine here, so the `docker info` of a live daemon is an example).
What is pinned here is behaviour, not the bytes: an engine that reports a DIFFERENT but
well-formed output still passes, and the M3 lie (`remoteSocket.exists: true` while the socket
file is gone) is pinned by `socket_alive`, which is the only verdict the code trusts.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError  # noqa: E402
from stack import runtimes  # noqa: E402
from stack.engine import compose_version  # noqa: E402
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
from tests.isolation import hermetic_env  # noqa: E402
from tests.stack_fakes import FakeRunner, FakeRuntime  # noqa: E402

# -- measured on this machine, 2026-10-06 -----------------------------------

#: The fields of the real `podman info --format json` that `engine()` reads.
PODMAN_INFO = {
    "version": {"Version": "5.7.0"},
    "host": {
        "os": "linux",
        "arch": "amd64",
        "kernel": "7.0.0-34-generic",
        "memTotal": 132484689920,  # measured again 2026-10-07, same command
        "security": {"rootless": True},
        "remoteSocket": {
            "path": "/run/user/1000/podman/podman.sock",
            "exists": True,  # the lie measured as M3: the file is not there
        },
    },
}

#: The real `podman compose version` here (measured 2026-10-07, stdout and stderr each sent to
#: a file): the banner announces the external docker-compose provider in stderr, underlined
#: with ANSI escapes even when stderr is not a terminal.
PODMAN_COMPOSE_VERSION = Completed(
    0, "Docker Compose version v5.2.0\n",
    '\x1b[4m>>>> Executing external compose provider "/usr/local/bin/docker-compose". '
    'Please see podman-compose(1) for how to disable this message. <<<<\n\n\x1b[0m')
#: `podman compose version` when podman finds no compose provider, measured 2026-10-07 with
#: both providers hidden: `env PATH=/usr/bin:/bin podman compose version` exits 125. The one
#: change is the HOME in the first path, which is the repo's placeholder.
PODMAN_COMPOSE_VERSION_FAILED = Completed(125, "", (
    "Error: looking up compose provider failed\n"
    "7 errors occurred:\n"
    "\t* exec: \"/home/me/.docker/cli-plugins/docker-compose\": stat "
    "/home/me/.docker/cli-plugins/docker-compose: no such file or directory\n"
    "\t* exec: \"/usr/local/lib/docker/cli-plugins/docker-compose\": stat "
    "/usr/local/lib/docker/cli-plugins/docker-compose: no such file or directory\n"
    "\t* exec: \"/usr/local/libexec/docker/cli-plugins/docker-compose\": stat "
    "/usr/local/libexec/docker/cli-plugins/docker-compose: no such file or directory\n"
    "\t* exec: \"/usr/lib/docker/cli-plugins/docker-compose\": stat "
    "/usr/lib/docker/cli-plugins/docker-compose: no such file or directory\n"
    "\t* exec: \"/usr/libexec/docker/cli-plugins/docker-compose\": stat "
    "/usr/libexec/docker/cli-plugins/docker-compose: no such file or directory\n"
    "\t* exec: \"docker-compose\": executable file not found in $PATH\n"
    "\t* exec: \"podman-compose\": executable file not found in $PATH\n"))

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
    "MemTotal": 33406828544,
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
#: The plugin's `docker compose version`, an example with a version of its own: the plugin is
#: the same program as the standalone binary, whose line is measured below.
DOCKER_COMPOSE_VERSION = Completed(0, "Docker Compose version v2.35.0\n")
#: `docker compose version` with no compose plugin, measured 2026-10-07 with the Docker CLI
#: 27.5.1 and a HOME without cli-plugins: exit 1, two stderr lines.
DOCKER_COMPOSE_VERSION_FAILED = Completed(
    1, "", "docker: 'compose' is not a docker command.\nSee 'docker --help'\n")
#: The standalone binary, measured 2026-10-07: `docker-compose version` prints the plugin's
#: line, `Docker Compose version v5.2.0`, and nothing on stderr.
DOCKER_COMPOSE_BINARY_VERSION = Completed(0, "Docker Compose version v5.2.0\n")

#: The real `podman stats --no-stream --format json`: a JSON LIST of objects.
PODMAN_STATS = json.dumps([
    {"name": "memories-plugin-qdrant", "mem_usage": "190.7MB / 132.5GB"},
    {"name": "memories-plugin-embed", "mem_usage": "2.0GiB / 132.5GB"},
])

#: `podman stats` of a container that does not exist, measured 2026-10-07 (podman 5.7.0):
#:   podman stats --no-stream --format json qctx-r3-missing-ctr
#: exits 125 with this ONE stderr line and nothing on stdout.
PODMAN_STATS_NO_SUCH_CONTAINER = Completed(
    125, "",
    'Error: unable to get list of containers: unable to look up container qctx-r3-missing-ctr: '
    'no container with name or ID "qctx-r3-missing-ctr" found: no such container\n')

#: The Docker shape, an example (there is no Docker engine here): one JSON object per line.
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


# -- real processes, for the process-group tests of the runner ----------------

#: The shell (the command) starts `sleep 30` (its child: a provider's backend in real life),
#: records the child's pid in "$1/pid" and waits on it. A background job of a non-interactive
#: shell starts with SIGINT ignored, so the child outlives any SIGINT: only a SIGKILL to the
#: group takes it.
GRANDCHILD = 'sleep 30 & echo $! > "$1/pid"; wait'

#: A command that stops cleanly on SIGINT: it traps it, leaves "$1/stopped" and exits.
TRAPS_SIGINT = ("trap 'echo stopped > \"$1/stopped\"; exit 0' INT; echo $$ > \"$1/pid\"; "
                "while :; do sleep 0.1; done")

#: The driver a Ctrl-C test runs as a real process: argv is the repo, the script, the temp
#: directory and the mode. It installs the SIGINT handler an interactive Python has, because
#: a parent that is a non-interactive shell's background job hands SIGINT down as IGNORED,
#: and then the signal never arrives and the test proves nothing (R3-1).
CTRL_C_DRIVER = """
import signal, sys
sys.path.insert(0, sys.argv[1])
signal.signal(signal.SIGINT, signal.default_int_handler)
from stack.process import SubprocessRunner
SubprocessRunner().run(["sh", "-c", sys.argv[2], "sh", sys.argv[3]],
                       timeout=60.0, stream=sys.argv[4] == "stream")
"""

#: The grace the ruling gives a command between the forwarded SIGINT and the SIGKILL (R3-1).
GRACE = 5.0


def recorded_pid(pid_file: Path, timeout: float = 10.0) -> int | None:
    """The pid a shell wrote, once its whole line is there; None if it never comes."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            text = pid_file.read_text()
        except OSError:
            text = ""
        if text.endswith("\n"):
            return int(text)
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.02)


def gone_by(pid: int, deadline: float) -> bool:
    """Whether `pid` no longer exists by `deadline`; signal 0 asks without signalling."""
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def kill_if_alive(pid: int) -> None:
    """A failed run must not leave its command behind, as the RED run of R3-1 would."""
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


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
        # measured 2026-10-07, what a podman that talks to a dead socket prints (on a Mac
        # the client always does): `podman --remote --url unix:///nonexistent.sock info
        # --format json` exits 125, with the client part (not JSON) on stdout
        dead = Completed(
            125, "OS: linux/amd64\nbuildOrigin: Ubuntu\nprovider: qemu\nversion: 5.7.0\n\n",
            "Cannot connect to Podman. Please verify your connection to the Linux system "
            "using `podman system connection list`, or try `podman machine init` and "
            "`podman machine start` to manage a new Linux VM\n"
            "Error: unable to connect to Podman socket: Get "
            "\"http://d/v5.7.0/libpod/_ping\": dial unix /nonexistent.sock: connect: no such "
            "file or directory: unix:///nonexistent.sock\n")
        no_info = Podman(FakeRunner({("podman", "info", "--format", "json"): dead}),
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
        # an invented failure: only the exit code of `podman machine info` is read
        responses = {**MACOS_RESPONSES,
                     ("podman", "machine", "info", "--format", "json"): Completed(125)}
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
        # The banner names podman-compose, which drives the `podman` CLI and needs no API
        # socket, so a dead socket is no obstacle. Measured 2026-10-07 with docker-compose
        # hidden and podman-compose on the PATH (its Homebrew prefix):
        #   env PATH=/usr/bin:/bin:/home/linuxbrew/.linuxbrew/bin podman compose version
        runtime = _podman_provider(
            _info(False),
            which={"podman": "/usr/bin/podman"},
            alive=lambda path: False,
            compose_version=Completed(
                0, "podman version 5.7.0\npodman-compose version 1.6.0\n",
                '\x1b[4m>>>> Executing external compose provider '
                '"/home/linuxbrew/.linuxbrew/bin/podman-compose". Please see '
                'podman-compose(1) for how to disable this message. <<<<\n\n\x1b[0m'))
        info = runtime.compose_provider()

        self.assertEqual(info.provider.argv, ("podman", "compose"))
        self.assertEqual(info.provider.version, "1.6.0")  # the compose line, not podman's

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
        self.assertEqual(info.provider.version, "v5.2.0")

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
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StackError):
                SubprocessRunner().run(["sh", "-c", GRANDCHILD, "sh", tmp], timeout=1.0)
            grandchild = recorded_pid(Path(tmp) / "pid", timeout=0.0)
        if grandchild is None:
            self.fail("the command was killed before it recorded the grandchild pid")
        # the grandchild must be reaped by the group kill, not linger
        if not gone_by(grandchild, time.monotonic() + 2.0):
            kill_if_alive(grandchild)
            self.fail(f"grandchild {grandchild} survived the timeout kill")


class TestACtrlC(unittest.TestCase):
    """R3-1: a Ctrl-C reaches Python only, because the command runs in its own session. The
    runner stops the command's whole group before the interrupt goes on: SIGINT to the
    group, up to `GRACE` seconds, then SIGKILL. Each test runs a real driver process with
    the default SIGINT handler and sends the SIGINT to the driver only, as a terminal sends
    it to the foreground group the driver is in."""

    def run_driver(self, tmp: str, script: str, mode: str) -> subprocess.Popen:
        driver = subprocess.Popen(
            [sys.executable, "-c", CTRL_C_DRIVER, str(REPO), script, tmp, mode],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=hermetic_env(tmp))
        self.addCleanup(self.reap, driver)
        return driver

    @staticmethod
    def reap(driver: subprocess.Popen) -> None:
        if driver.poll() is None:
            driver.kill()
        driver.wait()

    def recorded(self, tmp: str) -> int:
        pid = recorded_pid(Path(tmp) / "pid")
        if pid is None:
            self.fail("the command never recorded a pid")
        return pid

    def assert_the_group_is_gone_after_a_ctrl_c(self, mode: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            driver = self.run_driver(tmp, GRANDCHILD, mode)
            grandchild = self.recorded(tmp)

            os.kill(driver.pid, signal.SIGINT)
            sent = time.monotonic()

            if not gone_by(grandchild, sent + GRACE + 2.0):
                kill_if_alive(grandchild)
                self.fail(f"grandchild {grandchild} survived the Ctrl-C ({mode} mode)")
            # the interrupt goes on once the group is stopped: the driver dies of it
            driver.wait(timeout=10.0)
            self.assertEqual(driver.returncode, -signal.SIGINT)

    def test_a_ctrl_c_stops_the_whole_group_in_capture_mode(self):
        self.assert_the_group_is_gone_after_a_ctrl_c("capture")

    def test_a_ctrl_c_stops_the_whole_group_in_stream_mode(self):
        self.assert_the_group_is_gone_after_a_ctrl_c("stream")

    def test_the_command_gets_the_sigint_first_and_stops_on_its_own(self):
        # What the terminal would have delivered reaches the command: one that traps
        # SIGINT stops cleanly, long before the grace is up, with no SIGKILL needed.
        with tempfile.TemporaryDirectory() as tmp:
            driver = self.run_driver(tmp, TRAPS_SIGINT, "capture")
            command = self.recorded(tmp)

            os.kill(driver.pid, signal.SIGINT)
            sent = time.monotonic()
            try:
                driver.wait(timeout=GRACE + 2.0)
            except subprocess.TimeoutExpired:
                pass
            stopped = (Path(tmp) / "stopped").exists()
            if not gone_by(command, time.monotonic()):
                kill_if_alive(command)
            self.assertTrue(stopped, "the command never received the SIGINT")
            self.assertLess(time.monotonic() - sent, GRACE)
            self.assertEqual(driver.returncode, -signal.SIGINT)

    def test_a_second_ctrl_c_cuts_the_grace_short(self):
        # An impatient second Ctrl-C during the grace goes straight to the SIGKILL: the
        # child that ignores SIGINT is gone well before the grace would have ended.
        with tempfile.TemporaryDirectory() as tmp:
            driver = self.run_driver(tmp, GRANDCHILD, "capture")
            grandchild = self.recorded(tmp)

            os.kill(driver.pid, signal.SIGINT)
            first = time.monotonic()
            time.sleep(1.0)  # the runner is now inside the grace
            os.kill(driver.pid, signal.SIGINT)

            if not gone_by(grandchild, first + GRACE - 1.0):
                kill_if_alive(grandchild)
                self.fail(f"grandchild {grandchild} outlived the second Ctrl-C")
            driver.wait(timeout=10.0)
            self.assertEqual(driver.returncode, -signal.SIGINT)


class TestDiscover(unittest.TestCase):
    def test_discover_skips_a_runtime_whose_engine_does_not_answer(self):
        # docker: binary present, `docker info` does not answer (a CLI from 28.1, which
        # exits 1 on a connection error) -> skipped.
        # podman: binary present, `podman info` answers -> kept.
        responses = {
            ("docker", "info", "--format", DOCKER_JSON): DOCKER_INFO_DEAD_DAEMON_28_1,
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
# but the DAEMON does not answer. Measured with the static Docker CLI 27.5.1 run
# against a missing socket, on 2026-10-06 and again on 2026-10-07 (same bytes):
#   DOCKER_HOST=unix:///nonexistent.sock docker info --format '{{json .}}'; echo rc=$?
# The CLI EXITS 0 and prints the client part with zero-valued server fields and
# "SecurityOptions": null (R2 items C1 and I3): iterating SecurityOptions then
# raises TypeError, and discover crashes although Podman works. The guard is
# therefore the JSON (a non-empty ServerVersion and no ServerErrors), not the
# exit code. The values are the measured ones, in the order printed, trimmed to the
# keys `engine()` reads plus ServerErrors and SecurityOptions.
DOCKER_INFO_DEAD_DAEMON = {
    "KernelVersion": "",
    "OSType": "",
    "Architecture": "",
    "MemTotal": 0,
    "ServerVersion": "",
    "SecurityOptions": None,
    "ServerErrors": ["Cannot connect to the Docker daemon at unix:///nonexistent.sock. "
                     "Is the docker daemon running?"],
}
#: The same dead daemon under a Docker CLI from 28.1, which EXITS 1 instead. Read in
#: docker/cli v28.1.0: cli/command/system/info.go (`addServerInfo` returns a connection
#: error instead of appending it to ServerErrors, a field tagged `json:",omitempty"`, and
#: `runInfo` still prints the format of the zero-valued `system.Info`) and cmd/docker/
#: docker.go (prints the error on stderr, exits 1). The stderr line is moby v28.1.0
#: client/errors.go `connectionFailed`, at the default host of client/client_unix.go.
DOCKER_INFO_DEAD_DAEMON_28_1 = Completed(
    1,
    json.dumps({"KernelVersion": "", "OSType": "", "Architecture": "", "MemTotal": 0,
                "ServerVersion": "", "SecurityOptions": None}),
    "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
    "Is the docker daemon running?\n")
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
        self.assertEqual(docker.engine().memory_bytes, 33406828544)

        podman = _podman_provider(_info(True), which={"podman": "/usr/bin/podman"},
                                  alive=lambda path: True,
                                  compose_version=PODMAN_COMPOSE_VERSION)
        self.assertEqual(podman.engine().memory_bytes, 132484689920)  # measured here
        # an engine that publishes no figure: None, not a guess
        without = _info(True)
        without["host"].pop("memTotal", None)
        podman2 = _podman_provider(without, which={"podman": "/usr/bin/podman"},
                                   alive=lambda path: True,
                                   compose_version=PODMAN_COMPOSE_VERSION)
        self.assertIsNone(podman2.engine().memory_bytes)

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

    def test_banner_off_with_a_v1_docker_compose_needs_the_socket_too(self):
        # R3-m5: a v1 docker-compose is a wrapper that needs the API socket as well, and
        # its stdout spells itself `docker-compose version` (docker/compose 1.29.2,
        # compose/cli/utils.py `get_version_info`: 'docker-compose version {}, build {}';
        # the build is the short sha of the tag's commit, 5becea4ca9f6).
        runtime = _podman_provider(
            _info(False),
            which={"podman": "/usr/bin/podman"},  # no podman-compose on PATH
            alive=lambda path: False,
            compose_version=Completed(0, "docker-compose version 1.29.2, build 5becea4c\n",
                                      ""))
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
        # the real answer with both providers hidden (PODMAN_COMPOSE_VERSION_FAILED). The
        # socket is NOT the cause: `version` never touches it.
        runtime = _podman_provider(
            _info(True),  # a LIVE socket: the old code blamed the socket anyway
            which={"podman": "/usr/bin/podman"},  # no podman-compose on PATH
            alive=lambda path: True,
            compose_version=PODMAN_COMPOSE_VERSION_FAILED)
        info = runtime.compose_provider()
        self.assertIsNone(info.provider)
        self.assertEqual(info.problem, "podman compose found no compose provider: "
                                       "Error: looking up compose provider failed")
        self.assertEqual(info.fix, "install podman-compose (or docker-compose)")

    # ---- m5: a failed stats carries the tool's own first stderr line ----
    def test_a_failed_stats_carries_the_tools_first_stderr_line(self):
        podman = Podman(FakeRunner({("podman", "stats", "--no-stream", "--format", "json"):
                                    PODMAN_STATS_NO_SUCH_CONTAINER}),
                        which={"podman": "/usr/bin/podman"})
        with self.assertRaises(StackError) as ctx:
            podman.stats(["qctx-r3-missing-ctr"])
        self.assertEqual(ctx.exception.step, "runtime")
        self.assertEqual(str(ctx.exception), "runtime: podman stats failed: "
                         + PODMAN_STATS_NO_SUCH_CONTAINER.stderr.strip())

    def test_a_failed_docker_stats_is_a_stack_error_not_a_traceback(self):
        # measured 2026-10-07 with the Docker CLI 27.5.1 against a missing socket:
        #   DOCKER_HOST=unix:///nonexistent.sock docker stats --no-stream \
        #     --format '{{json .}}' memories-plugin-qdrant
        # exits 1 with this one stderr line
        dead = Completed(1, "", "Cannot connect to the Docker daemon at "
                                "unix:///nonexistent.sock. Is the docker daemon running?\n")
        docker = Docker(FakeRunner({("docker", "stats", "--no-stream", "--format",
                                     DOCKER_JSON): dead}),
                        which={"docker": "/usr/bin/docker"})
        with self.assertRaises(StackError) as ctx:
            docker.stats(["memories-plugin-qdrant"])
        self.assertEqual(ctx.exception.step, "runtime")
        self.assertEqual(str(ctx.exception),
                         "runtime: docker stats failed: " + dead.stderr.strip())

    # ---- m7: the compose version token strips a trailing comma ----
    def test_the_compose_version_strips_a_trailing_comma(self):
        # v1 docker-compose prints `docker-compose version 1.29.2, build 5becea4c`:
        # the token after `version` must drop the comma.
        self.assertEqual(compose_version("docker-compose version 1.29.2, build 5becea4c"),
                         "1.29.2")
        self.assertEqual(compose_version("Docker Compose version v5.2.0"), "v5.2.0")
        self.assertEqual(compose_version("podman-compose version 1.6.0"), "1.6.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
