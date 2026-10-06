"""Docker and Podman behind one runtime contract: the engine, the compose provider, the stats.

The real command outputs below were measured on this machine on 2026-10-06 (spec, "Detecção"):
a `podman info --format json` reduced to the fields that are read, a `podman compose version`
whose docker-compose banner lands in stderr, and a `docker info --format json`. What is pinned
here is behaviour, not the bytes: an engine that reports a DIFFERENT but well-formed output
still passes, and the M3 lie (`remoteSocket.exists: true` while the socket file is gone) is
pinned by `socket_alive`, which is the only verdict the code trusts.
"""
import json
import socket
import sys
import tempfile
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
        "kernel": "6.14.0-37-generic",
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

#: The real `podman-compose version` here.
PODMAN_COMPOSE_VERSION_STANDALONE = Completed(
    0, "podman-compose version 1.6.0\npodman version 5.7.0")

#: A well-formed `docker info --format json` (example, per the brief).
DOCKER_INFO = {
    "ServerVersion": "28.3.2",
    "OperatingSystem": "Ubuntu 24.04.3 LTS",
    "Architecture": "x86_64",
    "SecurityOptions": ["name=seccomp,profile=default", "name=rootless"],
}
DOCKER_INFO_ROOTFUL = {
    "ServerVersion": "28.3.2",
    "OperatingSystem": "Ubuntu 24.04.3 LTS",
    "Architecture": "x86_64",
    "SecurityOptions": ["name=seccomp,profile=default"],
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
        self.assertEqual(engine.kernel, "6.14.0-37-generic")
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
        docker = Docker(FakeRunner({("docker", "info", "--format", "json"):
                                    Completed(0, json.dumps(DOCKER_INFO))}),
                        which={"docker": "/usr/bin/docker"})
        engine = docker.engine()

        self.assertEqual(engine.name, "docker")
        self.assertEqual(engine.version, "28.3.2")
        self.assertEqual(engine.os, "Ubuntu 24.04.3 LTS")
        self.assertEqual(engine.arch, "amd64")  # x86_64 normalized
        self.assertTrue(engine.rootless)

        rootful = Docker(FakeRunner({("docker", "info", "--format", "json"):
                                     Completed(0, json.dumps(DOCKER_INFO_ROOTFUL))}),
                         which={"docker": "/usr/bin/docker"})
        self.assertFalse(rootful.engine().rootless)

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

        docker = Docker(FakeRunner({("docker", "stats", "--no-stream", "--format", "json"):
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

    def test_the_runner_captures_when_not_streaming(self):
        completed = SubprocessRunner().run(["echo", "hi"], timeout=5.0)

        self.assertTrue(completed.ok)
        self.assertEqual(completed.stdout.strip(), "hi")


class TestDiscover(unittest.TestCase):
    def test_discover_skips_a_runtime_whose_engine_does_not_answer(self):
        # docker: binary present, `docker info` does not answer -> skipped.
        # podman: binary present, `podman info` answers -> kept.
        responses = {
            ("docker", "info", "--format", "json"): Completed(1, "", "permission denied"),
            ("podman", "info", "--format", "json"): Completed(0, json.dumps(_info(True))),
        }
        runtimes_found = discover(FakeRunner(responses),
                                  which={"docker": "/usr/bin/docker",
                                         "podman": "/usr/bin/podman"})

        self.assertEqual([r.name for r in runtimes_found], ["podman"])

    def test_discover_lists_docker_before_podman(self):
        responses = {
            ("docker", "info", "--format", "json"): Completed(0, json.dumps(DOCKER_INFO)),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
