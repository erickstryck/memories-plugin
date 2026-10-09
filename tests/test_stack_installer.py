"""The provisioning use case: the twelve steps of the spec's "O que ela faz, em
ordem", orchestrated by `stack.installer.provision`.

The tests run the whole flow against the shared fakes (`FakeRuntime`,
`FakeTransport`, `ScriptedPrompter`, `RecordingReporter`, `FakeConfigSink`), never
against a live engine or the network. The catalogue's models are patched to 1 MiB
each with the real sha256 of the bytes served, so the download step runs the real
`fetch` (hash check included) in milliseconds; the `diagnose` and `calibrate`
callables are injected at the seam `Deps` gives them, shaped like the real ones
(`core/setup.py`'s `diagnose` return value, read at 045136e; `stack.verify`'s
`calibrate` signature).

The `--list-devices` fixtures say, beside them, how they were obtained: the AMD,
Intel and `(none)` shapes were measured on this machine 2026-10-06 (global context
M7); the NVIDIA and Apple ones are examples, because this machine has neither a
NVIDIA card nor a Mac, in the same `Vulkan<n>: name (total MiB, free MiB free)`
shape the parser was written for.
"""
import hashlib
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError, catalog, compose, installer, state, verify  # noqa: E402
from stack.engine import EngineInfo, Provider, ProviderInfo  # noqa: E402
from stack.facts import Gpu, HostFacts, NvidiaFacts  # noqa: E402
from stack.runtimes import Completed  # noqa: E402
from core.setup import Check  # noqa: E402
from tests.stack_fakes import (FakeConfigSink, FakeRuntime, FakeTransport,  # noqa: E402
                               RecordingReporter, ScriptedPrompter)

MIB = 2 ** 20
GIB = 2 ** 30

# -- the --list-devices fixtures of the probe step ---------------------------

#: Measured on this machine 2026-10-06 (global context M7), with /dev/dri in the
#: container.
LIST_AMD = ("Available devices:\n"
            "  Vulkan0: AMD Radeon RX 6900 XT (RADV NAVI21) (16368 MiB, 4018 MiB free)\n")
LIST_INTEL = ("Available devices:\n"
              "  Vulkan0: Intel(R) Graphics (BMG G31) (32656 MiB, 29268 MiB free)\n")
LIST_NONE = "Available devices:\n  (none)\n"
#: No NVIDIA card on this machine: an example in the same shape the parser reads
#: (the line format is `common/arg.cpp:1141` in b11382).
LIST_NVIDIA = ("Available devices:\n"
               "  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 23000 MiB free)\n")
#: No Mac on this machine: an example; the name is the one krunkit's Virtio-GPU
#: Venus driver prints, which is what `vendor_of` reads as `apple`.
LIST_APPLE = ("Available devices:\n"
              "  Vulkan0: Apple M2 Max (Venus) (32768 MiB, 30000 MiB free)\n")
#: Two GPUs of ONE vendor in a single `--list-devices`: the shape this host
#: measured on 2026-10-06 (two Arc B70, Vulkan0 and Vulkan2, 29268/29289 MiB
#: free); the two lines and the MiB figures below are an example in that same
#: shape, with distinct free memory so a pick by free memory is assertable.
LIST_INTEL_TWO = ("Available devices:\n"
                  "  Vulkan0: Intel(R) Graphics (BMG G31) (32656 MiB, 1000 MiB free)\n"
                  "  Vulkan1: Intel(R) Graphics (BMG G31) (32656 MiB, 30000 MiB free)\n")

# -- the patched catalogue models: 1 MiB each, the REAL hash of the bytes -----
#
# `fetch` checks size AND sha256 against the catalogue, so the bytes served must
# be exactly what the (patched) catalogue pins. The real 400 MiB files are not
# available in a hermetic test, and their digests are the catalogue's, not any
# synthetic bytes'.


class _ModelShim:
    def __init__(self, role, filename, size, filler):
        self.role, self.filename, self.size = role, filename, size
        self.repo, self.revision = "gpustack", "0000"
        self.sha256 = hashlib.sha256(filler * size).hexdigest()
        self.license = "MIT"

    def url(self):
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"


_EMBED = _ModelShim("embed", "bge-m3-Q4_K_M.gguf", MIB, b"x")
_RERANK = _ModelShim("rerank", "bge-reranker-v2-m3-Q4_K_M.gguf", MIB + 1, b"y")


class UrlTransport:
    """A `Transport` that answers each URL from its own `FakeTransport`: the two
    models are different sizes and one `FakeTransport` serves exactly one file.
    `.starts` records every `(url, start)` call, so a test proves nothing was
    downloaded by finding it empty."""

    def __init__(self, transports):
        self.transports = transports
        self.starts = []

    def get(self, url, *, start=0):
        self.starts.append((url, start))
        acc, size = [], 0
        for chunk in self.transports[url].get(url, start=start):
            acc.append(chunk)
            size += len(chunk)
            if size >= 8 * MIB:
                yield b"".join(acc)
                acc, size = [], 0
        if acc:
            yield b"".join(acc)


def good_diagnose(cfg):
    """A `diagnose` answer shaped like the real one (`core/setup.py` at 045136e:
    the keys `ready`, `checks` as a list of `asdict(c)` rows, `blockers`,
    `warnings`, `detected_dim`, `memory_suggestions`), with the stack's three
    checks healthy."""
    rows = [Check("Qdrant", True, "1 collection at http://127.0.0.1:6333"),
            Check("Embedding", True, "bge-m3 returns 1024 dimensions"),
            Check("Re-rank", True, "bge-reranker-v2-m3 answers in sigmoid 0..1")]
    return {"ready": True, "checks": [asdict(c) for c in rows], "blockers": [],
            "warnings": [], "detected_dim": 1024, "memory_suggestions": []}


def failing_diagnose(cfg):
    """The same shape, with Qdrant failing: the functional check must stop the
    flow and leave the config untouched."""
    rows = [Check("Qdrant", False, "did not answer: connection refused"),
            Check("Embedding", True, "bge-m3 returns 1024 dimensions"),
            Check("Re-rank", True, "bge-reranker-v2-m3 answers in sigmoid 0..1")]
    return {"ready": False, "checks": [asdict(c) for c in rows],
            "blockers": [asdict(rows[0])], "warnings": [], "detected_dim": 1024,
            "memory_suggestions": []}


def default_calibrate(cfg, budgets, *, clock=None, memory=None):
    """A `calibrate` answer in the real contract's shape: one ok `Check` per
    host budget and an info dict. `memory` is accepted but not called: the
    `FakeRuntime` carries no `stats`, and the real `calibrate` with the real
    `stats` is covered by `tests/test_stack_verify.py` and
    `tests/test_stack_runtimes.py`."""
    checks = [Check(b.host, True, f"{b.host}: embed 0.3 s (budget {b.embed_s:.1f} s), "
                                  f"rerank 0.5 s (budget {b.rerank_s:.1f} s)")
              for b in budgets]
    info = {b.host: {"embed_s": 0.3, "rerank_s": 0.5, "memory": None} for b in budgets}
    return checks, info


def raising_calibrate(cfg, budgets, *, clock=None, memory=None):
    """A `calibrate` that fails the way the real one does when the measurement
    cannot be taken: the embedder cannot reach the server, and `core.embedding`
    raises its `EmbeddingError` (a `CoreError`). The spec says calibration NEVER
    blocks, so this must become a warning, not an aborted install."""
    from core.embedding import EmbeddingError
    raise EmbeddingError("could not reach http://127.0.0.1:8003/v1/embeddings")


def timed_calibrate(embed_s: float, rerank_s: float):
    """The REAL `verify.calibrate`, over the shared embedder/reranker fakes of
    `tests/test_stack_verify.py` and a clock scripted to the given durations, with
    the `memory` callable the installer builds passed straight through: so a test
    sees what the installer does with a measured timing AND a failing `stats`."""
    from tests.test_stack_verify import FakeClock, FakeEmbedder, FakeReranker

    def calibrate(cfg, budgets, *, clock=None, memory=None):
        return verify.calibrate(cfg, budgets, embedder=FakeEmbedder(),
                                reranker=FakeReranker(docs=verify.CALIBRATION_RERANK_DOCS),
                                clock=FakeClock([0.0, embed_s, 0.0, rerank_s]),
                                memory=memory)
    return calibrate


def facts_for(system="linux", arch="amd64", gpus=(), nodes=(), ram: int | None = 16 * GIB,
              disk=400 * GIB, nvidia=None, selinux=False, wsl=False) -> HostFacts:
    return HostFacts(system=system, arch=arch, wsl=wsl, ram_bytes=ram,
                     disk_free_bytes=disk,
                     gpus=tuple(Gpu(vendor=v, card=c) for v, c in gpus),
                     render_nodes=tuple(nodes), selinux=selinux,
                     nvidia=nvidia or NvidiaFacts())


def docker_engine(memory=16 * GIB, os_name="linux", arch="amd64",
                  version="28.3.2") -> EngineInfo:
    # The shape of `docker info --format '{{json .}}'` (ServerVersion, OSType,
    # KernelVersion, Architecture, MemTotal): no Docker engine on this machine,
    # so an example, not a measurement.
    return EngineInfo("docker", version, os_name, arch, True,
                      kernel="6.8.0-45-generic", memory_bytes=memory)


def podman_engine(memory=16 * GIB, os_name="linux", arch="amd64", version="5.7.0",
                  vm=None, kernel="7.0.0-34-generic") -> EngineInfo:
    # The fields of `podman info --format json` that `engine()` reads (global
    # context M7 and the runtimes test): version.Version, host.os, host.arch,
    # host.kernel, host.memTotal; `vm` is the macOS machine's Host.VMType.
    return EngineInfo("podman", version, os_name, arch, True, kernel=kernel,
                      vm=vm, memory_bytes=memory)


def make_runtime(name, eng, list_devices=None, fail=None, argv=("docker", "compose"),
                 pname=None, note=None) -> FakeRuntime:
    provider = Provider(tuple(argv), pname or f"{name} compose", "5.2.0")
    return FakeRuntime(name, eng, ProviderInfo(provider, note=note),
                       dict(list_devices or {}), fail)


def run_case(tmp, *, request, runtimes, facts, prompter, reporter, config,
             env_extra=None, port_free=None, status=None, diagnose=None,
             calibrate=None, clock=None, sleep=None, budgets=None):
    """Builds the `Deps` over a temp stack directory and runs `provision` with
    the small patched models. Returns the `StackState`, `None`, or the
    `StackError` that escaped, so a test asserts on whichever it got."""
    transport = UrlTransport({_EMBED.url(): FakeTransport(b"x" * _EMBED.size),
                              _RERANK.url(): FakeTransport(b"y" * _RERANK.size)})
    env = {"HOME": str(tmp)}
    env.update(env_extra or {})
    deps = installer.Deps(
        runtimes=list(runtimes), facts=facts, prompter=prompter, reporter=reporter,
        config=config, transport=transport, stack_dir=Path(tmp) / "stack",
        budgets=list(budgets or [verify.Budget("hermes", 2.0, 2.0)]), env=env,
        port_free=port_free if port_free is not None else (lambda p: True),
        status=status if status is not None else (lambda url: 200),
        diagnose=diagnose or good_diagnose,
        calibrate=calibrate or default_calibrate,
        clock=clock if clock is not None else (lambda: 0.0),
        sleep=sleep if sleep is not None else (lambda seconds: None))
    with mock.patch.multiple(catalog, MODELS=(_EMBED, _RERANK),
                             MODELS_BYTES=_EMBED.size + _RERANK.size):
        try:
            return installer.provision(request, deps)
        except StackError as exc:
            return exc


class ProvisionTestCase(unittest.TestCase):
    """The shared harness: one temp directory per test, the stack dir inside it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.stack = Path(self.tmp) / "stack"


class TestChoosePorts(ProvisionTestCase):
    def test_only_the_busy_port_moves(self):
        wanted = dict(catalog.PORTS)
        got = installer.choose_ports(wanted, lambda p: p != 6333)
        self.assertEqual(got, {"qdrant": 6333 + 10000, "embed": 8003, "rerank": 8004})

    def test_the_next_free_port_is_taken_when_plus_10000_is_busy(self):
        wanted = dict(catalog.PORTS)
        got = installer.choose_ports(wanted, lambda p: p not in (6333, 6333 + 10000))
        self.assertEqual(got["qdrant"], 6333 + 10000 + 1)
        self.assertEqual(got["embed"], 8003)
        self.assertEqual(got["rerank"], 8004)

    def test_no_service_gets_a_port_another_service_was_given(self):
        # The embed range (18003..) and the rerank range (18004..) overlap: a
        # host that holds 8003, 8004 and 18003 must not let both land on
        # 18004, because `compose up -d` would fail binding the second one.
        wanted = dict(catalog.PORTS)
        got = installer.choose_ports(wanted, lambda p: p not in (8003, 8004, 18003))
        self.assertEqual(len(set(got.values())), 3,
                         f"two services share a port: {got}")
        self.assertEqual(got["embed"], 18004)
        self.assertEqual(got["rerank"], 18005)


class TestRuntimeAndPlatform(ProvisionTestCase):
    def test_two_runtimes_question_says_what_each_serves(self):
        # macOS with both runtimes: docker serves only the cpu (the apple is
        # "Podman only"), podman with a libkrun machine serves cpu and apple.
        # Each line of the question names the profiles it serves here (the ones
        # READY or MISSING on it).
        facts = facts_for(system="macos", arch="arm64", ram=None)
        docker = make_runtime("docker", docker_engine(os_name="linux", arch="arm64"))
        podman = make_runtime("podman", podman_engine(arch="arm64", vm="libkrun"))
        prompter = ScriptedPrompter([0, 0, True])  # docker, then cpu, then proceed
        reporter = RecordingReporter()
        result = run_case(self.tmp, request=installer.Request(),
                          runtimes=[docker, podman], facts=facts, prompter=prompter,
                          reporter=reporter, config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(len(prompter.choices), 2)
        title, lines, default = prompter.choices[0]
        self.assertEqual(default, 0)  # docker is the default of the question
        self.assertIn("cpu", lines[0])
        self.assertNotIn("apple", lines[0])
        self.assertIn("apple", lines[1])
        self.assertEqual(result.runtime, "docker")

    def test_yes_picks_docker(self):
        facts = facts_for()
        docker = make_runtime("docker", docker_engine())
        podman = make_runtime("podman", podman_engine())
        prompter = ScriptedPrompter([])
        result = run_case(self.tmp, request=installer.Request(yes=True),
                          runtimes=[docker, podman], facts=facts, prompter=prompter,
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(prompter.choices, [])  # no question was asked
        self.assertEqual(result.runtime, "docker")

    def test_stack_apple_without_yes_is_not_refused_and_one_option_asks_nothing(self):
        # An explicit experimental profile is not refused: the user asked for it
        # by name. Without --yes the reduced menu's default comes from the
        # profile's own options, not the whole-menu rule (which skips the
        # experimental option and falls back to a cpu line the reduced menu does
        # not have). With ONE option there is nothing to choose: no menu at all.
        facts = facts_for(system="macos", arch="arm64", ram=None)
        podman = make_runtime("podman", podman_engine(arch="arm64", vm="libkrun"),
                              list_devices={"apple": LIST_APPLE})
        prompter = ScriptedPrompter([True])  # only the summary's proceed
        result = run_case(self.tmp, request=installer.Request(profile="apple"),
                          runtimes=[podman], facts=facts, prompter=prompter,
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(result.profile, "apple")
        self.assertEqual(result.device, "Vulkan0")
        self.assertEqual(prompter.choices, [], "one option: nothing to choose")

    def test_an_experimental_profile_with_two_devices_opens_its_menu(self):
        # The reduced menu of an explicit experimental profile must take its
        # default from the profile's own options: the whole-menu rule skips the
        # experimental option and falls back to a cpu line this menu does not
        # have (it died with "the menu has no cpu option to fall back to"). Two
        # devices are needed to reach the menu at all (one is taken without
        # asking). The two lines are an EXAMPLE in the LIST_APPLE shape; no Mac
        # here, and a libkrun machine with two GPUs is not a measured case.
        two_apple = ("Available devices:\n"
                     "  Vulkan0: Apple M2 Max (Venus) (32768 MiB, 1000 MiB free)\n"
                     "  Vulkan1: Apple M2 Max (Venus) (32768 MiB, 30000 MiB free)\n")
        facts = facts_for(system="macos", arch="arm64", ram=None)
        podman = make_runtime("podman", podman_engine(arch="arm64", vm="libkrun"),
                              list_devices={"apple": two_apple})
        prompter = ScriptedPrompter([1, True])  # take the second line, proceed
        result = run_case(self.tmp, request=installer.Request(profile="apple"),
                          runtimes=[podman], facts=facts, prompter=prompter,
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        _title, lines, default = prompter.choices[0]
        self.assertEqual(len(lines), 2)
        self.assertEqual(default, 1, "the default is the profile's most free device")
        self.assertEqual(result.device, "Vulkan1")

    def test_stack_apple_goes_to_podman_without_asking(self):
        # The apple profile is Podman-only: it goes to Podman without asking,
        # even when Docker also answers.
        facts = facts_for(system="macos", arch="arm64", ram=None)
        docker = make_runtime("docker", docker_engine(os_name="linux", arch="arm64"))
        podman = make_runtime("podman", podman_engine(arch="arm64", vm="libkrun"),
                              list_devices={"apple": LIST_APPLE})
        prompter = ScriptedPrompter([])
        result = run_case(self.tmp, request=installer.Request(profile="apple", yes=True),
                          runtimes=[docker, podman], facts=facts, prompter=prompter,
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(prompter.choices, [])  # no runtime question
        self.assertEqual(result.runtime, "podman")
        self.assertEqual(result.profile, "apple")

    def test_stack_apple_with_runtime_docker_is_refused(self):
        facts = facts_for(system="macos", arch="arm64", ram=None)
        docker = make_runtime("docker", docker_engine(os_name="linux", arch="arm64"))
        podman = make_runtime("podman", podman_engine(arch="arm64", vm="libkrun"))
        err = run_case(self.tmp,
                       request=installer.Request(profile="apple", runtime="docker"),
                       runtimes=[docker, podman], facts=facts,
                       prompter=ScriptedPrompter([]), reporter=RecordingReporter(),
                       config=FakeConfigSink())
        self.assertIsInstance(err, StackError)
        self.assertEqual(err.step, "runtime")
        self.assertIn("apple runs on Podman only", str(err))
        self.assertNotIn("only only", str(err))  # the label already carries it

    def test_wsl2_is_the_windows_platform_and_not_refused(self):
        # WSL is read from the kernel (the `wsl` fact here stands for the
        # `microsoft` in osrelease): `platform_of` classifies it as windows,
        # and step 2 ACCEPTS that -- the refusal of the phase-1 plan is gone,
        # with it the pointer at the README's manual path. With cpu READY on
        # windows (Task 2) the flow no longer dies at the profile step: the
        # install completes, still naming the platform.
        facts = facts_for(wsl=True)
        docker = make_runtime("docker", docker_engine())
        result = run_case(self.tmp, request=installer.Request(profile="cpu", yes=True),
                          runtimes=[docker], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError,
                                 "cpu is READY on windows, so the install completes")
        self.assertEqual("windows", result.platform)
        self.assertEqual("cpu", result.profile)

    def test_wsl2_with_no_runtime_aborts_at_the_runtime_step(self):
        # WSL2, no runtime answering (Docker Desktop off, or WSL integration
        # unchecked, or podman absent): the abort is step 1's -- the runtime
        # one -- not the platform refusal (which native Windows gets from the
        # CLI's one-line gate and never reaches the installer). The existing
        # refusal names the install.
        facts = facts_for(wsl=True)
        err = run_case(self.tmp, request=installer.Request(profile="cpu", yes=True),
                       runtimes=[], facts=facts, prompter=ScriptedPrompter([]),
                       reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertIsInstance(err, StackError)
        self.assertEqual(err.step, "runtime")
        self.assertTrue(err.fix, "no runtime must still name the install")
        self.assertIn("install", err.fix.lower())
        self.assertNotIn("Local models", str(err),
                         "the abort is about the runtime, not the manual path")

    def test_a_dead_podman_socket_uses_podman_compose_and_records_it(self):
        # M3: with the API socket dead, `podman compose` (which runs
        # docker-compose) fails, so the provider is the standalone
        # podman-compose, and the provider used is what stack.json records.
        facts = facts_for()
        podman = make_runtime(
            "podman", podman_engine(), argv=("podman-compose",),
            pname="podman-compose",
            note="podman compose runs docker-compose, which needs the API socket; "
                 "falling back to the standalone podman-compose")
        reporter = RecordingReporter()
        result = run_case(self.tmp,
                          request=installer.Request(profile="cpu", yes=True),
                          runtimes=[podman], facts=facts,
                          prompter=ScriptedPrompter([]), reporter=reporter,
                          config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(result.provider, ["podman-compose"])
        saved = state.load(self.stack)
        self.assertEqual(saved.provider, ["podman-compose"])
        # the fallback was said out loud, and every compose call used the
        # standalone provider
        infos = [t for m, t in reporter.calls if m == "info"]
        self.assertTrue(any("falling back" in t for t in infos))
        self.assertTrue(all(argv[0] == "podman-compose"
                            for (argv, _p, _a, _t, _s) in podman.calls))


class TestMenu(ProvisionTestCase):
    def test_an_option_the_runtime_does_not_serve_says_what_it_needs_and_asks_again(self):
        # macOS with Docker Desktop: the apple line says "Podman only"; choosing
        # it repeats what it needs and the menu comes back.
        facts = facts_for(system="macos", arch="arm64", ram=None)
        docker = make_runtime("docker", docker_engine(os_name="linux", arch="arm64"))
        prompter = ScriptedPrompter([4, 0, True])  # apple (index 4), then cpu, proceed
        reporter = RecordingReporter()
        result = run_case(self.tmp, request=installer.Request(),
                          runtimes=[docker], facts=facts, prompter=prompter,
                          reporter=reporter, config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(len(prompter.choices), 2)
        lines = prompter.choices[0][1]
        self.assertIn("Podman only", lines[4])
        # the line names what it needs without a self-contradiction: the "not X"
        # clause used to derive X from the backend's own label, rendering "runs on
        # podman here, not podman" (review round on T10).
        self.assertIn("runs on podman here", lines[4])
        self.assertNotIn("not podman", lines[4])
        warns = [t for m, t in reporter.calls if m == "warn"]
        self.assertTrue(any("podman" in w for w in warns))
        self.assertEqual(result.profile, "cpu")

    def test_explicit_profile_with_no_proved_gpu_stops_with_reason_and_fix(self):
        # The host has the AMD card (READY on the host side), but the proof
        # fails: the explicit --stack amd stops with the reason (the tail of the
        # stderr) and the fix, instead of falling back to the cpu.
        facts = facts_for(gpus=[("amd", "0000:44:00.0")], nodes=("renderD128",))
        podman = make_runtime(
            "podman", podman_engine(), argv=("podman", "compose"),
            fail={"run": Completed(1, "", "Error: failed to start container: no render node")})
        err = run_case(self.tmp, request=installer.Request(profile="amd", yes=True),
                       runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                       reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertIsInstance(err, StackError)
        self.assertEqual(err.step, "profile")
        self.assertIn("no render node", str(err))

    def test_a_host_gpu_the_container_does_not_see_is_listed_unavailable(self):
        # The host has the AMD card and the driver, but the container sees
        # `(none)`: the profile is demoted to MISSING, the warning names it, and
        # `auto` falls back to the cpu instead of picking it.
        facts = facts_for(gpus=[("amd", "0000:44:00.0")], nodes=("renderD128",))
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                              list_devices={"amd": LIST_NONE})
        reporter = RecordingReporter()
        result = run_case(self.tmp, request=installer.Request(yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=reporter, config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(result.profile, "cpu")
        warns = [t for m, t in reporter.calls if m == "warn"]
        self.assertTrue(any("(none)" in w for w in warns))

    def test_auto_picks_the_proved_gpu_with_most_free_memory(self):
        # Both the AMD and the Intel prove a device; the default is the proven
        # GPU with the most FREE memory (M5): 29268 MiB on the Intel beats
        # 4018 MiB on the AMD.
        facts = facts_for(gpus=[("amd", "0000:44:00.0"), ("intel", "0000:02:00.0")],
                          nodes=("renderD128", "renderD129"))
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                              list_devices={"amd": LIST_AMD, "intel": LIST_INTEL})
        result = run_case(self.tmp, request=installer.Request(profile="auto", yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(result.profile, "intel")
        self.assertEqual(result.device, "Vulkan0")

    def test_build_options_gives_one_deviceless_line_per_profile(self):
        # The pre-proof menu is built from the availability alone: one line per
        # profile, no device. The device rows are added later by `_prove` (one
        # per proven GPU), so a two-GPU profile still shows ONE line here.
        from stack.backends import Availability, READY
        availability = {b: Availability(READY) for b in ("cpu", "amd", "intel",
                                                          "nvidia", "apple", "dzn")}
        options = installer.build_options(availability)
        self.assertEqual([o.backend for o in options],
                         ["cpu", "amd", "intel", "nvidia", "apple", "dzn"],
                         "catalogue order, cpu first, one line each")
        self.assertTrue(all(o.device is None for o in options),
                        "no device before _prove")

    def test_auto_picks_the_second_gpu_when_it_has_more_free_memory(self):
        # ONE vendor with TWO GPUs in the same --list-devices (this host's
        # shape: two Arc B70). The default is the proven GPU with the most
        # free memory -- the second line, not the first (spec: "o padrão é a
        # GPU provada de maior memória livre").
        facts = facts_for(gpus=[("intel", "0000:02:00.0")], nodes=("renderD128",))
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                              list_devices={"intel": LIST_INTEL_TWO})
        result = run_case(self.tmp, request=installer.Request(profile="auto", yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(result.profile, "intel")
        self.assertEqual(result.device, "Vulkan1")

    def test_the_menu_lists_each_proved_gpu_and_defaults_to_the_most_free(self):
        # The menu is one line PER proven GPU (spec: "cada GPU provada"), so
        # both Intel lines appear, and the default is the one with the most
        # free memory.
        facts = facts_for(gpus=[("intel", "0000:02:00.0")], nodes=("renderD128",))
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                              list_devices={"intel": LIST_INTEL_TWO})
        # the menu is cpu (0), amd unavailable (1), intel Vulkan0 (2), intel
        # Vulkan1 (3), nvidia (4), apple (5): pick the default, the most free
        # (Vulkan1, index 3), then proceed.
        prompter = ScriptedPrompter([3, True])
        result = run_case(self.tmp, request=installer.Request(),
                          runtimes=[podman], facts=facts, prompter=prompter,
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        title, lines, default = prompter.choices[0]
        self.assertEqual(result.device, "Vulkan1")
        self.assertEqual(default,
                         max(i for i, line in enumerate(lines)
                             if line.startswith("intel: Vulkan1")))
        self.assertTrue(any(line.startswith("intel: Vulkan0") for line in lines),
                        f"the first proven GPU is missing from the menu: {lines}")
        self.assertTrue(any(line.startswith("intel: Vulkan1") for line in lines),
                        f"the second proven GPU is missing from the menu: {lines}")

    def test_an_explicit_profile_reduces_the_menu_to_its_gpus(self):
        # `--stack intel` keeps the menu, reduced to the profile's GPUs (both
        # of them, no cpu line): the pick decides, and `--yes` takes the
        # default (the most free) without asking.
        facts = facts_for(gpus=[("intel", "0000:02:00.0")], nodes=("renderD128",))
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                              list_devices={"intel": LIST_INTEL_TWO})
        prompter = ScriptedPrompter([0, True])  # the first line (Vulkan0), then proceed
        result = run_case(self.tmp, request=installer.Request(profile="intel"),
                          runtimes=[podman], facts=facts, prompter=prompter,
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        title, lines, default = prompter.choices[0]
        self.assertEqual(len(lines), 2, f"the menu is not reduced: {lines}")
        self.assertFalse(any(line.startswith("cpu") for line in lines))
        self.assertTrue(lines[default].startswith("intel: Vulkan1"),
                        "the reduced menu's default is the profile's most free GPU")
        self.assertEqual(result.device, "Vulkan0", "the pick decides")
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                              list_devices={"intel": LIST_INTEL_TWO})
        result = run_case(self.tmp, request=installer.Request(profile="intel", yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(result.device, "Vulkan1", "--yes takes the most free")

    def test_auto_never_picks_apple(self):
        # The apple is the only proven GPU and it is experimental: `auto` still
        # falls back to the cpu.
        facts = facts_for(system="macos", arch="arm64", ram=None)
        podman = make_runtime("podman", podman_engine(arch="arm64", vm="libkrun"),
                              list_devices={"apple": LIST_APPLE})
        result = run_case(self.tmp, request=installer.Request(profile="auto", yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(result.profile, "cpu")

    def test_auto_without_a_proved_gpu_falls_back_to_cpu(self):
        facts = facts_for(gpus=[("amd", "0000:44:00.0")], nodes=("renderD128",))
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                              list_devices={"amd": LIST_NONE})
        result = run_case(self.tmp, request=installer.Request(profile="auto", yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(result.profile, "cpu")

    def test_declining_the_summary_downloads_nothing(self):
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        prompter = ScriptedPrompter([0, False])  # the cpu in the menu, then decline
        reporter = RecordingReporter()
        config = FakeConfigSink()
        outcome = run_case(self.tmp, request=installer.Request(profile="cpu", yes=False),
                           runtimes=[podman], facts=facts, prompter=prompter,
                           reporter=reporter, config=config)
        self.assertIsNone(outcome)
        self.assertEqual(config.saves, [])
        self.assertIsNone(state.load(self.stack))
        self.assertFalse((self.stack / "compose.yaml").exists())

    def test_cpu_plan_pins_no_device_even_when_the_engine_injects_gpus(self):
        # M1: the engine here is configured to inject a GPU into EVERY
        # container (the FakeRuntime answers the probe with the three GPUs of
        # this machine, whatever the file), so the cpu plan must carry `-dev
        # none` in the rendered compose, or the server would take the GPU.
        facts = facts_for(gpus=[("amd", "0000:44:00.0"), ("intel", "0000:02:00.0")],
                          nodes=("renderD128", "renderD129"))
        three_gpus = ("Available devices:\n"
                      "  Vulkan0: Intel(R) Graphics (BMG G31) (32656 MiB, 29268 MiB free)\n"
                      "  Vulkan1: AMD Radeon RX 6900 XT (RADV NAVI21) (16368 MiB, 4018 MiB free)\n"
                      "  Vulkan2: Intel(R) Graphics (BMG G31) (32656 MiB, 29289 MiB free)\n")
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                              list_devices={"cpu": three_gpus, "amd": LIST_AMD,
                                            "intel": LIST_INTEL})
        result = run_case(self.tmp, request=installer.Request(profile="cpu", yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        text = (self.stack / "compose.yaml").read_text()
        self.assertIn('- "-dev"\n      - "none"', text)


class TestProvision(ProvisionTestCase):
    def test_happy_path_cpu_writes_compose_state_and_config_in_order(self):
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        prompter = ScriptedPrompter([])
        reporter = RecordingReporter()
        config = FakeConfigSink()
        result = run_case(self.tmp, request=installer.Request(yes=True),
                          runtimes=[podman], facts=facts, prompter=prompter,
                          reporter=reporter, config=config)
        self.assertNotIsInstance(result, StackError)
        # stack.json: the verified stack, in the running phase
        self.assertEqual(result.phase, "running")
        self.assertEqual(result.profile, "cpu")
        self.assertEqual(result.ports, dict(catalog.PORTS))
        saved = state.load(self.stack)
        self.assertEqual(saved.phase, "running")
        # compose.yaml: exactly the render of the chosen plan
        expected = compose.Plan(
            platform="linux", runtime="podman", backend="cpu", device=None,
            gpu_index=None, ports=dict(catalog.PORTS), stack_dir=self.stack,
            images=catalog.resolve_images({}, {"HOME": self.tmp}), selinux=False)
        self.assertEqual((self.stack / "compose.yaml").read_text(), compose.dump(expected))
        # the config: written ONCE, with the stack's URLs, and only AFTER the
        # functional check was reported (the Qdrant line precedes the diff)
        self.assertEqual(len(config.saves), 1)
        self.assertEqual(config.file["api_base_url"], "http://127.0.0.1:8003/v1")
        self.assertEqual(config.file["embed_url"], "")
        self.assertEqual(config.file["rerank_url"], "http://127.0.0.1:8004/v1/rerank")
        self.assertEqual(config.file["qdrant_url"], "http://127.0.0.1:6333")
        self.assertEqual(config.file["vector_size"], 1024)
        calls = [t for m, t in reporter.calls]
        i_functional = next(i for i, t in enumerate(calls) if t.startswith("Qdrant:"))
        i_config = next(i for i, t in enumerate(calls) if t.startswith("config:"))
        self.assertLess(i_functional, i_config)
        # the download happened (the real fetch, hash verified) and the probe
        # files are gone
        self.assertTrue((self.stack / "models" / "bge-m3-Q4_K_M.gguf").exists())
        self.assertTrue((self.stack / "models" / "bge-reranker-v2-m3-Q4_K_M.gguf").exists())
        self.assertFalse((self.stack / "probe").exists())
        # the reboot hint was printed (podman on linux: the user service)
        infos = [t for m, t in reporter.calls if m == "info"]
        self.assertTrue(any("podman-restart.service" in t for t in infos))

    def test_happy_path_each_gpu_profile(self):
        cases = {
            "amd": (facts_for(gpus=[("amd", "0000:44:00.0")], nodes=("renderD128",)),
                    make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                                 list_devices={"amd": LIST_AMD})),
            "intel": (facts_for(gpus=[("intel", "0000:02:00.0")], nodes=("renderD129",)),
                      make_runtime("podman", podman_engine(), argv=("podman", "compose"),
                                   list_devices={"intel": LIST_INTEL})),
            "nvidia": (facts_for(
                           gpus=[("nvidia", "0000:01:00.0")],
                           nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",),
                                              icd=True, docker_hook=True)),
                       make_runtime("docker", docker_engine(),
                                    list_devices={"nvidia-0": LIST_NVIDIA})),
            "apple": (facts_for(system="macos", arch="arm64", ram=None),
                      make_runtime("podman", podman_engine(arch="arm64", vm="libkrun"),
                                   list_devices={"apple": LIST_APPLE})),
        }
        for profile, (facts, runtime) in cases.items():
            with self.subTest(profile=profile):
                tmp = tempfile.TemporaryDirectory()
                self.addCleanup(tmp.cleanup)
                stack = Path(tmp.name) / "stack"
                reporter = RecordingReporter()
                result = run_case(tmp.name,
                                  request=installer.Request(profile=profile, yes=True),
                                  runtimes=[runtime], facts=facts,
                                  prompter=ScriptedPrompter([]), reporter=reporter,
                                  config=FakeConfigSink())
                self.assertNotIsInstance(result, StackError)
                self.assertEqual(result.profile, profile)
                self.assertEqual(result.device, "Vulkan0")
                if profile == "nvidia":
                    self.assertEqual(result.gpu_index, 0)
                    text = (stack / "compose.yaml").read_text()
                    self.assertIn("device_ids", text)
                else:
                    self.assertIsNone(result.gpu_index)
                # the models were downloaded and the probe files are gone
                self.assertTrue((stack / "models" / "bge-m3-Q4_K_M.gguf").exists())
                self.assertFalse((stack / "probe").exists())

    def test_env_trap_names_the_variable_and_its_value(self):
        # A legacy QDRANT_URL in the shell rc beats the file (ENV_ALIASES): after
        # the save, the installer names the variable, its value, and says to
        # remove the export.
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        reporter = RecordingReporter()
        result = run_case(self.tmp, request=installer.Request(yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=reporter, config=FakeConfigSink(),
                          env_extra={"QDRANT_URL": "http://elsewhere:6333"})
        self.assertNotIsInstance(result, StackError)
        warns = [t for m, t in reporter.calls if m == "warn"]
        trap = [w for w in warns if "QDRANT_URL" in w]
        self.assertEqual(len(trap), 1)
        self.assertIn("http://elsewhere:6333", trap[0])
        self.assertIn("remove its export from your shell rc", trap[0])

    def test_config_is_written_only_after_verification(self):
        # The functional check fails: the flow stops, the config is left
        # untouched, and the state stays in the `compose` phase. The models
        # were already downloaded (the download comes before the verification).
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        config = FakeConfigSink()
        err = run_case(self.tmp, request=installer.Request(yes=True),
                       runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                       reporter=RecordingReporter(), config=config,
                       diagnose=failing_diagnose)
        self.assertIsInstance(err, StackError)
        self.assertEqual(err.step, "verify")
        self.assertEqual(config.saves, [])
        self.assertEqual(state.load(self.stack).phase, "compose")
        self.assertTrue((self.stack / "models" / "bge-m3-Q4_K_M.gguf").exists())

    def test_a_failed_calibration_is_a_warning_not_an_aborted_install(self):
        # Calibration NEVER blocks (spec): the functional check already
        # passed, so the stack works; a measurement that failed (the
        # runtime's `stats` on a rootless host without cgroups v2, the
        # embedder refusing the 6000-char text) must come out as a warning,
        # not an aborted install. An abort would leave the config unwritten
        # and the state in `compose`, so every re-run re-provisioned to the
        # same point, forever.
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        config = FakeConfigSink()
        reporter = RecordingReporter()
        result = run_case(self.tmp, request=installer.Request(yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=reporter, config=config,
                          calibrate=raising_calibrate)
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(len(config.saves), 1, "the config is written")
        self.assertEqual(result.phase, "running")
        warns = [text for method, text in reporter.calls if method == "warn"]
        self.assertTrue(any("calibration" in text for text in warns),
                        f"the failure must come out as a warning: {warns}")

    def test_a_failed_stats_keeps_the_measured_timing(self):
        # Only the memory reading fails (the runtime's `stats`, e.g. rootless
        # without cgroups v2): the timing was already measured and must still be
        # reported -- here a 9 s embed against hermes' 2 s budget -- and the
        # warning names only what was not read. A blanket "speed and memory were
        # not checked" would be false and would hide the over-budget warning.
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        podman.stats_answer = StackError("podman stats failed: cgroups v1", step="runtime")
        reporter = RecordingReporter()
        config = FakeConfigSink()
        result = run_case(self.tmp, request=installer.Request(yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=reporter, config=config,
                          calibrate=timed_calibrate(embed_s=9.0, rerank_s=0.5))
        self.assertNotIsInstance(result, StackError)
        said = [text for method, text in reporter.calls if method == "warn"]
        self.assertTrue(any("embed 9.0 s over" in t for t in said),
                        f"the measured timing must survive a failed stats: {said}")
        self.assertTrue(any("memory" in t and "cgroups v1" in t for t in said),
                        f"the warning names what was not read: {said}")
        self.assertFalse(any("speed and memory were not checked" in t for t in said))
        self.assertEqual(len(config.saves), 1)

    def test_a_programming_error_in_calibration_is_not_swallowed(self):
        # "Never blocks" is about a measurement that fails. A bug (here a
        # TypeError) must surface, not be reported as a failed measurement.
        def buggy(cfg, budgets, *, clock=None, memory=None):
            raise TypeError("unsupported operand type(s)")
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        with self.assertRaises(TypeError):
            run_case(self.tmp, request=installer.Request(yes=True), runtimes=[podman],
                     facts=facts, prompter=ScriptedPrompter([]),
                     reporter=RecordingReporter(), config=FakeConfigSink(),
                     calibrate=buggy)

    def test_a_failed_functional_check_points_at_re_running_the_install(self):
        # The functional check runs BEFORE the config is written. When it fails,
        # the stack is up (readiness passed) but not functional, the config is
        # untouched, and the state is still `compose`. The fix must point at the
        # command that RE-VERIFIES and (only if it passes) writes the config --
        # `qctx install`, which resumes. `qctx stack up` never re-verifies or
        # writes the config, so "run qctx stack up again" was a fix the command
        # could not carry out: it would start an already-running stack and the
        # config would stay unwritten, forever.
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        config = FakeConfigSink()
        err = run_case(self.tmp, request=installer.Request(yes=True),
                       runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                       reporter=RecordingReporter(), config=config,
                       diagnose=failing_diagnose)
        self.assertIsInstance(err, StackError)
        self.assertEqual(err.step, "verify")
        self.assertEqual(config.saves, [])
        self.assertIn("qctx install", err.fix,
                      f"the fix must name the command that re-verifies: {err.fix!r}")
        self.assertNotIn("stack up again", err.fix,
                         f"`stack up` never re-verifies: {err.fix!r}")

    def test_the_ram_warning_carries_the_measured_stack_figures(self):
        # The two llama-servers measured ~7.3 GiB together in the stack's own
        # configuration (embed ~2.3 GiB, rerank ~5.0 GiB), not the ~4.6 GiB of the
        # opening check. A claim must be true: the warning carries a figure that
        # is real, and the threshold sits at or above the measured need (a 6.5
        # GiB machine cannot hold a 7.3 GiB stack, so it must be warned). The
        # doc already carries these figures (1b35a87); the code caught up here.
        facts = facts_for(ram=5 * GIB)
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        reporter = RecordingReporter()
        result = run_case(self.tmp, request=installer.Request(profile="cpu", yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=reporter, config=FakeConfigSink())
        self.assertNotIsInstance(result, StackError)
        warns = [text for method, text in reporter.calls
                 if method == "warn" and "RAM" in text]
        self.assertEqual(len(warns), 1, "a 5 GiB machine must get the RAM warning")
        self.assertNotIn("4.6", warns[0], "the opening-check figure is stale")
        self.assertIn("7.3", warns[0], "the measured stack figure")
        # 7 GiB cannot hold the measured ~7.3 GiB: it must warn too (a 6 GiB
        # threshold would stay silent here)
        reporter7 = RecordingReporter()
        run_case(self.tmp, request=installer.Request(profile="cpu", yes=True),
                 runtimes=[make_runtime("podman", podman_engine(), argv=("podman", "compose"))],
                 facts=facts_for(ram=7 * GIB), prompter=ScriptedPrompter([]),
                 reporter=reporter7, config=FakeConfigSink())
        self.assertTrue(any(method == "warn" and "RAM" in text
                            for method, text in reporter7.calls),
                        "a 7 GiB machine must get the RAM warning")

    def test_no_runtime_at_all_names_a_fix(self):
        # spec: every refusal carries the correction. When NO runtime binary
        # answers (the common "nothing installed" case), the fallback fix is
        # empty, so the error said `step: runtime: no container runtime with a
        # compose provider answers` with no way forward. The empty-runtime case
        # must name the install, the way a present-but-providerless runtime does.
        facts = facts_for()
        err = run_case(self.tmp, request=installer.Request(profile="cpu"),
                       runtimes=[], facts=facts, prompter=ScriptedPrompter([]),
                       reporter=RecordingReporter(), config=FakeConfigSink())
        self.assertIsInstance(err, StackError)
        self.assertEqual(err.step, "runtime")
        self.assertTrue(err.fix, "no runtime must still name the install")
        self.assertIn("install", err.fix.lower())

    def test_a_hanging_logs_does_not_mask_a_failed_install_start(self):
        # The installer's failed start shows the log tail of the services that
        # did not come up. A `compose logs` that itself RAISES -- the provider
        # hangs and the 60s bound trips, SubprocessRunner raising
        # StackError(step="runtime") -- must not mask the readiness error: the
        # tail degrades to "(no log)" and the error keeps step="up". (The
        # existing test covers the non-zero-exit path, where the provider's
        # output IS the tail.)
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))

        def compose_raises_logs(provider, project, file, *args, timeout, stream=False):
            if args and args[0] == "logs":
                raise StackError("timed out after 60.0s: compose logs", step="runtime")
            return Completed(0)
        podman.compose = compose_raises_logs
        config = FakeConfigSink()
        ticks = iter([0.0, 601.0])
        err = run_case(self.tmp, request=installer.Request(profile="cpu", yes=True),
                       runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                       reporter=RecordingReporter(), config=config,
                       status=lambda url: None,
                       clock=lambda: next(ticks, 601.0))
        self.assertIsInstance(err, StackError)
        self.assertEqual(err.step, "up",
                         "the readiness error must survive a failing logs fetch")
        self.assertNotIn("timed out", str(err))

    def test_the_log_tail_survives_a_hanging_provider_lookup(self):
        # The tail asks the runtime for its provider first; on podman that runs
        # `podman compose version`, which can hang and raise like `logs` itself.
        # That raise must degrade the tail to "(no log)" too, not escape.
        from types import SimpleNamespace

        class HungLookup:
            def compose_provider(self):
                raise StackError("timed out after 30.0s: podman compose version",
                                 step="runtime")
        ctx = SimpleNamespace(runtime=HungLookup(), request=SimpleNamespace(project="p"))
        self.assertEqual(installer._log_tail(ctx, Path("/x/compose.yaml"), "embed"),
                         "(no log)")

    def test_replacing_a_non_empty_value_asks_first(self):
        # The file already points elsewhere (a non-empty qdrant_url): the diff
        # is shown, and the replacement ASKS before it saves.
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        prompter = ScriptedPrompter([0, True, True])  # cpu, proceed, replace
        config = FakeConfigSink(current={"qdrant_url": "http://old:6333"})
        result = run_case(self.tmp, request=installer.Request(),
                          runtimes=[podman], facts=facts, prompter=prompter,
                          reporter=RecordingReporter(), config=config)
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(len(config.saves), 1)
        self.assertEqual(len(prompter.confirms), 2)
        self.assertIn("proceed", prompter.confirms[0])
        self.assertIn("replace", prompter.confirms[1])
        self.assertEqual(config.file["qdrant_url"], "http://127.0.0.1:6333")

    def test_an_existing_embed_url_is_cleared_by_the_patch(self):
        # A stale embed_url would keep sending the embeddings to the old address
        # (it beats api_base_url in resolved_embed_url): the patch clears it.
        facts = facts_for()
        podman = make_runtime("podman", podman_engine(), argv=("podman", "compose"))
        config = FakeConfigSink(current={"embed_url": "http://stale:8003/v1/embeddings"})
        result = run_case(self.tmp, request=installer.Request(yes=True),
                          runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                          reporter=RecordingReporter(), config=config)
        self.assertNotIsInstance(result, StackError)
        self.assertEqual(config.saves[0]["embed_url"], "")
        self.assertEqual(config.file["embed_url"], "")

    def test_a_failed_up_shows_the_service_log_tail(self):
        # `compose up -d` fails: the log tail of every service that is not
        # ready is shown, the state stays in the `compose` phase, and nothing
        # is written to the config.
        facts = facts_for()
        podman = make_runtime(
            "podman", podman_engine(), argv=("podman", "compose"),
            fail={"up": Completed(1, "", "Error: failed to create container")})
        config = FakeConfigSink()
        err = run_case(self.tmp, request=installer.Request(profile="cpu", yes=True),
                       runtimes=[podman], facts=facts, prompter=ScriptedPrompter([]),
                       reporter=RecordingReporter(), config=config,
                       status=lambda url: None)
        self.assertIsInstance(err, StackError)
        self.assertEqual(err.step, "up")
        log_calls = [list(args) for (argv, project, args, timeout, stream) in podman.calls
                     if args and args[0] == "logs"]
        self.assertEqual(log_calls,
                         [["logs", "--tail", "50", "qdrant"],
                          ["logs", "--tail", "50", "embed"],
                          ["logs", "--tail", "50", "rerank"]])
        self.assertIn("qctx install", err.fix,
                      "a failed start leaves the state in `compose` and the config "
                      "unwritten; only the install resumes, verifies and writes it "
                      "(`qctx stack up` would mark it running and skip both)")
        self.assertNotIn("stack up", err.fix)
        self.assertEqual(config.saves, [])
        self.assertEqual(state.load(self.stack).phase, "compose")


if __name__ == "__main__":
    unittest.main()
