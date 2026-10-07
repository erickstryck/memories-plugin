"""The lifecycle of an already-provisioned stack: `status`, `up`, `down`, `remove`.

The tests run the four verbs (and the `runtime_for`, `check_minor` and `boot_status`
helpers) against the shared fakes (`FakeRuntime`, `FakeRunner`, `ScriptedPrompter`,
`RecordingReporter`, `FakeConfigSink`) and a temp stack directory -- never a live engine,
the network or real time. The state is written with the real `state.save`, so what the
lifecycle reads is exactly what a real install leaves behind.

The `systemctl` and `loginctl` answers are READS (is-enabled, show-user), not changes:
the `podman-restart.service` unit line is the form `systemctl --user is-enabled` prints
(measured on this machine, a unit that is enabled answers `enabled`), and the linger
line is `loginctl show-user <user> --property=Linger`, which prints `Linger=yes` or
`Linger=no`. The values are the two states the tool actually emits, not a guess.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError, catalog, compose, lifecycle, state, verify  # noqa: E402
from stack.engine import EngineInfo, Provider, ProviderInfo  # noqa: E402
from stack.runtimes import Completed  # noqa: E402
from tests.stack_fakes import (FakeConfigSink, FakeRunner, FakeRuntime,  # noqa: E402
                               RecordingReporter, ScriptedPrompter)


# -- fixtures -----------------------------------------------------------------


def make_state(**overrides) -> state.StackState:
    """A recorded install, the shape a real `qctx install` leaves in `stack.json`. The
    defaults are a healthy cpu/podman stack on the standard ports; a test overrides the
    field it is about."""
    base = dict(
        role="local", listen="127.0.0.1", platform="linux", runtime="podman",
        provider=["podman-compose"], profile="cpu", device=None, gpu_index=None,
        ports={"qdrant": 6333, "embed": 8003, "rerank": 8004},
        images={"llama": catalog.LLAMA_IMAGE, "qdrant": catalog.QDRANT_IMAGE},
        models={"bge-m3-Q4_K_M.gguf": "x", "bge-reranker-v2-m3-Q4_K_M.gguf": "y"},
        qdrant_version="1.19.2", selinux=False, phase="running",
        created_at="2026-10-06T00:00:00Z", updated_at="2026-10-06T00:00:00Z",
        project=catalog.PROJECT, schema=1)
    base.update(overrides)
    return state.StackState(**base)


def make_engine(name="podman") -> EngineInfo:
    # The fields `engine()` reports; the lifecycle never reads them, so a minimal one
    # is enough (the installer's tests carry the measured shapes).
    return EngineInfo(name, "5.7.0", "linux", "amd64", True,
                      kernel="7.0.0-34-generic", memory_bytes=16 * 2 ** 30)


def make_runtime(name="podman", provider_name="podman-compose",
                 argv=("podman", "compose"), gone=False) -> FakeRuntime:
    """A `FakeRuntime` for the lifecycle: it answers `compose` (always ok) and reports a
    fixed compose provider -- or `None` when `gone`, the provider that is no longer there."""
    if gone:
        info = ProviderInfo(None, fix="install podman-compose")
    else:
        info = ProviderInfo(Provider(tuple(argv), provider_name, "5.2.0"))
    return FakeRuntime(name, make_engine(name), info, {})


def make_deps(stack_dir: Path, *, state_obj=None, runtimes=None, status_fn=None,
              config=None, runner=None, prompter=None, reporter=None):
    """A `LifeDeps` over a temp stack directory. The state is written with the real
    `state.save` when given, and the probes default to the all-200 http, a frozen clock
    and a no-op sleep, so `wait_ready` passes in one round without touching the network."""
    if state_obj is not None:
        state.save(stack_dir, state_obj)
        # A provisioned stack has both files; render the real compose so the
        # remove/down tests see the install the way the lifecycle does.
        (stack_dir / state.COMPOSE_FILE).write_text(
            compose.dump(state.plan_of(state_obj, stack_dir)), encoding="utf-8")
    if runtimes is None:
        runtimes = [make_runtime()]
    if status_fn is None:
        status_fn = lambda url: 200
    if config is None:
        config = FakeConfigSink()
    if prompter is None:
        prompter = ScriptedPrompter([])
    if reporter is None:
        reporter = RecordingReporter()
    if runner is None:
        user = os.environ.get("USER", "unknown")
        # The default answers say the unit is off and linger is off: a `status` reports
        # the host even when the boot hooks are not on, and these tests do not assert the
        # boot line (test_boot_status... overrides this runner).
        runner = FakeRunner({
            ("systemctl", "--user", "is-enabled", "podman-restart.service"):
                Completed(0, "disabled\n"),
            ("loginctl", "show-user", user, "--property=Linger"):
                Completed(0, "Linger=no\n")})
    return lifecycle.LifeDeps(
        runtimes=runtimes, reporter=reporter, prompter=prompter, config=config,
        stack_dir=stack_dir, runner=runner, status=status_fn,
        clock=lambda: 0.0, sleep=lambda seconds: None)


def compose_args_of(runtime) -> list:
    """The `args` of each recorded `compose` call, so a test asserts what was run."""
    return [call[2] for call in runtime.calls]


# -- the tests ----------------------------------------------------------------


class _LifecycleCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.stack = Path(tmp.name) / "stack"


class TestStatus(_LifecycleCase):
    def test_status_without_a_stack_exits_0(self):
        deps = make_deps(self.stack)  # no state file
        result, code = lifecycle.status(deps)
        self.assertEqual(result, {"managed": False})
        self.assertEqual(code, 0)

    def test_status_healthy_exits_0(self):
        st = make_state()
        deps = make_deps(self.stack, state_obj=st, status_fn=lambda url: 200)
        result, code = lifecycle.status(deps)
        self.assertEqual(code, 0)
        self.assertTrue(result["managed"])
        self.assertTrue(result["healthy"])
        self.assertEqual(result["phase"], state.PHASE_RUNNING)
        urls = verify.stack_urls(st.ports)
        for service in catalog.SERVICES:
            self.assertEqual(result["services"][service]["status"], 200)
        self.assertEqual(result["services"]["qdrant"]["url"], urls["qdrant_url"])
        self.assertEqual(result["services"]["embed"]["url"], urls["api_base_url"])
        self.assertEqual(result["services"]["rerank"]["url"], urls["rerank_url"])
        self.assertFalse(result["outdated_pins"])
        self.assertIn("boot", result)

    def test_one_endpoint_down_exits_1(self):
        st = make_state()
        down_url = verify.stack_urls(st.ports)["api_base_url"]
        deps = make_deps(self.stack, state_obj=st,
                         status_fn=lambda url: None if url == down_url else 200)
        result, code = lifecycle.status(deps)
        self.assertEqual(code, 1)
        self.assertFalse(result["healthy"])
        self.assertIsNone(result["services"]["embed"]["status"])
        self.assertEqual(result["services"]["qdrant"]["status"], 200)

    def test_outdated_pins_are_listed(self):
        # the recorded images differ from the catalogue -> both roles are listed
        st = make_state(images={"llama": "old-llama:1", "qdrant": "old-qdrant:1"})
        deps = make_deps(self.stack, state_obj=st, status_fn=lambda url: 200)
        result, _ = lifecycle.status(deps)
        self.assertIn("llama", result["outdated_pins"])
        self.assertIn("qdrant", result["outdated_pins"])
        # the recorded images match the catalogue -> nothing is listed
        fresh = make_deps(self.stack / "fresh", state_obj=make_state(),
                          status_fn=lambda url: 200)
        result_fresh, _ = lifecycle.status(fresh)
        self.assertEqual(result_fresh["outdated_pins"], [])

    def test_config_points_here_is_reported_both_ways(self):
        st = make_state()
        urls = verify.stack_urls(st.ports)
        pointing = FakeConfigSink(current={
            "qdrant_url": urls["qdrant_url"], "api_base_url": urls["api_base_url"],
            "rerank_url": urls["rerank_url"]})
        deps_point = make_deps(self.stack, state_obj=st, config=pointing,
                               status_fn=lambda url: 200)
        result_point, _ = lifecycle.status(deps_point)
        self.assertTrue(result_point["config_points_here"])

        elsewhere = FakeConfigSink(current={
            "qdrant_url": "http://127.0.0.1:9999",
            "api_base_url": "http://127.0.0.1:9998/v1",
            "rerank_url": "http://127.0.0.1:9997/v1/rerank"})
        deps_else = make_deps(self.stack / "else", state_obj=st, config=elsewhere,
                              status_fn=lambda url: 200)
        result_else, _ = lifecycle.status(deps_else)
        self.assertFalse(result_else["config_points_here"])

    def test_status_on_a_corrupt_state_exits_1_with_the_fix(self):
        self.stack.mkdir(parents=True, exist_ok=True)
        (self.stack / state.STATE_FILE).write_text("not valid json {", encoding="utf-8")
        deps = make_deps(self.stack)
        result, code = lifecycle.status(deps)
        self.assertEqual(code, 1)
        self.assertTrue(result["managed"])
        self.assertIn("error", result)
        self.assertEqual(result["fix"], "qctx stack remove")


class TestUp(_LifecycleCase):
    def test_up_without_upgrade_repeats_the_recorded_images(self):
        recorded = {"llama": "custom-llama:1", "qdrant": "custom-qdrant:1"}
        st = make_state(images=recorded)
        runtime = make_runtime()
        deps = make_deps(self.stack, state_obj=st, runtimes=[runtime],
                         status_fn=lambda url: 200)
        result = lifecycle.up(deps)
        text = (self.stack / state.COMPOSE_FILE).read_text(encoding="utf-8")
        self.assertIn("custom-llama:1", text)
        self.assertIn("custom-qdrant:1", text)
        self.assertNotIn(catalog.LLAMA_IMAGE, text)
        self.assertNotIn(catalog.QDRANT_IMAGE, text)
        self.assertFalse(any("pull" in args for args in compose_args_of(runtime)))
        self.assertEqual(result.phase, state.PHASE_RUNNING)

    def test_up_upgrade_takes_the_catalogue_and_overrides_and_pulls(self):
        st = make_state(images={"llama": "old-llama:1", "qdrant": "old-qdrant:1"})
        runtime = make_runtime()
        deps = make_deps(self.stack, state_obj=st, runtimes=[runtime],
                         status_fn=lambda url: 200)
        result = lifecycle.up(deps, upgrade=True, images={"llama": "override-llama:1"})
        text = (self.stack / state.COMPOSE_FILE).read_text(encoding="utf-8")
        self.assertIn("override-llama:1", text)
        self.assertIn(catalog.QDRANT_IMAGE, text)
        self.assertNotIn("old-llama:1", text)
        self.assertTrue(any("pull" in args for args in compose_args_of(runtime)))
        self.assertEqual(result.images["llama"], "override-llama:1")
        self.assertEqual(result.images["qdrant"], catalog.QDRANT_IMAGE)


class TestMinorGuard(_LifecycleCase):
    def test_the_minor_guard_refuses_a_skip_and_allows_the_next(self):
        # 1.19 -> 1.20 is the next minor: allowed, no raise.
        lifecycle.check_minor("1.19.2", "1.20.0")
        # 1.19 -> 1.21 skips the intermediate 1.20: refused, naming it in the fix.
        with self.assertRaises(StackError) as ctx:
            lifecycle.check_minor("1.19.2", "1.21.0")
        self.assertIn("1.20", ctx.exception.fix)


class TestDown(_LifecycleCase):
    def test_down_keeps_volume_and_models_and_marks_stopped(self):
        st = make_state()
        models = self.stack / state.MODELS_DIR
        models.mkdir(parents=True)
        (models / "bge-m3-Q4_K_M.gguf").write_bytes(b"x")
        runtime = make_runtime()
        deps = make_deps(self.stack, state_obj=st, runtimes=[runtime])
        result = lifecycle.down(deps)
        downs = [args for args in compose_args_of(runtime) if args and args[0] == "down"]
        self.assertEqual(len(downs), 1)
        self.assertNotIn("-v", downs[0])
        self.assertTrue((models / "bge-m3-Q4_K_M.gguf").exists())
        self.assertEqual(result.phase, state.PHASE_STOPPED)
        self.assertEqual(state.load(self.stack).phase, state.PHASE_STOPPED)


class TestRemove(_LifecycleCase):
    def test_remove_deletes_files_and_keeps_models_and_data_by_default(self):
        st = make_state()
        models = self.stack / state.MODELS_DIR
        models.mkdir(parents=True)
        (models / "bge-m3-Q4_K_M.gguf").write_bytes(b"x")
        runtime = make_runtime()
        deps = make_deps(self.stack, state_obj=st, runtimes=[runtime])
        deleted = lifecycle.remove(deps)
        self.assertFalse((self.stack / state.COMPOSE_FILE).exists())
        self.assertFalse((self.stack / state.STATE_FILE).exists())
        self.assertIsNone(state.load(self.stack))
        self.assertTrue((models / "bge-m3-Q4_K_M.gguf").exists())
        downs = [args for args in compose_args_of(runtime) if args and args[0] == "down"]
        self.assertEqual(len(downs), 1)
        self.assertNotIn("-v", downs[0])
        self.assertEqual(len(deleted), 2)

    def test_purge_data_requires_the_project_name_or_yes(self):
        # a wrong name: nothing is deleted and no down -v runs
        st = make_state()
        runtime = make_runtime()
        prompter = ScriptedPrompter(["wrong-name"])
        deps = make_deps(self.stack, state_obj=st, runtimes=[runtime], prompter=prompter)
        deleted = lifecycle.remove(deps, purge_data=True)
        self.assertEqual(deleted, [])
        self.assertTrue((self.stack / state.STATE_FILE).exists())
        self.assertTrue((self.stack / state.COMPOSE_FILE).exists())
        self.assertNotIn("down", compose_args_of(runtime))

        # --yes answers for the user and proceeds with a down -v
        st2 = make_state()
        runtime2 = make_runtime()
        deps2 = make_deps(self.stack / "yes", state_obj=st2, runtimes=[runtime2])
        lifecycle.remove(deps2, purge_data=True, yes=True)
        downs = [args for args in compose_args_of(runtime2) if args and args[0] == "down"]
        self.assertEqual(len(downs), 1)
        self.assertIn("-v", downs[0])
        self.assertFalse((self.stack / "yes" / state.STATE_FILE).exists())

    def test_remove_never_touches_the_config_and_says_what_still_points_here(self):
        st = make_state()
        urls = verify.stack_urls(st.ports)
        config = FakeConfigSink(current={
            "qdrant_url": urls["qdrant_url"], "api_base_url": urls["api_base_url"],
            "rerank_url": urls["rerank_url"]})
        runtime = make_runtime()
        reporter = RecordingReporter()
        deps = make_deps(self.stack, state_obj=st, runtimes=[runtime],
                         config=config, reporter=reporter)
        lifecycle.remove(deps)
        self.assertEqual(config.saves, [])
        said = " ".join(text for _method, text in reporter.calls)
        self.assertIn("127.0.0.1", said)

    def test_remove_cleans_even_with_a_corrupt_state(self):
        self.stack.mkdir(parents=True, exist_ok=True)
        (self.stack / state.STATE_FILE).write_text("not json {", encoding="utf-8")
        (self.stack / state.COMPOSE_FILE).write_text("services: {}\n", encoding="utf-8")
        podman = make_runtime("podman", "podman-compose")
        docker = make_runtime("docker", "docker-compose", argv=("docker", "compose"))
        deps = make_deps(self.stack, runtimes=[podman, docker])
        deleted = lifecycle.remove(deps)
        for runtime in (podman, docker):  # every discovered runtime got a down
            downs = [args for args in compose_args_of(runtime)
                     if args and args[0] == "down"]
            self.assertEqual(len(downs), 1, f"{runtime.name} did not get a down")
        self.assertFalse((self.stack / state.STATE_FILE).exists())
        self.assertFalse((self.stack / state.COMPOSE_FILE).exists())
        self.assertEqual(len(deleted), 2)


class TestRuntimeFor(_LifecycleCase):
    def test_a_recorded_provider_that_is_gone_is_an_error_with_a_fix(self):
        st = make_state()  # runtime=podman, provider=[podman-compose]
        gone = make_runtime("podman", gone=True)  # the provider no longer answers
        with self.assertRaises(StackError) as ctx:
            lifecycle.runtime_for(st, [gone])
        self.assertTrue(ctx.exception.fix)


class TestBootStatus(_LifecycleCase):
    def test_boot_status_for_podman_reads_the_unit_and_linger(self):
        user = os.environ.get("USER", "unknown")
        runner = FakeRunner({
            ("systemctl", "--user", "is-enabled", "podman-restart.service"):
                Completed(0, "enabled\n"),
            ("loginctl", "show-user", user, "--property=Linger"):
                Completed(0, "Linger=yes\n")})
        line = lifecycle.boot_status(make_state(), runner)
        called = [call[0] for call in runner.calls]
        self.assertIn(["systemctl", "--user", "is-enabled", "podman-restart.service"],
                      called)
        self.assertIn(["loginctl", "show-user", user, "--property=Linger"], called)
        self.assertIn("enabled", line)
        self.assertIn("yes", line)


if __name__ == "__main__":
    unittest.main()
