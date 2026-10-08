"""Opt-in end-to-end integration against a REAL container runtime (Task 14).

The unit suite never touches a runtime: every engine, port, probe and clock is
injected. This file is the one place the whole flow runs against the machine's
actual compose provider (Podman here: `podman compose` resolves to the standalone
`podman-compose`, because the podman API socket is not connectable on this box),
so it is SKIPPED unless QCTX_STACK_IT=1 is set. It is meant to be run BY the
controller on this machine, in the order the plan's Task 14 fixes, and to leave
the machine as it was found (T15 asserts that).

Isolation: it provisions under a THROWAWAY project (memories-plugin-it) on
ports that collide with nothing (46333/48003/48004; this box's production is on
8003/8004), its stack directory and its config are temp files, and the two
models are SEEDED by hardlink/copy from QCTX_STACK_IT_MODELS (verified against
the catalogue pin) so the 836 MiB download never happens. The pinned images are
addressed by digest, so a machine that already has them (this one does) does not
download them either.

The two legs (Step 1 and 2 of the plan):
  test_cpu_profile_end_to_end   profile cpu: provision -> running; status 0;
                                down -> status 1; up -> 0; remove(purge) -> no
                                container, no volume of the it project.
  test_gpu_profile_end_to_end   QCTX_STACK_IT_GPU=amd|intel|nvidia: the recorded
                                profile is the one asked, the device starts with
                                Vulkan, and the rerank measured in calibration is
                                FASTER than the cpu rerank measured earlier in
                                the SAME run.

Run it (from the repo root, this machine):
  QCTX_STACK_IT=1 QCTX_STACK_IT_RUNTIME=podman \
      QCTX_STACK_IT_MODELS=<dir holding the two pinned Q4_K_M ggufs> \
      TMPDIR=/tmp python3 -m unittest tests.test_stack_integration -v
"""
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

IT = os.environ.get("QCTX_STACK_IT", "") == "1"

PROJECT = "memories-plugin-it"
#: The default it ports; QCTX_STACK_IT_PORTS (comma list) overrides them. They
#: must collide with nothing the machine runs (this box's production is on
#: 8003/8004, so these are clear).
DEFAULT_PORTS = {"qdrant": 46333, "embed": 48003, "rerank": 48004}
MODELS_DIR = os.environ.get("QCTX_STACK_IT_MODELS", "")
#: The one budget the calibration runs under: generous, so an over-budget warning
#: (which never raises) cannot mask a real failure, and the host name is what the
#: measurement dict is keyed by.
_BUDGET_HOST = "it"


def _it_ports() -> dict:
    raw = os.environ.get("QCTX_STACK_IT_PORTS", "")
    if not raw.strip():
        return dict(DEFAULT_PORTS)
    wanted = [int(p) for p in raw.split(",") if p.strip()]
    if len(wanted) != 3:
        raise AssertionError(f"QCTX_STACK_IT_PORTS wants 3 ports, got {wanted!r}")
    return {"qdrant": wanted[0], "embed": wanted[1], "rerank": wanted[2]}


def _it_runtime() -> str:
    rt = os.environ.get("QCTX_STACK_IT_RUNTIME", "podman").strip().lower()
    if rt not in ("docker", "podman"):
        raise AssertionError(f"QCTX_STACK_IT_RUNTIME must be docker or podman, got {rt!r}")
    return rt


def _it_gpu() -> "str | None":
    gpu = os.environ.get("QCTX_STACK_IT_GPU", "").strip().lower()
    if gpu and gpu not in ("amd", "intel", "nvidia"):
        raise AssertionError(
            f"QCTX_STACK_IT_GPU must be amd, intel or nvidia, got {gpu!r}")
    return gpu or None


def _seed_models(dest: Path) -> None:
    """Copy the two pinned models into the throwaway stack's models directory by
    hardlink (same filesystem) or copy, so the fetch step finds them present and
    never downloads the 836 MiB. Each seed is verified against the catalogue pin,
    so a wrong file is refused before anything runs."""
    import stack.catalog as catalog
    import stack.state as state
    dest = dest / state.MODELS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    for model in catalog.MODELS:
        target = dest / model.filename
        if target.exists():
            continue
        source = None
        if MODELS_DIR:
            candidate = Path(MODELS_DIR) / model.filename
            if candidate.exists():
                source = candidate
        if source is None:
            # The last place a finished install on this machine left them.
            source = Path.home() / ".hermes" / "cache" / "scratch" \
                / "stack-work" / "models" / model.filename
            source = source if source.exists() else None
        if source is None:
            raise AssertionError(
                f"no local copy of the pinned model {model.filename}: set "
                "QCTX_STACK_IT_MODELS to a directory holding it (the models are "
                "seeded, not downloaded)")
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        blob = target.read_bytes()
        if hashlib.sha256(blob).hexdigest() != model.sha256 or len(blob) != model.size:
            target.unlink(missing_ok=True)
            raise AssertionError(f"seeded {model.filename} is not the pinned file")


class _FileSink:
    """A ConfigSink over the temp config the integration writes: the installer's
    `provision` calls only `current_file` and `save` on it (it never calls
    `effective`), and the lifecycle verbs read it back the same way."""

    def __init__(self, path: Path):
        self.path = path

    def current_file(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def save(self, patch: dict) -> None:
        data = self.current_file()
        data.update(patch)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def effective(self):
        import core.config as core_config
        if self.path.exists():
            return core_config.load(self.path, env={})
        return core_config.load(None, env={})


def _config_path(workdir: Path) -> Path:
    return workdir / "it-config.json"


def _runtimes():
    import stack.facts as facts
    import stack.process as process
    import stack.runtimes as runtimes_mod
    import platform
    return runtimes_mod.discover(process.SubprocessRunner(),
                                 host_system=facts.normalize_system(
                                     platform.system()))


def _provision(profile: str, workdir: Path) -> None:
    """Provision a real stack under the throwaway project and drive it to
    running. `profile` is the catalogue option (cpu, amd, intel, nvidia)."""
    import stack.cli as stack_cli
    import stack.fetch as fetch
    import stack.facts as facts
    import stack.installer as installer
    import stack.state as state
    import stack.verify as verify

    stack_dir = workdir / "stack"
    stack_dir.mkdir(parents=True, exist_ok=True)
    _seed_models(stack_dir)

    host = facts.collect(facts.Probe(), stack_dir)
    deps = installer.Deps(
        runtimes=_runtimes(), facts=host, prompter=stack_cli.TerminalPrompter(),
        reporter=stack_cli.TerminalReporter(), config=_FileSink(_config_path(workdir)),
        transport=fetch.UrllibTransport(), stack_dir=stack_dir,
        budgets=[verify.Budget(_BUDGET_HOST, 60.0, 120.0)], env=dict(os.environ),
        port_free=facts.port_free)
    request = installer.Request(profile=profile, runtime=_it_runtime(), yes=True,
                                ports=_it_ports(), project=PROJECT)
    with contextlib.redirect_stdout(sys.stderr):
        installer.provision(request, deps)
    saved = state.load(stack_dir)
    if saved is None or saved.phase != state.PHASE_RUNNING:
        raise AssertionError(f"provision did not reach running (phase="
                             f"{getattr(saved, 'phase', None)!r})")


def _life(workdir: Path):
    import stack.cli as stack_cli
    import stack.lifecycle as lifecycle
    import stack.process as process
    import stack.state as state
    return lifecycle.LifeDeps(
        runtimes=_runtimes(), reporter=stack_cli.TerminalReporter(),
        prompter=stack_cli.TerminalPrompter(), config=_FileSink(_config_path(workdir)),
        stack_dir=workdir / "stack", runner=process.SubprocessRunner())


def _remove(workdir: Path) -> None:
    import stack.lifecycle as lifecycle
    with contextlib.redirect_stdout(sys.stderr):
        lifecycle.remove(_life(workdir), purge_models=True, purge_data=True, yes=True)


def _status_code(workdir: Path) -> int:
    import stack.lifecycle as lifecycle
    _section, code = lifecycle.status(_life(workdir))
    return code


def _measure_rerank_s(workdir: Path) -> float:
    """Time the warm rerank of the running stack once, the same way `calibrate`
    does. It reads the URLs the stack actually WROTE (the temp config), not the
    default ports: `choose_ports` moves a busy port to the first free one, so the
    stack can legitimately run on a different port than the one it was asked for,
    and a measurement aimed at the default port would hit nothing."""
    import stack.verify as verify
    cfg = _FileSink(_config_path(workdir)).effective()
    _checks, info = verify.calibrate(cfg, [verify.Budget(_BUDGET_HOST, 60.0, 120.0)])
    return info[_BUDGET_HOST]["rerank_s"]


def _engine_containers() -> list:
    import subprocess
    out = subprocess.run(["podman", "ps", "-a", "--format", "{{.Names}}"],
                         capture_output=True, text=True, timeout=60)
    return [n for n in out.stdout.split() if n]


def _engine_volumes() -> list:
    import subprocess
    out = subprocess.run(["podman", "volume", "ls", "--format", "{{.Name}}"],
                         capture_output=True, text=True, timeout=60)
    return [n for n in out.stdout.split() if n]


def _gpu_available(gpu: str) -> bool:
    import stack.facts as facts
    host = facts.collect(facts.Probe(), Path(tempfile.gettempdir()))
    return any(g.vendor == gpu for g in host.gpus)


@unittest.skipUnless(IT, "opt-in integration: set QCTX_STACK_IT=1 to run it")
class TheRealRuntime(unittest.TestCase):
    """The whole flow against the machine's real compose provider, on throwaway
    ports and a throwaway project, leaving the machine as it was found."""

    def test_cpu_profile_end_to_end(self):
        with tempfile.TemporaryDirectory(prefix="stack-it-cpu-") as raw:
            workdir = Path(raw)
            _provision("cpu", workdir)
            try:
                self.assertEqual(_status_code(workdir), 0,
                                 "a just-provisioned cpu stack must be healthy")
                import stack.lifecycle as lifecycle
                with contextlib.redirect_stdout(sys.stderr):
                    lifecycle.down(_life(workdir))
                self.assertEqual(_status_code(workdir), 1,
                                 "a stopped stack must report unhealthy")
                with contextlib.redirect_stdout(sys.stderr):
                    lifecycle.up(_life(workdir))
                self.assertEqual(_status_code(workdir), 0,
                                 "the stack must be healthy again after up")
                data = json.loads(_config_path(workdir).read_text())
                ports = _it_ports()
                self.assertEqual(data["api_base_url"],
                                 f"http://127.0.0.1:{ports['embed']}/v1")
                self.assertEqual(data["rerank_url"],
                                 f"http://127.0.0.1:{ports['rerank']}/v1/rerank")
                self.assertEqual(data["qdrant_url"],
                                 f"http://127.0.0.1:{ports['qdrant']}")
                self.assertEqual(data.get("embed_url"), "")
                self.assertEqual(data.get("vector_size"), 1024,
                                 "bge-m3 embeds to 1024 dimensions")
            finally:
                _remove(workdir)
            self.assertEqual([n for n in _engine_containers()
                              if n.startswith(PROJECT)], [],
                             "remove must leave no container of the it project")
            self.assertNotIn(f"{PROJECT}_memories-plugin-qdrant", _engine_volumes(),
                             "remove --purge-data must delete the it volume")

    def test_gpu_profile_end_to_end(self):
        gpu = _it_gpu()
        if not gpu:
            self.skipTest("set QCTX_STACK_IT_GPU=amd|intel|nvidia to run the gpu leg")
        if not _gpu_available(gpu):
            self.skipTest(f"no {gpu} GPU on this host: the leg cannot run")
        # The cpu baseline, in the SAME run, is what the gpu rerank is compared to.
        with tempfile.TemporaryDirectory(prefix="stack-it-cpu-") as raw:
            cpu_dir = Path(raw)
            _provision("cpu", cpu_dir)
            try:
                cpu_rerank_s = _measure_rerank_s(cpu_dir)
            finally:
                _remove(cpu_dir)
        with tempfile.TemporaryDirectory(prefix="stack-it-gpu-") as raw:
            workdir = Path(raw)
            _provision(gpu, workdir)
            try:
                import stack.state as state
                saved = state.load(workdir / "stack")
                self.assertIsNotNone(saved)
                self.assertEqual(saved.profile, gpu,
                                 "the recorded profile must be the one asked")
                self.assertIsNotNone(saved.device,
                                     "a gpu stack records the device it runs on")
                self.assertTrue(str(saved.device).startswith("Vulkan"),
                                f"the device must be a Vulkan node, got "
                                f"{saved.device!r}")
                self.assertEqual(_status_code(workdir), 0,
                                 "the gpu stack must be healthy")
                gpu_rerank_s = _measure_rerank_s(workdir)
                print(f"\n  [it] rerank cpu {cpu_rerank_s:.2f}s -> gpu "
                      f"{gpu_rerank_s:.2f}s", file=sys.stderr)
                self.assertLess(gpu_rerank_s, cpu_rerank_s,
                                f"the {gpu} rerank ({gpu_rerank_s:.2f}s) was not "
                                f"faster than the cpu rerank ({cpu_rerank_s:.2f}s)")
            finally:
                _remove(workdir)
            self.assertEqual([n for n in _engine_containers()
                              if n.startswith(PROJECT)], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
