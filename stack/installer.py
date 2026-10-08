"""The provisioning use case: `qctx install` stands up the stack.

`provision` is the whole step, in the order the spec fixes ("O que ela faz,
em ordem"): pick the runtime, check the platform, the availability of every
profile, ports, disk and RAM, render and validate the probe compose files,
pull the images, prove the GPUs, run the menu, confirm the summary, download
the models, render the real compose and bring the stack up, verify and
calibrate it, write the config, mark the state running.

This module ORCHESTRATES: every piece of knowledge it needs lives in the
module that owns it — the profiles and their availability in `backends`, the
render in `compose`, the download in `fetch`, the readiness in `health`, the
functional and calibration semantics in `verify`, the state schema in
`state`. A step here calls those and translates their answers into terminal
lines and state fields; it computes no device list, no port, no hash of its
own.

The three `Protocol`s are the seam the tests and the CLI (Task 11) run
against: the prompts, the output and the config file are injected, so the
whole flow is exercised with fakes and a temp directory, no engine, no
network. The flow's own data travels in a private `_Ctx`.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol

from core import setup as core_setup
from core.config import Config
from core.errors import CoreError

from . import StackError, catalog, compose, fetch, facts, health, state, verify
from .backends import (BACKENDS, MISSING, Availability, Option, READY,
                       default_option, runtime_label)
from .engine import ContainerRuntime, log_tail
from .facts import HostFacts
from .fetch import Transport

#: The headroom the disk must still carry beyond the models: the Qdrant
#: volume grows in it, and a download that fills the disk to the byte cannot
#: resume.
DISK_HEADROOM = 400 * 2 ** 20
#: Below this the RAM warning fires: the two servers measured ~7.3 GiB of RAM
#: together in the stack's own configuration (cpu embed ~2.3 GiB, cpu rerank
#: ~5.0 GiB, cgroup, 2026-10-08), so less than 8 GiB may not hold them plus
#: Qdrant. (The ~4.6 GiB the warning used to carry was the opening-check
#: figure, not the stack's: 1b35a87 fixed the doc, this fixes the code.)
RAM_WARN_BYTES = 8 * 2 ** 30
#: The manual path a host phase 1 cannot serve is pointed at.
README_PATH = "README.md, section 'Local models'"
#: The BARE host the endpoints are published on (phase 1); the ports live in
#: the state's `ports` dict, one per service (a host:port here would conflate
#: the two, and the phase-2 exposure classifier works on hosts).
LISTEN_HOST = "127.0.0.1"


class Prompter(Protocol):
    """The terminal questions: a free answer, a yes/no, a pick from a menu."""

    def ask(self, prompt: str) -> str: ...

    def confirm(self, prompt: str, *, default: bool = False) -> bool: ...

    def choose(self, title: str, lines: list[str], default: int) -> int: ...


class Reporter(Protocol):
    """The terminal output, in the wizard's style. `bar` hands the download
    its progress line (decision 16); the implementation decides the TTY
    shape."""

    def step(self, text: str) -> None: ...

    def ok(self, text: str) -> None: ...

    def info(self, text: str) -> None: ...

    def warn(self, text: str) -> None: ...

    def fail(self, text: str) -> None: ...

    def bar(self, total: int, label: str, start: int) -> fetch.ProgressBar: ...


class ConfigSink(Protocol):
    """The plugin's config file, as the installer reads and writes it."""

    def current_file(self) -> dict: ...

    def save(self, patch: dict) -> None: ...

    def effective(self) -> Config: ...


@dataclass
class Request:
    """What the invocation asked for: the profile and runtime flags, `--yes`,
    the `--image` overrides, the port choices and the project."""
    profile: str | None = None
    runtime: str | None = None
    yes: bool = False
    images: dict[str, str] = field(default_factory=dict)
    ports: dict[str, int] = field(default_factory=lambda: dict(catalog.PORTS))
    project: str = catalog.PROJECT


@dataclass
class Deps:
    """Everything `provision` may touch, injected. `facts` is read here, not
    re-collected: gathering is the CLI's job (Task 11). The probes (port,
    http, diagnose, calibrate, clock, sleep) default to the real ones; a test
    swaps them for fakes, so no engine, network or real time is needed."""
    runtimes: list[ContainerRuntime]
    facts: HostFacts
    prompter: Prompter
    reporter: Reporter
    config: ConfigSink
    transport: Transport
    stack_dir: Path
    budgets: list[verify.Budget]
    env: Mapping[str, str]
    port_free: Callable[[int], bool]
    status: Callable[[str], int | None] = health.http_status
    diagnose: Callable[[Config], dict] = core_setup.diagnose
    calibrate: Callable[..., tuple] = verify.calibrate
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


@dataclass
class _Ctx:
    """The settled data the later steps read instead of re-deriving: the
    runtime (step 1), the platform (step 2), the ports and the images."""
    request: Request
    deps: Deps
    runtime: ContainerRuntime
    platform: str = ""
    ports: dict[str, int] = field(default_factory=dict)
    images: dict[str, str] = field(default_factory=dict)


def provision(request: Request, deps: Deps) -> "state.StackState | None":
    """The twelve steps, in the order the spec fixes. Returns the state on
    success, or None when the user declines the summary (nothing was
    downloaded, nothing written)."""
    ctx = _Ctx(request, deps, _choose_runtime(request, deps)[0])
    ctx.platform = _check_platform(deps)
    _ports(ctx)
    _check_disk_and_ram(ctx)
    _write_probes(ctx)
    _pull(ctx)
    options = _prove(_options(ctx), ctx)
    option = _pick_option(ctx, options)
    plan = _plan(ctx, option)
    if not _summary(ctx, plan):
        return None
    _download(ctx)
    _write_compose_and_start(ctx, plan)
    dim = _verify_functional(ctx)
    _calibrate(ctx, plan)
    _save_config(ctx, plan, dim)
    return _finish(ctx)


def choose_ports(wanted: dict[str, int],
                 port_free: Callable[[int], bool]) -> dict[str, int]:
    """The ports to bind: the wanted one when free, else the FIRST free port
    in `range(p + 10000, p + 10100)`. A port already given to another service
    does not count as free: the fallback ranges of two services overlap (embed
    18003..18102 and rerank 18004..18103), so on a host that holds both wanted
    ports the two would otherwise land on the SAME candidate, and `compose up
    -d` would fail binding the second one. No free port in the range is an
    error that names its step: a stack that cannot bind is not a stack."""
    got: dict[str, int] = {}
    taken: set[int] = set()
    for service, port in wanted.items():
        for candidate in [port, *(port + 10000 + i for i in range(100))]:
            if candidate not in taken and port_free(candidate):
                got[service] = candidate
                taken.add(candidate)
                break
        else:
            raise StackError(
                f"port {port} and none of {port + 10000}..{port + 10099} is free "
                f"for {service}", step="ports",
                fix="free one of those ports, or pass the free one")
    return got


def build_options(platform: str, host: HostFacts, engine, runtime: str,
                  proofs: Mapping[str, list], availability: Mapping[str, Availability]
                  ) -> list[Option]:
    """The menu: one option per profile, and for a proven GPU profile, ONE
    option per proven device in `proofs` (spec: the menu lists each GPU the
    `--list-devices` showed). The order is the catalogue's, cpu first; the
    availability is shared by the profile's options. `provision` calls it
    BEFORE the proofs, with an empty `proofs` (one deviceless option per
    profile), and `_prove` then expands each proven profile into one option
    per device the same way."""
    options: list[Option] = []
    for backend in BACKENDS:
        av = availability[backend]
        if av.state == READY and backend != "cpu":
            seen = list(proofs.get(backend, ()))
            if seen:
                for device in seen:
                    options.append(Option(backend, device, None, av))
                continue
        options.append(Option(backend, None, None, av))
    return options


def option_line(option: Option, platform: str) -> str:
    """One line of the menu: the profile, where it runs (the label the
    compatibility table gives), and what it stands at here — the device with
    its free memory, the reason and fix when it is missing, the runtime it
    needs when this one does not serve it."""
    av = option.availability
    where = runtime_label(BACKENDS[option.backend].runtimes(platform))
    if av.state == READY:
        if option.device is not None:
            return (f"{option.backend}: {option.device.id} "
                    f"({option.device.name}, {option.device.total_mib} MiB, "
                    f"{option.device.free_mib} MiB free) [{where}]")
        return f"{option.backend}: no device (the servers run on the cpu) [{where}]"
    if av.state == "runtime":
        # the line says what the profile needs and where, and the bracket the
        # compatibility table gives; it does NOT name the runtime in use as the
        # thing it does not run on, because that runtime is not passed here (the
        # brief's signature is option_line(option, platform)) and deriving it
        # from the backend's own label rendered "runs on podman here, not podman".
        return f"{option.backend}: runs on {av.needs} here ({av.reason}) [{where}]"
    reason = av.reason or "unavailable"
    fix = f"; {av.fix}" if av.fix else ""
    return f"{option.backend}: unavailable here — {reason}{fix} [{where}]"


def config_patch(ports: dict[str, int], dim: int | None) -> dict:
    """The config fields the stack writes: its URLs (M2: `api_base_url` ends
    in `/v1` and `embed_url` is cleared, or a stale one keeps winning over
    it) and the dimension the embedder answered."""
    patch = dict(verify.stack_urls(ports))
    if dim:
        patch["vector_size"] = dim
    return patch


def config_diff(current: dict, patch: dict) -> list[tuple[str, str, str]]:
    """Only what changes: (field, current, new). A field the file does not
    have shows an empty current, the way the wizard's own diff does."""
    return [(key, str(current.get(key, "")), str(new))
            for key, new in patch.items() if str(current.get(key, "")) != str(new)]


def reboot_hint(runtime: str, platform: str) -> list[str]:
    """How the stack comes back after a reboot on this runtime: the lines the
    installer prints without executing (spec, "Reboot")."""
    if runtime == "docker":
        return ["after a reboot: restart: always brings the containers back; "
                "keep the docker daemon starting at boot (on Docker Desktop, "
                "'start at login')"]
    if platform == "linux":
        return ["after a reboot: systemctl --user enable --now "
                "podman-restart.service (it restarts the containers), and "
                "linger for the session (loginctl enable-linger $USER)"]
    return ["after a reboot: podman machine start brings the VM back, and "
            "restart: always brings the containers back with it"]


# -- the twelve steps, in the order the spec fixes ---------------------------


def _choose_runtime(request: Request, deps: Deps) -> tuple[ContainerRuntime, object]:
    """Step 1: the runtime that will serve the stack. The ones that do not
    answer (no engine, no compose provider) are refused with the fix they
    carry. With two that do, `--runtime` decides, else the question (or
    `--yes`, whose default is the docker discovered first). `apple` is served
    by Podman alone: it goes to Podman without asking, and a `--runtime docker`
    that asked for it is refused here, not discovered at the menu."""
    candidates = [r for r in deps.runtimes if r.compose_provider().provider is not None]
    if not candidates:
        # The per-runtime fixes, when any runtime answers without a provider;
        # the general install otherwise (no binary at all): a refusal must
        # never end with an empty correction.
        fixes = sorted({(r.compose_provider().fix or "install Docker or Podman")
                        for r in deps.runtimes}) or ["install Docker or Podman"]
        raise StackError("no container runtime with a compose provider answers",
                         step="runtime", fix="; ".join(fixes))
    if request.runtime:
        for runtime in candidates:
            if runtime.name == request.runtime:
                return _engine_of(request, deps, runtime)
        named = [r for r in deps.runtimes if r.name == request.runtime]
        if named:
            fix = named[0].compose_provider().fix or "install a compose provider"
            raise StackError(
                f"--runtime {request.runtime}: it answers but has no compose "
                f"provider", step="runtime", fix=fix)
        raise StackError(f"--runtime {request.runtime}: that runtime does not "
                         f"answer here", step="runtime",
                         fix="use one of the runtimes that do")
    if request.profile == "apple":
        for runtime in candidates:
            if runtime.name == "podman":
                deps.reporter.info("apple runs on Podman only: using podman")
                return _engine_of(request, deps, runtime)
        raise StackError("apple runs on Podman only, and podman does not answer",
                         step="runtime", fix="start podman, or pick another profile")
    if len(candidates) > 1:
        runtime = candidates[0] if request.yes else _ask_runtime(deps, candidates)
    else:
        runtime = candidates[0]
    return _engine_of(request, deps, runtime)


def _engine_of(request: Request, deps: Deps, runtime: ContainerRuntime) -> tuple:
    """The engine of the chosen runtime, with its provider's `note` (the M3
    fallback, when one happened) said to the terminal, and the refusal of a
    `--stack` profile that this runtime does not serve."""
    info = runtime.compose_provider()
    if info.note:
        deps.reporter.info(info.note)
    engine = runtime.engine()
    if engine is None:
        raise StackError(f"{runtime.name} stopped answering", step="runtime",
                         fix="check the engine and run again")
    profile = request.profile
    if profile in BACKENDS:
        needs = BACKENDS[profile].runtimes(facts.platform_of(deps.facts))
        if needs and runtime.name not in needs:
            raise StackError(f"{profile} runs on {runtime_label(needs)}",
                             step="runtime", fix=f"use --runtime {sorted(needs)[0]}")
    return runtime, engine


def _engine(ctx: _Ctx):
    """The engine step 1 settled (it answered there, and the engines the
    runtimes report are immutable for the life of the install): the later
    steps read it through this rather than re-asking the runtime."""
    engine = ctx.runtime.engine()
    assert engine is not None  # step 1 checked it
    return engine


def _ask_runtime(deps: Deps, candidates: list[ContainerRuntime]) -> ContainerRuntime:
    """The question that names, per runtime, the profiles it serves HERE: the
    ones READY or MISSING on it. An option this runtime does not serve is not
    one it serves, so it is not on the line."""
    platform = facts.platform_of(deps.facts)
    lines = []
    for runtime in candidates:
        engine = runtime.engine()
        if engine is None:
            continue
        served = [b for b in BACKENDS
                  if runtime.name in BACKENDS[b].runtimes(platform)
                  and BACKENDS[b].availability(platform, deps.facts, engine,
                                                runtime.name).state in (READY, MISSING)]
        lines.append(f"{runtime.name}: serves " + ", ".join(served))
    return candidates[deps.prompter.choose("which runtime should run the stack?",
                                           lines, 0)]


def _check_platform(deps: Deps) -> str:
    """Step 2, first: the platform. Windows (WSL included) enters in phase 3;
    until then the step is not offered, and the refusal points at the manual
    path the README keeps for it."""
    platform = facts.platform_of(deps.facts)
    if platform not in ("linux", "macos"):
        raise StackError("the stack runs on linux and macos in this version",
                         step="platform",
                         fix=f"until windows arrives (phase 3), set it up by hand: "
                             f"{README_PATH}")
    return platform


def _ports(ctx: _Ctx) -> None:
    """Step 2: the wanted ports, with a busy one moved to the first free
    `port + 10000` (the move is reported, the result settles the flow)."""
    ctx.ports = choose_ports(ctx.request.ports, ctx.deps.port_free)
    for service, moved in ctx.ports.items():
        if moved != ctx.request.ports[service]:
            ctx.deps.reporter.info(f"port {ctx.request.ports[service]} is busy: "
                                   f"{service} moves to {moved}")


def _check_disk_and_ram(ctx: _Ctx) -> None:
    """Step 2: the disk gate (below the models plus headroom, minus what is
    already downloaded — a re-run only fetches the missing model) and the RAM
    warning, on the figure the containers GET: the engine's on macOS (the
    machine's VM, not the host's RAM), the host's MemAvailable on linux."""
    free = ctx.deps.facts.disk_free_bytes
    if free is not None:
        have = 0
        models = ctx.deps.stack_dir / "models"
        if models.is_dir():
            have = sum(p.stat().st_size for p in models.iterdir() if p.is_file())
        need = catalog.MODELS_BYTES - have + DISK_HEADROOM
        if free < need:
            raise StackError(
                f"only {free / 2 ** 20:.0f} MiB free on disk; the download needs "
                f"{need / 2 ** 20:.0f} MiB (the models minus what is already here, "
                f"plus headroom)", step="disk",
                fix="free disk space and run again; a .part file resumes")
    engine = _engine(ctx)
    ram = (engine.memory_bytes if ctx.deps.facts.system == "macos"
           else ctx.deps.facts.ram_bytes)
    if ram is not None and ram < RAM_WARN_BYTES:
        ctx.deps.reporter.warn(
            f"only {ram / 2 ** 20:.0f} MiB of RAM for the containers: the two "
            f"llama-servers measured ~7.3 GiB together in the cpu profile, plus qdrant")


def _availability(ctx: _Ctx) -> dict[str, Availability]:
    """Each profile's availability on this host and runtime — the decision
    lives in the backend, this only asks it, per profile."""
    return {b: BACKENDS[b].availability(ctx.platform, ctx.deps.facts,
                                        _engine(ctx), ctx.runtime.name)
            for b in BACKENDS}


def _probe_plan(ctx: _Ctx, backend: str, gpu_index: int | None) -> compose.Plan:
    """A probe render of `backend`: the device is still unknown at this
    point (the proof step names it), so it renders without one; `gpu_index`
    is the nvidia-smi index the nvidia profile needs per file."""
    return compose.Plan(
        platform=ctx.platform, runtime=ctx.runtime.name, backend=backend,
        device=None, gpu_index=gpu_index, ports=ctx.ports,
        stack_dir=ctx.deps.stack_dir, images=ctx.images,
        selinux=ctx.deps.facts.selinux, project=ctx.request.project)


def _write_probes(ctx: _Ctx) -> None:
    """Step 3: `models/` exists; every profile READY (the nvidia excepted,
    its files are per index and the proof step writes them) gets its probe
    compose in `<stack>/probe/`, and each one passes `compose config` — the
    provider validating the file before it runs."""
    deps = ctx.deps
    ctx.images = catalog.resolve_images(ctx.request.images, deps.env)
    probe_dir = deps.stack_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    (deps.stack_dir / "models").mkdir(parents=True, exist_ok=True)
    for backend, av in _availability(ctx).items():
        if av.state != READY or backend == "nvidia":
            continue
        file = probe_dir / f"{backend}.yaml"
        file.write_text(compose.dump(_probe_plan(ctx, backend, None)))
        _run_compose(ctx, file, "config")


def _run_compose(ctx: _Ctx, file: Path, *args: str, step: str = "install",
                 timeout: float = 60.0, stream: bool = False) -> "object":
    """One compose command through the runtime that serves the stack: its
    provider, the project (M6: the volume name depends on it), the timeout of
    the plan's table. A failure names the command and shows its output."""
    provider = ctx.runtime.compose_provider().provider
    if provider is None:
        raise StackError(f"{ctx.runtime.name} has no compose provider",
                         step="runtime", fix="install a compose provider")
    out = ctx.runtime.compose(provider, ctx.request.project, file, *args,
                              timeout=timeout, stream=stream)
    if not out.ok:
        detail = out.stderr.strip() or out.stdout.strip() or "(no output)"
        raise StackError(f"compose {' '.join(args)} failed: {detail}", step=step,
                         fix="check the runtime logs for that service")
    return out


def _pull(ctx: _Ctx) -> None:
    """Step 4: the images, with the provider's progress on the terminal
    (`stream=True` inherits the terminal's own stdout, the 1800 s bound of
    the plan). The pull does not depend on the choice — every profile runs
    the same image — so it happens before the menu, and the menu only offers
    what the container will in fact see."""
    file = ctx.deps.stack_dir / "probe" / "cpu.yaml"
    if not file.exists():
        file = ctx.deps.stack_dir / "probe" / "pull.yaml"
        file.write_text(compose.dump(_probe_plan(ctx, "cpu", None)))
    _run_compose(ctx, file, "pull", timeout=1800.0, stream=True)


def _options(ctx: _Ctx) -> list[Option]:
    """The menu before the proofs: every profile as an option, each with its
    availability on this runtime and no device yet (step 5 fills them)."""
    return build_options(ctx.platform, ctx.deps.facts, _engine(ctx),
                         ctx.runtime.name, {}, _availability(ctx))


def _prove(options: list[Option], ctx: _Ctx) -> list[Option]:
    """Step 5: each GPU profile READY on the host side runs
    `compose run -T --rm --no-deps embed --list-devices` (M4: the `-T`, the
    output otherwise carries the TTY's `\\r\\n`) in its probe file — the same
    service definition that will run, so what the container sees here is what
    it will see later. The nvidia is proven once per nvidia-smi index, in its
    own file. A failed proof demotes the profile to MISSING with the tail of
    its stderr; a proof that sees no GPU demotes it with the reason that says
    why the host's card did not reach the container."""
    out: list[Option] = []
    for option in options:
        av = option.availability
        if av.state != READY or option.backend == "cpu":
            out.append(option)
            continue
        if option.backend == "nvidia":
            out.extend(_prove_nvidia(ctx, option))
            continue
        file = ctx.deps.stack_dir / "probe" / f"{option.backend}.yaml"
        devices, tail = _probe_devices(ctx, option.backend, file)
        if devices:
            # ONE option per proven GPU (spec: the menu lists each GPU the
            # `--list-devices` showed): the default, the menu and an explicit
            # `--stack <profile>` all see every line, not only the first.
            out.extend(Option(option.backend, device, None, av) for device in devices)
        else:
            _demote(option.backend, tail, ctx)
            out.append(Option(option.backend, None, None,
                              Availability(MISSING, tail,
                                           "check the gpu driver and the /dev/dri nodes")))
    return out


def _probe_devices(ctx: _Ctx, backend: str, file: Path) -> "tuple[list, str]":
    """The `--list-devices` of one probe file: the devices the backend keeps
    (its vendor), and the TAIL of the stderr when the run itself failed."""
    provider = ctx.runtime.compose_provider().provider
    if provider is None:  # pragma: no cover - step 1 guarantees a provider
        return [], "the runtime has no compose provider"
    out = ctx.runtime.compose(provider, ctx.request.project, file,
                              "run", "-T", "--rm", "--no-deps", "embed",
                              "--list-devices", timeout=180.0)
    if not out.ok:
        lines = [l for l in out.stderr.strip().splitlines() if l.strip()]
        return [], f"the gpu proof failed: {lines[-1] if lines else '(no output)'}"
    devices = BACKENDS[backend].devices_seen(out.stdout)
    if not devices:
        # The container saw nothing (the `(none)` line): the host's card did
        # not reach it, and the menu must say why with the probable reason.
        return [], ("the container sees no gpu (list-devices printed (none); the "
                    "host's card did not reach it — check the driver and the "
                    "/dev/dri nodes)")
    return devices, ""


def _prove_nvidia(ctx: _Ctx, option: Option) -> list[Option]:
    """One option per nvidia-smi index the host lists: each is rendered in
    its own probe file with the index it needs (device_ids on docker, the CDI
    device on podman) and proven there, because the index is what the real
    service patch takes (and `service_patch` refuses None)."""
    out: list[Option] = []
    for index in range(len(ctx.deps.facts.nvidia.gpus)):
        file = ctx.deps.stack_dir / "probe" / f"nvidia-{index}.yaml"
        file.write_text(compose.dump(_probe_plan(ctx, "nvidia", index)))
        devices, tail = _probe_devices(ctx, "nvidia", file)
        if devices:
            out.append(Option("nvidia", devices[0], index, option.availability))
        else:
            _demote(f"nvidia-{index}", tail, ctx)
            out.append(Option("nvidia", None, index,
                              Availability(MISSING, tail, "check the nvidia toolkit")))
    return out


def _demote(backend: str, tail: str, ctx: _Ctx) -> None:
    ctx.deps.reporter.warn(f"{backend}: {tail}")


def _pick_option(ctx: _Ctx, options: list[Option]) -> Option:
    """Step 6: the menu. An explicit profile that is not READY here stops
    with its reason and fix — it does NOT fall back to the cpu, because the
    user asked for it by name; only `auto` (and `--yes` alone, which takes
    the default without asking) fall back. An explicit profile that IS ready
    reduces the menu to the profile's GPUs (spec: "o menu se reduz às GPUs
    daquele perfil") and the pick decides; with `--yes` the profile's default
    (its most free GPU) applies without asking. A profile that is None is the
    menu itself, and a pick of an option this runtime does not serve repeats
    its correction and asks again (the step does not install runtimes)."""
    request = ctx.request
    if request.profile is not None and request.profile != "auto":
        ready = [o for o in options
                 if o.backend == request.profile and o.availability.state == READY]
        if not ready:
            option = next(o for o in options if o.backend == request.profile)
            av = option.availability
            fix = av.fix or (f"use {av.needs}" if av.needs else None)
            raise StackError(
                f"{request.profile}: {av.reason or 'unavailable here'}",
                step="profile", fix=fix)
        if request.yes:
            return _default_within(ready)
        if len(ready) == 1:
            return ready[0]  # one option: there is nothing to choose
        return _menu(ctx, ready, default=_default_within(ready))
    if request.profile == "auto" or request.yes:
        return default_option(options)
    return _menu(ctx, options)


def _default_within(options: list[Option]) -> Option:
    """The profile's own default when the user named the profile and passed
    `--yes`: the most free GPU among the profile's ready options. The
    whole-menu rules of `default_option` do not apply here: the experimental
    option the user asked for by name is not refused, and there is no cpu in
    the reduced list to fall back to (a `--stack cpu` is just the cpu)."""
    best, best_free = options[0], -1
    for option in options:
        free = option.device.free_mib if option.device is not None else 0
        if free > best_free:
            best, best_free = option, free
    return best


def _menu(ctx: _Ctx, options: list[Option], default: Option | None = None) -> Option:
    """The menu loop: one line per option; an unavailable pick repeats its
    correction and the menu comes back. The default is the one the caller
    passes -- an explicit profile's reduced menu passes its own most free GPU
    -- else the whole-menu rule (the proven, non-experimental GPU with the
    most free memory, else the cpu)."""
    if default is None:
        default = default_option(options)
    while True:
        lines = [option_line(o, ctx.platform) for o in options]
        pick = ctx.deps.prompter.choose("which profile should run the stack?",
                                        lines, options.index(default))
        option = options[pick]
        if option.availability.state == READY:
            return option
        av = option.availability
        text = f"{option.backend} is not available here — {av.reason}"
        if av.fix:
            text += f"; {av.fix}"
        elif av.needs:
            text += f"; it runs on {av.needs}"
        ctx.deps.reporter.warn(text)


def _plan(ctx: _Ctx, option: Option) -> compose.Plan:
    """The render of the chosen option: its device (None on the cpu, which
    the server command turns into `-dev none` — M1: an engine can inject a
    GPU into every container) and its nvidia index."""
    return compose.Plan(
        platform=ctx.platform, runtime=ctx.runtime.name, backend=option.backend,
        device=option.device.id if option.device else None,
        gpu_index=option.gpu_index, ports=ctx.ports, stack_dir=ctx.deps.stack_dir,
        images=ctx.images, selinux=ctx.deps.facts.selinux, project=ctx.request.project)


def _summary(ctx: _Ctx, plan: compose.Plan) -> bool:
    """Step 7: what will happen — the models and the MiB still to download,
    the ports, the directory, the volume — and the `proceed? [y/N]`, skipped
    with `--yes`. A decline is not an error: it ends the step and downloads
    nothing."""
    missing = 0
    for model in catalog.MODELS:
        if not (ctx.deps.stack_dir / "models" / model.filename).exists():
            missing += model.size
    deps = ctx.deps
    deps.reporter.info(f"downloads: {missing / 2 ** 20:.0f} MiB of models into "
                       f"{deps.stack_dir / 'models'}")
    deps.reporter.info("ports: " + ", ".join(f"{k} on {v}"
                                             for k, v in plan.ports.items()))
    deps.reporter.info(f"stack directory: {deps.stack_dir}")
    deps.reporter.info(f"qdrant volume: {compose.volume_name(plan.project)}")
    if ctx.request.yes:
        return True
    return deps.prompter.confirm("proceed? [y/N]", default=False)


def _download(ctx: _Ctx) -> None:
    """Step 8: the two models through the real `fetch` (resume + sha256), the
    progress on the reporter's bar. What is already the pinned file is
    skipped — the step is idempotent."""
    for model in catalog.MODELS:
        fetch.fetch(model, ctx.deps.stack_dir / "models", ctx.deps.transport,
                    bar=lambda total, start, label=model.filename: \
                        ctx.deps.reporter.bar(total, label, start),
                    log=ctx.deps.reporter.ok)


def _log_tail(ctx: _Ctx, file: Path, service: str) -> str:
    """The log tail of one service of this install (`engine.log_tail`, the one
    copy `qctx stack up` uses too)."""
    return log_tail(ctx.runtime, ctx.request.project, file, service)


def _write_compose_and_start(ctx: _Ctx, plan: compose.Plan) -> None:
    """Step 9: the real `compose.yaml`, the state in the `compose` phase,
    `compose up -d` (the 600 s bound) and the readiness wait (the 10 minutes
    of the spec). A failure — the `up` itself, or the wait — shows the log
    tail of every service that is NOT ready and stops with the state still in
    the `compose` phase: nothing is verified, and nothing reaches the config.
    Services that ARE ready are recorded by `wait_ready` as they come up, so
    their logs are not the evidence."""
    deps = ctx.deps
    deps.reporter.step("writing compose.yaml and starting the stack")
    compose_file = deps.stack_dir / "compose.yaml"
    compose_file.write_text(compose.dump(plan))
    state.save(deps.stack_dir, _state(ctx, plan, state.PHASE_COMPOSE))
    ready: dict[str, float] = {}
    try:
        _run_compose(ctx, compose_file, "up", "-d", timeout=600.0, step="up")
        health.wait_ready(health.endpoints(plan.ports), status=deps.status,
                          clock=deps.clock, sleep=deps.sleep, ready=ready)
    except StackError:
        for service in catalog.SERVICES:
            if service not in ready:
                deps.reporter.info(f"{service}:\n{_log_tail(ctx, compose_file, service)}")
        raise StackError(
            "the stack is not ready; the log tail of the services that did not "
            "come up is above", step="up",
            fix="fix what the log names, then re-run the install: qctx install (it "
                "resumes, verifies, then writes the config)")
    for service in ready:
        deps.reporter.ok(f"{service} is ready")


def _state(ctx: _Ctx, plan: compose.Plan, phase: str) -> "state.StackState":
    """The state the step writes: the provider used is recorded for the status
    section and the error text (informational: the lifecycle re-resolves it
    from the recorded runtime, see `lifecycle`), `listen` the BARE host the
    endpoints are published on (the ports live in `ports`), and the models
    the catalogue pins (filename to sha256)."""
    provider = ctx.runtime.compose_provider().provider
    assert provider is not None
    now = state.now()
    return state.StackState(
        role="local", listen=LISTEN_HOST, platform=ctx.platform,
        runtime=ctx.runtime.name, provider=[provider.name],
        profile=plan.backend, device=plan.device, gpu_index=plan.gpu_index,
        ports=dict(plan.ports), images=dict(plan.images),
        models={m.filename: m.sha256 for m in catalog.MODELS},
        qdrant_version=catalog.qdrant_version(plan.images["qdrant"]),
        selinux=ctx.deps.facts.selinux, phase=phase,
        created_at=now, updated_at=now, project=plan.project)


def _verify_functional(ctx: _Ctx) -> int | None:
    """Step 10, first half: the functional check, reusing `core.setup.diagnose`
    through `verify.functional` (the three services this step stood up, and
    only they; the rerank failure blocks here, unlike in `diagnose`). A
    failure stops the flow with the config left untouched: a config that
    points at a stack that does not work is worse than no config."""
    cfg = verify.stack_config(ctx.ports)
    checks, dim = verify.functional(cfg, diagnose=ctx.deps.diagnose)
    if not verify.functional_ok(checks):
        for check in checks:
            if not check.ok:
                ctx.deps.reporter.fail(f"{check.name}: {check.detail}")
                if check.fix_hint:
                    ctx.deps.reporter.info(f"fix: {check.fix_hint}")
        raise StackError("the stack did not pass its functional check",
                         step="verify",
                         fix="check what the failing lines name, then re-run the "
                             "install: qctx install (it re-verifies, and only then "
                             "writes the config)")
    for check in checks:
        ctx.deps.reporter.ok(f"{check.name}: {check.detail}")
    return dim


def _calibrate(ctx: _Ctx, plan: compose.Plan) -> None:
    """Step 10, second half: the calibration, one check per host budget (a
    warning when it is over — an over-budget stack works, it just misses a
    deadline), the memory through the runtime's `stats` (Ruling 4: the callable
    is built here, and `calibrate` reads it once per host). Nothing here aborts
    the install (spec: calibration never blocks; the functional check already
    passed, so the stack works, and an abort would leave the config unwritten
    and every re-run re-provisioning to the same point). Two failures, two
    warnings that each say only what is true: the memory reading fails alone
    (the runtime's `stats`, e.g. rootless podman without cgroups v2) and the
    timing is still reported; the timing itself fails (the embedder or the
    reranker raises its CoreError) and neither speed nor memory was checked. A
    non-CoreError is a bug and is not caught."""
    names = [compose.container_name(plan.project, role) for role in ("embed", "rerank")]
    unread: list[str] = []

    def memory() -> dict:
        # The memory reading is the last part of the measurement and the most
        # fragile (`stats` needs cgroups v2 on rootless podman): when it fails,
        # the timing already measured must survive, so the failure is recorded
        # and said once below, not raised through the calibration.
        try:
            return ctx.runtime.stats(names)
        except StackError as exc:
            if not unread:
                unread.append(str(exc))
            return {}

    cfg = verify.stack_config(ctx.ports)
    try:
        checks, info = ctx.deps.calibrate(cfg, ctx.deps.budgets, clock=ctx.deps.clock,
                                          memory=memory)
    except CoreError as exc:
        # The measurement itself failed (the embedder or the reranker could not
        # be timed). A CoreError only: a bug must surface, not pass for this.
        ctx.deps.reporter.warn(f"calibration: the measurement failed ({exc}); "
                               "the stack is up, but its speed and memory were not checked")
        return
    for check in checks:
        (ctx.deps.reporter.ok if check.ok else ctx.deps.reporter.warn)(
            f"{check.name}: {check.detail}")
    for host, row in info.items():
        for name, used in (row.get("memory") or {}).items():
            ctx.deps.reporter.info(f"{host}: {name} uses {used / 2 ** 20:.0f} MiB")
    if unread:
        ctx.deps.reporter.warn(f"calibration: the containers' memory was not read "
                               f"({unread[0]}); the timing above was measured")


def _save_config(ctx: _Ctx, plan: compose.Plan, dim: int | None) -> None:
    """Step 11: the config, written ONLY after the verification passed (this
    is where it is written, and nowhere earlier). The diff is shown; a
    non-empty value that is replaced asks first (`--yes` answers yes without
    asking); a decline leaves the file untouched and prints the `config set`
    of every field the stack would have written. Then the env trap: each
    variable that still points elsewhere is named with its value, and told to
    remove its export from the shell rc (the rc is the user's; the installer
    never edits it)."""
    deps = ctx.deps
    patch = config_patch(plan.ports, dim)
    diff = config_diff(deps.config.current_file(), patch)
    if not diff:
        deps.reporter.ok("config: already points at the stack")
        return
    for key, old, new in diff:
        deps.reporter.info(f"config: {key}: {old or '(empty)'} -> {new}")
    if any(old for _key, old, _new in diff) and not (
            ctx.request.yes or deps.prompter.confirm(
                "replace the non-empty value(s) above? [y/N]",
                default=ctx.request.yes)):
        for key, _old, new in diff:
            deps.reporter.info(f"not written; set it with: qctx config set "
                               f"{key.replace('_', '-')} {new}")
        return
    deps.config.save(patch)
    deps.reporter.ok(f"config: saved {len(diff)} field(s)")
    for name, value in verify.env_overrides(deps.env,
                                            {k: str(v) for k, v in patch.items()}):
        deps.reporter.warn(f"env trap: {name}={value} still points elsewhere; "
                           f"remove its export from your shell rc")


def _finish(ctx: _Ctx) -> "state.StackState":
    """Step 12: the state in `running`, the probe files gone (their job is
    done and they are not part of the stack), and how the stack comes back
    after a reboot on this runtime."""
    deps = ctx.deps
    saved = state.load(deps.stack_dir)
    assert saved is not None  # step 9 wrote it
    saved.phase = state.PHASE_RUNNING
    saved.updated_at = state.now()
    state.save(deps.stack_dir, saved)
    probe = deps.stack_dir / "probe"
    for file in probe.iterdir():
        file.unlink()
    probe.rmdir()
    deps.reporter.ok("stack.json: running")
    for line in reboot_hint(ctx.runtime.name, ctx.platform):
        deps.reporter.info(line)
    return saved
