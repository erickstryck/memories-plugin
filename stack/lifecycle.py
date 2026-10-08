"""The lifecycle of an ALREADY-provisioned stack: `qctx stack status|up|down|remove`.

`installer` builds a stack from scratch; this module operates one that exists. `status`
reports it, `up` starts it again (repeating the recorded images, or pulling the catalogue's
with `--upgrade`), `down` stops it (volume and models intact), `remove` deletes it.

TWO RULES KEEP IT SMALL. It never RE-PROBES the host: the CLI (Task 12) gathers the
runtimes and passes them in, and `runtime_for` picks the runtime the state RECORDED (it
does not re-run discovery to pick a different RUNTIME than the stack was built with). It
never RE-DOES the download: `up --upgrade` pulls images, it does not fetch models. The
four verbs talk only to the seams in `LifeDeps`, so they run on fakes and a temp directory,
no engine.

The COMPOSE PROVIDER is a different case, and it is worth saying plainly: it is re-resolved
from the recorded runtime every verb runs, NOT read out of `stack.json`. That is safe, not
a bug, because the reason to record it is already gone. The two podman providers label
containers differently only in their DEFAULT names, and every service sets an explicit
`container_name` (M6, spec:348) so the down/up/remove that follows a restart address the
same named containers and volume whichever provider created them. Re-resolving also stays
correct across a socket-state change: when the API socket goes dead, `podman compose` (the
docker-compose wrapper) must hand over to the standalone `podman-compose`, and the recorded
name would point at the wrong one. `st.provider` is therefore informational -- the status
section and the error text surface it -- and the verbs never act on it.

`remove` is the one verb that must not trust its state: a corrupt `stack.json` still has a
compose file and a volume to clean, so with no readable state it tries a `down` on every
runtime that answers and deletes the files. It never touches the config -- a config that
pointed at the stack now points at nothing, and `remove` says so.
"""
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from core.config import Config

from . import StackError, catalog, compose, health, state, verify
from .engine import ContainerRuntime, Provider, first_line
from .installer import ConfigSink, Prompter, Reporter
from .process import Runner


@dataclass
class LifeDeps:
    """Everything a lifecycle verb may touch, injected. `runtimes` is the CLI's discovery
    (not re-run here); `runner` is what `boot_status` reads `systemctl`/`loginctl`
    through; `env` is what `up --upgrade` resolves the `QCTX_STACK_IMAGE_*` overrides
    from (the spec applies them to `--upgrade` too, the same way the install does). It
    is REQUIRED, not defaulted: a construction that forgot it would silently resolve
    from nothing and bring back the ignored-override bug. `status`/`clock`/`sleep`
    are the readiness probes, defaulted to the real ones so a test swaps them for
    scripts and a frozen clock."""
    runtimes: list[ContainerRuntime]
    reporter: Reporter
    prompter: Prompter
    config: ConfigSink
    stack_dir: Path
    runner: Runner
    env: Mapping[str, str]
    status: Callable[[str], int | None] = health.http_status
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


def status(deps: LifeDeps) -> tuple[dict, int]:
    """The picture of a managed stack, and the exit code `qctx stack status` returns.

    No stack: `{"managed": False}` and 0. A stack: its phase, runtime and provider, the
    per-service URL and live status, whether every endpoint answers 200, whether the
    catalogue has moved past the recorded images, whether the config still points at these
    URLs, and the boot line -- exit 1 when it is not healthy. A state the loader refuses is
    reported as corrupt with the one fix that deletes it, and exit 1: `status` is the verb
    that must name a broken install, not guess over it.
    """
    try:
        st = state.load(deps.stack_dir)
    except StackError as exc:
        return {"managed": True, "error": str(exc), "fix": exc.fix}, 1
    if st is None:
        return {"managed": False}, 0
    urls = verify.stack_urls(st.ports)
    endpoints = health.endpoints(st.ports)
    # The displayed `url` is the config URL the plugin points at (what Task 12
    # shows); the live probe is the READINESS endpoint, the same one `up` waits
    # on. The pinned b11382 has no GET route for `/v1` or `/v1/rerank` (POST-only)
    # or a bare `/`, so probing the config URL 404s on a healthy llama-server --
    # a stack that just passed `up` would report unhealthy forever (R6).
    services = {name: {"url": urls[{"qdrant": "qdrant_url", "embed": "api_base_url",
                                    "rerank": "rerank_url"}[name]],
                       "status": deps.status(endpoints[name])}
                for name in catalog.SERVICES}
    healthy = all(s["status"] == 200 for s in services.values())
    outdated = [role for role in catalog.IMAGES
                if st.images.get(role) != catalog.IMAGES[role]]
    return {
        "managed": True,
        "phase": st.phase,
        "runtime": st.runtime,
        "provider": list(st.provider),
        "profile": st.profile,
        "device": st.device,
        "ports": dict(st.ports),
        "services": services,
        "healthy": healthy,
        "outdated_pins": outdated,
        "config_points_here": all(got == want for _f, got, want
                                  in verify.config_url_pairs(deps.config.effective(), st.ports)),
        "boot": boot_status(st, deps.runner),
    }, (0 if healthy else 1)


def up(deps: LifeDeps, *, upgrade: bool = False,
       images: dict[str, str] | None = None) -> state.StackState:
    """Start the stack again, from the recorded state.

    Without `--upgrade` it repeats EXACTLY the images `stack.json` holds (a `--image`
    override at install time is not undone by a re-run). With `--upgrade` it takes the
    catalogue's pins (or the `--image` overrides passed in), passes them through the minor
    guard, re-renders, pulls and starts. Either way it re-renders the compose, brings it up
    and waits on readiness before it marks the state running: a stack that is not ready is
    not reported as running.
    """
    st = _require_state(deps)
    if st.phase == state.PHASE_COMPOSE:
        # The last install did not finish: nothing verified the stack and nothing
        # wrote the config. `up` does neither, so starting it and marking it running
        # would hide that for good (the next install would see a healthy running
        # stack and only report it). The install resumes and does both.
        raise StackError("the last install did not finish (stack.json is still in the "
                         "compose phase), and up neither verifies the stack nor writes "
                         "the config", step="up",
                         fix="re-run the install: qctx install (it resumes, verifies, "
                             "then writes the config)")
    runtime, provider = runtime_for(st, deps.runtimes)
    if upgrade:
        resolved = catalog.resolve_images(images or {}, deps.env)
        check_minor(st.qdrant_version, catalog.qdrant_version(resolved["qdrant"]))
    else:
        resolved = dict(st.images)
    file = deps.stack_dir / state.COMPOSE_FILE
    file.write_text(compose.dump(replace(state.plan_of(st, deps.stack_dir), images=resolved)))
    if upgrade:
        _compose(deps, runtime, provider, st.project, file, "pull", timeout=1800.0, stream=True)
    _compose(deps, runtime, provider, st.project, file, "up", "-d", timeout=600.0)
    # A readiness wrapper records which service first answered 200, so a failure
    # can show the log tail of exactly the ones that did NOT come up (the
    # installer's own failure path shows the same evidence).
    targets = health.endpoints(st.ports)
    ready: dict[str, float] = {}
    by_url = {url: name for name, url in targets.items()}

    def status(url: str) -> int | None:
        answer = deps.status(url)
        if answer == 200:
            ready.setdefault(by_url[url], deps.clock())
        return answer

    try:
        health.wait_ready(targets, status=status, clock=deps.clock, sleep=deps.sleep)
    except StackError:
        for service in catalog.SERVICES:
            if service not in ready:
                tail = _log_tail(deps, runtime, provider, st.project, file, service)
                deps.reporter.info(f"{service}:\n{tail}")
        raise StackError(
            "the stack is not ready; the log tail of the services that did not "
            "come up is above", step="up",
            fix="fix what the log names, then run qctx stack up again") from None
    st.images = dict(resolved)
    if upgrade:
        st.qdrant_version = catalog.qdrant_version(resolved["qdrant"])
    st.phase = state.PHASE_RUNNING
    st.updated_at = state.now()
    state.save(deps.stack_dir, st)
    return st


def down(deps: LifeDeps) -> state.StackState:
    """Stop the stack, keeping its volume and models: a `down` without `-v`, and the state
    marked `stopped` so `status` can say it is down and point at `qctx stack up`."""
    st = _require_state(deps)
    runtime, provider = runtime_for(st, deps.runtimes)
    _compose(deps, runtime, provider, st.project,
             deps.stack_dir / state.COMPOSE_FILE, "down", timeout=300.0)
    st.phase = state.PHASE_STOPPED
    st.updated_at = state.now()
    state.save(deps.stack_dir, st)
    return st


def remove(deps: LifeDeps, *, purge_models: bool = False, purge_data: bool = False,
           yes: bool = False) -> list[str]:
    """Delete the stack's files, and (only when asked) its models and its Qdrant volume.

    `--purge-data` is the one that destroys the archive, so it asks for the project name
    typed back (or `--yes`) before it runs `down -v`; a wrong answer deletes nothing.
    `remove` NEVER touches the config: it reports the fields that still point at the stack
    it just deleted. A corrupt state still has a compose file and volume to clean, so with
    no readable state it tries a `down` on every runtime that answers.
    """
    st = _safe_load(deps.stack_dir)
    project = st.project if st is not None else catalog.PROJECT
    down_args = ("down", "-v") if purge_data else ("down",)
    if purge_data and not yes:
        answer = deps.prompter.ask(
            f"this deletes the Qdrant volume (the archive); type {project!r} to confirm: ")
        if answer.strip() != project:
            deps.reporter.info("aborted: nothing was deleted")
            return []
    file = deps.stack_dir / state.COMPOSE_FILE
    if st is None:
        # No readable state: the project name is the fixed one, and every runtime that
        # answers gets a best-effort down -- a dead provider must not stop the cleanup.
        for runtime in deps.runtimes:
            provider = runtime.compose_provider().provider
            if provider is None:
                continue
            try:
                runtime.compose(provider, project, file, *down_args, timeout=300.0)
            except StackError:
                pass
    else:
        runtime, provider = runtime_for(st, deps.runtimes)
        _compose(deps, runtime, provider, st.project, file, *down_args, timeout=300.0)
    deleted = _delete_files(deps, purge_models)
    if st is not None:
        _report_still_points_here(deps.reporter, deps.config.effective(), st.ports)
    return deleted


def check_minor(installed: str, target: str) -> None:
    """The Qdrant storage guard on `up --upgrade`: refuse a jump that skips a minor.

    Storage is compatible only between CONSECUTIVE minors, so `1.19 -> 1.21` is refused,
    naming the intermediate `1.20` to install first. The next minor (and the same or an
    older one) is allowed.
    """
    i_major, i_minor = _minor(installed)
    _t_major, t_minor = _minor(target)
    if t_minor > i_minor + 1:
        next_minor = f"{i_major}.{i_minor + 1}"
        raise StackError(
            f"Qdrant {installed} -> {target} skips minor {next_minor}; storage is "
            f"compatible only between consecutive minors",
            step="lifecycle",
            fix=f"upgrade to a Qdrant tagged v{next_minor} first, then to the new pin "
                f"(e.g. --image qdrant=<ref>:v{next_minor})")


def runtime_for(st: state.StackState,
                runtimes: list[ContainerRuntime]) -> tuple[ContainerRuntime, Provider]:
    """The runtime the STATE recorded, and the compose provider THAT runtime now
    answers with.

    The runtime is never re-discovered to a different one: a re-run that guessed
    docker when the stack was built on podman (or vice versa) would start the
    wrong containers. A recorded runtime that is no longer here, or whose
    provider has gone, is an error with a fix. The provider, by contrast, IS
    read live from that runtime (see the module note): `st.provider` is
    informational, and the live answer is what stays correct when the socket
    state changes.
    """
    named = [r for r in runtimes if r.name == st.runtime]
    if not named:
        raise StackError(f"the recorded runtime {st.runtime!r} is not discovered here",
                         step="lifecycle", fix="start that runtime, or qctx stack remove")
    runtime = named[0]
    info = runtime.compose_provider()
    if info.provider is None:
        raise StackError(
            f"{st.runtime}'s compose provider ({', '.join(st.provider)}) is no longer "
            f"available: {info.problem or 'no provider answered'}",
            step="lifecycle", fix=info.fix or "install a compose provider")
    return runtime, info.provider


def boot_status(st: state.StackState, runner: Runner) -> str:
    """How the stack comes back after a reboot, read live where it can be read.

    Podman on Linux has two user-level hooks the operator may not have set, so `status`
    reads them: whether `podman-restart.service` is enabled, and whether the session has
    linger. Docker and the non-Linux podman cases are a fixed line, because nothing on this
    machine reports them.
    """
    if st.runtime == "podman" and st.platform == "linux":
        unit = first_line(runner.run(
            ["systemctl", "--user", "is-enabled", "podman-restart.service"],
            timeout=30.0).stdout) or "not found"
        linger = _linger_value(runner.run(
            ["loginctl", "show-user", os.environ.get("USER", "unknown"),
             "--property=Linger"], timeout=30.0).stdout) or "unknown"
        return f"podman-restart.service is {unit}; linger is {linger}"
    if st.runtime == "docker":
        return ("the docker daemon must start at boot; restart: always brings the "
                "containers back")
    return "podman machine start brings the stack back"


# -- the seams the verbs share ------------------------------------------------


def _require_state(deps: LifeDeps) -> state.StackState:
    st = state.load(deps.stack_dir)
    if st is None:
        raise StackError("there is no managed stack here", step="lifecycle",
                         fix="run qctx install --stack first")
    return st


def _safe_load(directory: Path) -> state.StackState | None:
    """A state read that turns the loader's corrupt-state error into `None`: `remove`
    must clean a corrupt install, so it cannot let the read stop it."""
    try:
        return state.load(directory)
    except StackError:
        return None


def _compose(deps: LifeDeps, runtime: ContainerRuntime, provider: Provider,
             project: str, file: Path, *args: str, timeout: float,
             stream: bool = False) -> None:
    """One compose command through the recorded runtime and provider. A failure names the
    command and shows the output the provider produced."""
    out = runtime.compose(provider, project, file, *args, timeout=timeout, stream=stream)
    if not out.ok:
        detail = out.stderr.strip() or out.stdout.strip() or "(no output)"
        raise StackError(f"compose {' '.join(args)} failed: {detail}", step="lifecycle",
                         fix="check the runtime's logs for that service")


def _log_tail(deps: LifeDeps, runtime: ContainerRuntime, provider: Provider,
              project: str, file: Path, service: str) -> str:
    """The last 50 lines of one service: the evidence a failed `up` shows (the same
    evidence the installer's failed start shows). A container that never started has
    no log, and a `logs` that fails -- a non-zero exit (the provider's own output is
    the tail) or a RAISE (the provider hangs and the bound trips) -- degrades to
    "(no log)": the log tail is evidence, not a dependency, and it must never mask
    the readiness error, which is the one the operator needs."""
    try:
        out = runtime.compose(provider, project, file, "logs", "--tail", "50", service,
                              timeout=60.0)
    except StackError:
        return "(no log)"
    return out.stdout.strip() or out.stderr.strip() or "(no log)"


def _delete_files(deps: LifeDeps, purge_models: bool) -> list[str]:
    """The compose and state files, and (only when asked) the models, deleted one by one.
    The directory itself is left: it is the user's data directory, not the stack's."""
    deleted = []
    for name in (state.COMPOSE_FILE, state.STATE_FILE):
        path = deps.stack_dir / name
        if path.exists():
            path.unlink()
            deleted.append(str(path))
    if purge_models:
        models = deps.stack_dir / state.MODELS_DIR
        if models.is_dir():
            for path in models.iterdir():
                if path.is_file():
                    path.unlink()
                    deleted.append(str(path))
            models.rmdir()
    return deleted


def _report_still_points_here(reporter: Reporter, eff: Config, ports: dict[str, int]) -> None:
    """`remove` never edits the config, so a config that pointed at the stack now points at
    nothing: each field that still carries the stack's URL is named with its value."""
    matching = [(name, want) for name, got, want in verify.config_url_pairs(eff, ports)
                if got == want]
    if matching:
        for name, want in matching:
            reporter.warn(f"your config still points at the deleted stack: {name}={want}")
    else:
        reporter.info("your config does not point at the deleted stack")


def _minor(version: str) -> tuple[str, int]:
    """The major and minor of an `x.y.z` (or `x.y`) version, for the guard."""
    parts = version.strip().split(".")
    if len(parts) < 2 or not parts[1].isdigit():
        raise StackError(f"no Qdrant minor in: {version}", step="lifecycle",
                         fix="check the recorded qdrant_version in stack.json")
    return parts[0], int(parts[1])


def _linger_value(text: str) -> str:
    """The value off a `loginctl show-user ... --property=Linger` answer (`Linger=yes`)."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Linger="):
            return line.split("=", 1)[1]
    return ""
