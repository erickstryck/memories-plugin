"""The lifecycle of an ALREADY-provisioned stack: `qctx stack status|up|down|remove`.

`installer` builds a stack from scratch; this module operates one that exists. `status`
reports it, `up` starts it again (repeating the recorded images, or pulling the catalogue's
with `--upgrade`), `down` stops it (volume and models intact), `remove` deletes it.

TWO RULES KEEP IT SMALL. It never RE-PROBES the host: the CLI (Task 12) gathers the
runtimes and passes them in, and `runtime_for` picks the one the state RECORDED -- it does
not re-run discovery to guess a different provider than the stack was built with. It never
RE-DOES the download: `up --upgrade` pulls images, it does not fetch models. The four verbs
talk only to the seams in `LifeDeps`, so they run on fakes and a temp directory, no engine.

`remove` is the one verb that must not trust its state: a corrupt `stack.json` still has a
compose file and a volume to clean, so with no readable state it tries a `down` on every
runtime that answers and deletes the files. It never touches the config -- a config that
pointed at the stack now points at nothing, and `remove` says so.
"""
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from core.config import Config

from . import StackError, catalog, compose, health, state, verify
from .engine import ContainerRuntime, Provider, first_line
from .installer import ConfigSink, Prompter, Reporter
from .process import Runner


@dataclass
class LifeDeps:
    """Everything a lifecycle verb may touch, injected. `runtimes` is the CLI's discovery
    (not re-run here); `runner` is what `boot_status` reads `systemctl` and `loginctl`
    through; `status`/`clock`/`sleep` are the readiness probes, defaulted to the real ones
    so a test swaps them for scripts and a frozen clock."""
    runtimes: list[ContainerRuntime]
    reporter: Reporter
    prompter: Prompter
    config: ConfigSink
    stack_dir: Path
    runner: Runner
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
    url_by_service = {"qdrant": urls["qdrant_url"], "embed": urls["api_base_url"],
                      "rerank": urls["rerank_url"]}
    services = {name: {"url": url_by_service[name], "status": deps.status(url_by_service[name])}
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
        "config_points_here": _config_points_here(deps.config.effective(), st.ports),
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
    runtime, provider = runtime_for(st, deps.runtimes)
    if upgrade:
        resolved = catalog.resolve_images(images or {}, {})
        check_minor(st.qdrant_version, catalog.qdrant_version(resolved["qdrant"]))
    else:
        resolved = dict(st.images)
    file = deps.stack_dir / state.COMPOSE_FILE
    file.write_text(compose.dump(replace(state.plan_of(st, deps.stack_dir), images=resolved)))
    if upgrade:
        _compose(deps, runtime, provider, st.project, file, "pull", timeout=1800.0, stream=True)
    _compose(deps, runtime, provider, st.project, file, "up", "-d", timeout=600.0)
    try:
        health.wait_ready(health.endpoints(st.ports), status=deps.status,
                          clock=deps.clock, sleep=deps.sleep)
    except StackError:
        raise StackError("the stack is not ready", step="up",
                         fix="check the runtime's logs for that container") from None
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
    """The runtime and provider the STATE recorded, never a re-discovery.

    It picks the runtime named in `stack.json` from the ones the CLI discovered and uses
    the compose provider THAT runtime answers with. A recorded runtime that is no longer
    here, or whose provider has gone, is an error with a fix: re-running discovery to find
    a different provider than the stack was built with would change what `up` starts.
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


def _config_points_here(eff: Config, ports: dict[str, int]) -> bool:
    """Whether the effective config still points at the stack's own URLs -- the three that
    name a host, `embed_url` kept out because the stack always clears it to `""`."""
    urls = verify.stack_urls(ports)
    pairs = [("qdrant_url", eff.qdrant_url, urls["qdrant_url"]),
             ("api_base_url", eff.api_base_url, urls["api_base_url"]),
             ("rerank_url", eff.rerank_url, urls["rerank_url"])]
    return all(got == want for _name, got, want in pairs)


def _report_still_points_here(reporter: Reporter, eff: Config, ports: dict[str, int]) -> None:
    """`remove` never edits the config, so a config that pointed at the stack now points at
    nothing: each field that still carries the stack's URL is named with its value."""
    urls = verify.stack_urls(ports)
    pairs = [("qdrant_url", eff.qdrant_url, urls["qdrant_url"]),
             ("api_base_url", eff.api_base_url, urls["api_base_url"]),
             ("rerank_url", eff.rerank_url, urls["rerank_url"])]
    matching = [(name, want) for name, got, want in pairs if got == want]
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
