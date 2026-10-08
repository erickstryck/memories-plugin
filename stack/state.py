"""The persisted state of one stack install: `stack.json` in the stack directory.

This module owns the FILE and its schema -- the constants, the `StackState`
dataclass, where the stack directory lives, the tolerant `load`, the atomic
`save`, and the `plan_of` mapping that hands a stored state back to `compose`.
It does not render, and it does not touch a runtime: `plan_of` builds the `Plan`
dataclass from the stored fields and stops; `render` is compose's job.

LOAD IS TOLERANT OF ABSENCE, NOT OF UNTRUSTWORTHINESS. No file means no managed
stack (`None`): `qctx stack status` says so and exits clean. But a file that is
not valid JSON, that has the wrong shape, or that carries a `schema` this code
has never seen is a NAMED error, not a guess, because the two callers that read
state most often are `remove` (which must still clean up a corrupt install,
review focus 4) and `status` (which must report it). Both therefore get the same
`fix`, `qctx stack remove`, which points the operator at the one command that
deletes the file and can repair the rest.

SAVE IS ATOMIC AND OWNER-ONLY, and it raises when the bytes do not land.
`core.statefile.write_json` publishes the file behind a rename, so a reader never
sees it half-written, and it creates the directory and the file owner-only
(`0o700` / `0o600`). That module reports a failed write as `False` rather than
raising, and `save` turns that `False` into a `StackError` here: a state that did
not land cannot be reported as saved, because `up` without `--upgrade` repeats
exactly what `stack.json` holds.

THE DIRECTORY PRECEDENCE is `$QCTX_STACK_DIR`, then
`${XDG_DATA_HOME:-~/.local/share}/mnemosine/stack` (spec, "Portas, caminhos
e nomes"). A blank value counts as absent, the way `core.config` reads every
environment knob, so `QCTX_STACK_DIR="  "` does not silently become an empty
path. The result is `~`-expanded and made absolute: `compose.render` refuses a
non-absolute source path, so the directory is settled here, before anything
renders it.
"""
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import core.statefile as statefile
from . import StackError, catalog
from .compose import Plan

#: The schema version of `stack.json`. A file at a HIGHER version is refused, not read:
#: a newer field would be a guess, and reading it would carry an unknown shape forward
#: into `plan_of` and `render`.
SCHEMA = 1

STATE_FILE = "stack.json"
COMPOSE_FILE = "compose.yaml"
MODELS_DIR = "models"

#: The phases a `StackState` sits in, in the order `up` moves through them: `compose`
#: holds the file written but the containers not yet running, `running` is the verified
#: stack, and `stopped` is what `down` leaves (volume and models intact).
PHASE_COMPOSE = "compose"
PHASE_RUNNING = "running"
PHASE_STOPPED = "stopped"

#: The one fix for every untrustworthy file: the single command that deletes the file
#: and can repair the rest. The message names what is wrong; the fix stays uniform.
_REMOVED_FIX = "qctx stack remove"


def now() -> str:
    """The current UTC time as ISO-8601 with a trailing `Z`, for `created_at`/`updated_at`."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class StackState:
    """What `qctx stack up` repeats when it runs without `--upgrade`: the exact install
    that was verified, stored so a re-run rebuilds the same endpoints.

    `role` is `local` in phase 1 (`server` arrives in phase 2); `listen` is the host the
    endpoints are published on (the bare address, not a `host:port`: the ports already live
    in `ports`, one per service, and the phase-2 exposure classifier reads hosts). `profile`
    is the backend chosen at the menu; `device` the
    `-dev` value (None on cpu, which always pins `-dev none`); `gpu_index` the nvidia-smi
    index the nvidia profile needs. `provider` is the list of compose providers the runtime
    answered with, in the order the lifecycle uses them. `images` and `models` are what was
    actually used (a `--image` override beats the catalogue), and `qdrant_version` is the
    installed `x.y.z` that the minor guard on `up --upgrade` refuses to skip past.
    """
    role: str
    listen: str
    platform: str
    runtime: str
    provider: list[str]
    profile: str
    device: str | None
    gpu_index: int | None
    ports: dict[str, int]
    images: dict[str, str]
    models: dict[str, str]
    qdrant_version: str
    selinux: bool
    phase: str
    created_at: str
    updated_at: str
    project: str = catalog.PROJECT
    schema: int = SCHEMA


def stack_dir(env: Mapping[str, str] = os.environ) -> Path:
    """The directory that holds `stack.json`, `compose.yaml` and `models/`.

    Precedence: `$QCTX_STACK_DIR`, then
    `${XDG_DATA_HOME:-~/.local/share}/mnemosine/stack`. A blank value counts
    as absent, so it falls through to the next source rather than becoming an empty
    path. The result is `~`-expanded and made absolute: `compose.render` refuses a
    non-absolute source path, so the directory is settled here, before anything
    renders it. A leading `~` in the override expands against the same `HOME` the
    fallback uses, which is what the injected `env` names.
    """
    override = (env.get("QCTX_STACK_DIR") or "").strip()
    if override:
        # A `~` in the override expands against the env's `HOME`; a plain path does not
        # need a home at all, so only the tilde case may raise for a missing `HOME`.
        return _home_of(override, _home(env)) if override.startswith("~") \
            else Path(override).absolute()
    xdg = (env.get("XDG_DATA_HOME") or "").strip()
    if xdg:
        return Path(xdg, catalog.PROJECT, "stack").absolute()
    return (_home(env) / ".local" / "share" / catalog.PROJECT / "stack").absolute()


def _home(env: Mapping[str, str]) -> Path:
    """The home the fallback is built from. `HOME` must name a place: without it there is
    no `~/.local/share` to fall back to, and reaching for the real environment would break
    the hermetic test that injects `env` (see `tests/isolation.py`)."""
    home = (env.get("HOME") or "").strip()
    if not home:
        raise StackError(
            "no stack directory: QCTX_STACK_DIR, XDG_DATA_HOME and HOME are all absent",
            step="state", fix="set QCTX_STACK_DIR to an absolute path")
    return Path(home)


def _home_of(value: str, home: Path) -> Path:
    """A `~`-prefixed path resolved against `home`: `~` alone is `home`, `~/...` is
    `home/...`. The caller only passes `~` values (see `stack_dir`), and `home` is
    already absolute, so the result is absolute too."""
    if value == "~":
        return home
    return home / value[2:]


def load(directory: Path) -> StackState | None:
    """The state at `directory/stack.json`, or `None` when the file is not there.

    Raises `StackError(step="state")` when the file is present but untrustworthy --
    not valid JSON, the wrong shape, or a schema this code has not seen -- because the
    callers that hit one (`status`, `remove`) must name the file and the fix rather
    than act on a guess.
    """
    path = Path(directory) / STATE_FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StackError(f"{path} is not valid JSON: {exc}",
                         step="state", fix=_REMOVED_FIX) from exc
    return _coerce(payload, path)


def _coerce(payload: object, path: Path) -> StackState:
    """A parsed `stack.json` payload into a `StackState`, or a named error.

    Each check names the file, because the caller that hits one is `status` or
    `remove` and its output is the only thing the operator sees.
    """
    if not isinstance(payload, dict):
        raise StackError(f"{path} is not a JSON object; it may be corrupt",
                         step="state", fix=_REMOVED_FIX)
    _check_schema(payload.get("schema", SCHEMA), path)
    _check_containers(payload, path)
    try:
        return StackState(**payload)
    except TypeError as exc:
        # A missing required field, an unknown field, or a field the dataclass cannot
        # take: the file is not the state this code reads, so it is corrupt.
        raise StackError(f"{path} is not a stack.json state: {exc}",
                         step="state", fix=_REMOVED_FIX) from exc


def _check_schema(schema: object, path: Path) -> None:
    """That `schema` is the version this code reads. A newer one is refused by name (a
    guess is worse than no read); any other value is corrupt."""
    if not isinstance(schema, int) or isinstance(schema, bool):
        raise StackError(f"{path} has no integer schema; it may be corrupt",
                         step="state", fix=_REMOVED_FIX)
    if schema > SCHEMA:
        raise StackError(
            f"{path} was written by a newer version (schema {schema}); "
            f"this version reads schema {SCHEMA}",
            step="state", fix=_REMOVED_FIX)
    if schema < SCHEMA:
        raise StackError(f"{path} has an older schema ({schema}); it may be corrupt",
                         step="state", fix=_REMOVED_FIX)


#: The four container fields of `StackState`, by the container they must hold. `ports`
#: and `images` are what `render` indexes; `models` and `provider` are what the config
#: patch and the lifecycle read. Validating their shape at load keeps a hand-typed file
#: from building a state whose shape lies to those readers.
_CONTAINERS = {"ports": dict, "images": dict, "models": dict, "provider": list}


def _check_containers(payload: dict, path: Path) -> None:
    for name, kind in _CONTAINERS.items():
        value = payload.get(name)
        if not isinstance(value, kind):
            raise StackError(
                f"{path} has a wrong shape: {name} must be a {kind.__name__}",
                step="state", fix=_REMOVED_FIX)
    # ports is a mapping to integers, the only place the value type matters to render.
    if any(not isinstance(v, int) or isinstance(v, bool) for v in payload["ports"].values()):
        raise StackError(f"{path} has a wrong shape: ports must map to integers",
                         step="state", fix=_REMOVED_FIX)


def save(directory: Path, state: StackState) -> None:
    """Writes `state` to `directory/stack.json` atomically, owner-only.

    Raises `StackError(step="state")` when the write does not land: `write_json` reports
    a failure as `False` instead of raising, and a state that did not land cannot be
    reported as saved, because `up` without `--upgrade` repeats it.
    """
    path = Path(directory) / STATE_FILE
    if not statefile.write_json(path, asdict(state), make_parents=True):
        raise StackError(f"could not write {path}; the state was not saved",
                         step="state", fix="make the stack directory writable")


def plan_of(state: StackState, directory: Path) -> Plan:
    """The `compose.Plan` that re-renders a stored state: `backend` is the stored
    `profile`, `stack_dir` is the directory argument, and every other field is carried
    through. This builds the dataclass and stops; `render` is compose's job."""
    return Plan(platform=state.platform, runtime=state.runtime, backend=state.profile,
                device=state.device, gpu_index=state.gpu_index, ports=state.ports,
                stack_dir=Path(directory), images=state.images, selinux=state.selinux,
                project=state.project)
