"""The stack's compose file: rendered from the catalogue and a plan, written by an emitter of
its own.

What it is FOR: one `compose.yaml` per install, read the same way by Docker and Podman, where
the only thing a platform, runtime or profile changes is what the backend's `service_patch`
adds. `render` builds the document as a plain dict and is pure (no file, no runtime); `emit`
writes it; `dump` is the two together. The golden fixtures in `tests/fixtures/stack/` are `dump`
of each combination phase 1 serves, and they are the per-OS, per-hardware compose the
repository shows.

WHY AN EMITTER AND NOT A YAML LIBRARY: the plugin is stdlib-only. The emitter writes block style
only, and every scalar goes through `json.dumps`, so a value is a double-quoted string, a number,
`true`, `false` or `null`, and never a plain word for the parser to guess at. podman-compose
reads this file with PyYAML, a YAML 1.1 parser, where a plain `no` is False and a plain `22:22`
is the integer 1342 (measured). A key is written bare only when it is an identifier and not one
of YAML 1.1's bool or null words; every other key goes through `json.dumps` too, which is why
`"run.oci.keep_original_groups"` and `"mnemosine-qdrant"` come out quoted.

THE NAMES (M6): docker-compose names a container `<project>-<service>-1` and podman-compose
`<project>_<service>_1`, so every service sets `container_name`, and whatever reads stats or
logs later does not have to know which provider created it. Both providers prefix the named
volume with the project, so the Qdrant data lives in `volume_name(project)`, not in `VOLUME`.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import StackError, catalog
from .backends import BACKENDS

#: The keys of a service, in the order they are written. A key is written only with a value.
_SERVICE_KEYS = ("container_name", "image", "restart", "command", "ports", "volumes",
                 "environment", "devices", "deploy", "annotations", "cap_drop", "security_opt")

_BARE_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: The YAML 1.1 type registry's bool and null words. A key spelled like one is quoted even
#: though it is an identifier, or a 1.1 parser may read it as a bool or as null.
_YAML11_BOOL_AND_NULL = frozenset(
    "y Y yes Yes YES n N no No NO true True TRUE false False FALSE "
    "on On ON off Off OFF null Null NULL ~".split())


@dataclass(frozen=True)
class Plan:
    """What one compose file is rendered from. `render` reads every field but `platform`,
    which names where the plan runs and changes nothing in a phase-1 file.

    `device` is the `-dev` value. None is `-dev none`, which the cpu profile always carries
    (M1); a GPU profile is not refused without one, because the plan's probe step renders a
    profile before `--list-devices` has named its device. `gpu_index` is the nvidia-smi
    index the nvidia profile needs.
    """
    platform: str
    runtime: str
    backend: str
    device: str | None
    gpu_index: int | None
    ports: dict[str, int]
    stack_dir: Path
    images: dict[str, str]
    selinux: bool = False
    project: str = catalog.PROJECT


def container_name(project: str, service: str) -> str:
    return f"{project}-{service}"


def volume_name(project: str) -> str:
    """The Qdrant volume as the engine names it: the file declares `VOLUME` and the
    provider prefixes the project (M6)."""
    return f"{project}_{catalog.VOLUME}"


def render(plan: Plan) -> dict:
    """The compose document of `plan`, keys in the order they are written."""
    _check_stack_dir(plan.stack_dir)
    services = {}
    for service in catalog.SERVICES:
        fields = _qdrant(plan) if service == "qdrant" else _server(plan, service)
        services[service] = {key: fields[key] for key in _SERVICE_KEYS if key in fields}
    return {"services": services, "volumes": {catalog.VOLUME: {}}}


#: One fix for every refused directory: the check below is the whole policy.
_STACK_DIR_FIX = ("set QCTX_STACK_DIR to an absolute path without ':', '$' or characters "
                  "outside the basic multilingual plane")


def _check_stack_dir(stack_dir: Path) -> None:
    r"""Refuse a `stack_dir` the short volume syntax `<host>:/models:ro` cannot carry in both
    providers.

    The emitter writes the path through `json.dumps`, which escapes every non-ASCII
    character as a `\uXXXX` sequence; the providers read those escapes back differently,
    so every one of these was measured on throwaway compose files (both providers,
    `config` only, never `up`):
    2026-10-06 (R2-9): a `:` anywhere breaks the parse in BOTH; a code point above 0xFFFF
    is rejected by docker-compose and garbled by podman-compose. 2026-10-07 (R4):
    `docker-compose v5.2.0` and `podman-compose 1.6.0` interpolate `$VAR` and `${VAR}` in the
    host path (`/x/$HOME/stack` resolved to the real home in both, exit 0, no warning;
    `a$zzqq` lost the variable name; `a$$b` stayed `a$$b` in docker-compose but became
    `a$b` in podman-compose, so even writing `$$` is not portable); a relative path fails with
    a misleading "undefined volume rel/stack/models" in both; a leading `~` is expanded by
    docker-compose and kept literally by podman-compose; a lone surrogate (surrogateescape
    for a non-UTF-8 byte) is rejected by docker-compose and kept by podman-compose.
    Accented characters inside the BMP work in both and stay accepted.
    """
    path = str(stack_dir)
    if not path.startswith("/"):
        # A relative or `~` path is not the directory the user means: docker-compose
        # expands a leading `~` and a relative path becomes a project volume.
        raise StackError(
            f"the stack directory {path!r} is not absolute",
            step="compose", fix=_STACK_DIR_FIX)
    for char in path:
        code = ord(char)
        if char == ":" or char == "$" or 0xD800 <= code <= 0xDFFF or code > 0xFFFF:
            # ':' and '$' are special to the volume syntax and to provider
            # interpolation; surrogates and non-BMP code points break the file in one
            # provider or the other.
            raise StackError(
                f"the stack directory {path!r} cannot be used as a compose source path",
                step="compose", fix=_STACK_DIR_FIX)


def dump(plan: Plan) -> str:
    """The `compose.yaml` text of `plan`."""
    return emit(render(plan))


def _common(plan: Plan, service: str) -> dict:
    # 127.0.0.1 is what keeps the servers off the network: inside the container they
    # listen on 0.0.0.0, which the port mapping requires.
    port = f"127.0.0.1:{plan.ports[service]}:{catalog.CONTAINER_PORTS[service]}"
    return {"container_name": container_name(plan.project, service),
            "restart": "always",
            "ports": [port],
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"]}


def _qdrant(plan: Plan) -> dict:
    return {**_common(plan, "qdrant"),
            "image": plan.images["qdrant"],
            "volumes": [f"{catalog.VOLUME}:/qdrant/storage"],
            "environment": {"QDRANT__TELEMETRY_DISABLED": "true"}}


def _server(plan: Plan, role: str) -> dict:
    label = "ro,z" if plan.selinux else "ro"
    fields = {**_common(plan, role),
              "image": plan.images["llama"],
              "command": catalog.server_command(role, plan.device),
              "volumes": [f"{plan.stack_dir}/models:/models:{label}"]}
    patch = BACKENDS[plan.backend].service_patch(plan.runtime, plan.gpu_index)
    for key in patch:
        # Merged over the base, the key would replace the command or the models mount;
        # left out of the fixed order, it would vanish from the file without a word.
        if key in fields or key not in _SERVICE_KEYS:
            raise StackError(f"the {plan.backend} profile cannot set {key!r} on a service",
                             step="compose")
    return {**fields, **patch}


def emit(doc: dict) -> str:
    """`doc` as block YAML: two-space indent, list items as `- item`, every scalar through
    `json.dumps`, and an empty map or list in flow form, because a bare `key:` is null."""
    if not doc:
        return "{}\n"
    return "".join(f"{line}\n" for line in _mapping(doc, 0))


def _mapping(mapping: dict, depth: int):
    pad = "  " * depth
    for key, value in mapping.items():
        head = f"{pad}{_key(key)}:"
        if isinstance(value, (dict, list)) and value:
            yield head
            yield from _block(value, depth + 1)
        else:
            yield f"{head} {_flow(value)}"


def _sequence(items: list, depth: int):
    pad = "  " * depth
    for item in items:
        if isinstance(item, (dict, list)) and item:
            # the item's first line takes the dash in place of its own indent
            first, *rest = _block(item, depth + 1)
            yield f"{pad}- {first[len(pad) + 2:]}"
            yield from rest
        else:
            yield f"{pad}- {_flow(item)}"


def _block(value, depth: int):
    return _mapping(value, depth) if isinstance(value, dict) else _sequence(value, depth)


def _flow(value) -> str:
    """What goes on the line of a key or a dash: a scalar, or an EMPTY map or list."""
    if isinstance(value, dict):
        return "{}"
    if isinstance(value, list):
        return "[]"
    return json.dumps(value)


def _key(key: str) -> str:
    if _BARE_KEY.fullmatch(key) and key not in _YAML11_BOOL_AND_NULL:
        return key
    return json.dumps(key)
