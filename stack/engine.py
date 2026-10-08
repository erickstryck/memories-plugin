"""The runtime contract, and what every engine shares.

Everything else in `stack` talks to a `ContainerRuntime`, never to a binary: `engine()` reports
what the host runs, `compose_provider()` picks the compose front-end that will actually work,
`compose()` runs one compose command, and `stats()` reads a container's memory. A new engine is
one module and one class (`stack.docker`, `stack.podman`), not a branch.

The rest of this module is what both engines share: the binary lookup, the socket verdict, the
size parser, and the compose helpers. The helpers carry no leading underscore because the engine
modules import them (review round R3, item R3-7).
"""
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from . import StackError
from .process import Completed

#: `podman info` and the version commands are cheap; the bounds come from the plan's Global
#: Constraints (30 s for info/version).
INFO_TIMEOUT = 30.0


def as_which(which: Callable | Mapping[str, str]):
    """The binary lookup as a callable `name -> path | None`. `shutil.which` is the production
    lookup; a mapping is a test double (`dict.get` already answers `None` for a missing name)."""
    if callable(which):
        return which
    return which.get


def normalize_arch(machine: str) -> str:
    """The engine's machine name folded to the OCI arch. `x86_64` and `amd64` are the same arch."""
    name = machine.lower()
    if name in ("x86_64", "amd64"):
        return "amd64"
    if name in ("aarch64", "arm64"):
        return "arm64"
    return name


@dataclass(frozen=True)
class EngineInfo:
    name: str
    version: str
    os: str
    arch: str
    rootless: bool
    kernel: str = ""
    vm: str | None = None
    socket: str | None = None
    #: The memory the CONTAINERS actually get, in bytes. On macOS that is the machine VM's
    #: (2048 MiB by default), not the Mac's, so the engine reports it, not `hw.memsize`
    #: (R2 item I6). None when the engine does not publish a figure.
    memory_bytes: int | None = None


@dataclass(frozen=True)
class Provider:
    argv: tuple[str, ...]
    name: str
    version: str


@dataclass(frozen=True)
class ProviderInfo:
    provider: Provider | None
    problem: str | None = None
    fix: str | None = None
    note: str | None = None


class ContainerRuntime(Protocol):
    name: str

    def engine(self) -> EngineInfo | None: ...

    def compose_provider(self) -> ProviderInfo: ...

    def compose(self, provider: Provider, project: str, file: Path, *args: str,
                timeout: float, stream: bool = False) -> Completed: ...

    def stats(self, names: list[str]) -> dict[str, int]: ...


def socket_alive(path: str | None, timeout: float = 2.0) -> bool:
    """The only socket verdict the code trusts (see `stack.podman`). A connect is the test;
    `podman info`'s `exists` field is not read because it lies when the socket file is gone."""
    if not path:
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(timeout)
    try:
        probe.connect(path)
        return True
    except OSError:
        return False
    finally:
        probe.close()


_SIZE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*$")
#: decimal units (podman stats reports `190.7MB` as 10^6) and binary units (`2.0GiB` as 2^30)
_UNITS = {"B": 1, "KB": 10**3, "K": 10**3, "MB": 10**6, "M": 10**6, "GB": 10**9, "G": 10**9,
          "TB": 10**12, "T": 10**12,
          "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}


def parse_size(text: str) -> int | None:
    """A `podman stats`/`docker stats` memory figure in bytes, or None when it is not a figure."""
    match = _SIZE.match(text)
    if match is None:
        return None
    value, unit = match.groups()
    factor = _UNITS.get(unit or "B")
    if factor is None:
        return None
    return int(float(value) * factor)


def first_line(text: str) -> str:
    """The first line of a tool's output that is not blank, stripped; "" when there is none."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def stats_failed(tool: str, out: Completed) -> StackError:
    """A failed `stats`: the tool's own first stderr line is the message (R2 item m5)."""
    message = f"{tool} stats failed"
    first = first_line(out.stderr)
    if first:
        message += f": {first}"
    return StackError(message, step="runtime")


def compose_argv(provider: Provider, project: str, file: Path, args: tuple[str, ...]) -> list[str]:
    return [*provider.argv, "-p", project, "-f", str(file), *args]


def log_tail(runtime: "ContainerRuntime", project: str, file: Path, service: str,
             provider: Provider | None = None) -> str:
    """The last 50 lines of one service's log: the evidence a failed start shows,
    in the installer and in `qctx stack up` alike (one copy, so the two cannot
    drift). The tail is evidence, not a dependency, and it must never mask the
    readiness error the caller is about to raise:
    - a `logs` that exits non-zero still has output, and that output is the tail
      (the provider's own complaint); with no output at all it reads "(no log)";
    - a call that RAISES -- the provider lookup (`podman compose version`) or the
      `logs` itself hanging past its bound, which the runner turns into a
      StackError -- reads "(no log)" too.
    `provider` is the one the caller already resolved; without it, it is looked
    up here, inside the same guard."""
    try:
        if provider is None:
            provider = runtime.compose_provider().provider
        if provider is None:
            return "(no log)"
        out = runtime.compose(provider, project, file, "logs", "--tail", "50", service,
                              timeout=60.0)
    except StackError:
        return "(no log)"
    return out.stdout.strip() or out.stderr.strip() or "(no log)"


def compose_version(stdout: str) -> str:
    """The token after `version` on the line that names compose: "Docker Compose version
    v5.2.0", or the second line of podman-compose's "podman version 5.7.0\\npodman-compose
    version 1.6.0" (measured here: the podman line comes first)."""
    for line in stdout.splitlines():
        parts = line.split()
        if "compose" in line.lower() and "version" in parts:
            index = parts.index("version")
            if index + 1 < len(parts):
                return parts[index + 1].rstrip(",")
    return ""


def before_slash(text: str) -> str:
    # "190.7MB / 132.5GB" -> "190.7MB": the used part, not the total
    return text.split(" / ", 1)[0]
