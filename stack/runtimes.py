"""Docker and Podman behind one runtime contract: the engine, the compose provider, the stats.

This module is where the stack meets a container engine. Everything else in `stack` talks to a
`ContainerRuntime`, never to a binary: `engine()` reports what the host runs, `compose_provider()`
picks the compose front-end that will actually work, `compose()` runs one compose command, and
`stats()` reads a container's memory. A new engine is one class, not a branch.

The command runner (`Completed`, `Runner`, `SubprocessRunner`) lives in `stack.process`; it is
re-exported here so the earlier tasks and the tests keep importing it from `stack.runtimes`
(review round R2, item m10).

The compose provider has two DISTINCT failure modes (review round R2):
  * `podman compose version` FAILS: podman looked for a compose provider, found none, and never
    touched the API socket. The cause is the missing provider; the fix is to install one.
  * `podman compose version` SUCCEEDS but runs an EXTERNAL docker-compose: that wrapper needs the
    live API socket. `podman info` reports `remoteSocket.exists: true` even when the socket file
    is not there (the lie logged as M3), so `exists` is never read; the only verdict trusted is
    `socket_alive`, which connects to the path with a short timeout. When the socket is dead and
    the wrapper would fail, `podman-compose` (standalone, needs no socket) is the fallback; when
    it is not installed, the fix starts the socket.
"""
import json
import os
import re
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from . import StackError
# Re-exported so `from stack.runtimes import Completed, Runner, SubprocessRunner` keeps working
# (R2 item m10): the runner itself lives in `stack.process`.
from .process import Completed, Runner, SubprocessRunner  # noqa: F401

#: `podman info` and the version commands are cheap; the bounds come from the plan's Global
#: Constraints (30 s for info/version).
_INFO_TIMEOUT = 30.0
#: `--format '{{json .}}'` works on every Docker CLI; the `--format json` shorthand exists only
#: since Docker 23, and an older CLI prints the literal word `json` instead.
_DOCKER_JSON = "{{json .}}"


def _as_which(which: Callable | Mapping[str, str]):
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
    """The only socket verdict the code trusts (see the module docstring). A connect is the test;
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


_BANNER = re.compile(r'Executing external compose provider "([^"]+)"')


def _docker_compose_via(stdout: str, stderr: str) -> bool:
    """Whether `podman compose` runs an EXTERNAL docker-compose (R2 item I1).

    Either signal suffices: the stderr banner names a `docker-compose` binary, or stdout starts
    with `Docker Compose version`. The banner can be switched off (`compose_warning_logs = false`
    in containers.conf, or `PODMAN_COMPOSE_WARNING_LOGS=false`), but the stdout line stays
    (measured 2026-10-06 with `PODMAN_COMPOSE_WARNING_LOGS=false podman compose version`).
    """
    banner = _BANNER.search(stderr)
    if banner is not None and os.path.basename(banner.group(1)).startswith("docker-compose"):
        return True
    return stdout.lstrip().startswith("Docker Compose version")


def _tool_failed(tool: str, out: Completed) -> StackError:
    """A failed `stats`: the tool's own first stderr line is the message (R2 item m5)."""
    first = next((line.strip() for line in out.stderr.splitlines() if line.strip()), "")
    message = f"{tool} stats failed"
    if first:
        message += f": {first}"
    return StackError(message, step="runtime")


class Docker:
    name = "docker"

    def __init__(self, runner: Runner, which=shutil.which):
        self.runner = runner
        self.which = _as_which(which)

    def engine(self) -> EngineInfo | None:
        if self.which("docker") is None:
            return None
        out = self.runner.run(["docker", "info", "--format", _DOCKER_JSON],
                              timeout=_INFO_TIMEOUT)
        if not out.ok:
            return None
        try:
            info = json.loads(out.stdout)
        except json.JSONDecodeError:
            return None  # an engine whose info is not JSON does not answer
        if not isinstance(info, dict):
            return None
        # R2 items C1 and I3: Docker CLI 23.0 to 28.0 exits 0 when the daemon does not answer,
        # printing zero-valued fields and "SecurityOptions": null; and a `docker` that is a
        # podman-docker shim prints podman's info, which has no ServerVersion. Neither is a
        # Docker engine answering, so the guard is the JSON, not the exit code: the engine
        # counts only when ServerVersion is non-empty and ServerErrors is empty.
        if not str(info.get("ServerVersion") or "").strip():
            return None
        if info.get("ServerErrors"):
            return None
        rootless = any("rootless" in opt for opt in (info.get("SecurityOptions") or []))
        # `OSType` is the engine's OS (`OperatingSystem` is a label such as "Docker
        # Desktop"); the kernel is what tells WSL apart (`platform_of`). `MemTotal` is what
        # the containers get (R2 item I6).
        return EngineInfo("docker", info.get("ServerVersion", ""), info.get("OSType", ""),
                          normalize_arch(info.get("Architecture", "")), rootless,
                          kernel=info.get("KernelVersion", ""),
                          memory_bytes=info.get("MemTotal"))

    def compose_provider(self) -> ProviderInfo:
        out = self.runner.run(["docker", "compose", "version"], timeout=_INFO_TIMEOUT)
        if out.ok:
            return ProviderInfo(Provider(("docker", "compose"), "docker compose",
                                         _compose_version(out.stdout)))
        if self.which("docker-compose") is not None:
            binary = self.runner.run(["docker-compose", "version"], timeout=_INFO_TIMEOUT)
            if binary.ok:
                return ProviderInfo(Provider(("docker-compose",), "docker-compose",
                                             _compose_version(binary.stdout)))
        return ProviderInfo(None,
                            problem="no docker compose provider answers "
                                    "(`docker compose version` failed and there is no "
                                    "`docker-compose` binary)",
                            fix="install the Docker Compose plugin (docker-compose-plugin)")

    def compose(self, provider: Provider, project: str, file: Path, *args: str,
                timeout: float, stream: bool = False) -> Completed:
        return self.runner.run(_compose_argv(provider, project, file, args),
                               timeout=timeout, stream=stream)

    def stats(self, names: list[str]) -> dict[str, int]:
        if not names:
            return {}
        out = self.runner.run(["docker", "stats", "--no-stream", "--format", _DOCKER_JSON,
                               *names],
                              timeout=_INFO_TIMEOUT)
        if not out.ok:
            raise _tool_failed("docker", out)
        # one JSON object per line
        result: dict[str, int] = {}
        try:
            rows = [json.loads(line) for line in out.stdout.splitlines()
                    if line.strip()]
        except json.JSONDecodeError:
            raise StackError(
                "docker stats did not return JSON", step="runtime",
                fix="check the containers are running and try again") from None
        for row in rows:
            mem = parse_size(_before_slash(row.get("MemUsage", "")))
            if mem is not None:
                result[row["Name"]] = mem
        return result


class Podman:
    name = "podman"

    def __init__(self, runner: Runner, which=shutil.which, host_system: str = "linux",
                 socket_alive=socket_alive):
        self.runner = runner
        self.which = _as_which(which)
        self.host_system = host_system
        self.socket_alive = socket_alive

    def engine(self) -> EngineInfo | None:
        if self.which("podman") is None:
            return None
        info = self._json(["podman", "info", "--format", "json"])
        if not isinstance(info, dict):
            return None  # no answer, or an answer that is not JSON
        host = info.get("host", {})
        vm = None
        socket_path = host.get("remoteSocket", {}).get("path")
        if self.host_system == "macos":
            # On a Mac `podman info` answers from INSIDE the VM, so its socket is the
            # VM's path; the VM type and the host side socket come from the machine.
            vm, socket_path = self._machine()
        return EngineInfo("podman", info.get("version", {}).get("Version", ""),
                          host.get("os", ""), normalize_arch(host.get("arch", "")),
                          host.get("security", {}).get("rootless", False),
                          kernel=host.get("kernel", ""), vm=vm, socket=socket_path,
                          memory_bytes=host.get("memTotal"))

    def _machine(self) -> tuple[str | None, str | None]:
        """The VM type and the host side API socket of the machine in use (macOS).

        Read in the podman source at v5.7.0 and v6.0.0: `podman machine inspect` carries
        no VMType; `podman machine info` names the provider (`Host.VMType`) and the
        machine in use (`Host.CurrentMachine`); and the socket `podman compose` hands
        docker-compose is that machine's `ConnectionInfo.PodmanSocket.Path`.
        """
        info = self._json(["podman", "machine", "info", "--format", "json"])
        host = info.get("Host") if isinstance(info, dict) else None
        if not isinstance(host, dict):
            return None, None
        vm = str(host.get("VMType") or "").lower() or None
        name = host.get("CurrentMachine") or ""
        if not name:
            return vm, None
        machines = self._json(["podman", "machine", "inspect", name])
        if not (isinstance(machines, list) and machines and isinstance(machines[0], dict)):
            return vm, None
        podman_socket = (machines[0].get("ConnectionInfo") or {}).get("PodmanSocket") or {}
        return vm, podman_socket.get("Path") or None

    def _json(self, argv: list[str]):
        """The JSON a podman command prints, or None when it fails or prints something else."""
        out = self.runner.run(argv, timeout=_INFO_TIMEOUT)
        if not out.ok:
            return None
        try:
            return json.loads(out.stdout)
        except json.JSONDecodeError:
            return None

    def compose_provider(self) -> ProviderInfo:
        out = self.runner.run(["podman", "compose", "version"], timeout=_INFO_TIMEOUT)
        if not out.ok:
            # R2 item I2: a failing `version` means podman found no compose provider at all.
            # podman looks the provider up before `version` runs, and `version` never touches
            # the API socket, so the socket is not the cause here.
            return self._no_provider_or_standalone(out.stderr)
        if _docker_compose_via(out.stdout, out.stderr):
            # the docker-compose wrapper needs the live API socket; a native provider does not
            socket_path = self._socket_path()
            if not self.socket_alive(socket_path):
                return self._socket_problem(socket_path)
        return ProviderInfo(Provider(("podman", "compose"), "podman compose",
                                     _compose_version(out.stdout)))

    def _socket_path(self) -> str | None:
        engine = self.engine()
        return engine.socket if engine else None

    def _no_provider_or_standalone(self, stderr: str) -> ProviderInfo:
        """`podman compose version` failed: no compose provider. Try the standalone
        `podman-compose`; when that is absent too, the fix is to install one (R2 item I2)."""
        if self.which("podman-compose") is not None:
            out = self.runner.run(["podman-compose", "version"], timeout=_INFO_TIMEOUT)
            if out.ok:
                return ProviderInfo(Provider(("podman-compose",), "podman-compose",
                                             _compose_version(out.stdout)))
        first = next((line.strip() for line in stderr.splitlines() if line.strip()), "")
        problem = "podman compose found no compose provider"
        if first:
            problem += f": {first}"
        return ProviderInfo(None, problem=problem,
                            fix="install podman-compose (or docker-compose)")

    def _socket_problem(self, socket_path: str | None) -> ProviderInfo:
        """The wrapper is docker-compose and the API socket is dead: try the standalone
        `podman-compose`; when it is absent too, the fix starts the socket (M3)."""
        if self.which("podman-compose") is not None:
            out = self.runner.run(["podman-compose", "version"], timeout=_INFO_TIMEOUT)
            if out.ok:
                return ProviderInfo(Provider(("podman-compose",), "podman-compose",
                                             _compose_version(out.stdout)),
                                    note="podman compose runs docker-compose, which needs the "
                                         "API socket; falling back to the standalone "
                                         "podman-compose")
        problem = ("podman compose needs the API socket at " + str(socket_path)) \
            if socket_path else "podman compose needs the API socket"
        fix = ("systemctl --user enable --now podman.socket" if self.host_system == "linux"
               else "podman machine start")
        return ProviderInfo(None, problem=problem, fix=fix)

    def compose(self, provider: Provider, project: str, file: Path, *args: str,
                timeout: float, stream: bool = False) -> Completed:
        return self.runner.run(_compose_argv(provider, project, file, args),
                               timeout=timeout, stream=stream)

    def stats(self, names: list[str]) -> dict[str, int]:
        if not names:
            return {}
        out = self.runner.run(["podman", "stats", "--no-stream", "--format", "json", *names],
                              timeout=_INFO_TIMEOUT)
        if not out.ok:
            raise _tool_failed("podman", out)
        # a JSON list of objects (lowercase keys), unlike docker's one-object-per-line
        try:
            rows = json.loads(out.stdout)
        except json.JSONDecodeError:
            raise StackError(
                "podman stats did not return JSON", step="runtime",
                fix="check the containers are running and try again") from None
        result: dict[str, int] = {}
        for row in rows:
            mem = parse_size(_before_slash(row.get("mem_usage", "")))
            if mem is not None:
                result[row["name"]] = mem
        return result


def _compose_argv(provider: Provider, project: str, file: Path, args: tuple[str, ...]) -> list[str]:
    return [*provider.argv, "-p", project, "-f", str(file), *args]


def _compose_version(stdout: str) -> str:
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


def _before_slash(text: str) -> str:
    # "190.7MB / 132.5GB" -> "190.7MB": the used part, not the total
    return text.split(" / ", 1)[0]


def discover(runner: Runner, which=shutil.which,
             host_system: str = "linux") -> list["ContainerRuntime"]:
    """Docker before Podman, only the engines whose binary exists and answers `info`. A binary
    present but whose `info` does not answer, or hangs, is skipped, not an error: the other
    engine may work (R2 item m1)."""
    which = _as_which(which)
    found: list[ContainerRuntime] = []
    for factory in (lambda: Docker(runner, which=which),
                    lambda: Podman(runner, which=which, host_system=host_system)):
        runtime = factory()
        try:
            engine = runtime.engine()
        except StackError:
            engine = None  # a hung `info` (timeout) is not an answering engine
        if engine is not None:
            found.append(runtime)
    return found
