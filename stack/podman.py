"""Podman behind the runtime contract of `stack.engine`: the compose provider choice, the API
socket, and the macOS machine.

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
from pathlib import Path

from . import StackError
from .engine import (INFO_TIMEOUT, EngineInfo, Provider, ProviderInfo, as_which, before_slash,
                     compose_argv, compose_version, normalize_arch, parse_size, socket_alive,
                     stats_failed)
from .process import Completed, Runner

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


class Podman:
    name = "podman"

    def __init__(self, runner: Runner, which=shutil.which, host_system: str = "linux",
                 socket_alive=socket_alive):
        self.runner = runner
        self.which = as_which(which)
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
        out = self.runner.run(argv, timeout=INFO_TIMEOUT)
        if not out.ok:
            return None
        try:
            return json.loads(out.stdout)
        except json.JSONDecodeError:
            return None

    def compose_provider(self) -> ProviderInfo:
        out = self.runner.run(["podman", "compose", "version"], timeout=INFO_TIMEOUT)
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
                                     compose_version(out.stdout)))

    def _socket_path(self) -> str | None:
        engine = self.engine()
        return engine.socket if engine else None

    def _no_provider_or_standalone(self, stderr: str) -> ProviderInfo:
        """`podman compose version` failed: no compose provider. Try the standalone
        `podman-compose`; when that is absent too, the fix is to install one (R2 item I2)."""
        if self.which("podman-compose") is not None:
            out = self.runner.run(["podman-compose", "version"], timeout=INFO_TIMEOUT)
            if out.ok:
                return ProviderInfo(Provider(("podman-compose",), "podman-compose",
                                             compose_version(out.stdout)))
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
            out = self.runner.run(["podman-compose", "version"], timeout=INFO_TIMEOUT)
            if out.ok:
                return ProviderInfo(Provider(("podman-compose",), "podman-compose",
                                             compose_version(out.stdout)),
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
        return self.runner.run(compose_argv(provider, project, file, args),
                               timeout=timeout, stream=stream)

    def stats(self, names: list[str]) -> dict[str, int]:
        if not names:
            return {}
        out = self.runner.run(["podman", "stats", "--no-stream", "--format", "json", *names],
                              timeout=INFO_TIMEOUT)
        if not out.ok:
            raise stats_failed("podman", out)
        # a JSON list of objects (lowercase keys), unlike docker's one-object-per-line
        try:
            rows = json.loads(out.stdout)
        except json.JSONDecodeError:
            raise StackError(
                "podman stats did not return JSON", step="runtime",
                fix="check the containers are running and try again") from None
        result: dict[str, int] = {}
        for row in rows:
            mem = parse_size(before_slash(row.get("mem_usage", "")))
            if mem is not None:
                result[row["name"]] = mem
        return result
