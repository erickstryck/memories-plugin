"""Docker behind the runtime contract of `stack.engine`.

What is Docker's own: `docker info --format '{{json .}}'`, where the engine counts only when the
JSON says a daemon answered (the exit code does not say it, see `Docker.engine`); the compose
plugin before the standalone `docker-compose`; and `docker stats`, one JSON object per line.
"""
import json
import shutil
from pathlib import Path

from . import StackError
from .engine import (INFO_TIMEOUT, EngineInfo, Provider, ProviderInfo, as_which, before_slash,
                     compose_argv, compose_version, normalize_arch, parse_size, stats_failed)
from .process import Completed, Runner

#: `--format '{{json .}}'` works on every Docker CLI; the `--format json` shorthand exists only
#: since Docker 23, and an older CLI prints the literal word `json` instead.
_DOCKER_JSON = "{{json .}}"


class Docker:
    name = "docker"

    def __init__(self, runner: Runner, which=shutil.which):
        self.runner = runner
        self.which = as_which(which)

    def engine(self) -> EngineInfo | None:
        if self.which("docker") is None:
            return None
        out = self.runner.run(["docker", "info", "--format", _DOCKER_JSON],
                              timeout=INFO_TIMEOUT)
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
        out = self.runner.run(["docker", "compose", "version"], timeout=INFO_TIMEOUT)
        if out.ok:
            return ProviderInfo(Provider(("docker", "compose"), "docker compose",
                                         compose_version(out.stdout)))
        if self.which("docker-compose") is not None:
            binary = self.runner.run(["docker-compose", "version"], timeout=INFO_TIMEOUT)
            if binary.ok:
                return ProviderInfo(Provider(("docker-compose",), "docker-compose",
                                             compose_version(binary.stdout)))
        return ProviderInfo(None,
                            problem="no docker compose provider answers "
                                    "(`docker compose version` failed and there is no "
                                    "`docker-compose` binary)",
                            fix="install the Docker Compose plugin (docker-compose-plugin)")

    def compose(self, provider: Provider, project: str, file: Path, *args: str,
                timeout: float, stream: bool = False) -> Completed:
        return self.runner.run(compose_argv(provider, project, file, args),
                               timeout=timeout, stream=stream)

    def stats(self, names: list[str]) -> dict[str, int]:
        if not names:
            return {}
        out = self.runner.run(["docker", "stats", "--no-stream", "--format", _DOCKER_JSON,
                               *names],
                              timeout=INFO_TIMEOUT)
        if not out.ok:
            raise stats_failed("docker", out)
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
            mem = parse_size(before_slash(row.get("MemUsage", "")))
            if mem is not None:
                result[row["Name"]] = mem
        return result
