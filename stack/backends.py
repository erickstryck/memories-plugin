"""The backend profiles: how the GPU (or no GPU) reaches the llama-server
container, and on which runtime each profile runs.

This module DECIDES. It does not gather facts (that is `stack.facts`): it takes
the `HostFacts` and the `EngineInfo` and answers the three questions the menu
must show. WHERE does a profile run (`runtimes`, the spec's "Compatibilidade"
table, and the single source for the label and the docs' table)? IS it ready
HERE (`availability`, each missing piece with its named fix)? and WHAT does the
service receive (`service_patch`, the compose diff per runtime)?

The GPU DEFAULT is also decided here, on free memory alone: the `ggml_vulkan
uma:` line never comes out of `--list-devices` (M5), so there is no discrete
versus integrated signal, and the ready, non-experimental device with the most
free MiB is the default. The apple profile is experimental, so it is never the
default even when it is the only ready GPU.
"""
import re
from dataclasses import dataclass
from typing import Protocol

from . import StackError
# `parse_devices` and `vendor_of` are re-exported for the importers of `stack.backends`.
from .devices import Device, parse_devices, vendor_of  # noqa: F401
from .engine import EngineInfo
from .facts import HostFacts

#: The platforms this version serves. Windows means WSL2 with a container
#: runtime: Task 1 classifies such a host as the `windows` platform, and the cpu
#: profile runs there on the official image (spec 2026-10-09). A BARE Windows
#: (no WSL2) is still refused, but that one-liner now lives in `install_step`
#: (`facts.is_native_windows`), not here -- so this table offers the cpu.
_SUPPORTED_PLATFORMS = ("linux", "macos", "windows")

#: The documented CDI generation (NVIDIA Container Toolkit), run as root because it
#: writes under /etc/cdi. Without `--output` it prints the spec to stdout and writes
#: nothing, so it would fix nothing.
CDI_GENERATE = "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml (as root)"

#: The states the menu shows for an option. A `runtime` state means the profile
#: runs, but on the OTHER runtime: it carries `needs`, not a fix.
READY, MISSING, RUNTIME, UNSUPPORTED = "ready", "missing", "runtime", "unsupported"


@dataclass(frozen=True)
class Availability:
    """Where an option stands for THIS host and runtime, and the next move when
    it is not ready. `fix` is a command or package; `needs` is the runtime the
    profile would run on instead."""
    state: str
    reason: str = ""
    fix: str | None = None
    needs: str | None = None


class Backend(Protocol):
    id: str
    experimental: bool
    vendor: str | None
    image_role: str | None

    def runtimes(self, platform: str) -> frozenset[str]: ...

    def availability(self, platform: str, facts: HostFacts, engine: EngineInfo,
                     runtime: str) -> Availability: ...

    def service_patch(self, runtime: str, gpu_index: int | None) -> dict: ...

    def devices_seen(self, output: str) -> list[Device]: ...


class Cpu:
    """No device at all: it runs everywhere, always, and needs nothing. The
    `-dev none` that keeps it off an injected GPU (M1) lives in the server
    command, not in the service: the compose diff for this profile is empty."""
    id = "cpu"
    experimental = False
    vendor: str | None = None
    image_role: str | None = None

    def runtimes(self, platform: str) -> frozenset[str]:
        if platform in _SUPPORTED_PLATFORMS:
            return frozenset({"docker", "podman"})
        return frozenset()

    def availability(self, platform: str, facts: HostFacts, engine: EngineInfo,
                     runtime: str) -> Availability:
        if platform not in _SUPPORTED_PLATFORMS:
            return Availability(state=UNSUPPORTED,
                                reason="the stack runs on linux, macos and windows in this version")
        return Availability(state=READY, reason="cpu")

    def service_patch(self, runtime: str, gpu_index: int | None) -> dict:
        return {}

    def devices_seen(self, output: str) -> list[Device]:
        return []


class DriGpu:
    """The AMD and Intel profiles: `/dev/dri` in, and on Podman the annotation
    that keeps the user's groups (required rootless, harmless rootful, so there
    is no branch on rootless: the plan's ruling)."""
    id: str = "dri"
    experimental: bool = False
    vendor: str | None = None
    image_role: str | None = "llama"

    def __init__(self, vendor: str):
        self.id = vendor
        self.vendor = vendor

    def runtimes(self, platform: str) -> frozenset[str]:
        if platform == "linux":
            return frozenset({"docker", "podman"})
        return frozenset()

    def availability(self, platform: str, facts: HostFacts, engine: EngineInfo,
                     runtime: str) -> Availability:
        if platform != "linux":
            return Availability(state=UNSUPPORTED,
                                reason="dri gpu profiles run on linux only")
        if not any(g.vendor == self.vendor for g in facts.gpus):
            return Availability(state=UNSUPPORTED,
                                reason=f"no {self.vendor} gpu on the host")
        if not facts.render_nodes:
            return Availability(state=MISSING,
                                reason="no /dev/dri/renderD* node",
                                fix=f"install the {self.vendor} GPU driver")
        return Availability(state=READY)

    def service_patch(self, runtime: str, gpu_index: int | None) -> dict:
        # the explicit host:container form, as in the spec's example and the nine
        # shapes the opening verification ran through both providers
        patch: dict = {"devices": ["/dev/dri:/dev/dri"]}
        if runtime == "podman":
            patch["annotations"] = {"run.oci.keep_original_groups": "1"}
        return patch

    def devices_seen(self, output: str) -> list[Device]:
        return [d for d in parse_devices(output) if d.vendor == self.vendor]


class NvidiaGpu:
    """The NVIDIA profile: the Vulkan ICD and the libraries come from the host
    driver, injected by the toolkit. Docker takes them through one of its two
    hooks; Podman through a CDI spec. Each missing piece has its named fix
    (spec, "Detecção")."""
    id: str = "nvidia"
    experimental: bool = False
    vendor: str | None = "nvidia"
    image_role: str | None = "llama"

    def runtimes(self, platform: str) -> frozenset[str]:
        if platform == "linux":
            return frozenset({"docker", "podman"})
        return frozenset()

    def availability(self, platform: str, facts: HostFacts, engine: EngineInfo,
                     runtime: str) -> Availability:
        if platform != "linux":
            return Availability(state=UNSUPPORTED, reason="nvidia runs on linux only")
        if not any(g.vendor == "nvidia" for g in facts.gpus):
            # offered only with an NVIDIA GPU on the host (spec, "Perfis de backend")
            return Availability(state=UNSUPPORTED, reason="no nvidia gpu on the host")
        if not facts.nvidia.gpus:
            # the card is on the bus, but no driver lists it (nouveau, or none)
            return Availability(state=MISSING, reason="nvidia-smi lists no gpu",
                                fix="install the NVIDIA driver")
        if not facts.nvidia.icd:
            # The spec's named correction: the driver's GL/Vulkan component,
            # `libnvidia-gl-<version>` on Ubuntu; on Podman the CDI spec generated
            # before it lacks the icd, so it is generated again.
            fix = "install the driver's GL/Vulkan package (libnvidia-gl-<version> on Ubuntu)"
            if runtime == "podman":
                fix += ", then " + CDI_GENERATE
            return Availability(state=MISSING,
                                reason="no nvidia vulkan icd on the host", fix=fix)
        if runtime == "docker":
            # A Docker daemon accepts a GPU request through one of its two hooks:
            # `nvidia-container-runtime-hook` on any version, or `nvidia-cdi-hook`,
            # but only from 29.2 (the daemon did not recognise the CDI hook before
            # then; read in moby at v28.3.2, v28.5.2, docker-v29.1.0 and
            # docker-v29.2.0, review round R2 item m3). An unparseable version
            # therefore does not count the CDI hook.
            hook_ok = facts.nvidia.docker_hook
            if not hook_ok and facts.nvidia.cdi_hook:
                hook_ok = _docker_ge_29_2(engine.version)
            if not hook_ok:
                return Availability(state=MISSING,
                                    reason="no nvidia container runtime hook",
                                    fix="install nvidia-container-toolkit")
        elif not facts.nvidia.cdi_spec:
            # On Podman the CDI spec is the hook. Generating it needs `nvidia-ctk`;
            # when the toolkit is not installed, the fix says so first (R2 item m4).
            fix = (CDI_GENERATE if facts.nvidia.ctk
                   else "install nvidia-container-toolkit, then " + CDI_GENERATE)
            return Availability(state=MISSING, reason="no nvidia cdi spec", fix=fix)
        return Availability(state=READY)

    def service_patch(self, runtime: str, gpu_index: int | None) -> dict:
        if gpu_index is None:
            # `nvidia.com/gpu=None` would reach the compose file and fail at `up`
            raise StackError("the nvidia profile needs the nvidia-smi index of a gpu",
                             step="backends")
        if runtime == "podman":
            # Podman ignores the `deploy` block (podman #28309, #28436): the
            # CDI device is the only form it accepts.
            return {"devices": [f"nvidia.com/gpu={gpu_index}"]}
        # `graphics` is explicit: without it the daemon leaves the ICD out
        # (spec, "NVIDIA no Docker").
        return {"deploy": {"resources": {"reservations": {"devices": [
            {"driver": "nvidia", "device_ids": [str(gpu_index)],
             "capabilities": ["gpu", "compute", "utility", "graphics"]}
        ]}}}}

    def devices_seen(self, output: str) -> list[Device]:
        return [d for d in parse_devices(output) if d.vendor == "nvidia"]


class AppleGpu:
    """The experimental Apple profile: a libkrun machine, whose `/dev/dri`
    (krunkit's Virtio-GPU Venus) is the only GPU path on macOS. Docker Desktop
    passes no GPU to a container at all, so the profile runs on Podman only,
    and the fix for a non-libkrun machine is the one for the installed podman
    version (spec, "Compatibilidade")."""
    id: str = "apple"
    experimental: bool = True
    vendor: str | None = "apple"
    image_role: str | None = "llama"

    def runtimes(self, platform: str) -> frozenset[str]:
        if platform == "macos":
            return frozenset({"podman"})
        return frozenset()

    def availability(self, platform: str, facts: HostFacts, engine: EngineInfo,
                     runtime: str) -> Availability:
        if platform != "macos":
            return Availability(state=UNSUPPORTED, reason="apple runs on macos only")
        if "arm64" not in (facts.arch, engine.arch):
            # An Intel Mac has no container GPU path at all (spec, "Detecção").
            # Either arch saying arm64 is enough: Python under Rosetta reports
            # x86_64 on Apple Silicon, and the engine's arch (the VM's) is the one
            # the spec says counts on macOS.
            return Availability(state=UNSUPPORTED,
                                reason="no container gpu path on an intel mac")
        if runtime == "docker":
            return Availability(state=RUNTIME,
                                reason="docker desktop passes no gpu to a container",
                                needs="podman")
        if engine.vm != "libkrun":
            # ONE fix for podman 5 and 6 (review round R2, item I7): the provider
            # must persist. A one-shot `CONTAINERS_MACHINE_PROVIDER=libkrun podman
            # machine init` does not, because `podman machine info` (Host.VMType)
            # and `podman machine start` read the provider CONFIGURED in
            # containers.conf or the environment for each command, so the next
            # `info` says applehv again and the start cannot find the new machine.
            # On podman 6 applehv appears only when config or env pins it, which
            # `--provider libkrun` does not change. The durable fix is to set it in
            # the [machine] table, then re-create the machine.
            fix = ("set provider = \"libkrun\" in the [machine] table of "
                   "~/.config/containers/containers.conf (and unset "
                   "CONTAINERS_MACHINE_PROVIDER), then podman machine init --now")
            return Availability(state=MISSING,
                                reason="the podman machine is not libkrun", fix=fix)
        return Availability(state=READY)

    def service_patch(self, runtime: str, gpu_index: int | None) -> dict:
        patch: dict = {"devices": ["/dev/dri:/dev/dri"]}
        if runtime == "podman":
            # the same annotation the dri profiles carry on Podman: the machine runs
            # rootless by default, and the annotation is required rootless and
            # harmless rootful (review round R2, item R2-8). Unmeasured on a Mac;
            # the cost of being wrong is one annotation line.
            patch["annotations"] = {"run.oci.keep_original_groups": "1"}
        return patch

    def devices_seen(self, output: str) -> list[Device]:
        return [d for d in parse_devices(output) if d.vendor == "apple"]


def _docker_ge_29_2(version: str) -> bool:
    """Whether a Docker version is 29.2 or newer, so the `nvidia-cdi-hook` counts.

    `29.2`/`29.3.1`/`30.0` are yes; `28.5.2`, `29.1.0` and an unparseable string
    are no (review round R2 item m3): when the version cannot be read, the CDI
    hook must not count, only the runtime hook may."""
    match = re.match(r"^(\d+)\.(\d+)", version.strip())
    if match is None:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return major > 29 or (major == 29 and minor >= 2)


#: The menu's order: the safe default first, then the GPU profiles.
BACKENDS: dict[str, Backend] = {
    "cpu": Cpu(),
    "amd": DriGpu("amd"),
    "intel": DriGpu("intel"),
    "nvidia": NvidiaGpu(),
    "apple": AppleGpu(),
}

#: The profiles that take a device: everything but `cpu`.
GPU_PROFILES = ("amd", "intel", "nvidia", "apple")


def runtime_label(runtimes: frozenset[str]) -> str:
    """The menu's label for where a profile runs (decision 15)."""
    if {"docker", "podman"} <= set(runtimes):
        return "Docker and Podman"
    if runtimes == frozenset({"podman"}):
        return "Podman only"
    if runtimes == frozenset({"docker"}):
        return "Docker only"
    return "not available here"


@dataclass(frozen=True)
class Option:
    """One entry of the menu: a backend, the device it got (or none), the
    `-dev` index, and where it stands on this host."""
    backend: str
    device: Device | None
    gpu_index: int | None
    availability: Availability


def default_option(options: list[Option]) -> Option:
    """The profile to start with: among the ready, non-experimental options
    that got a device, the one with the most free memory (M5). No such option,
    the `cpu` one, which the menu always carries."""
    best: Option | None = None
    best_free = -1
    for option in options:
        device = option.device
        if option.availability.state != READY or device is None:
            continue
        if BACKENDS[option.backend].experimental:
            continue
        if device.free_mib > best_free:
            best, best_free = option, device.free_mib
    if best is not None:
        return best
    for option in options:
        if option.backend == "cpu":
            return option
    raise StackError("the menu has no cpu option to fall back to",
                     step="backends", fix=None)
