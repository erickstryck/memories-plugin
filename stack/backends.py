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
from .facts import HostFacts
from .runtimes import EngineInfo

#: The platforms phase 1 serves. Windows (WSL included) enters in phase 3, and
#: until then the step refuses it instead of offering the CPU (spec, "Detecção").
_PHASE1_PLATFORMS = ("linux", "macos")

#: The documented CDI generation (NVIDIA Container Toolkit), run as root because it
#: writes under /etc/cdi. Without `--output` it prints the spec to stdout and writes
#: nothing, so it would fix nothing.
CDI_GENERATE = "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml (as root)"

#: The states the menu shows for an option. A `runtime` state means the profile
#: runs, but on the OTHER runtime: it carries `needs`, not a fix.
READY, MISSING, RUNTIME, UNSUPPORTED = "ready", "missing", "runtime", "unsupported"

#: One line of `--list-devices`: `Vulkan1: AMD Radeon RX 6900 XT (RADV NAVI21)
#: (16368 MiB, 4018 MiB free)`. The `Vulkan<n>` prefix is what the server
#: command's `-dev` takes, so it is parsed, not assumed to be in order.
_DEVICE = re.compile(
    r"^Vulkan(\d+):\s+(.*?)\s*\((\d+) MiB,\s*(\d+) MiB free\)\s*$")


@dataclass(frozen=True)
class Device:
    """One Vulkan device the container saw in `--list-devices`."""
    index: int
    name: str
    total_mib: int
    free_mib: int

    @property
    def id(self) -> str:
        """The `-dev` argument: `Vulkan<index>`."""
        return f"Vulkan{self.index}"

    @property
    def vendor(self) -> str | None:
        """The vendor, read from the name the driver prints. `None` for a CPU
        device (llvmpipe) and for a name no profile knows."""
        return vendor_of(self.name)


def parse_devices(output: str) -> list[Device]:
    """The `Vulkan<n>` lines of `--list-devices`, in the order printed.

    The output comes from `compose run` (M4: a TTY, so `\r\n`) and may carry
    log lines in the middle (the `ggml_vulkan` banner): a line that does not
    match is noise, and the `(none)` shape gives an empty list.
    """
    devices = []
    for line in output.splitlines():
        match = _DEVICE.match(line.strip())
        if match is None:
            continue
        devices.append(Device(int(match.group(1)), match.group(2),
                              int(match.group(3)), int(match.group(4))))
    return devices


def vendor_of(name: str) -> str | None:
    """The vendor of a device name, by the tokens the drivers print, matched
    case-sensitively: a name is the driver's, not a free text. `llvmpipe` is a
    CPU, and a name no profile knows belongs to no profile."""
    if "NVIDIA" in name:
        return "nvidia"
    if "AMD" in name or "RADV" in name:
        return "amd"
    if "Intel" in name:
        return "intel"
    if "Virtio" in name or "Venus" in name or "Apple" in name:
        return "apple"
    return None


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
        if platform in _PHASE1_PLATFORMS:
            return frozenset({"docker", "podman"})
        return frozenset()

    def availability(self, platform: str, facts: HostFacts, engine: EngineInfo,
                     runtime: str) -> Availability:
        if platform not in _PHASE1_PLATFORMS:
            return Availability(state=UNSUPPORTED,
                                reason="the stack runs on linux and macos in this version")
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
            # The spec's named correction: the GL/Vulkan component of the driver,
            # `libnvidia-gl-<version>` on Ubuntu; on Podman the CDI spec generated
            # before it lacks the icd, so it is generated again.
            fix = "install libnvidia-gl-<version>"
            if runtime == "podman":
                fix += ", then " + CDI_GENERATE
            return Availability(state=MISSING,
                                reason="no nvidia vulkan icd on the host", fix=fix)
        if runtime == "docker" and not facts.nvidia.docker_hook:
            return Availability(state=MISSING,
                                reason="no nvidia container runtime hook",
                                fix="install nvidia-container-toolkit")
        if runtime == "podman" and not facts.nvidia.cdi_spec:
            return Availability(state=MISSING, reason="no nvidia cdi spec",
                                fix=CDI_GENERATE)
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
            if _major(engine.version) >= 6:
                fix = "podman machine init --provider libkrun"
            else:
                fix = "CONTAINERS_MACHINE_PROVIDER=libkrun podman machine init"
            return Availability(state=MISSING,
                                reason="the podman machine is not libkrun", fix=fix)
        return Availability(state=READY)

    def service_patch(self, runtime: str, gpu_index: int | None) -> dict:
        return {"devices": ["/dev/dri:/dev/dri"]}

    def devices_seen(self, output: str) -> list[Device]:
        return [d for d in parse_devices(output) if d.vendor == "apple"]


def _major(version: str) -> int:
    """The major of a podman version like `5.7.0` or `6.0.0`."""
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError):
        return 5


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
