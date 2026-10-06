"""The facts the install step reads about the host: the OS, the RAM, the disk, the
GPUs, the SELinux mode, and the NVIDIA readiness.

This module GATHERS facts. It does not CHOOSE a GPU: choosing is Task 5's job, and
it chooses on free memory alone. That is why the GPU TYPE (discrete versus
integrated) is deliberately NOT read here. Measured on this machine (2026-10-06,
spec "Escolha da GPU"), the `ggml_vulkan ... uma:` line never comes with
`--list-devices`, so a type flag read here would be fiction (M5): the facts carry
what exists, and nothing the choice will not use.

Everything is read through a `Probe`: files under `probe.root`, `platform` and
`shutil` as injected callables, and a `Runner` for the commands. A production
`collect` uses the default `Probe()`; a test points `root` at a fake tree in a
temp directory. The GPU walk is shaped like this machine: two Intel cards, one AMD
card, an ASPEED BMC card that is not a GPU, and a `card0-DP-1` connector that is
not a card at all.
"""
import platform
import re
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .runtimes import Runner, normalize_arch

#: The PCI vendor ids the kernel prints in `/sys`, mapped to the name the menu uses.
#: Only these are GPUs: the ASPEED BMC (0x1a03) and every other onboard device
#: that lives in `/sys/class/drm` is ignored.
VENDORS = {"0x1002": "amd", "0x8086": "intel", "0x10de": "nvidia"}

_CARD = re.compile(r"^card(\d+)$")


@dataclass(frozen=True)
class Gpu:
    """One GPU the host can see: its vendor and the readable name of its card."""
    vendor: str
    card: str


@dataclass(frozen=True)
class NvidiaFacts:
    """Whether an NVIDIA GPU could reach a container. Filled only when a nvidia
    GPU is present; the empty default is the "no nvidia here" answer."""
    gpus: tuple[str, ...] = ()
    icd: bool = False
    docker_hook: bool = False
    cdi_spec: bool = False


@dataclass(frozen=True)
class HostFacts:
    """The host, as the install step sees it. `ram_bytes` and `disk_free_bytes`
    are `None` when the fact could not be read, not zero."""
    system: str
    arch: str
    wsl: bool
    ram_bytes: int | None
    disk_free_bytes: int | None
    gpus: tuple[Gpu, ...]
    render_nodes: tuple[str, ...]
    selinux: bool
    nvidia: NvidiaFacts = NvidiaFacts()


@dataclass
class Probe:
    """The injected primitives every fact is read through. A production `collect`
    uses the defaults; a test points `root` at a fake tree and swaps the
    callables, so no real file, command or PATH lookup happens."""
    root: Path = Path("/")
    system: Callable[[], str] = platform.system
    machine: Callable[[], str] = platform.machine
    which: Callable[[str], str | None] = shutil.which
    runner: Runner | None = None


def normalize_system(name: str) -> str:
    """`platform.system` folded to the one word the rest of the stack uses."""
    if name == "Linux":
        return "linux"
    if name == "Darwin":
        return "macos"
    if name == "Windows":
        return "windows"
    return name.lower()


def collect(probe: Probe, stack_dir: Path) -> HostFacts:
    """Read every fact of the host through `probe`, at the directory `stack_dir`
    will live in. `stack_dir` itself need not exist yet: the disk is measured at
    the nearest ancestor that does."""
    system = normalize_system(probe.system())
    arch = normalize_arch(probe.machine())
    gpus = _gpus(probe)
    nvidia = _nvidia_facts(probe, gpus)
    return HostFacts(
        system=system,
        arch=arch,
        wsl=_wsl(probe),
        ram_bytes=_ram(probe, system),
        disk_free_bytes=_disk_free(probe, stack_dir),
        gpus=gpus,
        render_nodes=_render_nodes(probe),
        selinux=_selinux(probe),
        nvidia=nvidia,
    )


def _gpus(probe: Probe) -> tuple[Gpu, ...]:
    """The real cards in `/sys/class/drm`: a `cardN` (all digits) whose vendor is
    in `VENDORS`. A `cardN-<connector>` is a connector, not a card, and a vendor
    outside `VENDORS` (the ASPEED BMC) is not a GPU."""
    drm = probe.root / "sys" / "class" / "drm"
    if not drm.is_dir():
        return ()
    found = []
    for entry in drm.iterdir():
        match = _CARD.match(entry.name)
        if match is None or not entry.is_dir():
            continue
        vendor_file = entry / "device" / "vendor"
        try:
            vendor = vendor_file.read_text().strip().lower()
        except OSError:
            continue
        if vendor not in VENDORS:
            continue
        label_file = entry / "device" / "label"
        try:
            card = label_file.read_text().strip()
        except OSError:
            card = entry.name
        if not card:
            card = entry.name
        found.append((int(match.group(1)), Gpu(vendor=VENDORS[vendor], card=card)))
    found.sort(key=lambda pair: pair[0])
    return tuple(gpu for _, gpu in found)


def _render_nodes(probe: Probe) -> tuple[str, ...]:
    """The `renderD*` nodes under `/dev/dri`, sorted by name. Empty when the
    directory is absent: the GPU list is the source of truth, not the nodes."""
    dri = probe.root / "dev" / "dri"
    if not dri.is_dir():
        return ()
    return tuple(sorted(e.name for e in dri.iterdir()
                        if e.name.startswith("renderD")))


def _wsl(probe: Probe) -> bool:
    """WSL is in the KERNEL release: `...-microsoft-standard-WSL2` on WSL2,
    `...-Microsoft` on WSL1. `/etc/os-release` is the distro's own file and names no
    WSL, so it is not read. Absent is not WSL."""
    osrelease = probe.root / "proc" / "sys" / "kernel" / "osrelease"
    try:
        return "microsoft" in osrelease.read_text().lower()
    except OSError:
        return False


def _ram(probe: Probe, system: str) -> int | None:
    """The free RAM the host can actually use. On linux the kernel publishes it
    as `MemAvailable`; on macos the only reading is `sysctl hw.memsize`."""
    if system == "linux":
        try:
            meminfo = (probe.root / "proc" / "meminfo").read_text()
        except OSError:
            return None
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                # "MemAvailable:    4194304 kB" -> 4194304 * 1024 bytes
                return int(line.split()[1]) * 1024
        return None
    if system == "macos":
        if probe.runner is None:
            return None
        out = probe.runner.run(["sysctl", "hw.memsize"], timeout=30.0)
        if not out.ok:
            return None
        # "hw.memsize: 17179869184"
        for line in out.stdout.splitlines():
            if line.startswith("hw.memsize:"):
                return int(line.split(":")[1].strip())
        return None
    return None


def _disk_free(probe: Probe, stack_dir: Path) -> int | None:
    """The free bytes at the nearest EXISTING ancestor of `stack_dir`. The stack
    directory is created later, so the path itself is usually absent: walk up
    until something is there, and read its free space."""
    candidate = Path(stack_dir)
    while True:
        try:
            return shutil.disk_usage(candidate).free
        except OSError:
            if candidate == candidate.parent:
                return None  # not even the root answered
            candidate = candidate.parent


def _selinux(probe: Probe) -> bool:
    """Enforcing is the `1` in `/sys/fs/selinux/enforce`. Absent is not
    enforcing: permissive and disabled both say `0` or nothing."""
    enforce = probe.root / "sys" / "fs" / "selinux" / "enforce"
    try:
        return enforce.read_text().strip() == "1"
    except OSError:
        return False


def _nvidia_facts(probe: Probe, gpus: tuple[Gpu, ...]) -> NvidiaFacts:
    """The NVIDIA readiness, filled only when a nvidia GPU is present. A missing
    runner skips the `nvidia-smi` read (the file-based parts still hold) rather
    than crash: the file facts are enough to say the driver is half there."""
    if not any(g.vendor == "nvidia" for g in gpus):
        return NvidiaFacts()
    names = ()
    if probe.runner is not None:
        out = probe.runner.run(["nvidia-smi", "-L"], timeout=30.0)
        if out.ok:
            names = tuple(_nvidia_name(line)
                          for line in out.stdout.splitlines()
                          if (n := _nvidia_name(line)))
    icd = _nvidia_icd(probe)
    docker_hook = probe.which("nvidia-container-runtime-hook") is not None \
        or probe.which("nvidia-cdi-hook") is not None
    cdi = _cdi_spec(probe)
    return NvidiaFacts(gpus=names, icd=icd, docker_hook=docker_hook, cdi_spec=cdi)


def _nvidia_icd(probe: Probe) -> bool:
    """The NVIDIA Vulkan ICD on the host (spec, "Detecção"): it ships in
    `/usr/share/vulkan/icd.d` on a driver install and in `/etc/vulkan/icd.d`
    when the admin drops it there, so either one makes the profile possible."""
    for path in ("usr/share/vulkan/icd.d", "etc/vulkan/icd.d"):
        if (probe.root / path / "nvidia_icd.json").is_file():
            return True
    return False


def _nvidia_name(line: str) -> str:
    """`GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-0)` -> the NAME. The line after
    the `GPU <i>: ` prefix is what `nvidia-smi -L` gives, but a name is not its
    UUID: the trailing ` (UUID: ...)` is metadata, so it is dropped. A line
    without the prefix is not a GPU."""
    match = re.match(r"^GPU\s+\d+:\s+(.*)$", line.strip())
    if match is None:
        return ""
    name = match.group(1).strip()
    return re.sub(r"\s*\((?:UUID|uuid):\s*[^)]*\)\s*$", "", name).strip()


def _cdi_spec(probe: Probe) -> bool:
    """A CDI spec naming an nvidia GPU (spec, "Detecção"): it lives in `/etc/cdi`
    or `/var/run/cdi`, in either container runtime's hand, so either directory
    makes the profile possible: any file containing `nvidia.com/gpu`."""
    for cdi_dir in (probe.root / "etc" / "cdi", probe.root / "var" / "run" / "cdi"):
        if not cdi_dir.is_dir():
            continue
        for entry in cdi_dir.iterdir():
            if not entry.is_file():
                continue
            try:
                if "nvidia.com/gpu" in entry.read_text():
                    return True
            except OSError:
                continue
    return False


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    """True when a socket can bind `(host, port)`. This is the only real I/O the
    module does, and it is loopback: a bind on `127.0.0.1` reaches no other
    machine. The socket is closed before the answer leaves, and a failure to even
    open it is a False, not a raise: the caller is asking whether the port is
    usable, and "could not ask" is not "usable"."""
    sock = None
    try:
        sock = socket.socket()
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        if sock is not None:
            sock.close()


def platform_of(facts: HostFacts, engine_kernel: str = "") -> str:
    """The platform the install step runs on, from the facts and the engine
    kernel. WSL is Windows even when the host reports linux: the kernel says
    `microsoft`. The macos branch comes first because the engine, not the host,
    is what the image must run on."""
    if facts.system == "macos":
        return "macos"
    if facts.system == "windows" or facts.wsl or "microsoft" in engine_kernel:
        return "windows"
    return "linux"
