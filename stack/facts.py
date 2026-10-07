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
temp directory.

The GPU walk reads the PCI bus, not `/sys/class/drm` (review round R2, item I4): a
`cardN` appears there only once a kernel driver is bound, so a card with no driver
(a driverless NVIDIA, the case the spec names) is invisible there. A PCI device is a
GPU when its `class` starts with `0x03` (display controller: `0x030000` VGA,
`0x030200` 3D, `0x038000` other) and its `vendor` is in `VENDORS`; its `card` name is
the PCI address. Render nodes still come from `/dev/dri`.
"""
import platform
import re
import shutil
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import StackError
from .engine import normalize_arch
from .process import Runner, SubprocessRunner

#: The PCI vendor ids the kernel prints in `/sys/bus/pci/devices/*/vendor`, mapped to
#: the name the menu uses. Only these are GPUs: the ASPEED BMC (0x1a03) and every
#: other display-class device are ignored.
VENDORS = {"0x1002": "amd", "0x8086": "intel", "0x10de": "nvidia"}

#: A display controller (PCI class 0x03): `0x030000` VGA, `0x030200` 3D, `0x038000`
#: other. The audio functions that share a GPU's card number are class `0x040300`
#: (multimedia) and start with `0x04`, so they are not matched here (measured
#: 2026-10-06: the AMD card's HDMI audio is `0000:44:00.1`, class `0x040300`).
_DISPLAY_CLASS_PREFIX = "0x03"


@dataclass(frozen=True)
class Gpu:
    """One GPU the host can see: its vendor and the PCI address of the card."""
    vendor: str
    card: str


@dataclass(frozen=True)
class NvidiaFacts:
    """Whether an NVIDIA GPU could reach a container. Filled only when a nvidia
    GPU is present; the empty default is the "no nvidia here" answer.

    `docker_hook` is `nvidia-container-runtime-hook` on the PATH (any Docker
    version). `cdi_hook` is `nvidia-cdi-hook` on the PATH: it counts for Docker only
    from 29.2 (the daemon did not recognise it before), so the backend combines it
    with the Docker version (review round R2, item m3). `ctk` is `nvidia-ctk`, the
    tool that generates a missing CDI spec.
    """
    gpus: tuple[str, ...] = ()
    icd: bool = False
    docker_hook: bool = False
    cdi_hook: bool = False
    cdi_spec: bool = False
    ctk: bool = False


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
    callables, so no real file read happens. `runner` defaults to a real
    `SubprocessRunner` (review round R2, item I5: a production `collect` must run
    `nvidia-smi`); a test injects a fake. `disk_usage` is injected so the disk test
    never reads a real filesystem (review round R2, item m8)."""
    root: Path = Path("/")
    system: Callable[[], str] = platform.system
    machine: Callable[[], str] = platform.machine
    which: Callable[[str], str | None] = shutil.which
    runner: Runner = field(default_factory=SubprocessRunner)
    disk_usage: Callable[[Path], Any] = shutil.disk_usage


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
    """The display controllers on the PCI bus, by vendor (review round R2, item I4).

    A device counts when its `class` starts with `0x03` (a display controller) and
    its `vendor` is in `VENDORS`. Reading the bus, not `/sys/class/drm`, is what
    keeps a driverless card visible: `/sys/class/drm` lists a `cardN` only once a
    driver is bound. The card's name is the PCI address, and the list is sorted by
    address (the addresses are zero-padded, so lexicographic order is numeric).
    """
    pci = probe.root / "sys" / "bus" / "pci" / "devices"
    if not pci.is_dir():
        return ()
    found: list[Gpu] = []
    for dev in sorted(pci.iterdir()):
        try:
            cls = (dev / "class").read_text().strip()
            vendor = (dev / "vendor").read_text().strip().lower()
        except OSError:
            continue
        if not cls.startswith(_DISPLAY_CLASS_PREFIX):
            continue
        if vendor not in VENDORS:
            continue
        found.append(Gpu(vendor=VENDORS[vendor], card=dev.name))
    return tuple(found)


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
    """The free RAM the host can actually use, for the LINUX case. On linux the
    kernel publishes it as `MemAvailable`. On macOS this returns `None` (review
    round R2, item I6): the host's `hw.memsize` is not what the containers get.
    They run in the machine VM (2048 MiB by default), so the figure that matters
    is the engine's own, read from `EngineInfo.memory_bytes` by the caller."""
    if system != "linux":
        return None
    try:
        meminfo = (probe.root / "proc" / "meminfo").read_text()
    except OSError:
        return None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            # "MemAvailable:    4194304 kB" -> 4194304 * 1024 bytes
            return int(line.split()[1]) * 1024
    return None


def _disk_free(probe: Probe, stack_dir: Path) -> int | None:
    """The free bytes at the nearest EXISTING ancestor of `stack_dir`. The stack
    directory is created later, so the path itself is usually absent: walk up
    until something is there, and read its free space. The measurement goes through
    `probe.disk_usage` so a test injects a fake and never reads a real filesystem
    (review round R2, item m8)."""
    candidate = Path(stack_dir)
    while True:
        try:
            return probe.disk_usage(candidate).free
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
    """The NVIDIA readiness, filled only when a nvidia GPU is present.

    A hung `nvidia-smi` (a broken driver can hang it, and the runner then raises
    `StackError`) or a failing one is a driver that lists no GPU: `gpus` stays empty,
    which the nvidia profile reads as "install the NVIDIA driver". It is not the end of
    host detection, because the CPU stays on offer on every host (review round R3,
    item R3-6).
    """
    if not any(g.vendor == "nvidia" for g in gpus):
        return NvidiaFacts()
    names: tuple[str, ...] = ()
    try:
        out = probe.runner.run(["nvidia-smi", "-L"], timeout=30.0)
    except StackError:
        pass  # hung: no GPU listed
    else:
        if out.ok:
            names = tuple(name for name in map(_nvidia_name, out.stdout.splitlines()) if name)
    icd = _nvidia_icd(probe)
    docker_hook = probe.which("nvidia-container-runtime-hook") is not None
    cdi_hook = probe.which("nvidia-cdi-hook") is not None
    ctk = probe.which("nvidia-ctk") is not None
    cdi = _cdi_spec(probe)
    return NvidiaFacts(gpus=names, icd=icd, docker_hook=docker_hook,
                       cdi_hook=cdi_hook, cdi_spec=cdi, ctk=ctk)


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
    makes the profile possible. The files are read as bytes and searched for the
    marker `nvidia.com/gpu`: the directories may hold any file, and one that is not
    UTF-8 crashed a text read (review round R2, item m9)."""
    marker = b"nvidia.com/gpu"
    for cdi_dir in (probe.root / "etc" / "cdi", probe.root / "var" / "run" / "cdi"):
        if not cdi_dir.is_dir():
            continue
        for entry in cdi_dir.iterdir():
            if not entry.is_file():
                continue
            try:
                if marker in entry.read_bytes():
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
    `microsoft` (case-insensitive, review round R2, item m6). The macos branch
    comes first because the engine, not the host, is what the image must run on."""
    if facts.system == "macos":
        return "macos"
    if facts.system == "windows" or facts.wsl or "microsoft" in engine_kernel.lower():
        return "windows"
    return "linux"
