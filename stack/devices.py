"""What a container's `--list-devices` printed: the Vulkan devices, and whose each one is.

`parse_devices` reads the `Vulkan<n>` lines and `vendor_of` names the vendor from the name the
driver prints. The backend profiles (`stack.backends`) keep the devices of their vendor, and the
GPU default reads the free memory (M5).
"""
import re
from dataclasses import dataclass

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
    """The vendor of a device name, by the tokens the native drivers print, matched
    case-sensitively: a name is the driver's, not a free text. `llvmpipe` is a
    CPU, and a name no profile knows belongs to no profile. The dzn device (Windows,
    phase 3) wraps the adapter's own name, `Microsoft Direct3D12 (<adapter>)` (Mesa
    `dzn_device.c`), so the same tokens apply to it."""
    if "NVIDIA" in name:
        return "nvidia"
    if "AMD" in name or "RADV" in name:
        return "amd"
    if "Intel" in name:
        return "intel"
    if "Virtio" in name or "Venus" in name or "Apple" in name:
        return "apple"
    return None
