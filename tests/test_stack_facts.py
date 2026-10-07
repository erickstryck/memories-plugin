"""Host facts: the OS, the RAM, the disk, the GPUs, the SELinux mode, the NVIDIA
readiness.

The whole point is the injected `Probe`: every fact is read from files under
`probe.root`, every command through `probe.runner`, so a fake tree in a temp
directory stands in for the machine. Nothing here CHOOSES a GPU (that is Task 5)
and nothing here reads the GPU TYPE: measured on this machine (2026-10-06, spec
"Escolha da GPU"), the `--list-devices` output never carries a discrete/integrated
flag, so free memory is the only signal a choice may use, and the facts stay
typeless by design (M5).

The GPU list is shaped like this machine's PCI display devices, measured 2026-10-06
with `cat /sys/bus/pci/devices/*/class /sys/bus/pci/devices/*/vendor`: two Intel (xe),
one AMD (amdgpu), an ASPEED BMC that must be ignored, and the class-`0x04` audio
functions that share a GPU's card number, which must not be mistaken for GPUs.
"""
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import runtimes  # noqa: E402
from stack.facts import (  # noqa: E402
    Gpu,
    HostFacts,
    NvidiaFacts,
    Probe,
    VENDORS,
    collect,
    normalize_system,
    platform_of,
    port_free,
)
from stack.runtimes import normalize_arch  # noqa: E402
from tests.stack_fakes import FakeRunner  # noqa: E402


def make_pci(root: Path, addr: str, vendor: str, cls: str) -> None:
    """One PCI device under the fake `/sys/bus/pci/devices`, as the machine prints it.

    `vendor` is the hex id the kernel writes (lowercased here, as the code reads it),
    and `cls` the class code. A device counts as a GPU when `cls` starts with `0x03`
    (a display controller) and `vendor` is in `VENDORS`; the `0x04` audio functions
    that share a GPU's card number are not display controllers and must be ignored.
    """
    dev = root / "sys" / "bus" / "pci" / "devices" / addr
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(f"{vendor}\n")
    (dev / "class").write_text(f"{cls}\n")


def linux_probe(tmp: Path, **overrides) -> Probe:
    """A Probe aimed at `tmp` with the machine answered by the injected callables."""
    return Probe(root=tmp, system=lambda: "Linux", machine=lambda: "x86_64",
                 **overrides)


class TestGpuDiscovery(unittest.TestCase):
    def test_this_machines_shape_two_intel_one_amd_and_the_bmc_ignored(self):
        """The PCI display devices of this machine (measured 2026-10-06 with
        `cat /sys/bus/pci/devices/*/class /sys/bus/pci/devices/*/vendor`): two Intel
        (xe), one AMD (amdgpu), and the ASPEED BMC, plus the audio functions that
        share a GPU's card number, which are class `0x04` and not display
        controllers, so they are not GPUs."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_pci(root, "0000:03:00.0", "0x8086", "0x030000")  # Intel, xe
            make_pci(root, "0000:44:00.0", "0x1002", "0x030000")  # AMD, amdgpu
            make_pci(root, "0000:c8:00.0", "0x8086", "0x030000")  # Intel, xe
            make_pci(root, "0000:cb:00.0", "0x1a03", "0x030000")  # ASPEED BMC: not in VENDORS
            make_pci(root, "0000:04:00.0", "0x8086", "0x040300")  # Intel HDMI audio: class 0x04
            make_pci(root, "0000:44:00.1", "0x1002", "0x040300")  # AMD HDMI audio: class 0x04

            facts = collect(linux_probe(root), root / "stack")

        # the three display GPUs, sorted by PCI address; the BMC and both audio
        # functions are out
        self.assertEqual(len(facts.gpus), 3, f"BMC and audio must be out: {facts.gpus}")
        self.assertEqual(facts.gpus[0], Gpu(vendor="intel", card="0000:03:00.0"))
        self.assertEqual(facts.gpus[1], Gpu(vendor="amd", card="0000:44:00.0"))
        self.assertEqual(facts.gpus[2], Gpu(vendor="intel", card="0000:c8:00.0"))
        self.assertNotIn("0x1a03", VENDORS)

    def test_a_driverless_nvidia_is_on_the_bus_and_visible(self):
        """The case the spec names: a card with no driver bound. `/sys/class/drm`
        would not list it at all (a `cardN` appears only once a driver binds), but
        the PCI bus does, so it is in `gpus` and the availability step can say
        "install the driver" (review round R2, item I4)."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_pci(root, "0000:65:00.0", "0x10de", "0x030000")  # NVIDIA, no driver

            facts = collect(linux_probe(root), root / "stack")

        self.assertEqual(facts.gpus, (Gpu(vendor="nvidia", card="0000:65:00.0"),))

    def test_a_non_display_function_is_not_a_gpu(self):
        """A device with a GPU vendor but a non-display class (the AMD HDMI audio
        function, class `0x040300`) is not a GPU, even though the vendor is known."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_pci(root, "0000:44:00.1", "0x1002", "0x040300")

            facts = collect(linux_probe(root), root / "stack")

        self.assertEqual(facts.gpus, ())

    def test_render_nodes_are_listed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "dev" / "dri").mkdir(parents=True)
            (root / "dev" / "dri" / "renderD129").write_text("")
            (root / "dev" / "dri" / "renderD128").write_text("")
            (root / "dev" / "dri" / "card0").write_text("")  # not a render node

            facts = collect(linux_probe(root), root / "stack")

        self.assertEqual(facts.render_nodes, ("renderD128", "renderD129"))


class TestTheOsFacts(unittest.TestCase):
    def test_wsl_is_read_from_the_kernel_release(self):
        """WSL is the kernel's, not the distro's: `/proc/sys/kernel/osrelease` says
        `microsoft` (WSL2 `...-microsoft-standard-WSL2`, WSL1 `...-Microsoft`), while
        `/etc/os-release` is the distro's own file and names no WSL at all."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "proc" / "sys" / "kernel").mkdir(parents=True)
            osrelease = root / "proc" / "sys" / "kernel" / "osrelease"
            # a distro file that even says microsoft does not make a native kernel WSL
            (root / "etc").mkdir(parents=True)
            (root / "etc" / "os-release").write_text(
                'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu on microsoft hardware"\n')

            for release, wsl in (("7.0.0-34-generic\n", False),
                                 ("5.15.167.4-microsoft-standard-WSL2\n", True),
                                 ("4.4.0-19041-Microsoft\n", True)):
                with self.subTest(release=release.strip()):
                    osrelease.write_text(release)
                    self.assertEqual(collect(linux_probe(root), root / "stack").wsl, wsl)

            osrelease.unlink()
            self.assertFalse(collect(linux_probe(root), root / "stack").wsl)

    def test_ram_is_mem_available_on_linux(self):
        """`MemAvailable` is the number the kernel gives for what is actually free."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "proc").mkdir(parents=True)
            (root / "proc" / "meminfo").write_text(
                "MemTotal:       65684804 kB\n"
                "MemFree:        51200000 kB\n"
                "MemAvailable:   1000 kB\n"
                "Buffers:          4096 kB\n")

            facts = collect(linux_probe(root), root / "stack")

        self.assertEqual(facts.ram_bytes, 1000 * 1024)

    def test_ram_on_macos_is_not_the_host_figure(self):
        """On macOS the host's `hw.memsize` is NOT what the containers get: they run
        in the machine VM (2048 MiB by default), so `collect` does not read `sysctl`
        and returns None (review round R2, item I6). The figure the install step
        checks is the engine's own, `EngineInfo.memory_bytes`."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            probe = Probe(root=root, system=lambda: "Darwin", machine=lambda: "arm64")

            facts = collect(probe, root / "stack")

        self.assertEqual(facts.system, "macos")
        self.assertEqual(facts.arch, "arm64")
        self.assertIsNone(facts.ram_bytes)

    def test_disk_is_measured_at_the_nearest_existing_ancestor(self):
        """`stack_dir` does not exist yet: the free space is read at the ancestor
        that does. The measurement goes through the injected `disk_usage`, so this
        test never reads a real filesystem (review round R2, item m8)."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stack_dir = root / "a" / "b" / "stack"  # a and b do not exist

            def fake_disk_usage(path):
                return SimpleNamespace(free=424242)  # a figure, not a real read

            facts = collect(linux_probe(root, disk_usage=fake_disk_usage), stack_dir)

        self.assertEqual(facts.disk_free_bytes, 424242)

    def test_disk_walks_up_until_an_existing_path(self):
        """The walk records every candidate it tries: it walks up from the missing
        `stack` directory until one is measured, and only the first that answers is
        used (review round R2, item m8)."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stack_dir = root / "a" / "b" / "stack"
            asked: list = []

            def fake_disk_usage(path):
                asked.append(Path(path))
                if Path(path) == stack_dir or Path(path) == stack_dir.parent:
                    raise OSError("no such file")  # the not-yet-created levels
                return SimpleNamespace(free=1234)

            facts = collect(linux_probe(root, disk_usage=fake_disk_usage), stack_dir)

        self.assertEqual(facts.disk_free_bytes, 1234)
        # the walk tried the missing levels first, then the first existing one
        # (`root/a`), which answered, and stopped there
        self.assertEqual(asked, [stack_dir, stack_dir.parent, stack_dir.parent.parent])
        self.assertEqual(asked[-1], root / "a")

    def test_disk_returns_none_when_nothing_answers(self):
        """Even the root that the walk ends on raises: the answer is None, not a
        crash (review round R2, item m8)."""
        def refusing(path):
            raise OSError("no filesystem")

        facts = collect(linux_probe(Path(tempfile.mkdtemp()),
                                    disk_usage=refusing),
                        Path("/nonexistent/a/stack"))
        self.assertIsNone(facts.disk_free_bytes)

    def test_selinux_enforcing_is_read(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "sys" / "fs" / "selinux").mkdir(parents=True)

            (root / "sys" / "fs" / "selinux" / "enforce").write_text("1\n")
            self.assertTrue(collect(linux_probe(root), root / "stack").selinux)

            (root / "sys" / "fs" / "selinux" / "enforce").write_text("0\n")
            self.assertFalse(collect(linux_probe(root), root / "stack").selinux)

            (root / "sys" / "fs" / "selinux" / "enforce").unlink()
            self.assertFalse(collect(linux_probe(root), root / "stack").selinux)

    def test_the_system_and_arch_are_normalized(self):
        with tempfile.TemporaryDirectory() as raw:
            facts = collect(linux_probe(Path(raw)), Path(raw) / "stack")

        self.assertEqual((facts.system, facts.arch), ("linux", "amd64"))
        self.assertEqual(normalize_arch("x86_64"), facts.arch)


class TestNvidiaReadiness(unittest.TestCase):
    def test_nvidia_readiness_has_its_parts(self):
        """A nvidia card in the tree: every readiness part is read, nothing is skipped."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_pci(root, "0000:65:00.0", "0x10de", "0x030000")  # NVIDIA
            (root / "usr" / "share" / "vulkan" / "icd.d").mkdir(parents=True)
            (root / "usr" / "share" / "vulkan" / "icd.d" / "nvidia_icd.json").write_text(
                '{"file_format_version": "1.0.0"}\n')
            (root / "etc" / "cdi").mkdir(parents=True)
            (root / "etc" / "cdi" / "gpu.nvidia.yaml").write_text(
                # the device class an nvidia CDI spec is generated with; the code
                # looks for the `nvidia.com/gpu` marker in it (m7: `kind: RuntimeClass`
                # was an invention, this is the real nvidia-ctk output)
                "kind: nvidia.com/gpu\n"
                "metadata:\n  name: gpu0\n"
                "containers:\n  - type: cdiv1\n")

            def fake_which(name):
                if name in ("nvidia-container-runtime-hook", "nvidia-cdi-hook",
                            "nvidia-ctk"):
                    return f"/usr/bin/{name}"
                return None

            probe = linux_probe(root, runner=FakeRunner({
                ("nvidia-smi", "-L"):
                    runtimes.Completed(0,
                                       "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-0)\n"
                                       "GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-1)\n"),
            }), which=fake_which)

            facts = collect(probe, root / "stack")

        self.assertEqual(len(facts.gpus), 1)
        self.assertEqual(facts.gpus[0].vendor, "nvidia")
        self.assertEqual(facts.nvidia.gpus,
                         ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"))
        self.assertTrue(facts.nvidia.icd)
        self.assertTrue(facts.nvidia.docker_hook)  # nvidia-container-runtime-hook
        self.assertTrue(facts.nvidia.cdi_hook)     # nvidia-cdi-hook
        self.assertTrue(facts.nvidia.ctk)          # nvidia-ctk
        self.assertTrue(facts.nvidia.cdi_spec)

    def test_the_secondary_icd_and_cdi_paths_count_too(self):
        """The spec names a second home for each: the ICD in `/etc/vulkan/icd.d`
        (an admin drop) and the CDI spec in `/var/run/cdi` (the runtime's hand).
        With only the secondary paths present, both readiness parts still hold."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_pci(root, "0000:65:00.0", "0x10de", "0x030000")  # NVIDIA
            (root / "etc" / "vulkan" / "icd.d").mkdir(parents=True)
            (root / "etc" / "vulkan" / "icd.d" / "nvidia_icd.json").write_text(
                '{"file_format_version": "1.0.0"}\n')
            (root / "var" / "run" / "cdi").mkdir(parents=True)
            (root / "var" / "run" / "cdi" / "nvidia.yaml").write_text(
                "containers:\n  - name: nvidia.com/gpu\n")

            probe = linux_probe(root, runner=FakeRunner({
                ("nvidia-smi", "-L"): runtimes.Completed(0, "")}))

            facts = collect(probe, root / "stack")

        self.assertTrue(facts.nvidia.icd)
        self.assertTrue(facts.nvidia.cdi_spec)
        # the hooks and ctk are absent on this tree: only the file facts are set
        self.assertFalse(facts.nvidia.docker_hook)
        self.assertFalse(facts.nvidia.cdi_hook)
        self.assertFalse(facts.nvidia.ctk)

    def test_without_a_nvidia_card_the_facts_stay_the_empty_default(self):
        """No nvidia in the tree: no nvidia-smi call, no file reads, the default stands."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_pci(root, "0000:03:00.0", "0x8086", "0x030000")  # Intel
            runner = FakeRunner({("nvidia-smi", "-L"):  # must never be consulted
                                 runtimes.Completed(0, "GPU 0: never\n")})

            facts = collect(linux_probe(root, runner=runner), root / "stack")

        self.assertEqual(facts.nvidia, NvidiaFacts())
        self.assertEqual(runner.calls, [])


class TestNormalizeAndPlatformOf(unittest.TestCase):
    def test_normalize_system_folds_the_platform_names(self):
        self.assertEqual(normalize_system("Linux"), "linux")
        self.assertEqual(normalize_system("Darwin"), "macos")
        self.assertEqual(normalize_system("Windows"), "windows")
        self.assertEqual(normalize_system("linux"), "linux")

    def test_platform_of(self):
        base = dict(system="linux", arch="amd64", wsl=False, ram_bytes=None,
                    disk_free_bytes=None, gpus=(), render_nodes=(), selinux=False)
        self.assertEqual(platform_of(HostFacts(**base)), "linux")
        self.assertEqual(platform_of(HostFacts(system="macos", **{k: v for k, v in base.items()
                                                                 if k != "system"})), "macos")
        self.assertEqual(platform_of(HostFacts(system="windows", **{k: v for k, v in base.items()
                                                                   if k != "system"})), "windows")
        # WSL-on-Linux: the host says linux, the kernel says microsoft
        self.assertEqual(platform_of(HostFacts(**{**base, "wsl": True})), "windows")
        self.assertEqual(
            platform_of(HostFacts(**base), engine_kernel="5.10.16.3-microsoft-standard-WSL2"),
            "windows")

    def test_the_default_probe_points_at_the_real_root_and_runner(self):
        # R2 item I5: a production `Probe()` must run `nvidia-smi`, so `runner`
        # defaults to a real `SubprocessRunner`, not `None` (tests inject a fake).
        self.assertEqual(Probe().root, Path("/"))
        self.assertEqual(Probe().system(), "Linux")
        self.assertIsNotNone(Probe().which("sh"))
        self.assertIsInstance(Probe().runner, runtimes.SubprocessRunner)
        # R2 item m8: the disk measurement is injected, defaulting to `shutil.disk_usage`.
        self.assertIsNotNone(Probe().disk_usage)


class TestPortFree(unittest.TestCase):
    def test_port_free_says_no_for_a_bound_port(self):
        finder = socket.socket()
        finder.bind(("127.0.0.1", 0))
        free_port = finder.getsockname()[1]
        finder.close()

        holder = socket.socket()
        holder.bind(("127.0.0.1", free_port))
        try:
            self.assertFalse(port_free(free_port))
        finally:
            holder.close()
        # and a genuinely free port (re-bound by nobody) answers yes
        self.assertTrue(port_free(free_port))


if __name__ == "__main__":
    unittest.main(verbosity=2)
