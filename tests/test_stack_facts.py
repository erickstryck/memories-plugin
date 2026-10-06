"""Host facts: the OS, the RAM, the disk, the GPUs, the SELinux mode, the NVIDIA
readiness.

The whole point is the injected `Probe`: every fact is read from files under
`probe.root`, every command through `probe.runner`, so a fake tree in a temp
directory stands in for the machine. Nothing here CHOOSES a GPU (that is Task 5)
and nothing here reads the GPU TYPE: measured on this machine (2026-10-06, spec
"Escolha da GPU"), the `--list-devices` output never carries a discrete/integrated
flag, so free memory is the only signal a choice may use, and the facts stay
typeless by design (M5).

The GPU tree is shaped like this machine's: two Intel cards, one AMD card, and an
ASPEED BMC card that must be ignored, plus a connector directory that must not be
mistaken for a card.
"""
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

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


def make_card(root: Path, name: str, vendor: str | None, label: str | None) -> None:
    """One `cardN` under the fake `/sys/class/drm`, as the machine prints it.

    A `None` vendor or label leaves the file out: the fallbacks `collect` must
    apply are exactly what a real tree does (the BMC card has a vendor with no
    label here, and the labels are empty on some systems).
    """
    card = root / "sys" / "class" / "drm" / name
    (card / "device").mkdir(parents=True)
    if vendor is not None:
        (card / "device" / "vendor").write_text(f"{vendor}\n")
    if label is not None:
        (card / "device" / "label").write_text(f"{label}\n")


def linux_probe(tmp: Path, **overrides) -> Probe:
    """A Probe aimed at `tmp` with the machine answered by the injected callables."""
    return Probe(root=tmp, system=lambda: "Linux", machine=lambda: "x86_64",
                 **overrides)


class TestGpuDiscovery(unittest.TestCase):
    def test_this_machines_shape_two_intel_one_amd_and_the_bmc_ignored(self):
        """card0..card3 as on this machine: 0x8086, 0x1002, 0x8086, and the 0x1a03
        ASPEED BMC, plus a connector dir that is not a card at all."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_card(root, "card0", "0x8086", "Intel(R) Graphics (BMG G31)")
            make_card(root, "card1", "0x1002", "Radeon RX 6900 XT")
            make_card(root, "card2", "0x8086", "Intel(R) Graphics (BMG G31)")
            make_card(root, "card3", "0x1a03", None)  # the BMC: a vendor, no label
            (root / "sys" / "class" / "drm" / "card0-DP-1").mkdir(parents=True)

            facts = collect(linux_probe(root), root / "stack")

        self.assertEqual(len(facts.gpus), 3, f"BMC and connector must be out: {facts.gpus}")
        self.assertEqual(facts.gpus[0], Gpu(vendor="intel",
                                            card="Intel(R) Graphics (BMG G31)"))
        self.assertEqual(facts.gpus[1], Gpu(vendor="amd", card="Radeon RX 6900 XT"))
        self.assertEqual(facts.gpus[2], Gpu(vendor="intel",
                                            card="Intel(R) Graphics (BMG G31)"))
        self.assertNotIn("0x1a03", VENDORS)

    def test_a_card_without_a_label_falls_back_to_its_directory_name(self):
        """This machine's own cards carry no `device/label`: the fallback must hold."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_card(root, "card1", "0x8086", None)

            facts = collect(linux_probe(root), root / "stack")

        self.assertEqual(facts.gpus, (Gpu(vendor="intel", card="card1"),))

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

    def test_ram_on_macos_comes_from_sysctl(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            probe = Probe(root=root, system=lambda: "Darwin", machine=lambda: "arm64",
                          runner=FakeRunner({
                              ("sysctl", "hw.memsize"):
                                  runtimes.Completed(0, "hw.memsize: 17179869184\n")}))

            facts = collect(probe, root / "stack")

        self.assertEqual(facts.system, "macos")
        self.assertEqual(facts.arch, "arm64")
        self.assertEqual(facts.ram_bytes, 16 * 1024**3)

    def test_disk_is_measured_at_the_nearest_existing_ancestor(self):
        """`stack_dir` does not exist yet: the free space is read at the ancestor that does."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stack_dir = root / "a" / "b" / "stack"  # a and b do not exist
            facts = collect(linux_probe(root), stack_dir)
            # the nearest existing ancestor is `root` itself: capture its free space
            # while the tree still exists so the numbers below must agree
            expected = shutil.disk_usage(root).free

        self.assertIsNotNone(facts.disk_free_bytes)
        self.assertGreater(facts.disk_free_bytes, 0)
        self.assertEqual(facts.disk_free_bytes, expected)

    def test_disk_falls_back_to_the_filesystem_root(self):
        """Nothing on the way up exists but `/`: the root's free space is still the
        answer, not `None`."""
        missing = Path("/nonexistent-qctx-stack-review")
        facts = collect(linux_probe(missing), missing / "a" / "stack")

        self.assertIsNotNone(facts.disk_free_bytes)
        self.assertGreater(facts.disk_free_bytes, 0)

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
    def test_nvidia_readiness_has_its_three_parts(self):
        """A nvidia card in the tree: every readiness part is read, nothing is skipped."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_card(root, "card0", "0x10de", "NVIDIA GeForce RTX 4090")
            (root / "usr" / "share" / "vulkan" / "icd.d").mkdir(parents=True)
            (root / "usr" / "share" / "vulkan" / "icd.d" / "nvidia_icd.json").write_text(
                '{"file_format_version": "1.0.0"}\n')
            (root / "etc" / "cdi").mkdir(parents=True)
            (root / "etc" / "cdi" / "gpu.nvidia.yaml").write_text(
                "kind: RuntimeClass\nspec:\n  containers:\n    - name: nvidia.com/gpu\n")

            def fake_which(name):
                if name in ("nvidia-container-runtime-hook", "nvidia-cdi-hook"):
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
        self.assertTrue(facts.nvidia.docker_hook)
        self.assertTrue(facts.nvidia.cdi_spec)

    def test_the_secondary_icd_and_cdi_paths_count_too(self):
        """The spec names a second home for each: the ICD in `/etc/vulkan/icd.d`
        (an admin drop) and the CDI spec in `/var/run/cdi` (the runtime's hand).
        With only the secondary paths present, both readiness parts still hold."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_card(root, "card0", "0x10de", "NVIDIA GeForce RTX 4090")
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

    def test_without_a_nvidia_card_the_facts_stay_the_empty_default(self):
        """No nvidia in the tree: no nvidia-smi call, no file reads, the default stands."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_card(root, "card0", "0x8086", "Intel(R) Graphics")
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

    def test_the_default_probe_points_at_the_real_root(self):
        self.assertEqual(Probe().root, Path("/"))
        self.assertEqual(Probe().system(), "Linux")
        self.assertIsNotNone(Probe().which("sh"))
        self.assertIsNone(Probe().runner)


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
