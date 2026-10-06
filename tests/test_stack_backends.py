"""The backend profiles: what each one receives, where it runs, and which one is
the default.

The compatibility matrix here is ABSOLUTE: it pins `Backend.runtimes` to the
spec's "Compatibilidade" table (docs/superpowers/specs/2026-10-05-local-stack-
design.md), not to what the implementation happens to return. The availability
table is pinned the same way, including the fix strings the spec names. The
measured `--list-devices` output (global context, "Real `--list-devices` output")
is the parser's fixture, with the podman-compose `\r\n` (M4) and a stray log
line in the middle.

The default follows M5: the `uma:` line never comes out of `--list-devices`, so
free memory is the only signal, and an experimental profile (apple) is never
the default even when it is ready.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError  # noqa: E402
from stack.backends import (  # noqa: E402
    BACKENDS,
    GPU_PROFILES,
    MISSING,
    READY,
    RUNTIME,
    UNSUPPORTED,
    Availability,
    Cpu,
    DriGpu,
    NvidiaGpu,
    Option,
    AppleGpu,
    default_option,
    parse_devices,
    runtime_label,
    vendor_of,
)
from stack.facts import Gpu, HostFacts, NvidiaFacts  # noqa: E402
from stack.runtimes import EngineInfo  # noqa: E402


def host(system="linux", arch="amd64", gpus=(), render_nodes=(), nvidia=NvidiaFacts()) -> HostFacts:
    """A frozen host with everything empty except what the case sets: the
    availability tests each pin one fact at a time."""
    return HostFacts(system=system, arch=arch, wsl=False, ram_bytes=None,
                     disk_free_bytes=None, gpus=tuple(gpus),
                     render_nodes=tuple(render_nodes), selinux=False, nvidia=nvidia)


def engine(name="docker", version="28.0.0", os="linux", rootless=True, vm=None) -> EngineInfo:
    """A frozen engine; only the fields the cases pin differ."""
    return EngineInfo(name=name, version=version, os=os, arch="amd64",
                      rootless=rootless, kernel="6.14.0-37-generic", vm=vm,
                      socket=None)


#: The real `--list-devices` output measured on this machine (2026-10-06).
DEVICES_OUTPUT = """\
Available devices:
  Vulkan0: Intel(R) Graphics (BMG G31) (32656 MiB, 29268 MiB free)
  Vulkan1: AMD Radeon RX 6900 XT (RADV NAVI21) (16368 MiB, 4018 MiB free)
  Vulkan2: Intel(R) Graphics (BMG G31) (32656 MiB, 29289 MiB free)
"""

NONE_OUTPUT = "Available devices:\n  (none)\n"


class ParseDevicesTest(unittest.TestCase):
    def test_parse_devices_on_this_machines_output(self):
        devices = parse_devices(DEVICES_OUTPUT)
        self.assertEqual(3, len(devices))
        self.assertEqual(0, devices[0].index)
        self.assertEqual("Vulkan0", devices[0].id)
        self.assertEqual("Intel(R) Graphics (BMG G31)", devices[0].name)
        self.assertEqual(32656, devices[0].total_mib)
        self.assertEqual(29268, devices[0].free_mib)
        self.assertEqual("intel", devices[0].vendor)
        self.assertEqual(1, devices[1].index)
        self.assertEqual("Vulkan1", devices[1].id)
        self.assertEqual("AMD Radeon RX 6900 XT (RADV NAVI21)", devices[1].name)
        self.assertEqual(16368, devices[1].total_mib)
        self.assertEqual(4018, devices[1].free_mib)
        self.assertEqual("amd", devices[1].vendor)
        self.assertEqual(2, devices[2].index)
        self.assertEqual("Vulkan2", devices[2].id)
        self.assertEqual(32656, devices[2].total_mib)
        self.assertEqual(29289, devices[2].free_mib)
        self.assertEqual("intel", devices[2].vendor)

    def test_parse_devices_none_crlf_and_noise(self):
        self.assertEqual([], parse_devices(NONE_OUTPUT))
        self.assertEqual([], parse_devices(NONE_OUTPUT.replace("\n", "\r\n")))
        # The podman-compose output carries \r\n (M4): the same three lines parse
        # to the same devices.
        self.assertEqual(parse_devices(DEVICES_OUTPUT),
                         parse_devices(DEVICES_OUTPUT.replace("\n", "\r\n")))
        # A stray log line in the middle is not a device line: ignored, and the
        # indices keep coming from the `Vulkan<n>` prefix, not the position.
        noisy = DEVICES_OUTPUT.replace(
            "Vulkan1: ",
            "  ggml_backend_vk: something happened\r\n  Vulkan1: ", 1)
        self.assertEqual(parse_devices(DEVICES_OUTPUT), parse_devices(noisy))
        self.assertEqual([], parse_devices(""))
        self.assertEqual([], parse_devices("Vulkan: no index here (100 MiB, 10 MiB free)\n"))


class VendorOfTest(unittest.TestCase):
    def test_vendor_of(self):
        self.assertEqual("nvidia", vendor_of("NVIDIA GeForce RTX 4090"))
        self.assertEqual("amd", vendor_of("AMD Radeon RX 6900 XT (RADV NAVI21)"))
        self.assertEqual("amd", vendor_of("RADV NAVI21"))
        self.assertEqual("intel", vendor_of("Intel(R) Graphics (BMG G31)"))
        self.assertEqual("apple", vendor_of("Virtio-GPU Venus (Apple M2 Pro)"))
        self.assertEqual("apple", vendor_of("Venus (Apple M3)"))
        self.assertIsNone(vendor_of("llvmpipe (LLVM 19.1.7, 256 bits)"))
        self.assertIsNone(vendor_of("Microsoft Direct3D12 (RTX 4090)"))
        # The tokens are matched case-sensitively, as llama.cpp prints them.
        self.assertIsNone(vendor_of("nvidia geforce rtx 4090"))
        self.assertIsNone(vendor_of("intel(R) graphics"))


class CompatibilityMatrixTest(unittest.TestCase):
    def test_the_compatibility_matrix_is_the_specs(self):
        # Absolute: the spec's "Compatibilidade" table is the only source.
        # Every (platform, profile) pair the spec names, pinned; the rest must
        # be empty. Windows enters in phase 3, so nothing runs there yet, but
        # the CPU row of the spec says both runtimes, so the contract keeps
        # the promise when phase 3 lands.
        both = frozenset({"docker", "podman"})
        podman = frozenset({"podman"})
        expected = {
            ("linux", "cpu"): both,
            ("linux", "amd"): both,
            ("linux", "intel"): both,
            ("linux", "nvidia"): both,
            ("linux", "apple"): frozenset(),
            ("macos", "cpu"): both,
            ("macos", "amd"): frozenset(),
            ("macos", "intel"): frozenset(),
            ("macos", "nvidia"): frozenset(),
            ("macos", "apple"): podman,
            ("windows", "cpu"): both,
            ("windows", "amd"): frozenset(),
            ("windows", "intel"): frozenset(),
            ("windows", "nvidia"): frozenset(),
            ("windows", "apple"): frozenset(),
        }
        for platform in ("linux", "macos", "windows"):
            for profile in ("cpu", "amd", "intel", "nvidia", "apple"):
                with self.subTest(platform=platform, profile=profile):
                    self.assertEqual(expected[(platform, profile)],
                                     BACKENDS[profile].runtimes(platform))

    def test_backends_dict_is_ordered_and_complete(self):
        self.assertEqual(("cpu", "amd", "intel", "nvidia", "apple"),
                         tuple(BACKENDS))
        self.assertEqual(("amd", "intel", "nvidia", "apple"), GPU_PROFILES)
        self.assertIsInstance(BACKENDS["cpu"], Cpu)
        self.assertIsInstance(BACKENDS["amd"], DriGpu)
        self.assertIsInstance(BACKENDS["intel"], DriGpu)
        self.assertIsInstance(BACKENDS["nvidia"], NvidiaGpu)
        self.assertIsInstance(BACKENDS["apple"], AppleGpu)
        self.assertEqual("amd", BACKENDS["amd"].vendor)
        self.assertEqual("intel", BACKENDS["intel"].vendor)
        self.assertEqual("nvidia", BACKENDS["nvidia"].vendor)
        self.assertEqual("apple", BACKENDS["apple"].vendor)
        self.assertIsNone(BACKENDS["cpu"].vendor)
        self.assertEqual("llama", BACKENDS["amd"].image_role)
        self.assertEqual("llama", BACKENDS["nvidia"].image_role)
        self.assertEqual("llama", BACKENDS["apple"].image_role)
        self.assertIsNone(BACKENDS["cpu"].image_role)
        self.assertFalse(BACKENDS["cpu"].experimental)
        self.assertFalse(BACKENDS["amd"].experimental)
        self.assertFalse(BACKENDS["intel"].experimental)
        self.assertFalse(BACKENDS["nvidia"].experimental)
        self.assertTrue(BACKENDS["apple"].experimental)


class AvailabilityTableTest(unittest.TestCase):
    def test_availability_table(self):
        cases = [
            # name, profile, platform, facts, engine, runtime, state, reason, fix, needs
            ("dri without the vendor's gpu", "intel", "linux",
             host(gpus=(Gpu(vendor="amd", card="AMD Radeon RX 6900 XT"),),
                  render_nodes=("renderD128",)),
             engine("podman", "5.7.0"), "podman",
             UNSUPPORTED, "no intel gpu on the host", None, None),
            ("dri with no render node", "amd", "linux",
             host(gpus=(Gpu(vendor="amd", card="AMD Radeon RX 6900 XT"),), render_nodes=()),
             engine("docker"), "docker",
             MISSING, "no /dev/dri/renderD* node",
             "install the amd GPU driver", None),
            ("dri ready on linux", "intel", "linux",
             host(gpus=(Gpu(vendor="intel", card="Intel(R) Graphics (BMG G31)"),),
                  render_nodes=("renderD128", "renderD129")),
             engine("podman", "5.7.0", rootless=False), "podman",
             READY, "", None, None),
            ("nvidia without the icd", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="NVIDIA GeForce RTX 4090"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=False,
                                     docker_hook=True, cdi_spec=True)),
             engine("podman", "5.7.0"), "podman",
             MISSING, "no nvidia vulkan icd on the host",
             "libnvidia-gl-<version>", None),
            ("nvidia docker without the hook", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="NVIDIA GeForce RTX 4090"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=True,
                                     docker_hook=False, cdi_spec=True)),
             engine("docker"), "docker",
             MISSING, "no nvidia container runtime hook",
             "install nvidia-container-toolkit", None),
            ("nvidia podman without the cdi spec", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="NVIDIA GeForce RTX 4090"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=True,
                                     docker_hook=True, cdi_spec=False)),
             engine("podman", "5.7.0"), "podman",
             MISSING, "no nvidia cdi spec", "nvidia-ctk cdi generate", None),
            ("nvidia ready on podman", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="NVIDIA GeForce RTX 4090"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=True,
                                     docker_hook=True, cdi_spec=True)),
             engine("podman", "5.7.0"), "podman",
             READY, "", None, None),
            ("apple on docker", "apple", "macos",
             host(system="macos", arch="arm64"),
             engine("docker", "4.40.0", os="linux"), "docker",
             RUNTIME, "docker desktop passes no gpu to a container", None, "podman"),
            ("apple with an applehv vm on podman 5", "apple", "macos",
             host(system="macos", arch="arm64"),
             engine("podman", "5.7.0", os="linux", vm="applehv"), "podman",
             MISSING, "the podman machine is not libkrun",
             "CONTAINERS_MACHINE_PROVIDER=libkrun podman machine init", None),
            ("apple with an applehv vm on podman 6", "apple", "macos",
             host(system="macos", arch="arm64"),
             engine("podman", "6.0.0", os="linux", vm="applehv"), "podman",
             MISSING, "the podman machine is not libkrun",
             "podman machine init --provider libkrun", None),
            ("intel mac", "apple", "macos",
             host(system="macos", arch="amd64"),
             engine("podman", "5.7.0", os="linux"), "podman",
             UNSUPPORTED, "no container gpu path on an intel mac", None, None),
            ("apple with a libkrun vm on arm64", "apple", "macos",
             host(system="macos", arch="arm64"),
             engine("podman", "5.7.0", os="linux", vm="libkrun"), "podman",
             READY, "", None, None),
            ("dri is not offered off linux", "amd", "macos",
             host(system="macos", arch="arm64"),
             engine("podman", "5.7.0", os="linux"), "podman",
             UNSUPPORTED, "dri gpu profiles run on linux only", None, None),
            ("nvidia is not offered off linux", "nvidia", "macos",
             host(system="macos", arch="arm64",
                  nvidia=NvidiaFacts(icd=True, cdi_spec=True)),
             engine("podman", "5.7.0", os="linux"), "podman",
             UNSUPPORTED, "nvidia runs on linux only", None, None),
            ("apple is not offered off macos", "apple", "linux",
             host(gpus=(Gpu(vendor="amd", card="AMD Radeon"),),
                  render_nodes=("renderD128",)),
             engine("docker"), "docker",
             UNSUPPORTED, "apple runs on macos only", None, None),
        ]
        for (name, profile, platform, facts, engine_info, runtime, state, reason,
             fix, needs) in cases:
            with self.subTest(name=name):
                availability = BACKENDS[profile].availability(
                    platform, facts, engine_info, runtime)
                self.assertEqual(state, availability.state)
                self.assertEqual(reason, availability.reason)
                self.assertEqual(fix, availability.fix)
                self.assertEqual(needs, availability.needs)

    def test_cpu_is_ready_on_every_platform(self):
        for platform in ("linux", "macos", "windows"):
            with self.subTest(platform=platform):
                availability = BACKENDS["cpu"].availability(
                    platform, host(), engine("podman", "5.7.0"), "podman")
                self.assertEqual(READY, availability.state)
                self.assertEqual("cpu", availability.reason)
                self.assertIsNone(availability.fix)


class ServicePatchTest(unittest.TestCase):
    def test_service_patches(self):
        self.assertEqual({"devices": ["/dev/dri"],
                          "annotations": {"run.oci.keep_original_groups": "1"}},
                         BACKENDS["amd"].service_patch("podman", None))
        self.assertEqual({"devices": ["/dev/dri"]},
                         BACKENDS["intel"].service_patch("docker", None))
        self.assertEqual({"deploy": {"resources": {"reservations": {"devices":
                         [{"driver": "nvidia", "device_ids": ["1"],
                           "capabilities": ["gpu", "compute", "utility",
                                            "graphics"]}]}}}},
                         BACKENDS["nvidia"].service_patch("docker", 1))
        self.assertEqual({"devices": ["nvidia.com/gpu=1"]},
                         BACKENDS["nvidia"].service_patch("podman", 1))
        self.assertEqual({"devices": ["/dev/dri"]},
                         BACKENDS["apple"].service_patch("podman", None))
        self.assertEqual({}, BACKENDS["cpu"].service_patch("docker", None))
        # The annotation rides on EVERY podman service patch of a dri profile:
        # it is required rootless and harmless rootful, so there is no branch
        # on rootless.
        self.assertEqual({"devices": ["/dev/dri"],
                          "annotations": {"run.oci.keep_original_groups": "1"}},
                         BACKENDS["intel"].service_patch("podman", None))


class DevicesSeenTest(unittest.TestCase):
    def test_devices_seen_filters_by_vendor(self):
        self.assertEqual([], BACKENDS["cpu"].devices_seen(DEVICES_OUTPUT))
        self.assertEqual(1, len(BACKENDS["amd"].devices_seen(DEVICES_OUTPUT)))
        self.assertEqual("Vulkan1", BACKENDS["amd"].devices_seen(DEVICES_OUTPUT)[0].id)
        self.assertEqual(2, len(BACKENDS["intel"].devices_seen(DEVICES_OUTPUT)))
        self.assertEqual(["Vulkan0", "Vulkan2"],
                         [d.id for d in BACKENDS["intel"].devices_seen(DEVICES_OUTPUT)])
        self.assertEqual([], BACKENDS["nvidia"].devices_seen(DEVICES_OUTPUT))
        self.assertEqual([], BACKENDS["apple"].devices_seen(DEVICES_OUTPUT))
        venus = ("Available devices:\n  Vulkan0: Virtio-GPU Venus (Apple M2 Pro) "
                 "(16384 MiB, 8000 MiB free)\n")
        self.assertEqual(1, len(BACKENDS["apple"].devices_seen(venus)))
        self.assertEqual("apple", BACKENDS["apple"].devices_seen(venus)[0].vendor)


class DefaultOptionTest(unittest.TestCase):
    def test_default_is_the_most_free_memory_and_never_experimental(self):
        intel_a = Option("intel", parse_devices(DEVICES_OUTPUT)[0], 0,
                         Availability(state=READY, reason=""))
        amd = Option("amd", parse_devices(DEVICES_OUTPUT)[1], 1,
                     Availability(state=READY, reason=""))
        intel_b = Option("intel", parse_devices(DEVICES_OUTPUT)[2], 2,
                         Availability(state=READY, reason=""))
        options = [intel_a, amd, intel_b]
        self.assertEqual("Vulkan2", default_option(options).device.id)
        # The experimental apple profile is ready and has a device: it is still
        # never the default.
        venus = parse_devices("Available devices:\n  Vulkan0: Virtio-GPU Venus "
                              "(Apple M2 Pro) (16384 MiB, 9000 MiB free)\n")[0]
        apple = Option("apple", venus, 0, Availability(state=READY, reason=""))
        self.assertEqual("Vulkan2", default_option([apple, intel_a, amd, intel_b]).device.id)
        # No ready device at all: the cpu option, no matter what it carries. A
        # ready option with no device is what a menu shows before the probe
        # found a card.
        self.assertEqual("cpu", default_option([Option("amd", None, 1,
                                                       Availability(state=READY,
                                                                    reason="")),
                                                Option("cpu", None, None,
                                                       Availability(state=READY,
                                                                    reason="cpu"))]).backend)
        # The cpu option is always in the list of a menu: it is the fallback,
        # so a list with neither a ready device nor a cpu option is a caller
        # bug.
        with self.assertRaises(StackError):
            default_option([Option("amd", None, 1,
                                   Availability(state=READY, reason=""))])
        # A ready option with no device is no candidate: the free-memory rule
        # still picks the ready one with the most free MiB.
        self.assertEqual("Vulkan2",
                         default_option([Option("amd", None, 1,
                                                Availability(state=READY, reason="")),
                                         intel_b]).device.id)


class RuntimeLabelTest(unittest.TestCase):
    def test_runtime_label(self):
        self.assertEqual("Docker and Podman",
                         runtime_label(frozenset({"docker", "podman"})))
        self.assertEqual("Docker and Podman",
                         runtime_label(frozenset({"podman", "docker"})))
        self.assertEqual("Podman only", runtime_label(frozenset({"podman"})))
        self.assertEqual("Docker only", runtime_label(frozenset({"docker"})))
        self.assertEqual("not available here", runtime_label(frozenset()))


if __name__ == "__main__":
    unittest.main()
