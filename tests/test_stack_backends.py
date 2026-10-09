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
    DznGpu,
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


def engine(name="docker", version="28.0.0", os="linux", rootless=True, vm=None,
           arch="amd64") -> EngineInfo:
    """A frozen engine; only the fields the cases pin differ."""
    return EngineInfo(name=name, version=version, os=os, arch=arch,
                      rootless=rootless, kernel="6.14.0-37-generic", vm=vm,
                      socket=None)


#: The documented CDI generation: without `--output` it prints the spec to stdout
#: and writes nothing, so the profile would stay missing after the "fix".
CDI_GENERATE = "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml (as root)"

#: The real `--list-devices` output measured on this machine (2026-10-06).
DEVICES_OUTPUT = """\
Available devices:
  Vulkan0: Intel(R) Graphics (BMG G31) (32656 MiB, 29268 MiB free)
  Vulkan1: AMD Radeon RX 6900 XT (RADV NAVI21) (16368 MiB, 4018 MiB free)
  Vulkan2: Intel(R) Graphics (BMG G31) (32656 MiB, 29289 MiB free)
"""

NONE_OUTPUT = "Available devices:\n  (none)\n"

#: A dzn `--list-devices` run on WSL2 with the GPU passthrough: the adapter's
#: own name wrapped by Mesa's dzn driver (`Microsoft Direct3D12 (<adapter>)`).
DZN_OUTPUT = (
    "Available devices:\n"
    "  Vulkan0: Microsoft Direct3D12 (NVIDIA GeForce RTX 4090) "
    "(24564 MiB, 23000 MiB free)\n"
)

#: The same dzn container WITHOUT the WSL2 GPU passthrough: the Mesa Vulkan
#: driver falls back to `llvmpipe` (CPU software rendering), which is a CPU
#: device, not a dzn GPU.
DZN_LLVMPIPE_OUTPUT = (
    "Available devices:\n"
    "  Vulkan0: llvmpipe (LLVM 15.0.7, 256 bits) (0 MiB, 0 MiB free)\n"
)


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
        # The dzn device (Windows, phase 3) wraps the adapter's own name: Mesa's
        # src/microsoft/vulkan/dzn_device.c at mesa-26.0.3, line 1071, names it
        # "Microsoft Direct3D12 (%s)" with the adapter's description, and a WSL2
        # container printed `deviceName = Microsoft Direct3D12 (NVIDIA GeForce GTX
        # 1080)` (microsoft/wslg issue 1215, comment of 2026-02-20). The vendor's own
        # token is inside, so the same tokens apply.
        self.assertEqual("nvidia",
                         vendor_of("Microsoft Direct3D12 (NVIDIA GeForce GTX 1080)"))
        # The tokens are matched case-sensitively, as llama.cpp prints them.
        self.assertIsNone(vendor_of("nvidia geforce rtx 4090"))
        self.assertIsNone(vendor_of("intel(R) graphics"))


class CompatibilityMatrixTest(unittest.TestCase):
    def test_the_compatibility_matrix_is_the_specs(self):
        # Absolute: the spec's "Compatibilidade" table is the only source.
        # Every (platform, profile) pair this version serves, pinned; the rest
        # must be empty. The cpu profile now serves windows too (cross-platform
        # wizard, spec 2026-10-09): a WSL2 host with a runtime is the windows
        # platform, and the cpu runs the official image there. The GPU profiles
        # still do not (the dzn path is a separate profile); neither does a
        # platform the spec does not name.
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
            # dzn is the ONLY gpu path on windows: the official image never
            # reaches the gpu, so the gpu profiles (amd/intel/nvidia) run the
            # dzn image through it, on docker. Podman has no gpu path there
            # (no WSL2 gpu passthrough), so dzn is docker-only.
            ("windows", "dzn"): frozenset({"docker"}),
        }
        for platform in ("linux", "macos", "windows", "freebsd"):
            for profile in ("cpu", "amd", "intel", "nvidia", "apple", "dzn"):
                with self.subTest(platform=platform, profile=profile):
                    self.assertEqual(expected.get((platform, profile), frozenset()),
                                     BACKENDS[profile].runtimes(platform))

    def test_backends_dict_is_ordered_and_complete(self):
        self.assertEqual(("cpu", "amd", "intel", "nvidia", "apple", "dzn"),
                         tuple(BACKENDS))
        self.assertEqual(("amd", "intel", "nvidia", "apple", "dzn"), GPU_PROFILES)
        self.assertIsInstance(BACKENDS["cpu"], Cpu)
        self.assertIsInstance(BACKENDS["amd"], DriGpu)
        self.assertIsInstance(BACKENDS["intel"], DriGpu)
        self.assertIsInstance(BACKENDS["nvidia"], NvidiaGpu)
        self.assertIsInstance(BACKENDS["apple"], AppleGpu)
        self.assertIsInstance(BACKENDS["dzn"], DznGpu)
        self.assertEqual("amd", BACKENDS["amd"].vendor)
        self.assertEqual("intel", BACKENDS["intel"].vendor)
        self.assertEqual("nvidia", BACKENDS["nvidia"].vendor)
        self.assertEqual("apple", BACKENDS["apple"].vendor)
        self.assertIsNone(BACKENDS["cpu"].vendor)
        # dzn is the shared windows gpu profile: the vendor comes from the adapter
        # name at runtime (the Direct3D12 wrapper), not from the profile itself.
        self.assertIsNone(BACKENDS["dzn"].vendor)
        self.assertEqual("llama", BACKENDS["amd"].image_role)
        self.assertEqual("llama", BACKENDS["nvidia"].image_role)
        self.assertEqual("llama", BACKENDS["apple"].image_role)
        self.assertIsNone(BACKENDS["cpu"].image_role)
        # The dzn profile runs the OWN image, not the official `llama` one.
        self.assertEqual("llama-dzn", BACKENDS["dzn"].image_role)
        self.assertFalse(BACKENDS["cpu"].experimental)
        self.assertFalse(BACKENDS["amd"].experimental)
        self.assertFalse(BACKENDS["intel"].experimental)
        self.assertFalse(BACKENDS["nvidia"].experimental)
        self.assertTrue(BACKENDS["apple"].experimental)
        # dzn is offered, not experimental: its non-conformance is handled by the
        # numeric verification step (spec 2026-10-09), not by hiding it.
        self.assertFalse(BACKENDS["dzn"].experimental)


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
             host(gpus=(Gpu(vendor="nvidia", card="0000:65:00.0"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=False,
                                     docker_hook=True, cdi_spec=True)),
             engine("podman", "5.7.0"), "podman",
             MISSING, "no nvidia vulkan icd on the host",
             "install the driver's GL/Vulkan package (libnvidia-gl-<version> on Ubuntu), "
             "then " + CDI_GENERATE, None),
            # spec, "Detecção": on Docker the icd is the whole fix; on Podman the
            # CDI spec generated before it lacks the icd, so it is generated again
            ("nvidia without the icd on docker", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="0000:65:00.0"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=False,
                                     docker_hook=True, cdi_spec=True)),
             engine("docker"), "docker",
             MISSING, "no nvidia vulkan icd on the host",
             "install the driver's GL/Vulkan package (libnvidia-gl-<version> on Ubuntu)",
             None),
            # this machine: two Intel cards and an AMD one, no NVIDIA at all
            ("nvidia on a host with no nvidia gpu", "nvidia", "linux",
             host(gpus=(Gpu(vendor="intel", card="card0"), Gpu(vendor="amd", card="card1")),
                  render_nodes=("renderD128", "renderD129")),
             engine("podman", "5.7.0"), "podman",
             UNSUPPORTED, "no nvidia gpu on the host", None, None),
            # the card is on the bus, but no driver lists it (nouveau, or none)
            ("nvidia card that nvidia-smi does not list", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="0000:65:00.0"),), render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=(), icd=False)),
             engine("docker"), "docker",
             MISSING, "nvidia-smi lists no gpu", "install the NVIDIA driver", None),
            ("nvidia docker without the hook", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="0000:65:00.0"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=True,
                                     docker_hook=False, cdi_spec=True)),
             engine("docker"), "docker",
             MISSING, "no nvidia container runtime hook",
             "install nvidia-container-toolkit", None),
            # R2 item m3: the cdi hook counts for Docker only from 29.2. On 28.x it
            # does not, so a host with only the cdi hook and Docker 28.5 is missing.
            ("nvidia docker cdi hook but docker below 29.2", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="0000:65:00.0"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=True,
                                     docker_hook=False, cdi_hook=True, cdi_spec=True)),
             engine("docker", "28.5.2"), "docker",
             MISSING, "no nvidia container runtime hook",
             "install nvidia-container-toolkit", None),
            # the same host on Docker 29.2: the cdi hook counts, ready.
            ("nvidia docker cdi hook on docker 29.2", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="0000:65:00.0"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=True,
                                     docker_hook=False, cdi_hook=True, cdi_spec=True)),
             engine("docker", "29.2.0"), "docker",
             READY, "", None, None),
            ("nvidia podman without the cdi spec, with nvidia-ctk", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="0000:65:00.0"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=True,
                                     docker_hook=True, cdi_spec=False, ctk=True)),
             engine("podman", "5.7.0"), "podman",
             MISSING, "no nvidia cdi spec", CDI_GENERATE, None),
            # the toolkit is not installed: the fix names it first (R2 item m4)
            ("nvidia podman without the cdi spec, without nvidia-ctk", "nvidia", "linux",
             host(gpus=(Gpu(vendor="nvidia", card="0000:65:00.0"),),
                  render_nodes=("renderD128",),
                  nvidia=NvidiaFacts(gpus=("NVIDIA GeForce RTX 4090",), icd=True,
                                     docker_hook=True, cdi_spec=False, ctk=False)),
             engine("podman", "5.7.0"), "podman",
             MISSING, "no nvidia cdi spec",
             "install nvidia-container-toolkit, then " + CDI_GENERATE, None),
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
            # R2 item I7: ONE fix for podman 5 and 6. The provider must persist in
            # the [machine] table of containers.conf; a one-shot env at `init` does
            # not, because `machine info` and `machine start` read the configured
            # provider for each command.
            ("apple with an applehv vm on podman 5", "apple", "macos",
             host(system="macos", arch="arm64"),
             engine("podman", "5.7.0", os="linux", vm="applehv"), "podman",
             MISSING, "the podman machine is not libkrun",
             "set provider = \"libkrun\" in the [machine] table of "
             "~/.config/containers/containers.conf (and unset "
             "CONTAINERS_MACHINE_PROVIDER), then podman machine init --now", None),
            ("apple with an applehv vm on podman 6", "apple", "macos",
             host(system="macos", arch="arm64"),
             engine("podman", "6.0.0", os="linux", vm="applehv"), "podman",
             MISSING, "the podman machine is not libkrun",
             "set provider = \"libkrun\" in the [machine] table of "
             "~/.config/containers/containers.conf (and unset "
             "CONTAINERS_MACHINE_PROVIDER), then podman machine init --now", None),
            ("intel mac", "apple", "macos",
             host(system="macos", arch="amd64"),
             engine("podman", "5.7.0", os="linux"), "podman",
             UNSUPPORTED, "no container gpu path on an intel mac", None, None),
            # Python under Rosetta reports x86_64 on an Apple Silicon Mac; the
            # engine's arch (the VM's) is what the spec says counts on macOS
            ("apple under rosetta", "apple", "macos",
             host(system="macos", arch="amd64"),
             engine("podman", "5.7.0", os="linux", vm="libkrun", arch="arm64"), "podman",
             READY, "", None, None),
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
            # dzn is the windows-only gpu profile. A bare windows host's
            # /sys/bus/pci shows the Windows host's PCI bus, not what the dzn
            # driver sees, so availability does NOT gate on facts.gpus: the
            # gpu proof (does /dev/dxg actually expose a gpu) happens in the
            # installer's prove step, not here (spec 2026-10-09, decision 21).
            ("dzn is not offered off windows", "dzn", "linux",
             host(),
             engine("docker"), "docker",
             UNSUPPORTED, "dzn runs on windows only", None, None),
            ("dzn is not offered off windows (macos)", "dzn", "macos",
             host(system="macos", arch="arm64"),
             engine("podman", "5.7.0", os="linux"), "podman",
             UNSUPPORTED, "dzn runs on windows only", None, None),
            # On windows the gpu path is docker-only: podman's WSL2 integration
            # carries no gpu passthrough, so the profile points at docker.
            ("dzn on windows with podman", "dzn", "windows",
             host(system="windows", arch="amd64"),
             engine("podman", "5.7.0", os="linux"), "podman",
             RUNTIME, "podman has no gpu path on windows", None, "docker"),
            # docker on windows: ready, without requiring facts.gpus.
            ("dzn ready on windows docker without facts.gpus", "dzn", "windows",
             host(system="windows", arch="amd64", gpus=()),
             engine("docker", "28.0.0", os="linux"), "docker",
             READY, "", None, None),
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

    def test_cpu_is_ready_on_linux_macos_and_windows(self):
        # The cpu profile serves every platform this version supports: linux,
        # macos and windows (a WSL2 host with a runtime is the windows platform,
        # cross-platform wizard plan, Task 1). The one-line refusal for a bare
        # Windows lives in `install_step` (`facts.is_native_windows`), not here.
        for platform in ("linux", "macos", "windows"):
            with self.subTest(platform=platform):
                availability = BACKENDS["cpu"].availability(
                    platform, host(), engine("podman", "5.7.0"), "podman")
                self.assertEqual(READY, availability.state)
                self.assertEqual("cpu", availability.reason)
                self.assertIsNone(availability.fix)
        # A platform the spec does not name is still unsupported.
        for platform in ("freebsd",):
            with self.subTest(platform=platform):
                availability = BACKENDS["cpu"].availability(
                    platform, host(), engine("podman", "5.7.0"), "podman")
                self.assertEqual(UNSUPPORTED, availability.state)
                self.assertIsNone(availability.fix)

    def test_cpu_on_windows_runs_the_official_image_with_no_device(self):
        # Spec 2026-10-09: the cpu profile on WSL2 deliberately does NOT use the
        # dzn image (the GPU profiles do, and that is a separate profile). It
        # runs the official image with `-dev none`, exactly like on linux: the
        # compose diff is empty.
        self.assertEqual({}, BACKENDS["cpu"].service_patch("docker", None))
        self.assertEqual({}, BACKENDS["cpu"].service_patch("podman", None))


class ServicePatchTest(unittest.TestCase):
    def test_an_nvidia_patch_without_a_gpu_index_is_refused(self):
        # `nvidia.com/gpu=None` would reach the compose file and fail at `up`
        for runtime in ("docker", "podman"):
            with self.subTest(runtime=runtime):
                with self.assertRaises(StackError) as ctx:
                    BACKENDS["nvidia"].service_patch(runtime, None)
                self.assertEqual("backends", ctx.exception.step)

    def test_service_patches(self):
        self.assertEqual({"devices": ["/dev/dri:/dev/dri"],
                          "annotations": {"run.oci.keep_original_groups": "1"}},
                         BACKENDS["amd"].service_patch("podman", None))
        self.assertEqual({"devices": ["/dev/dri:/dev/dri"]},
                         BACKENDS["intel"].service_patch("docker", None))
        self.assertEqual({"deploy": {"resources": {"reservations": {"devices":
                         [{"driver": "nvidia", "device_ids": ["1"],
                           "capabilities": ["gpu", "compute", "utility",
                                            "graphics"]}]}}}},
                         BACKENDS["nvidia"].service_patch("docker", 1))
        self.assertEqual({"devices": ["nvidia.com/gpu=1"]},
                         BACKENDS["nvidia"].service_patch("podman", 1))
        self.assertEqual({"devices": ["/dev/dri:/dev/dri"]},
                         BACKENDS["apple"].service_patch("docker", None))
        # R2 item R2-8: the apple profile on Podman carries the same keep_original_groups
        # annotation the dri profiles do: the machine runs rootless by default.
        self.assertEqual({"devices": ["/dev/dri:/dev/dri"],
                          "annotations": {"run.oci.keep_original_groups": "1"}},
                         BACKENDS["apple"].service_patch("podman", None))
        self.assertEqual({}, BACKENDS["cpu"].service_patch("docker", None))
        # The annotation rides on EVERY podman service patch of a dri profile:
        # it is required rootless and harmless rootful, so there is no branch
        # on rootless.
        self.assertEqual({"devices": ["/dev/dri:/dev/dri"],
                          "annotations": {"run.oci.keep_original_groups": "1"}},
                         BACKENDS["intel"].service_patch("podman", None))
        # The dzn patch (spec-mestra line ~380, "Perfis de backend") is the
        # Windows gpu profile verbatim: /dev/dxg in, /usr/lib/wsl mounted read-
        # only, and LD_LIBRARY_PATH at the WSL-provided Vulkan loader libs. It
        # does not depend on the runtime or a gpu index: there is one gpu path,
        # the D3D12 device the WSL2 host exposes.
        dzn_patch = {"devices": ["/dev/dxg"],
                     "volumes": ["/usr/lib/wsl:/usr/lib/wsl:ro"],
                     "environment": ["LD_LIBRARY_PATH=/usr/lib/wsl/lib"]}
        self.assertEqual(dzn_patch, BACKENDS["dzn"].service_patch("docker", None))
        self.assertEqual(dzn_patch, BACKENDS["dzn"].service_patch("docker", 0))


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

    def test_dzn_devices_seen_keeps_the_direct3d12_adapter(self):
        # The dzn driver names the device `Microsoft Direct3D12 (<adapter>)`:
        # the adapter is the real gpu, and the vendor is read from the name
        # inside the parentheses (the NVIDIA token is inside, so `vendor_of`
        # resolves it). One dzn device in the output is one device seen.
        devices = BACKENDS["dzn"].devices_seen(DZN_OUTPUT)
        self.assertEqual(1, len(devices))
        self.assertEqual("Vulkan0", devices[0].id)
        self.assertEqual("Microsoft Direct3D12 (NVIDIA GeForce RTX 4090)",
                         devices[0].name)
        self.assertEqual("nvidia", devices[0].vendor)
        # The other vendor's adapters resolve through the same wrapper: AMD and
        # Intel tokens inside the parentheses.
        amd = ("Available devices:\n  Vulkan0: Microsoft Direct3D12 "
               "(AMD Radeon RX 7900 XTX) (24576 MiB, 22000 MiB free)\n")
        self.assertEqual("amd", BACKENDS["dzn"].devices_seen(amd)[0].vendor)
        intel = ("Available devices:\n  Vulkan0: Microsoft Direct3D12 "
                 "(Intel(R) Arc(TM) A770) (16384 MiB, 15000 MiB free)\n")
        self.assertEqual("intel", BACKENDS["dzn"].devices_seen(intel)[0].vendor)
        # On the plain (non-dzn) output the adapter name is not wrapped in
        # `Microsoft Direct3D12 (...)`, so it is not a dzn device.
        self.assertEqual([], BACKENDS["dzn"].devices_seen(DEVICES_OUTPUT))
        self.assertEqual([], BACKENDS["dzn"].devices_seen(NONE_OUTPUT))

    def test_dzn_devices_seen_excludes_llvmpipe(self):
        # Without the WSL2 gpu passthrough the dzn container lists `llvmpipe`
        # (CPU software rendering). That is not a dzn gpu and must not count:
        # the profile would otherwise claim a gpu the host never exposes.
        self.assertEqual([], BACKENDS["dzn"].devices_seen(DZN_LLVMPIPE_OUTPUT))


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
