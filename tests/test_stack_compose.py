"""The compose file: what `render` builds from a plan, how `emit` writes it, and the golden
fixtures that are the compose of each platform, runtime and profile of phase 1.

The fixtures are generated, never edited by hand: `python3 tests/test_stack_compose.py --regen`
rewrites them from `dump(plan)` and runs no test (step 3 of the catalogue's bump procedure).
WHICH fixtures exist is not a list typed here: it is derived from `Backend.runtimes(platform)`,
the spec's single source for where a profile runs ("Compatibilidade"), with `amd` and `intel`
folded into `dri` because they render the same file. The spec's nine names are pinned
separately, so the derivation cannot drift together with `runtimes()` -- and the two windows
cpu fixtures Task 2 serves (a WSL2 host runs the cpu on the official image) are pinned there
too.

The fixture plans live under the placeholder homes `/home/me` and `/Users/me`, never a real one.
"""
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError, catalog  # noqa: E402
from stack.backends import BACKENDS  # noqa: E402
from stack.compose import (  # noqa: E402
    Plan,
    container_name,
    dump,
    emit,
    render,
    volume_name,
)

FIXTURES = REPO / "tests" / "fixtures" / "stack"

#: Every platform `runtimes()` is asked about. Windows now serves the cpu profile
#: (a WSL2 host with a runtime is the windows platform, cross-platform wizard
#: plan); the GPU profiles arrive through the dzn backend (Task 3), so asking
#: anyway is what makes a windows profile show up here.
PLATFORMS = ("linux", "macos", "windows")

#: The spec's nine ("Fixtures golden"), plus the two windows cpu fixtures Task 2
#: serves (a WSL2 host runs the cpu on the official image, identical to linux).
#: There is no `macos-docker-apple`: Docker Desktop passes no GPU to a container.
#: The windows dzn fixtures join this set with Task 7.
PHASE1_FIXTURES = {
    "linux-docker-cpu", "linux-docker-dri", "linux-docker-nvidia",
    "linux-podman-cpu", "linux-podman-dri", "linux-podman-nvidia",
    "macos-docker-cpu", "macos-podman-cpu", "macos-podman-apple",
    "windows-docker-cpu", "windows-podman-cpu",
}

STACK_DIRS = {"linux": Path("/home/me/.local/share/mnemosine/stack"),
              "macos": Path("/Users/me/.local/share/mnemosine/stack"),
              # The WSL2 distro is a Linux home: the state and stack dir live in
              # the distro (decision 24), so the cpu fixture is identical to the
              # linux one.
              "windows": Path("/home/me/.local/share/mnemosine/stack")}

#: The service keys in the spec's order, pinned here rather than read from the module.
SERVICE_KEYS = ("container_name", "image", "restart", "command", "ports", "volumes",
                "environment", "devices", "deploy", "annotations", "cap_drop", "security_opt")

#: The plain scalars the YAML 1.1 type registry resolves to a bool or to null.
YAML11_BOOL_AND_NULL = ("y Y yes Yes YES n N no No NO true True TRUE false False FALSE "
                        "on On ON off Off OFF null Null NULL ~").split()

#: One emitted line: an optional dash, then an optional key (bare, or a JSON string) with its
#: colon, then the value, which is empty only on a line that opens a block.
LINE = re.compile(r' *(?:- )?(?:(?P<key>[A-Za-z_][A-Za-z0-9_]*|"(?:[^"\\]|\\.)*"):(?: |$))?'
                  r'(?P<value>.*)')


def fixture_name(platform: str, runtime: str, backend: str) -> str:
    profile = "dri" if backend in ("amd", "intel") else backend
    return f"{platform}-{runtime}-{profile}"


def fixture_plan(platform: str, runtime: str, backend: str, **changes) -> Plan:
    """The plan a fixture renders: the catalogue's ports and images, `Vulkan0` on a GPU
    profile, the first nvidia-smi index on nvidia, no SELinux. `changes` overrides a field."""
    fields = dict(platform=platform, runtime=runtime, backend=backend,
                  device=None if backend == "cpu" else "Vulkan0",
                  gpu_index=0 if backend == "nvidia" else None,
                  ports=dict(catalog.PORTS), stack_dir=STACK_DIRS[platform],
                  images=dict(catalog.IMAGES))
    fields.update(changes)
    return Plan(**fields)


def served_fixtures() -> dict[str, Plan]:
    """Every fixture some runtime serves, by name, with the plan it is rendered from.

    `amd` comes first in `BACKENDS`, so it fills the `dri` slot; `intel` renders the same
    text (`test_amd_and_intel_render_the_same_file`), so the choice changes no file.
    """
    served: dict[str, Plan] = {}
    for platform in PLATFORMS:
        for backend, profile in BACKENDS.items():
            for runtime in sorted(profile.runtimes(platform)):
                name = fixture_name(platform, runtime, backend)
                if name not in served:
                    served[name] = fixture_plan(platform, runtime, backend)
    return served


def scalar_offence(line: str) -> str | None:
    """What is wrong with one emitted line, or None: every value after `: ` or `- ` must be
    exactly what `json.dumps` writes for a scalar, or the flow form of an empty map or list."""
    match = LINE.fullmatch(line)
    if match is None:
        return "not a line"
    value = match["value"]
    if not value:
        return None if match["key"] else "neither a key nor a value"
    if value in ("{}", "[]"):
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return "not JSON"
    if isinstance(parsed, (dict, list)):
        return "not a scalar"
    if json.dumps(parsed) != value:
        return "not what json.dumps writes"
    return None


class TheFixturesTest(unittest.TestCase):
    maxDiff = None

    def test_fixtures_exist_iff_the_runtime_serves_the_profile(self):
        served = set(served_fixtures())
        on_disk = {path.stem for path in FIXTURES.glob("*.yaml")}
        self.assertEqual(served, on_disk)
        self.assertEqual(PHASE1_FIXTURES, served)
        self.assertNotIn("macos-docker-apple", on_disk)

    def test_each_fixture_is_what_render_produces(self):
        plans = served_fixtures()
        for name, plan in sorted(plans.items()):
            with self.subTest(fixture=name):
                self.assertEqual((FIXTURES / f"{name}.yaml").read_text(encoding="utf-8"),
                                 dump(plan))
        self.assertEqual(11, len(plans))

    def test_amd_and_intel_render_the_same_file(self):
        for runtime in ("docker", "podman"):
            with self.subTest(runtime=runtime):
                self.assertEqual(dump(fixture_plan("linux", runtime, "amd")),
                                 dump(fixture_plan("linux", runtime, "intel")))


class TheEmitterTest(unittest.TestCase):
    def test_every_scalar_is_json_quoted(self):
        offences, lines = [], 0
        for name, plan in sorted(served_fixtures().items()):
            for number, line in enumerate(dump(plan).splitlines(), 1):
                lines += 1
                offence = scalar_offence(line)
                if offence:
                    offences.append(f"{name}:{number}: {offence}: {line}")
        self.assertEqual([], offences)
        self.assertGreater(lines, 9 * 60)

    def test_the_yaml_1_1_traps_come_out_as_strings(self):
        # Plain, PyYAML (podman-compose's parser) reads `no` as False and `22:22` as 1342.
        self.assertEqual('a: "no"\nb: "22:22"\nc: "true"\nd: "x\\ny"\ne: null\nf: true\ng: 3\n',
                         emit({"a": "no", "b": "22:22", "c": "true", "d": "x\ny",
                               "e": None, "f": True, "g": 3}))

    def test_reserved_and_dotted_keys_are_quoted(self):
        self.assertEqual('"yes": 1\n"run.oci.keep_original_groups": "1"\na_b: 2\n',
                         emit({"yes": 1, "run.oci.keep_original_groups": "1", "a_b": 2}))
        for key in YAML11_BOOL_AND_NULL + ["mnemosine-qdrant", "8080", "", "a b", "x:y"]:
            with self.subTest(key=key):
                self.assertEqual(f"{json.dumps(key)}: 1\n", emit({key: 1}))
        for key in ("a_b", "_x", "QDRANT__TELEMETRY_DISABLED", "yes_no", "onion", "nulls"):
            with self.subTest(key=key):
                self.assertEqual(f"{key}: 1\n", emit({key: 1}))

    def test_an_empty_map_or_list_is_written_in_flow_form(self):
        # A bare `key:` would be null, which is neither.
        self.assertEqual("a: {}\nb: []\n", emit({"a": {}, "b": []}))
        self.assertEqual("{}\n", emit({}))

    def test_a_map_inside_a_list_hangs_off_its_dash(self):
        self.assertEqual('l:\n  - a: 1\n    b:\n      - "x"\n  - "y"\n',
                         emit({"l": [{"a": 1, "b": ["x"]}, "y"]}))


class TheRenderTest(unittest.TestCase):
    def test_the_keys_come_in_the_fixed_order_and_only_with_a_value(self):
        for name, plan in sorted(served_fixtures().items()):
            doc = render(plan)
            with self.subTest(fixture=name):
                self.assertEqual(["services", "volumes"], list(doc))
                self.assertEqual(["qdrant", "embed", "rerank"], list(doc["services"]))
                for service in doc["services"].values():
                    self.assertEqual([k for k in SERVICE_KEYS if k in service], list(service))
                    self.assertTrue(all(service.values()))

    def test_every_service_drops_its_privileges_and_restarts(self):
        for name, plan in sorted(served_fixtures().items()):
            for service, fields in render(plan)["services"].items():
                with self.subTest(fixture=name, service=service):
                    self.assertEqual("always", fields["restart"])
                    self.assertEqual(["ALL"], fields["cap_drop"])
                    self.assertEqual(["no-new-privileges:true"], fields["security_opt"])

    def test_cpu_service_is_pinned_to_no_device(self):
        # M1: an engine can inject a GPU into every container, so leaving the device out is
        # not enough; the command itself has to say `none`.
        seen = 0
        for platform in PLATFORMS:
            for runtime in sorted(BACKENDS["cpu"].runtimes(platform)):
                services = render(fixture_plan(platform, runtime, "cpu"))["services"]
                for role in ("embed", "rerank"):
                    with self.subTest(platform=platform, runtime=runtime, role=role):
                        self.assertEqual(["-dev", "none"], services[role]["command"][-2:])
                        self.assertNotIn("devices", services[role])
                        self.assertNotIn("deploy", services[role])
                    seen += 1
        self.assertEqual(12, seen)

    def test_ports_bind_loopback_only(self):
        for name, plan in sorted(served_fixtures().items()):
            for service, fields in render(plan)["services"].items():
                with self.subTest(fixture=name, service=service):
                    self.assertEqual(1, len(fields["ports"]))
                    self.assertTrue(fields["ports"][0].startswith("127.0.0.1:"))
        moved = {"qdrant": 16333, "embed": 18003, "rerank": 18004}
        services = render(fixture_plan("linux", "podman", "amd", ports=moved))["services"]
        self.assertEqual({"qdrant": ["127.0.0.1:16333:6333"],
                          "embed": ["127.0.0.1:18003:8080"],
                          "rerank": ["127.0.0.1:18004:8080"]},
                         {service: fields["ports"] for service, fields in services.items()})

    def test_selinux_adds_the_z_label(self):
        plain = render(fixture_plan("linux", "podman", "amd"))["services"]
        labelled = render(fixture_plan("linux", "podman", "amd", selinux=True))["services"]
        models = "/home/me/.local/share/mnemosine/stack/models:/models"
        for role in ("embed", "rerank"):
            with self.subTest(role=role):
                self.assertEqual([f"{models}:ro"], plain[role]["volumes"])
                self.assertEqual([f"{models}:ro,z"], labelled[role]["volumes"])
        self.assertEqual(plain["qdrant"], labelled["qdrant"])

    def test_container_and_volume_names_carry_the_project(self):
        # M6: each provider names containers its own way unless `container_name` is set, and
        # both prefix the volume with the project.
        self.assertEqual("mnemosine-embed", container_name("mnemosine", "embed"))
        self.assertEqual("mnemosine_mnemosine-qdrant",
                         volume_name("mnemosine"))
        self.assertEqual("qctx-it_mnemosine-qdrant", volume_name("qctx-it"))
        doc = render(fixture_plan("linux", "docker", "cpu", project="qctx-it"))
        self.assertEqual({"qdrant": "qctx-it-qdrant", "embed": "qctx-it-embed",
                          "rerank": "qctx-it-rerank"},
                         {service: fields["container_name"]
                          for service, fields in doc["services"].items()})
        # the file declares the bare volume; the provider adds the project
        self.assertEqual({"mnemosine-qdrant": {}}, doc["volumes"])
        self.assertEqual(["mnemosine-qdrant:/qdrant/storage"],
                         doc["services"]["qdrant"]["volumes"])
        default = render(fixture_plan("linux", "docker", "cpu"))
        self.assertEqual("mnemosine-qdrant",
                         default["services"]["qdrant"]["container_name"])

    def test_a_patch_the_service_has_no_place_for_is_refused(self):
        # Dropped, the key would vanish without a word; merged over the base, it would
        # replace the command or the models mount.
        class Patching:
            def __init__(self, patch):
                self.patch = patch

            def service_patch(self, runtime, gpu_index):
                return dict(self.patch)

        for patch in ({"group_add": ["video"]}, {"command": ["--list-devices"]},
                      {"volumes": ["/usr/lib/wsl:/usr/lib/wsl:ro"]}):
            with self.subTest(patch=patch):
                with mock.patch.dict(BACKENDS, {"odd": Patching(patch)}):
                    with self.assertRaises(StackError) as ctx:
                        render(fixture_plan("linux", "docker", "odd"))
                self.assertEqual("compose", ctx.exception.step)

    def test_a_stack_dir_with_a_colon_is_refused(self):
        # R2 item R2-9: the models mount is the short form `<host>:/models:ro`, which
        # both providers split on ':'. A ':' in the host path breaks the parse in BOTH
        # (measured 2026-10-06 with docker-compose v5.2.0 and podman-compose 1.6.0).
        for bad in (Path("/home/me/stack:1"), Path("/home/me/a:b/c")):
            with self.subTest(stack_dir=bad):
                with self.assertRaises(StackError) as ctx:
                    render(fixture_plan("linux", "docker", "cpu", stack_dir=bad))
                self.assertEqual("compose", ctx.exception.step)
                self.assertIn("QCTX_STACK_DIR", ctx.exception.fix)

    def test_a_relative_stack_dir_is_refused(self):
        # R4: both providers treat a relative host path as a project volume and fail
        # with a misleading "undefined volume rel/stack/models" (measured 2026-10-07).
        with self.assertRaises(StackError) as ctx:
            render(fixture_plan("linux", "docker", "cpu", stack_dir=Path("rel/stack")))
        self.assertEqual("compose", ctx.exception.step)
        self.assertIn("QCTX_STACK_DIR", ctx.exception.fix)

    def test_a_tilde_stack_dir_is_refused(self):
        # R4: docker-compose expands a leading '~' to the invoking user's home while
        # podman-compose keeps it literally (measured 2026-10-07). A '~' path is not
        # absolute, so it is refused too.
        with self.assertRaises(StackError) as ctx:
            render(fixture_plan("linux", "docker", "cpu", stack_dir=Path("~/stack")))
        self.assertEqual("compose", ctx.exception.step)
        self.assertIn("QCTX_STACK_DIR", ctx.exception.fix)

    def test_a_stack_dir_with_a_dollar_is_refused(self):
        # R4: both providers interpolate '$VAR' and '${VAR}' in the host path.
        # '/home/me/a$zzqq/stack' mounts '/home/me/a/stack', exit 0 in both, docker-compose
        # only warning that the variable is unset (measured 2026-10-07 with docker-compose
        # v5.2.0 and podman-compose 1.6.0). '$$' is not a workaround: docker-compose kept
        # 'a$$b' while podman-compose resolved it to 'a$b'.
        for bad in (Path("/home/me/a$zzqq/stack"), Path("/home/me/x/${HOME}/stack"),
                    Path("/home/me/x/$HOME/stack"), Path("/home/me/a$$b/stack")):
            with self.subTest(stack_dir=bad):
                with self.assertRaises(StackError) as ctx:
                    render(fixture_plan("linux", "docker", "cpu", stack_dir=bad))
                self.assertEqual("compose", ctx.exception.step)
                self.assertIn("QCTX_STACK_DIR", ctx.exception.fix)

    def test_a_stack_dir_with_a_surrogate_is_refused(self):
        # R4: a lone surrogate is what surrogateescape makes of a non-UTF-8 path byte.
        # docker-compose rejects the file ("invalid Unicode character escape code");
        # podman-compose exits 0 and keeps the escape (measured 2026-10-07).
        with self.assertRaises(StackError) as ctx:
            render(fixture_plan("linux", "docker", "cpu",
                                stack_dir=Path("/home/me/caf\udce9")))
        self.assertEqual("compose", ctx.exception.step)
        self.assertIn("QCTX_STACK_DIR", ctx.exception.fix)

    def test_a_stack_dir_with_a_non_bmp_character_is_refused(self):
        # R2 item R2-9: a code point above 0xFFFF (an emoji) is rejected by
        # docker-compose and garbled by podman-compose (measured 2026-10-06).
        with self.assertRaises(StackError) as ctx:
            render(fixture_plan("linux", "docker", "cpu",
                                stack_dir=Path("/home/me/\U0001F4A4")))  # an emoji
        self.assertEqual("compose", ctx.exception.step)
        self.assertIn("QCTX_STACK_DIR", ctx.exception.fix)

    def test_an_accented_stack_dir_is_accepted(self):
        # R2 item R2-9: an accented path (code points inside the BMP) works in both
        # providers, so it must NOT be refused.
        accented = Path("/home/me/Memórias/plugin")
        doc = render(fixture_plan("linux", "docker", "cpu", stack_dir=accented))
        models = "/home/me/Memórias/plugin/models:/models:ro"
        self.assertEqual([models], doc["services"]["embed"]["volumes"])
        self.assertEqual([models], doc["services"]["rerank"]["volumes"])


def regen() -> None:
    """Rewrites every fixture from `dump`. Runs no test."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, plan in sorted(served_fixtures().items()):
        path = FIXTURES / f"{name}.yaml"
        path.write_text(dump(plan), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    # `--regen` writes the fixtures and runs no test. One expression, because the hygiene test
    # (tests/test_hygiene_fixes.py) wants `unittest.main(` on the guard's first statement.
    regen() if "--regen" in sys.argv[1:] else unittest.main(verbosity=2)
