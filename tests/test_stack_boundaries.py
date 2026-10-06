"""The import direction `cli -> stack -> core`, held by assertions and not by habit.

Two rules. `core/`, `hooks/` and `hosts/` never import `stack`: the hooks load `core` on every
prompt, and they must not so much as load the subprocess, container and download code the stack
is made of. And `stack/` never imports `hooks/`, `hosts/` or `cli/`: it must not know which hosts
exist, so what differs per host must reach it as a parameter from the CLI. Beside the direction,
`stack/` imports nothing but the stdlib, `core` and itself.

Imports are read from the AST by `imported_packages`, the reader `test_core_is_portable.py`
proves catches a real import and ignores prose. `stack/` names `hooks/`, `hosts/` and `cli/` in
its own docstring to state this very rule, and a text scan would report that.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.errors import CoreError  # noqa: E402
from stack import StackError  # noqa: E402
from tests.test_core_is_portable import imported_packages  # noqa: E402

#: What `stack/` may never import: the hooks, the hosts, the CLI that calls it, and `agent`,
#: which is hermes' own package, so importing it is importing a host.
STACK_MAY_NOT_IMPORT = {"hooks", "hosts", "cli", "agent"}


class TestTheDependenciesPointOneWay(unittest.TestCase):
    def test_no_core_hooks_or_hosts_module_imports_stack(self):
        offenders = {str(p.relative_to(REPO)): "stack"
                     for d in ("core", "hooks", "hosts") for p in (REPO / d).rglob("*.py")
                     if "stack" in imported_packages(p)}

        self.assertEqual(offenders, {},
                         f"core/, hooks/ and hosts/ must not import the stack: {offenders}")

    def test_the_walk_saw_core_hooks_and_hosts(self):
        """Guard on the guard: a renamed or moved directory empties the walk above, and an
        empty walk passes."""
        for d in ("core", "hooks", "hosts"):
            with self.subTest(directory=d):
                self.assertTrue(any((REPO / d).rglob("*.py")), f"{d}/ has no module to walk")

    def test_stack_imports_no_host_and_no_cli(self):
        bad = {p.name: sorted(imported_packages(p) & STACK_MAY_NOT_IMPORT)
               for p in (REPO / "stack").glob("*.py")}

        self.assertEqual({k: v for k, v in bad.items() if v}, {},
                         "stack/ must not import hooks/, hosts/, cli/ or agent")

    def test_stack_imports_only_the_stdlib_core_and_itself(self):
        """The stdlib-only constraint, held mechanically. `plugin.yaml` declares no Python
        dependency, so nothing installs one: what `stack/` imports must already be in every
        Python it runs on."""
        allowed = set(sys.stdlib_module_names) | {"core", "stack"}
        extra = {p.name: sorted(imported_packages(p) - allowed)
                 for p in (REPO / "stack").glob("*.py")}

        self.assertEqual({k: v for k, v in extra.items() if v}, {},
                         "stack/ may import only the stdlib, core and itself")

    def test_the_walk_saw_the_stack_package(self):
        self.assertIn("__init__.py", [p.name for p in (REPO / "stack").glob("*.py")])


class TestAStackErrorNamesItsStepAndFix(unittest.TestCase):
    def test_stack_error_is_a_core_error_and_names_step_and_fix(self):
        exc = StackError("no runtime", step="runtime", fix="install Docker or Podman")

        self.assertIsInstance(exc, CoreError)
        self.assertEqual(str(exc), "runtime: no runtime (fix: install Docker or Podman)")
        self.assertEqual((exc.step, exc.fix), ("runtime", "install Docker or Podman"))

        bare = StackError("x", step="s")
        self.assertEqual(str(bare), "s: x")
        self.assertIsNone(bare.fix)
        # An empty fix is no fix: a list of collected fixes can join to "", and a message
        # ending in "(fix: )" would point at a correction that is not there.
        self.assertEqual(str(StackError("x", step="s", fix="")), "s: x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
