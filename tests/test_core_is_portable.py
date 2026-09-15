"""`core/` must not depend on a host. This is the structural claim the whole project rests on.

It was true in fact and held by NOTHING. The only direction guard in the tree covers one
module (`tests/test_quarantine.py::TestItDependsOnNothingHeavy`) and does it by searching that
one file's source for two strings. Proven by mutation: adding a module-level constant to
`core/bigfile.py` computed by reading `hosts/hermes/bigfile.py` off disk left all 1434 tests
green. A cruder violation — a circular import — did redden 40 tests, but through an
ImportError cascade rather than any assertion about direction, so the realistic version (a
`core` module quietly learning a host's constant, a path, an env var name) sails through CI.

This is the same class as the guard a previous review established by two explicit rulings and
that no test secured: removing it left the suite green, so a future cleanup would have deleted
it and passed.
"""
import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parent.parent
CORE = REPO / "core"

#: The packages `core/` may never import. `cli` is here for the same reason as the hosts: the
#: CLI is one host's entry point, and a core module reaching into it would make the other
#: host's behaviour depend on a command line nobody ran.
FORBIDDEN = {"hooks", "hosts", "cli", "agent"}


def imported_packages(path: Path) -> set:
    """Every top-level package `path` imports, by reading the AST rather than the text.

    A text scan would report the prose: these modules explain their own boundaries in
    comments, and several name `hosts/hermes` in a docstring precisely to say they do not
    import it. Only a real `import` statement counts.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `level > 0` is a relative import: it can only reach inside `core`, which is
            # the direction this guard exists to allow.
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])

    return found


class TestCoreDependsOnNoHost(unittest.TestCase):
    def test_no_core_module_imports_a_host(self):
        offenders = {}
        for path in sorted(CORE.glob("*.py")):
            bad = imported_packages(path) & FORBIDDEN
            if bad:
                offenders[path.name] = sorted(bad)

        self.assertEqual(offenders, {},
                         "core/ is portable BECAUSE it knows no host; these break that: "
                         f"{offenders}")

    def test_the_walk_actually_saw_the_package(self):
        """Guard on the guard. An empty derivation passes vacuously, and a rename or a moved
        directory would make the assertion above prove nothing while staying green."""
        modules = list(CORE.glob("*.py"))
        self.assertGreater(len(modules), 20,
                           f"only {len(modules)} core modules found — the walk is not looking "
                           f"where it thinks it is")
        self.assertTrue(any("knobs" in p.name for p in modules),
                        "a known core module is missing from the walk")

    def test_the_guard_would_CATCH_a_violation(self):
        """The mutation, as an assertion: the check must fail on a module that does import a
        host. Without this, a bug in `imported_packages` leaves the suite green forever."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            offender = Path(d) / "pretend_core_module.py"
            offender.write_text("from hosts.hermes import bigfile\n", encoding="utf-8")
            self.assertEqual(imported_packages(offender) & FORBIDDEN, {"hosts"})

            subtle = Path(d) / "subtle.py"
            subtle.write_text("import hooks.recall as r\n", encoding="utf-8")
            self.assertEqual(imported_packages(subtle) & FORBIDDEN, {"hooks"})

    def test_a_host_name_in_PROSE_is_not_a_violation(self):
        """These modules explain their boundaries in comments, and several name a host in a
        docstring exactly to say they do not import it. A text scan would report that prose
        and the guard would be abandoned as noisy."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            documented = Path(d) / "documented.py"
            documented.write_text('"""Never imports hosts.hermes; see the adapter."""\n'
                                  "# hooks/recall.py calls this, it does not live here\n"
                                  "import json\n", encoding="utf-8")
            self.assertEqual(imported_packages(documented) & FORBIDDEN, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
