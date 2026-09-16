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


def sibling_offenders(hosts_dir: Path) -> tuple[dict, int]:
    """Host modules that import ANOTHER host, plus how many comparisons were made.

    A FUNCTION AND NOT A LOOP INSIDE A TEST, because the version that lived inside the test
    asserted nothing on a one-host tree: the `for other in siblings` body never ran, and the
    test passed while making zero comparisons. Returning the count lets a caller prove the
    walk actually did the work — a guard whose own execution is unverified is a guard that
    reports success for the wrong reason.

    Imports are read from the AST via `imported_packages`, so a docstring naming another host
    in order to say "this does not import it" is not an offender.
    """
    siblings = sorted(p.name for p in hosts_dir.iterdir()
                      if p.is_dir() and not p.name.startswith("__"))
    offenders, compared = {}, 0
    for path in sorted(hosts_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        mine = path.relative_to(hosts_dir).parts[0]
        imported = imported_packages(path)
        text = path.read_text(encoding="utf-8")
        for other in siblings:
            if other == mine:
                continue
            compared += 1
            # BOTH conditions: the AST proves it is a real import and not prose, the text
            # says WHICH host. Counting the comparison before either test is deliberate —
            # the count measures work done, not violations found.
            if "hosts" in imported and f"hosts.{other}" in text:
                offenders.setdefault(str(path.relative_to(hosts_dir)), []).append(other)

    return offenders, compared


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


class TestNoHostImportsAnotherHost(unittest.TestCase):
    """A host adapter may depend on `core/`, never on a sibling host.

    THE DIRECTION GUARD ABOVE DOES NOT COVER THIS, and the gap had a measured cost. The
    table of invokable operations lived in `hosts/hermes/tools.py` although nothing in it
    was the hermes host's, so a third adapter had two ways to offer operations to a model
    and both were wrong: copy ~700 lines that no equivalence test would cover, or import
    the second host's package — which is worse than ugly, because that module's tuning
    fallback resolved the OTHER host's provider class and read ITS environment-tuned recall
    floors. A third host would have silently inherited the second host's retrieval policy,
    which is the exact divergence `tests/test_host_equivalence.py` exists to prevent.

    The table is in `core/operations.py` now, and this is what keeps it there.

    THE RULE IS A FUNCTION, not a loop body, for a measured reason: with one host package in
    the tree the sibling loop executes ZERO comparisons, so the assertion never ran and the
    test reported success while proving nothing. Instrumented and confirmed: 0 calls to
    `assertNotIn`. A rule that only runs when a second host happens to exist is dormant
    exactly until the day it matters. Now `sibling_offenders` takes the directory to walk,
    and the tests below run it against a SYNTHETIC two-host tree as well as the real one."""

    def test_each_host_package_imports_no_other_host(self):
        offenders, compared = sibling_offenders(REPO / "hosts")

        self.assertEqual(offenders, {},
                         f"a host imports a sibling host: {offenders}")

    def test_the_rule_CATCHES_a_sibling_import_on_a_two_host_tree(self):
        """The assertion above cannot fail while the tree has one host. This one can.

        A synthetic tree with two hosts, one importing the other, is the only way to exercise
        the comparison the real tree never reaches — and it stays honest when a second host
        is added for real."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            hosts = Path(d)
            (hosts / "alpha").mkdir()
            (hosts / "beta").mkdir()
            (hosts / "alpha" / "__init__.py").write_text("from core import operations\n",
                                                         encoding="utf-8")
            (hosts / "beta" / "__init__.py").write_text("from hosts.alpha import thing\n",
                                                        encoding="utf-8")

            offenders, compared = sibling_offenders(hosts)

            self.assertGreater(compared, 0, "the synthetic tree compared nothing")
            self.assertIn("beta/__init__.py", offenders,
                          f"the rule missed a host importing its sibling: {offenders}")
            self.assertNotIn("alpha/__init__.py", offenders,
                             "importing core/ was reported as a violation")

    def test_a_clean_two_host_tree_is_not_reported(self):
        """The other direction: two hosts that both import only `core/` are fine."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            hosts = Path(d)
            for name in ("alpha", "beta"):
                (hosts / name).mkdir()
                (hosts / name / "__init__.py").write_text(
                    "from core import operations\nfrom core import knobs\n", encoding="utf-8")

            offenders, compared = sibling_offenders(hosts)

            self.assertGreater(compared, 0, "the synthetic tree compared nothing")
            self.assertEqual(offenders, {}, f"a clean tree was reported: {offenders}")

    def test_PROSE_naming_a_sibling_is_not_a_violation(self):
        """A host that documents the boundary must not be reported for describing it.

        These modules explain themselves in comments, and the honest way to say "this does
        not import the other host" is to name it. A text-only rule reports that sentence and
        gets abandoned as noisy, which is how a guard dies. Measured: dropping the AST half
        of the condition leaves this the only failing test."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            hosts = Path(d)
            for name in ("alpha", "beta"):
                (hosts / name).mkdir()
            (hosts / "alpha" / "__init__.py").write_text("from core import operations\n",
                                                         encoding="utf-8")
            (hosts / "beta" / "__init__.py").write_text(
                '"""This adapter never imports hosts.alpha; the shared table is in core."""\n'
                "# hosts.alpha owns its own tuning, see core/operations.py::bind_tuning\n"
                "from core import operations\n", encoding="utf-8")

            offenders, compared = sibling_offenders(hosts)

            self.assertGreater(compared, 0, "the synthetic tree compared nothing")
            self.assertEqual(offenders, {},
                             f"prose naming a sibling was reported as an import: {offenders}")

    def test_the_guard_would_CATCH_a_host_importing_a_sibling(self):
        """Without this the test above passes on a tree where no host imports anything."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            offender = Path(d) / "offender.py"
            offender.write_text("from hosts.hermes import tools\n", encoding="utf-8")
            self.assertIn("hosts", imported_packages(offender))
            self.assertIn("hosts.hermes", offender.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
