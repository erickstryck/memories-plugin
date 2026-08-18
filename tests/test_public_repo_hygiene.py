"""This repository is PUBLIC (since 2026-08-18). Some mistakes stop being private mistakes.

None of these are about secrets — the keys never enter the tree, `config set` refuses them and
`.gitignore` has covered `config.json`, `.env`, `*.secret` and `*.key` since early on; a search of
all 169 commits for both real key VALUES found nothing. What these hold is the next tier down:
the things that are merely untidy in a private repo and are a small gift to a stranger in a public
one, plus the one legal detail a public repository actually needs.
"""
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def tracked_files():
    listed = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                            timeout=120)
    assert listed.returncode == 0, listed.stderr

    return [REPO / name for name in listed.stdout.split()]


def tracked_text():
    for path in tracked_files():
        if path.suffix in {".md", ".py", ".sh", ".yaml", ".yml", ".json", ".txt"} and path.is_file():
            yield path, path.read_text(errors="replace")


class TestNoRealHomeDirectoryIsCOMMITTED(unittest.TestCase):
    def test_no_tracked_file_embeds_the_home_of_whoever_wrote_it(self):
        """Checked against the CURRENT user's home rather than one hardcoded name, so it fails
        for the next person too — and so it says nothing about who that is.

        `/home/me/...` in the plans is deliberate fixture data and stays: it is obviously a
        placeholder, and rewriting example paths to protect a fictional user is theatre. What
        this catches is the real thing — a path that only exists on one machine, which is both a
        small disclosure and a line nobody else can run."""
        home = str(Path.home())
        offenders = [str(p.relative_to(REPO)) for p, text in tracked_text()
                     if home in text and p.name != Path(__file__).name]
        self.assertEqual(offenders, [],
                         f"these embed {home}, which is one machine's path:\n"
                         + "\n".join(offenders))


class TestTheExecutionWORKSPACEStaysOut(unittest.TestCase):
    """Ledgers, task briefs, implementer reports and review packages. The skill that creates
    them writes a `.gitignore` inside its own subdirectory, which works until the directory is
    made some other way — and then one `git add -A` publishes machine paths and internal notes."""

    def test_the_repository_ignores_it_ITSELF(self):
        rule = subprocess.run(["git", "check-ignore", "-v", ".superpowers/sdd/probe"],
                              cwd=REPO, capture_output=True, text=True, timeout=60)
        self.assertEqual(rule.returncode, 0, "`.superpowers/` is not ignored by anything")
        source = rule.stdout.split(":")[0]
        self.assertEqual(source, ".gitignore",
                         "it is ignored, but by a file that may not survive — the whole point is "
                         f"that the ROOT .gitignore owns this rule, not {source!r}. The first "
                         "version of this test accepted any file whose name contained "
                         "'.gitignore', which the inner one satisfies, so deleting the root rule "
                         "left the suite green.")

    def test_nothing_from_it_is_tracked(self):
        inside = [str(p.relative_to(REPO)) for p in tracked_files()
                  if str(p.relative_to(REPO)).startswith(".superpowers/")]
        self.assertEqual(inside, [], "execution artefacts are committed:\n" + "\n".join(inside))


class TestAPublicRepositorySaysWhatMayBeDoneWithIt(unittest.TestCase):
    def test_a_LICENSE_file_exists(self):
        """Without one, "all rights reserved" is the default — so a public repository with no
        license text is published code nobody may legally use, which is rarely the intent."""
        self.assertTrue((REPO / "LICENSE").is_file(), "no LICENSE in a public repository")

    def test_it_MATCHES_what_the_plugin_manifest_declares(self):
        """The manifest claimed MIT while no license shipped. One of the two had to move, and a
        claim that cannot be honoured is the one that should."""
        import json
        declared = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text()).get("license")
        self.assertTrue(declared, "the manifest declares no license")
        self.assertIn(declared.split("-")[0].upper(), (REPO / "LICENSE").read_text().upper(),
                      f"the manifest says {declared!r} and LICENSE does not say so")


class TestNoRealCREDENTIALShapeIsTracked(unittest.TestCase):
    """A backstop, not the defence. The defence is that `config set` refuses secrets and the
    ignore rules predate the first commit — verified by searching every commit for both real key
    values and finding none. This only stops the obvious from being ADDED later."""

    SHAPES = re.compile(
        r"""(?x)
        sk-[A-Za-z0-9]{20,} | ghp_[A-Za-z0-9]{20,} | gho_[A-Za-z0-9]{20,}
        | AKIA[0-9A-Z]{16} | xox[baprs]-[A-Za-z0-9-]{10,}
        | -----BEGIN\s+(RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----""")

    def test_no_tracked_file_carries_a_known_token_shape(self):
        offenders = [str(p.relative_to(REPO)) for p, text in tracked_text()
                     if p.name != Path(__file__).name and self.SHAPES.search(text)]
        self.assertEqual(offenders, [], "\n".join(offenders))


if __name__ == "__main__":
    unittest.main(verbosity=2)
