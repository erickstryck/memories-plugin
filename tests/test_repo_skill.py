"""The repos skill is the only discovery path on claude-code, so it must not lie.

There is no tool schema on this host to read: a model finds `qctx repos` because a skill file
describes it, or it does not find it at all. That makes the file a functional part of the
feature rather than documentation about it — and a skill naming a flag the CLI does not have
sends the model to a refusal it cannot diagnose.

WHAT THESE TESTS PIN, and why each: that every command the skill shows is a command that parses;
that the flags it names exist; and that the claims a user would ACT on — the scoped default, the
two meanings of --limit, permanence — are the ones the code actually implements. What they
deliberately do not pin is prose: a test that asserts sentences turns an editing pass red.
"""
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SKILL = REPO / "skills" / "repo-index" / "SKILL.md"
QCTX = REPO / "cli" / "qctx.py"


def skill_text() -> str:
    return SKILL.read_text()


def commands_in_skill() -> list[str]:
    """Every `qctx repos …` line inside a fenced block — the ones a reader will copy."""
    out = []
    for block in re.findall(r"```bash\n(.*?)```", skill_text(), re.S):
        for line in block.splitlines():
            line = line.split("#")[0].strip()
            if line.startswith("qctx repos"):
                out.append(line)

    return out


class TestTheSkillExistsAndIsDiscoverable(unittest.TestCase):
    def test_the_file_is_where_a_host_looks_for_skills(self):
        """Beside the other two, in `skills/<name>/SKILL.md`. A file anywhere else is a file
        no host loads, and the feature would stay undiscoverable with documentation written."""
        self.assertTrue(SKILL.is_file(), f"{SKILL} does not exist")
        self.assertTrue((REPO / "skills" / "memory" / "SKILL.md").is_file(),
                        "the layout this test assumes has changed")

    def test_the_frontmatter_carries_a_name_and_a_description(self):
        head = skill_text().split("---")[1]
        self.assertRegex(head, r"(?m)^name: repo-index$")
        self.assertRegex(head, r"(?m)^description: \S")

    def test_the_description_says_WHEN_to_use_it_and_not_only_what_it_is(self):
        """A description that only names the tool is one the model never matches against a
        situation. The other two skills both state their trigger; this must too."""
        head = skill_text().split("---")[1]
        self.assertRegex(head.lower(), r"use it when|use when",
                         "the description gives no trigger")


class TestEveryCommandTheSkillShowsIsREAL(unittest.TestCase):
    """The failure this prevents: a skill that sends the model to a flag that never existed.
    Each command is parsed by the real CLI, so a renamed flag fails here rather than in front
    of a user."""

    def test_every_verb_is_DOCUMENTED_somewhere_in_the_skill(self):
        """A verb the skill never names is a verb the model does not know exists.

        Measured against the whole text and not only the copyable blocks, because `drop` is
        deliberately NOT in one: it deletes an archive permanently with no undo, and a
        ready-to-paste destructive line is a line that gets pasted. It is documented in prose,
        with its `--yes`, which is enough to know it exists and not enough to run by reflex."""
        text = skill_text()
        for verb in ("search", "register", "add", "list", "drop"):
            with self.subTest(verb=verb):
                self.assertIn(f"repos {verb}", text, f"the skill never names `repos {verb}`")

    def test_the_destructive_verb_is_NOT_in_a_copyable_block(self):
        """The other half of the decision above, pinned so a later edit does not undo it by
        being helpful. Every other verb belongs in a block; this one does not."""
        self.assertNotIn("repos drop", " ".join(commands_in_skill()),
                         "`repos drop` became copy-pasteable")

    def test_every_flag_the_skill_names_is_a_flag_the_CLI_accepts(self):
        """Parsed from the real `--help`, not from a list kept here: a second list would be a
        second thing to forget."""
        for cmd in commands_in_skill():
            parts = cmd.split()
            verb = parts[2]
            help_out = subprocess.run([sys.executable, str(QCTX), "repos", verb, "--help"],
                                      capture_output=True, text=True, timeout=60)
            self.assertEqual(help_out.returncode, 0, help_out.stderr)
            for flag in (p for p in parts if p.startswith("--")):
                with self.subTest(cmd=cmd, flag=flag):
                    self.assertIn(flag, help_out.stdout,
                                  f"`{flag}` is in the skill but not in `repos {verb} --help`")


class TestTheClaimsAUserWouldACTOnAreTrue(unittest.TestCase):
    def test_the_scoped_default_refusal_is_the_one_the_core_raises(self):
        """The skill promises a refusal naming BOTH remedies rather than a silent widening.
        That sentence is written in the core; if it stops naming both, the skill is wrong."""
        from core.repos import RepoError, RepoIndex
        from tests.fakes import FakeEmbedder, FakeVectorStore
        ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
        with self.assertRaises(RepoError) as caught:
            ix.search_request("anything", cwd=str(REPO.parent))
        message = str(caught.exception)
        self.assertIn("--repo", message)
        self.assertIn("--all", message)

    def test_the_skill_says_the_index_is_permanent_and_drop_needs_confirmation(self):
        """Both are load-bearing for a reader deciding whether to index: an archive that
        silently expired, or a drop that did not ask, would each be a nasty surprise."""
        text = skill_text()
        self.assertIn("permanent", text.lower())
        self.assertIn("--yes", text)
        self.assertIn("no undo", text.lower())

    def test_the_skill_states_BOTH_meanings_of_the_limit(self):
        """`limit` trims hits when scoped and repositories when across — the one thing about
        this surface a careful reader gets wrong without being told."""
        text = skill_text()
        self.assertIn("--limit", text)
        self.assertRegex(text, r"(?s)--limit.*(repositor|group)")

    def test_the_skill_names_the_hermes_tools_so_the_hosts_stay_findable_together(self):
        """Equivalence is the promise this plugin makes about its two hosts. A skill that
        described only one would leave a reader on the other looking for a CLI they have not
        got."""
        text = skill_text()
        for tool in ("repos_search", "repos_list", "repos_register", "repos_add", "repos_drop"):
            with self.subTest(tool=tool):
                self.assertIn(tool, text)

    def test_every_hermes_repo_tool_the_skill_names_is_a_tool_that_EXISTS(self):
        """The same defence as the flags above, for the other host: the names come from the
        real schema list, so a renamed tool fails here."""
        from hosts.hermes import tools
        real = {s["name"] for s in tools.SCHEMAS}
        for tool in re.findall(r"`(repos_\w+)`", skill_text()):
            with self.subTest(tool=tool):
                self.assertIn(tool, real, f"the skill names `{tool}`, which is not a tool")


if __name__ == "__main__":
    unittest.main(verbosity=2)
