"""The README must describe THIS tree.

It stopped doing so, measured on 2026-08-19: it said "three hooks" over four, gave a test
count 29 short, and printed two different numbers of hermes tools on the same page. Every
one of those was true when written. Prose does not rot on its own — it rots because
nothing reads it, so this reads it.

Counts are written as DIGITS in the README on purpose. A test that has to parse "three"
would be a test nobody keeps working.
"""
import json
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

README = (REPO / "README.md").read_text()


def load_cli():
    """Imports cli/qctx.py as a module. It is a script, not a package member — the same
    helper `tests/test_cli_render.py` uses, and for the same reason."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("qctx_cli", REPO / "cli" / "qctx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def known_commands(parser, prefix=()):
    """Every (command, subcommand) pair argparse accepts."""
    import argparse
    found = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                found.add(prefix + (name,))
                found |= known_commands(sub, prefix + (name,))

    return found


class CitedCommandsExist(unittest.TestCase):
    def test_every_qctx_command_in_the_readme_is_real(self):
        known = known_commands(load_cli().build_parser())
        tops = {pair[0] for pair in known}
        for command, rest in re.findall(r"\bqctx ([a-z][a-z-]*)((?: [a-z][a-z-]*)?)",
                                        README):
            self.assertIn(command, tops, f"README cites `qctx {command}`, which does not "
                                         f"exist")
            second = rest.strip()
            subs = {pair[1] for pair in known if len(pair) == 2 and pair[0] == command}
            if subs and second:
                self.assertIn(second, subs,
                              f"README cites `qctx {command} {second}`, and {command} has "
                              f"no such subcommand")


class CountsMatchTheTree(unittest.TestCase):
    def counted(self, noun):
        r"""Every "<n> … <noun>" the README states, one line at a time.

        The modifier words between the number and the noun are counted too
        ("20 model-invokable tools" is a count of tools): a regex that only sees
        the number glued to the noun passes over the very lie it exists to catch.
        The separators are [ \t], not \s: a \s would let "22 tools" at the end of
        one line keep matching all the way into the "skills/" of the next, and
        the count test would die of a lie the README never told.
        """
        pattern = rf"\b(\d+)(?:[ \t]+[\w-]+){{0,3}}[ \t]{noun}s?\b"
        return {int(n) for n in re.findall(pattern, README)}

    def test_hooks(self):
        declared = json.loads((REPO / "hooks" / "hooks.json").read_text())["hooks"]
        real = sum(len(matcher.get("hooks", []))
                   for event in declared.values() for matcher in event)
        self.assertEqual(self.counted("hook"), {real})

    def test_skills(self):
        real = len(list((REPO / "skills").glob("*/SKILL.md")))
        self.assertEqual(self.counted("skill"), {real})

    def test_tools(self):
        from hosts.hermes import tools
        self.assertEqual(self.counted("tool"), {len(tools.SCHEMAS)})

    def test_collections(self):
        """The two places that said "three collections" over five.

        They survived the first pass because they were spelled as a word, and this test
        reads digits. Digits are the contract: the README states the count as a numeral
        so that a sixth collection field breaks this test the day it is added.
        """
        from core import config
        real = len([k for k in config.DEFAULTS if k.endswith("_collection")])
        self.assertEqual(self.counted("collection"), {real})


class TheHeadlineInstallCommandRuns(unittest.TestCase):
    """The first command a new user types has to survive this machine's cache.

    `~/.claude/plugins/cache/…/*/scripts/install.sh` expands to one directory per
    installed commit — five on the machine this was measured on. Bash then runs the
    first and hands the other four to `qctx install`, which exits 2 on unrecognised
    arguments. So the README's line is EXECUTED here, against a fake cache with five
    SHA directories, and the stub it reaches reports how many arguments it was given.
    """

    def readme_claude_line(self) -> str:
        for line in README.splitlines():
            if line.startswith("bash ") and ".claude/plugins/cache" in line:
                return line

        self.fail("the README no longer shows a claude-code install.sh command")

    def test_it_passes_exactly_one_path_with_five_cached_commits(self):
        import subprocess
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as home:
            cache = (Path(home) / ".claude" / "plugins" / "cache" / "memories-plugin"
                     / "memories-plugin")
            for sha in ("aaa1111", "bbb2222", "ccc3333", "ddd4444", "eee5555"):
                script = cache / sha / "scripts" / "install.sh"
                script.parent.mkdir(parents=True)
                script.write_text("#!/usr/bin/env bash\nprintf 'argc=%s\\n' \"$#\"\n")
                script.chmod(0o755)
            done = subprocess.run(["bash", "-c", self.readme_claude_line()],
                                  capture_output=True, text=True, timeout=60,
                                  env={"HOME": home, "PATH": "/usr/bin:/bin"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "argc=0",
                         "the command handed extra cache paths to install.sh")


class TheLocalStackSectionStaysTrue(unittest.TestCase):
    """`## Running Qdrant and the models` and `### Quick install` tell a reader on THEIR
    machine how to stand up the three endpoints with llama.cpp and the two gpustack GGUFs.

    What is pinned is the load-bearing surface, not the prose: the two model repositories
    by name, the serving switch the rerank route exists without, the batch flags the first
    real rerank fails on, and that every address in the section is localhost, because the
    one failure this section must not repeat is pointing a stranger at the author's
    machines. The em-dash ban is pinned file-wide: a style rule a reader will hold this
    file to, so a single reintroduction is a regression, not a preference.
    """

    def section(self, header: str) -> str:
        """The body of a heading, up to the next heading of the SAME or HIGHER level.

        A `## ` section runs to the next `## `, a `### ` section runs to the next `### `
        or `## `. Two traps the first version fell into: stopping only at `## ` made a
        `###` section swallow every later `###` sibling, and reading `#`-led lines without
        tracking code fences made a bash comment inside a block (like `# embedding, on
        :8003`) look like a level-1 heading and truncate the section there.
        """
        lines = README.splitlines()
        try:
            start = next(i for i, line in enumerate(lines) if line == header)
        except StopIteration:
            self.fail(f"the README no longer has a {header!r} section")
        level = len(header) - len(header.lstrip("#"))
        end = len(lines)
        in_fence = False
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = re.match(r"^(#+) ", line)
            if match and len(match.group(1)) <= level:
                end = i
                break
        return "\n".join(lines[start:end])

    def test_the_model_repositories_are_named(self):
        section = self.section("## Running Qdrant and the models")
        for repo in ("gpustack/bge-m3-GGUF", "gpustack/bge-reranker-v2-m3-GGUF"):
            self.assertIn(repo, section,
                          f"the local stack section no longer names {repo}")

    def test_the_rerank_serving_requirements_are_still_there(self):
        section = self.section("## Running Qdrant and the models")
        self.assertIn("--reranking", section,
                      "without this flag the rerank route answers 501; the section must say so")
        self.assertIn("8192", section,
                      "the rerank batch flags are the ones a first real rerank fails on")

    def test_every_address_is_localhost(self):
        section = self.section("## Running Qdrant and the models")
        quick = self.section("### Quick install")
        for host in re.findall(r"http://([A-Za-z0-9_.-]+)", section + quick):
            self.assertIn(host, ("127.0.0.1", "localhost"),
                          f"the local stack sections point at {host}, not at a reader's own machine")

    def test_quick_install_covers_all_three_oses(self):
        quick = self.section("### Quick install")
        for os_name in ("Linux", "macOS", "Windows"):
            self.assertRegex(quick, rf"(?m)^\*\*{os_name}\*\*$",
                             f"the quick install no longer has an {os_name} block")

    def test_the_readme_never_uses_the_spaced_em_dash(self):
        """A standing style rule, pinned because prose rots: the file had 98 of them until
        2026-08-25, and the one a reader remembers is the one that comes back first."""
        offenders = [i for i, line in enumerate(README.splitlines(), 1) if "\u2014" in line]
        self.assertEqual(offenders, [],
                         "these lines use the spaced em-dash, which is banned in this file:\n"
                         + "\n".join(f"  {i}: {README.splitlines()[i-1]}" for i in offenders[:10]))


if __name__ == "__main__":
    unittest.main()