"""The hermes adapter's half of the skills: registration, namespacing, and the pointer.

WHY THIS FILE EXISTS. The skills shipped in `skills/` were written for claude-code, which
scans a plugin's `skills/` directory on its own. Hermes does NOT: its loader hands the plugin
a `ctx` and registers only what `register(ctx)` asks it to register, and this adapter asked
for a memory provider and nothing else. Measured against the installed loader on 2026-09-16:
with the plugin enabled, `PluginManager._plugin_skills` was `{}` and `skill_view('memory')`
answered "not found" -- while `skills/memory/SKILL.md` sat on disk, 9,382 bytes of it,
unreachable.

WHY THAT MATTERED RATHER THAN BEING COSMETIC. `system_prompt_block()` is under a kilobyte and
says what memory IS. The skill says how to USE it, and the two things a session actually
needed -- "Search first, then save" and the failure mode "treat 'didn't we discuss this?' as
rhetorical" -- exist ONLY in the skill. A session lost exactly that way: it treated the user's
question as rhetoric and re-derived a decision that was already recorded.

The system prompt block cannot absorb the skill: it is injected on EVERY turn of EVERY session
and is deliberately tiny for that reason. A skill is the right shape, paid for only when
loaded. So it has to be REGISTERED, and it has to be NAMED somewhere the model will read.

WHAT IS TESTED WHERE. The catalogue itself (what exists on disk, in what order, with what
tolerance for a broken tree) is `core/skills.py`, driven by `tests/test_core_skills.py`. This
file tests only what the hermes adapter adds: handing the catalogue to a host, surviving hosts
that cannot take it, and pointing the model at the qualified name.
"""
import os
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import skills  # noqa: E402
from hosts.hermes import (  # noqa: E402
    PRIMARY_SKILL, SKILL_NAMESPACE, MemoriesProvider, qualified_skill, register)


class RecordingCtx:
    """The two registrar methods this plugin calls on hermes' PluginContext.

    Deliberately NOT a mock: the real ctx raises FileNotFoundError for a path that does not
    exist, ValueError for a name carrying ':', and AttributeError for a str where it wants a
    Path (measured in `hermes_cli/plugins.py::register_skill`). A permissive mock would accept
    a registration hermes refuses, which is the whole failure this file exists to catch.
    """

    def __init__(self):
        self.provider = None
        self.skills = {}

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_skill(self, name, path, description="", frontmatter=None):
        if ":" in name:
            raise ValueError(f"skill name {name!r} must not contain ':' (hermes namespaces it)")
        if not name or not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+.")
        if not isinstance(path, Path):
            raise AttributeError(
                f"register_skill requires a pathlib.Path, got {type(path).__name__} -- "
                "hermes calls path.exists() and silently disables the whole plugin")
        if not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")
        self.skills[name] = path


class TestTheCatalogueReachesTheHost(unittest.TestCase):
    def setUp(self):
        self.ctx = RecordingCtx()
        register(self.ctx)

    def test_every_skill_the_package_ships_is_registered(self):
        """The adapter registers the CATALOGUE, so a fourth skill needs no edit here."""
        self.assertEqual(sorted(self.ctx.skills), skills.names(),
                         "a skill on disk that register() never hands to the host is "
                         "unreachable by skill_view, which is how `memory` was lost")

    def test_each_registered_path_is_the_catalogue_s_path(self):
        self.assertEqual(
            {name: str(path) for name, path in self.ctx.skills.items()},
            skills.paths())

    def test_registering_the_skills_does_not_cost_the_memory_provider(self):
        """The provider is the reason the plugin exists; skills ride along, never replace."""
        self.assertIsInstance(self.ctx.provider, MemoriesProvider)


class TestTheSkillsNeverTakeTheProviderDown(unittest.TestCase):
    """hermes' loader turns an exception from register() into a disabled plugin.

    So an accessory that raises costs the memory provider -- the one thing that must survive.
    """

    def test_a_host_whose_ctx_has_no_register_skill_still_gets_the_provider(self):
        """claude-code's collector, and any hermes predating plugin skills, has no such method."""
        class OldCtx:
            def __init__(self):
                self.provider = None

            def register_memory_provider(self, provider):
                self.provider = provider

        old = OldCtx()
        register(old)
        self.assertIsInstance(old.provider, MemoriesProvider)

    def test_a_host_that_REFUSES_a_skill_still_gets_the_provider(self):
        """Refusal is not absence.

        The refusal modelled here is the one that is real for this plugin, MEASURED against
        the installed host: `ValueError: Invalid skill name` from `_NAMESPACE_RE`. An earlier
        version of this test blamed a duplicate registration on reload; hermes gates that
        raise on `manifest.portable`, which is False for a native `plugin.yaml` like ours --
        registering twice simply replaces the entry.
        """
        class RefusingCtx(RecordingCtx):
            def register_skill(self, name, path, description="", frontmatter=None):
                raise ValueError(f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+.")

        refusing = RefusingCtx()
        register(refusing)
        self.assertIsInstance(refusing.provider, MemoriesProvider)
        self.assertEqual(refusing.skills, {})

    def test_ONE_refused_skill_does_not_cost_THE_OTHERS(self):
        """The tolerance is per skill, not per batch: a `continue`, never a `break`."""
        class PickyCtx(RecordingCtx):
            def register_skill(self, name, path, description="", frontmatter=None):
                if name == skills.names()[0]:
                    raise ValueError("this one is already registered")
                return super().register_skill(name, path, description, frontmatter)

        picky = PickyCtx()
        register(picky)
        self.assertEqual(sorted(picky.skills), skills.names()[1:])

    def test_a_FAILING_provider_registration_is_NOT_swallowed(self):
        """The skills' tolerance must not spread to the provider.

        Found by mutation: wrapping `register_memory_provider` in the same best-effort
        try/except left the suite green, and the result is a plugin that loads "successfully"
        with no memory at all -- the one failure the user must see, made invisible. hermes
        turns this exception into a visible disabled plugin with the reason attached.
        """
        class BrokenCtx(RecordingCtx):
            def register_memory_provider(self, provider):
                raise RuntimeError("the archive is unreachable")

        with self.assertRaises(RuntimeError):
            register(BrokenCtx())

    def test_a_refusal_is_REPORTED_and_not_dropped_in_silence(self):
        """Tolerating a failure is not the same as hiding it.

        A swallowed refusal with no trace reproduces the very defect this change fixes, one
        layer down: a skill on disk that no host loads, with nothing anywhere saying why.
        The note names the skill and the reason, on fd 2 (stdout carries the big-file guard's
        JSON block and must not be written to).
        """
        class RefusingCtx(RecordingCtx):
            def register_skill(self, name, path, description="", frontmatter=None):
                raise ValueError("Invalid skill name 'x'. Must match [a-zA-Z0-9_-]+.")

        read_fd, write_fd = os.pipe()
        saved = os.dup(2)
        try:
            os.dup2(write_fd, 2)
            register(RefusingCtx())
            os.dup2(saved, 2)
            os.close(write_fd)
            captured = os.read(read_fd, 65536).decode()
        finally:
            os.close(read_fd)
            os.close(saved)

        for name in skills.names():
            with self.subTest(skill=name):
                self.assertIn(name, captured, "the note does not name the dropped skill")
        self.assertIn("Invalid skill name", captured, "the note does not carry the reason")


class TestTheNamespaceIsSpelledOnce(unittest.TestCase):
    """`memories:memory` is hermes' spelling, and it is built in one place."""

    def test_it_qualifies_a_bare_name_the_way_hermes_does(self):
        self.assertEqual(qualified_skill("memory"), "memories:memory")

    def test_the_namespace_matches_the_manifest_the_loader_reads(self):
        """hermes derives the namespace from the plugin's declared name; a rename there
        silently retargets every `skill_view` call this adapter advertises."""
        manifest = (REPO / "plugin.yaml").read_text(encoding="utf-8")
        self.assertIn(f"\nname: {SKILL_NAMESPACE}\n", manifest)

    def test_the_primary_skill_is_THE_MEMORY_ONE(self):
        """Pinned as a literal, and the literal is the point.

        Found by review, as a mutation the suite did not kill: `PRIMARY_SKILL = "doc-index"`
        left all 1644 tests green while every session, on every turn, was told
        `skill_view("memories:doc-index")` -- sent to the document-indexing skill instead of
        the memory one. The two tests that touched this were both blind to it: one asserted
        only that the name is among the three shipped (true for any of them), the other that
        the block contains `qualified_skill(PRIMARY_SKILL)` (tautological -- both sides
        derive from the same constant).

        A wrong pointer silently reinstates the defect this whole change exists to fix: the
        model follows the name it was given, never loads the write-side procedure, and the
        archive degrades with the registration test still green. `memory` is the skill that
        carries "Search first, then save"; doc-index and repo-index are read-side tools.
        """
        self.assertEqual(PRIMARY_SKILL, "memory")
        self.assertIn(PRIMARY_SKILL, skills.names(), "the pointer names a skill we ship")

    def test_the_primary_skill_is_the_one_that_teaches_WRITING(self):
        """The property behind the literal, so the pin is not purely a spelling check."""
        text = Path(skills.paths()[PRIMARY_SKILL]).read_text(encoding="utf-8").lower()
        for expected in ("search first", "save"):
            with self.subTest(phrase=expected):
                self.assertIn(expected, text)


class TestTheModelIsTOLDTheSkillExists(unittest.TestCase):
    """Registration makes a plugin skill loadable; it does NOT make it findable.

    hermes keeps plugin skills out of `<available_skills>` ("explicit loads only", per
    `register_skill`'s own docstring), so a model never told the qualified name will never
    call skill_view with it. The pointer is therefore part of the fix, not decoration.
    """

    def block(self) -> str:
        return MemoriesProvider().system_prompt_block()

    def test_the_block_names_the_qualified_skill(self):
        self.assertIn(qualified_skill(PRIMARY_SKILL), self.block())

    def test_the_block_stays_a_pointer_and_does_not_become_the_skill(self):
        """It is injected on every turn of every session; the skill carries the detail.

        The bound is deliberately loose -- the point is "a pointer, not the skill": the skill
        is ~9.3k and the block was 788 bytes before the pointer line.
        """
        self.assertLess(len(self.block()), 2000)

    def test_the_pointer_is_NOT_in_the_text_shared_with_the_other_host(self):
        """`core/prompts.py` is injected verbatim into claude-code, where this name does not
        resolve. Host-specific pointers belong in the adapter; this is the same rule
        `tests/test_host_equivalence.py` enforces from the other direction."""
        from core import prompts

        for name in ("INSTRUCTIONS", "CHECKPOINT_PROCEDURE"):
            with self.subTest(text=name):
                self.assertNotIn(SKILL_NAMESPACE + ":", getattr(prompts, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
