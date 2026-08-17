# tests/test_config_repos.py
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config as C  # noqa: E402
from core.config import ConfigError  # noqa: E402


def a_config(**over):
    base = dict(memory_collection="mem", docs_collection="tmp", library_collection="lib",
                repos_collection="repos", repos_registry_collection="reg")
    base.update(over)
    full = {f.name: getattr(C.Config, f.name, "") for f in __import__("dataclasses").fields(C.Config)}
    full.update({k: v for k, v in base.items()})

    return C.Config(**{k: v for k, v in full.items()})


class TestTheTwoNewCollections(unittest.TestCase):
    def test_the_defaults_are_named(self):
        self.assertEqual(C.DEFAULTS["repos_collection"], "memories_repos")
        self.assertEqual(C.DEFAULTS["repos_registry_collection"], "memories_repos_registry")

    def test_both_have_env_aliases(self):
        self.assertEqual(C.ENV_ALIASES["repos_collection"],
                         ("QCTX_REPOS_COLLECTION", "REPOS_COLLECTION"))
        self.assertEqual(C.ENV_ALIASES["repos_registry_collection"],
                         ("QCTX_REPOS_REGISTRY_COLLECTION", "REPOS_REGISTRY_COLLECTION"))

    def test_require_returns_the_name(self):
        cfg = a_config()
        self.assertEqual(cfg.require_repos_collection(), "repos")
        self.assertEqual(cfg.require_repos_registry_collection(), "reg")

    def test_an_unset_repos_collection_is_an_error_and_not_a_silent_default(self):
        with self.assertRaises(ConfigError):
            a_config(repos_collection="").require_repos_collection()


class TestTheFiveAreDistinct(unittest.TestCase):
    """Every collision degrades SILENTLY, which is why this raises instead of warning."""

    def test_repos_may_not_be_the_memory_collection(self):
        with self.assertRaises(ConfigError):
            a_config(repos_collection="mem").require_repos_collection()

    def test_repos_may_not_be_the_library(self):
        """The library is permanent and hand-picked; tens of thousands of automatic code
        chunks in it would drown exactly the archive that curation paid for."""
        with self.assertRaises(ConfigError):
            a_config(repos_collection="lib").require_repos_collection()

    def test_the_registry_may_not_be_the_chunk_archive(self):
        """They are separate so that no search ever has to filter registry rows out. Letting
        them collide reintroduces the filter this design exists to avoid."""
        with self.assertRaises(ConfigError):
            a_config(repos_registry_collection="repos").require_repos_registry_collection()

    def test_the_older_three_still_reject_the_new_two(self):
        with self.assertRaises(ConfigError):
            a_config(docs_collection="repos").require_docs_collection()


class TestTheDIAGNOSTICKnowsTheSameFiveNames(unittest.TestCase):
    """`setup.diagnose` enumerated three collections while the guard enumerated five.

    The consequence is not cosmetic: `qctx setup` answered "ready" on a configuration whose
    first effect, in `_require_distinct`'s own words, is repository chunks drowning the
    curated memory archive. The guard did fire — but only once a repos command ran, which
    is after the diagnostic said the install was fine.
    """

    def test_a_repos_collection_colliding_with_memory_is_a_BLOCKER(self):
        from core import setup
        rel = setup.diagnose(a_config(repos_collection="mem", qdrant_url=""))
        self.assertIn("repos_collection", {c["name"] for c in rel["blockers"]})

    def test_a_registry_colliding_with_memory_is_a_BLOCKER(self):
        from core import setup
        rel = setup.diagnose(a_config(repos_registry_collection="mem", qdrant_url=""))
        self.assertIn("repos_registry_collection", {c["name"] for c in rel["blockers"]})

    def test_the_diagnostic_reports_on_all_five_collections(self):
        """Not only on collisions: a check that never mentions a collection cannot report
        anything about it — a wrong dimension included."""
        from core import setup
        rel = setup.diagnose(a_config(qdrant_url=""))
        self.assertLessEqual(
            {"memory_collection", "docs_collection", "library_collection",
             "repos_collection", "repos_registry_collection"},
            {c["name"] for c in rel["checks"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
