"""The persisted `stack.json`: read tolerantly, written atomically, owner-only.

`load` is the reader that `qctx stack status` and `remove` rely on: a missing file
is no stack (`None`), but a file that cannot be trusted is a named error, not a
guess -- and the name of the fix is always `qctx stack remove`, because a corrupt
state is the one state `remove` must still be able to clean (review focus 4).
`save` is the writer: it goes through `core.statefile.write_json`, so the file
is published atomically and `0o600`, and a write that did not land is an error
instead of a saved state that never was.

The fixtures in `_write` stand for the bad files `load` must refuse: the corrupt
bytes are not JSON at all; the wrong shape is valid JSON with `ports` as a list;
the newer schema is valid JSON at `schema` 2. The placeholder home is `/home/me`,
never a real one.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError, catalog, state  # noqa: E402
from stack.compose import Plan  # noqa: E402


def make_state(**changes) -> state.StackState:
    """A full `StackState`, the way the installer would build one after the
    probe: `cpu` on Docker, the catalogue's ports and images, phase `running`.
    `changes` overrides a field."""
    fields = dict(role="local", listen="127.0.0.1:6333", platform="linux",
                  runtime="docker", provider=["docker-compose"], profile="cpu",
                  device=None, gpu_index=None, ports=dict(catalog.PORTS),
                  images=dict(catalog.IMAGES),
                  models={"embed": catalog.EMBED_MODEL.sha256,
                          "rerank": catalog.RERANK_MODEL.sha256},
                  qdrant_version="1.19.2", selinux=False, phase="running",
                  created_at="2026-10-06T12:00:00Z",
                  updated_at="2026-10-06T12:05:00Z")
    fields.update(changes)
    return state.StackState(**fields)


class TestTheRoundtrip(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="stack-state-"))

    def test_roundtrip(self):
        original = make_state()
        state.save(self.dir, original)
        self.assertEqual(original, state.load(self.dir))
        self.assertEqual(catalog.PROJECT, state.load(self.dir).project)

    def test_absent_is_none(self):
        self.assertIsNone(state.load(self.dir))


class TestTheUnreadableFile(unittest.TestCase):
    def _write(self, name, text):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="stack-state-"))

    def test_a_corrupt_file_names_itself_and_the_remove_fix(self):
        # The fixture stands for the bytes a crash in an older, non-atomic writer
        # would leave behind: not JSON at all.
        path = self._write(state.STATE_FILE, "{ this is not json")
        with self.assertRaises(StackError) as caught:
            state.load(self.dir)
        self.assertEqual("state", caught.exception.step)
        self.assertEqual("qctx stack remove", caught.exception.fix)
        self.assertIn(str(path), str(caught.exception))

    def test_a_wrong_shape_is_corrupt_too(self):
        # Valid JSON, but `ports` is a list: the shape a hand-edited file takes
        # when someone forgets the mapping. It is corrupt in the same way.
        payload = json.loads(json.dumps(make_state().__dict__))
        payload["ports"] = [6333, 8003, 8004]
        path = self._write(state.STATE_FILE, json.dumps(payload))
        with self.assertRaises(StackError) as caught:
            state.load(self.dir)
        self.assertEqual("state", caught.exception.step)
        self.assertEqual("qctx stack remove", caught.exception.fix)
        self.assertIn(str(path), str(caught.exception))

    def test_a_newer_schema_is_refused(self):
        # Valid JSON at a `schema` this code has never seen: reading it is a
        # guess, so it is refused, and the fix is still `remove`.
        payload = json.loads(json.dumps(make_state().__dict__))
        payload["schema"] = state.SCHEMA + 1
        self._write(state.STATE_FILE, json.dumps(payload))
        with self.assertRaises(StackError) as caught:
            state.load(self.dir)
        self.assertEqual("state", caught.exception.step)
        self.assertEqual("qctx stack remove", caught.exception.fix)


class TestTheWrite(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="stack-state-"))

    def test_the_file_is_owner_only(self):
        # Save into a directory that does not exist yet, so `write_json` runs its real
        # publish path: `ensure_dir` creates the stack directory owner-only (0o700) and
        # the staged temporary is created 0o600 before it is renamed over the target.
        # That is the contract that keeps a state file -- which names the ports, the
        # images and the models in use -- private to the user who installed it.
        fresh = self.dir / "not-yet" / catalog.PROJECT / "stack"
        state.save(fresh, make_state())
        self.assertEqual(0o700, fresh.stat().st_mode & 0o777)
        self.assertEqual(0o600, (fresh / state.STATE_FILE).stat().st_mode & 0o777)


class TestTheStackDirectory(unittest.TestCase):
    def test_stack_dir_precedence(self):
        # An empty value counts as absent: the test uses it to prove the
        # precedence, which a missing key would prove by accident.
        home = "/home/me"
        xdg = "/home/me/.local/share"
        cases = [
            ({"QCTX_STACK_DIR": "/custom/stack"}, Path("/custom/stack")),
            ({"QCTX_STACK_DIR": "~", "HOME": home}, Path(home)),
            ({"QCTX_STACK_DIR": "  ", "XDG_DATA_HOME": xdg, "HOME": home},
             Path(xdg) / catalog.PROJECT / "stack"),
            ({"XDG_DATA_HOME": "  ", "HOME": home},
             Path(home) / ".local" / "share" / catalog.PROJECT / "stack"),
            ({"QCTX_STACK_DIR": "  ", "XDG_DATA_HOME": "  ", "HOME": home},
             Path(home) / ".local" / "share" / catalog.PROJECT / "stack"),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertEqual(expected, state.stack_dir(env))

    def test_stack_dir_refuses_no_home_at_all(self):
        with self.assertRaises(StackError) as caught:
            state.stack_dir({})
        self.assertEqual("state", caught.exception.step)
        self.assertIn("QCTX_STACK_DIR", caught.exception.fix or str(caught.exception))


class TestThePlanOf(unittest.TestCase):
    def test_plan_of_carries_every_field(self):
        original = make_state(profile="nvidia", device="Vulkan1", gpu_index=1,
                              provider=["docker-compose", "podman-compose"])
        directory = Path("/custom/stack")
        plan = state.plan_of(original, directory)
        self.assertIsInstance(plan, Plan)
        # `backend` is the stored `profile`; `stack_dir` is the directory argument.
        self.assertEqual(original.profile, plan.backend)
        self.assertEqual(directory, plan.stack_dir)
        self.assertEqual(original.platform, plan.platform)
        self.assertEqual(original.runtime, plan.runtime)
        self.assertEqual(original.device, plan.device)
        self.assertEqual(original.gpu_index, plan.gpu_index)
        self.assertEqual(original.ports, plan.ports)
        self.assertEqual(original.images, plan.images)
        self.assertEqual(original.selinux, plan.selinux)
        self.assertEqual(catalog.PROJECT, plan.project)


if __name__ == "__main__":
    unittest.main(verbosity=2)
