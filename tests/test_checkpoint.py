"""Tests for the checkpoint hook.

It had no tests at all, and that is exactly how a mechanical identifier rename broke
it in silence: the `{intervalo}` placeholder in the template is a STRING, so the rename
left it alone while renaming the `intervalo=` keyword argument that fills it. The hook
raised `KeyError: 'intervalo'` on every checkpoint round, and nothing said so — the host
swallows a hook's stderr, and the other 172 tests never touch this file.

Hence the shape of these tests: they run the hook as the host does — a real process,
JSON on stdin, JSON on stdout — because that is the only way the failure was reachable.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / "hooks" / "checkpoint.py"


def run_hook(session: str, interval: str, state_dir: str, times: int = 1) -> list[str]:
    """Runs the hook `times` times over the same state dir, returning the stdout of each.

    The counter lives on disk, so repeated calls are what exercises the interval.
    """
    env = dict(os.environ, QCTX_STATE_DIR=state_dir, QCTX_CHECKPOINT_INTERVAL=interval)
    env.pop("QCTX_CHECKPOINT_DISABLED", None)
    outputs = []
    for _ in range(times):
        proc = subprocess.run([sys.executable, str(HOOK)], input=json.dumps({"session_id": session}),
                              capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise AssertionError(f"hook failed: {proc.stderr.strip()}")
        outputs.append(proc.stdout)

    return outputs


class TestCheckpointFires(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_the_procedure_is_emitted_and_fully_formatted(self):
        out = run_hook("s1", "1", self.tmp.name)[0]
        context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertIn("Interaction 1 of this conversation (every 1)", context,
                      "both placeholders have to be filled, not left as braces")
        self.assertNotIn("{", context.split("Mandatory metadata")[0],
                         "an unfilled placeholder means the format keys drifted")

    def test_the_metadata_example_survives_formatting(self):
        context = json.loads(run_hook("s2", "1", self.tmp.name)[0])["hookSpecificOutput"]["additionalContext"]
        # Doubled braces in the template: the JSON example is literal, not a placeholder.
        self.assertIn('{"type": "user|feedback|project|reference"', context)

    def test_it_stays_silent_between_checkpoints(self):
        outs = run_hook("s3", "3", self.tmp.name, times=3)
        self.assertEqual([o.strip() for o in outs[:2]], ["", ""],
                         "intermediate interactions must not inject anything")
        self.assertIn("memory checkpoint", outs[2])

    def test_zero_interval_never_fires(self):
        self.assertEqual(run_hook("s4", "0", self.tmp.name, times=4)[-1].strip(), "")

    def test_disabled_produces_nothing(self):
        env = dict(os.environ, QCTX_STATE_DIR=self.tmp.name, QCTX_CHECKPOINT_INTERVAL="1",
                   QCTX_CHECKPOINT_DISABLED="1")
        proc = subprocess.run([sys.executable, str(HOOK)], input="{}",
                              capture_output=True, text=True, env=env)
        self.assertEqual(proc.stdout.strip(), "")
        self.assertEqual(proc.returncode, 0)

    def test_sessions_count_independently(self):
        run_hook("alpha", "2", self.tmp.name)
        # A second session starting from zero must not inherit alpha's count.
        self.assertEqual(run_hook("beta", "2", self.tmp.name)[0].strip(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
