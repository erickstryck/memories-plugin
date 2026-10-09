"""The final summary block: `summary.render_summary`, the pure function the
installer prints at the end of a successful install, on every platform.

The tests build a `compose.Plan` with known ports and a plain config dict and
assert on the MEANINGFUL content of the lines (the URLs, the api-key text,
the engine-VM line) — never on the exact whitespace padding of the label
column. The stale-URL case is the one that pins Ruling R11: the three URLs
come from the plan's ports (the stack is running on them even when the user
declined the config write), while the api-key comes from the passed config.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import catalog, compose, summary, verify  # noqa: E402

#: The default api-key text the spec's block carries when the local Qdrant
#: has no key (pinned here, not imported: the spec is the authority).
NO_KEY_TEXT = "(nenhuma: Qdrant local sem chave)"


def make_plan(platform="linux", runtime="podman", backend="cpu",
              ports=None) -> compose.Plan:
    """A plan with known ports; `stack_dir` is never validated in the
    constructor (only in `render`), so a placeholder path is enough."""
    return compose.Plan(
        platform=platform, runtime=runtime, backend=backend, device=None,
        gpu_index=None, ports=dict(ports or catalog.PORTS),
        stack_dir=Path("/tmp/qctx-summary-stack"), images={}, selinux=False)


def _line(lines: list[str], label: str) -> str:
    """The line whose label column reads `label` (padding not pinned)."""
    return next(l for l in lines if l.lstrip().startswith(label))


class TestRenderSummary(unittest.TestCase):
    def test_headline_carries_the_runtime_and_the_backend(self):
        lines = summary.render_summary({}, make_plan(runtime="docker",
                                                     backend="cpu"))
        self.assertEqual(lines[0], "stack: running (docker, cpu)")
        lines = summary.render_summary({}, make_plan(runtime="podman",
                                                     backend="dzn"))
        self.assertEqual(lines[0], "stack: running (podman, dzn)")

    def test_the_urls_come_from_the_plan_not_the_saved_config(self):
        # R11: a saved config whose URLs point elsewhere (the decline case,
        # where the file keeps its old values) must not win over the plan:
        # the stack IS running on the plan's ports.
        plan = make_plan()
        stale = {"qdrant_url": "http://127.0.0.1:9999",
                 "api_base_url": "http://127.0.0.1:9999/v1",
                 "rerank_url": "http://127.0.0.1:9999/v1/rerank"}
        lines = summary.render_summary(stale, plan)
        urls = verify.stack_urls(plan.ports)
        self.assertIn(urls["qdrant_url"], _line(lines, "qdrant"))
        self.assertIn(urls["api_base_url"], _line(lines, "embed"))
        self.assertIn(urls["rerank_url"], _line(lines, "rerank"))
        self.assertFalse(any("9999" in l for l in lines))

    def test_the_urls_match_stack_urls_of_the_plans_ports(self):
        # The exact formats of `verify.stack_urls` (the embed line ends in
        # `/v1`, the rerank line in `/v1/rerank`), from non-default ports.
        ports = {"qdrant": 7000, "embed": 7001, "rerank": 7002}
        lines = summary.render_summary({}, make_plan(ports=ports))
        self.assertIn("http://127.0.0.1:7000", _line(lines, "qdrant"))
        self.assertIn("http://127.0.0.1:7001/v1", _line(lines, "embed"))
        self.assertIn("http://127.0.0.1:7002/v1/rerank", _line(lines, "rerank"))

    def test_no_api_key_shows_the_default(self):
        lines = summary.render_summary({}, make_plan())
        self.assertIn(NO_KEY_TEXT, _line(lines, "api-key"))

    def test_empty_api_key_shows_the_default(self):
        lines = summary.render_summary({"qdrant_api_key": ""}, make_plan())
        self.assertIn(NO_KEY_TEXT, _line(lines, "api-key"))

    def test_api_key_shows_the_key(self):
        lines = summary.render_summary({"qdrant_api_key": "the-key"},
                                       make_plan())
        key = _line(lines, "api-key")
        self.assertIn("the-key", key)
        self.assertNotIn(NO_KEY_TEXT, key)

    def test_the_summary_reads_the_config_it_is_given(self):
        # Two config objects, one plan: swapping the object swaps the
        # api-key line (the function reads what it is passed, not memory).
        plan = make_plan()
        with_key = summary.render_summary({"qdrant_api_key": "the-key"}, plan)
        without_key = summary.render_summary({}, plan)
        self.assertIn("the-key", _line(with_key, "api-key"))
        self.assertIn(NO_KEY_TEXT, _line(without_key, "api-key"))
        self.assertNotIn("the-key", _line(without_key, "api-key"))
        self.assertNotIn(NO_KEY_TEXT, _line(with_key, "api-key"))

    def test_windows_names_the_ram_as_the_engine_vms(self):
        lines = summary.render_summary(
            {}, make_plan(platform="windows", runtime="docker"))
        ram = [l for l in lines if l.lstrip().startswith("ram")]
        self.assertEqual(len(ram), 1)
        self.assertIn("engine VM", ram[0])
        self.assertIn("Windows host", ram[0])

    def test_linux_has_no_engine_vm_ram_line(self):
        lines = summary.render_summary({}, make_plan(platform="linux"))
        self.assertFalse(any("engine VM" in l for l in lines))

    def test_the_block_is_five_lines_and_windows_adds_a_sixth(self):
        lines = summary.render_summary({}, make_plan())
        self.assertEqual(len(lines), 5)
        self.assertEqual(len(summary.render_summary({},
                                                    make_plan(platform="windows"))), 6)
        labels = [l.lstrip().split()[0] for l in lines]
        self.assertEqual(labels, ["stack:", "qdrant", "embed", "rerank",
                                  "api-key"])


if __name__ == "__main__":
    unittest.main()
