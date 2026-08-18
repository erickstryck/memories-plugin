import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import http  # noqa: E402
from core import windowprobe  # noqa: E402


def a_listing(*models):
    return {"data": list(models)}


class TestTheFourShapesMeasuredAgainstRealServERS(unittest.TestCase):
    """Every fixture here is a shape measured against a live server on 2026-08-17, not a
    shape imagined. The values are the ones those servers returned."""

    def test_vLLM_puts_it_at_the_top_as_max_model_len(self):
        entry = {"id": "Qwen3.8-27B", "max_model_len": 524288}
        self.assertEqual(windowprobe.window_from(entry), 524288)

    def test_openrouter_aggregate_is_context_length(self):
        entry = {"id": "qwen/qwen3.8-27b", "context_length": 262144}
        self.assertEqual(windowprobe.window_from(entry), 262144)

    def test_llama_cpp_NESTS_it_under_meta(self):
        """A search that only looks at the top level misses this one entirely."""
        entry = {"id": "bge-m3", "meta": {"n_ctx": 8192, "n_ctx_train": 8192}}
        self.assertEqual(windowprobe.window_from(entry), 8192)

    def test_an_unknown_shape_yields_nothing_rather_than_a_guess(self):
        self.assertEqual(windowprobe.window_from({"id": "x", "window_size_tokens": 4096}), 0)

    def test_a_non_numeric_or_absurd_value_yields_nothing(self):
        self.assertEqual(windowprobe.window_from({"id": "x", "max_model_len": "big"}), 0)
        self.assertEqual(windowprobe.window_from({"id": "x", "max_model_len": 0}), 0)
        self.assertEqual(windowprobe.window_from({"id": "x", "max_model_len": -1}), 0)


class TestTheOrderThatOpenRouterForces(unittest.TestCase):
    def test_when_the_two_openrouter_fields_DISAGREE_the_provider_one_wins(self):
        """Measured: OpenRouter's two fields disagree on 34 of its 414 models.
        `context_length` is the best across every provider serving that model;
        `top_provider.context_length` is the one the request actually reaches. Taking the
        aggregate makes the guard sleep exactly when it needed to wake — a fixture where the
        two agree proves nothing about the order."""
        entry = {"id": "nvidia/nemotron-3.5-lightning",
                 "context_length": 1_000_000,
                 "top_provider": {"context_length": 262_144}}
        self.assertEqual(windowprobe.window_from(entry), 262_144)

    def test_a_broken_top_provider_falls_to_the_aggregate_rather_than_to_nothing(self):
        entry = {"id": "x", "context_length": 262_144, "top_provider": {}}
        self.assertEqual(windowprobe.window_from(entry), 262_144)


class TestProbing(unittest.TestCase):
    def test_it_finds_the_entry_whose_id_matches_the_model(self):
        listing = a_listing({"id": "other", "max_model_len": 111},
                            {"id": "wanted", "max_model_len": 222})
        got = windowprobe.probe("http://x/v1", "k", "wanted", fetch=lambda *a, **k: listing)
        self.assertEqual(got, 222)

    def test_a_model_absent_from_the_listing_yields_nothing(self):
        listing = a_listing({"id": "other", "max_model_len": 111})
        self.assertEqual(windowprobe.probe("http://x/v1", "k", "wanted",
                                           fetch=lambda *a, **k: listing), 0)

    def test_an_unreachable_endpoint_yields_nothing_and_does_not_raise(self):
        def boom(*a, **k):
            raise OSError("connection refused")

        self.assertEqual(windowprobe.probe("http://x/v1", "k", "m", fetch=boom), 0)

    def test_a_listing_that_is_not_a_listing_yields_nothing(self):
        for junk in ({}, {"data": "nope"}, {"data": [None]}, [], "text"):
            with self.subTest(junk=junk):
                self.assertEqual(windowprobe.probe("http://x/v1", "k", "m",
                                                   fetch=lambda *a, **k: junk), 0)

    def test_it_asks_the_models_route_of_the_base_url(self):
        seen = {}

        def spy(url, **kw):
            seen["url"] = url

            return a_listing()

        windowprobe.probe("http://x/v1", "k", "m", fetch=spy)
        self.assertEqual(seen["url"], "http://x/v1/models")

    def test_a_base_url_with_a_trailing_slash_does_not_double_it(self):
        seen = {}

        def spy(url, **kw):
            seen["url"] = url

            return a_listing()

        windowprobe.probe("http://x/v1/", "k", "m", fetch=spy)
        self.assertEqual(seen["url"], "http://x/v1/models")


@unittest.skipUnless(os.environ.get("QCTX_INTEGRATION") == "1",
                     "integration: needs a reachable model endpoint")
class TestARealServerAnswers(unittest.TestCase):
    """The half a fake cannot prove: that a real listing has the shape this module expects.
    Uses whatever endpoint the config points at; skips when it names no model we can check."""

    def test_the_listing_has_data_entries_with_ids(self):
        from core import load
        cfg = load()
        if not cfg.api_base_url:
            self.skipTest("no api_base_url configured")
        url = cfg.api_base_url.rstrip("/") + "/models"
        payload = http.request_json(url, headers=http.bearer(cfg.api_key), timeout=10.0)
        self.assertIsInstance(payload.get("data"), list)
        for entry in payload["data"]:
            self.assertIn("id", entry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
