"""The catalogue, pinned to the values the stack ships with.

The assertions are absolute on purpose: a bump is an edit of `stack/catalog.py` plus the
tables in this file, so changing one constant is what goes red. The digests live in the
tables, where a bump can be reviewed against them, and they are not secret-shaped.
"""
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError  # noqa: E402
from stack import catalog  # noqa: E402


AS_SHIPPED = {
    "llama": "ghcr.io/ggml-org/llama.cpp:server-vulkan-b11382"
             "@sha256:431561ee79ee67b3980a02ff47ed9dc19496127643b75671d9789ef693ca57f9",
    "qdrant": "docker.io/qdrant/qdrant:v1.19.2-unprivileged"
              "@sha256:efb96a9425a90d2d5a1a0a474156280df1892dcdf1af3e8515bd2589b1bfd88b",
    # tag only: the dzn pin carries no digest until the build workflow's first
    # publish (catalog, BUMP PROCEDURE); the digest is copied in by the bump
    "llama-dzn": "ghcr.io/erickstryck/llama-dzn:b11382-mesa26.0.3",
}

EMBED_AS_SHIPPED = {
    "role": "embed",
    "repo": "gpustack/bge-m3-GGUF",
    "revision": "2d48f1737679ad900d5c26c5aad5410e9c70fdca",
    "filename": "bge-m3-Q4_K_M.gguf",
    "size": 437778496,
    "sha256": "6d39681b26c61279ac1f82db35a04a05009e94c415b51c858ff571489a82fc06",
    "license": "MIT",
}

RERANK_AS_SHIPPED = {
    "role": "rerank",
    "repo": "gpustack/bge-reranker-v2-m3-GGUF",
    "revision": "3093af03b1a635e67b084b1d8c03c5f5e020fd05",
    "filename": "bge-reranker-v2-m3-Q4_K_M.gguf",
    "size": 438376864,
    "sha256": "e186a244ed455b4ab66ec64339ce7427a6ae13f5c0b5e544de96e50f0f8b3673",
    "license": "Apache-2.0",
}

#: The Qdrant reference with no tag at all: a digest alone says nothing about the version,
#: so `qdrant_version` must refuse it and the minor guard cannot lean on it.
QDRANT_REF_WITHOUT_VERSION = ("qdrant/qdrant@sha256:efb96a9425a90d2d5a1a0a474156280df1892"
                              "dcdf1af3e8515bd2589b1bfd88b")


class TestThePinnedValues(unittest.TestCase):
    def test_the_images_are_the_pinned_ones(self):
        self.assertEqual(catalog.IMAGES, AS_SHIPPED)
        self.assertEqual(catalog.LLAMA_IMAGE, AS_SHIPPED["llama"])
        self.assertEqual(catalog.QDRANT_IMAGE, AS_SHIPPED["qdrant"])
        self.assertEqual(catalog.LLAMA_DZN_IMAGE, AS_SHIPPED["llama-dzn"])
        # the dzn pin is a tag until the workflow's first publish: no digest yet
        self.assertNotIn("@", catalog.LLAMA_DZN_IMAGE)

    def test_the_image_env_names_are_the_pinned_ones(self):
        self.assertEqual(catalog.IMAGE_ENV,
                         {"llama": "QCTX_STACK_IMAGE_LLAMA",
                          "qdrant": "QCTX_STACK_IMAGE_QDRANT",
                          "llama-dzn": "QCTX_STACK_IMAGE_LLAMA_DZN"})

    def test_the_models_are_the_pinned_ones(self):
        self.assertEqual(asdict(catalog.EMBED_MODEL), EMBED_AS_SHIPPED)
        self.assertEqual(asdict(catalog.RERANK_MODEL), RERANK_AS_SHIPPED)
        self.assertEqual(catalog.MODELS, (catalog.EMBED_MODEL, catalog.RERANK_MODEL))

    def test_the_models_weigh_836_mib(self):
        self.assertEqual(catalog.MODELS_BYTES, 437778496 + 438376864)
        self.assertEqual(round(catalog.MODELS_BYTES / 2**20), 836)

    def test_the_urls_are_the_revision_pinned_resolve_urls(self):
        self.assertEqual(
            catalog.EMBED_MODEL.url(),
            "https://huggingface.co/gpustack/bge-m3-GGUF"
            "/resolve/2d48f1737679ad900d5c26c5aad5410e9c70fdca/bge-m3-Q4_K_M.gguf")
        self.assertEqual(
            catalog.RERANK_MODEL.url(),
            "https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF"
            "/resolve/3093af03b1a635e67b084b1d8c03c5f5e020fd05"
            "/bge-reranker-v2-m3-Q4_K_M.gguf")

    def test_the_ports_and_the_default_names_are_the_pinned_ones(self):
        self.assertEqual(catalog.PORTS, {"qdrant": 6333, "embed": 8003, "rerank": 8004})
        self.assertEqual(catalog.CONTAINER_PORTS,
                         {"qdrant": 6333, "embed": 8080, "rerank": 8080})
        self.assertEqual(catalog.PORT_FALLBACK_OFFSET, 10000)
        self.assertEqual(catalog.PROJECT, "mnemosine")
        self.assertEqual(catalog.SERVICES, ("qdrant", "embed", "rerank"))
        self.assertEqual(catalog.VOLUME, "mnemosine-qdrant")
        self.assertEqual(catalog.EMBED_DIM, 1024)
        self.assertEqual(catalog.CONTEXT, 8192)


class TestTheServerCommand(unittest.TestCase):
    def test_the_cpu_command_declares_no_device(self):
        # Not merely omitting the device: an engine can inject a GPU into every container,
        # so the cpu profile has to say `none` on its own.
        self.assertEqual(catalog.server_command("embed", None)[-2:], ["-dev", "none"])

    def test_the_cpu_commands_are_the_measured_ones(self):
        self.assertEqual(
            catalog.server_command("embed", None),
            ["-m", "/models/bge-m3-Q4_K_M.gguf", "--host", "0.0.0.0",
             "--port", "8080", "--embedding", "-c", "8192", "-b", "8192",
             "-ub", "8192", "--no-ui", "-dev", "none"])
        self.assertEqual(
            catalog.server_command("rerank", None),
            ["-m", "/models/bge-reranker-v2-m3-Q4_K_M.gguf", "--host", "0.0.0.0",
             "--port", "8080", "--reranking", "-c", "8192", "-b", "8192",
             "-ub", "8192", "--no-ui", "-dev", "none"])

    def test_a_gpu_command_names_its_device(self):
        self.assertEqual(catalog.server_command("rerank", "Vulkan2")[-2:],
                         ["-dev", "Vulkan2"])

    def test_both_roles_carry_the_8192_batch_and_the_current_ui_flag(self):
        for role, switch in (("embed", "--embedding"), ("rerank", "--reranking")):
            cmd = catalog.server_command(role, None)
            self.assertIn(switch, cmd)
            self.assertIn("--no-ui", cmd)
            self.assertNotIn("--no-webui", cmd)
            for flag in ("-c", "-b", "-ub"):
                self.assertEqual(cmd[cmd.index(flag) + 1], "8192")

    def test_an_unknown_role_is_refused(self):
        with self.assertRaises(StackError) as ctx:
            catalog.server_command("llm", None)
        self.assertEqual(ctx.exception.step, "catalog")


class TestTheQdrantVersion(unittest.TestCase):
    def test_qdrant_version_comes_from_the_tag(self):
        self.assertEqual(catalog.qdrant_version(catalog.QDRANT_IMAGE), "1.19.2")
        self.assertEqual(catalog.qdrant_version("qdrant/qdrant:v1.13.4-unprivileged"),
                         "1.13.4")
        # a registry with a port: the tag is the one on the LAST path segment
        self.assertEqual(catalog.qdrant_version(
            "localhost:5000/qdrant/qdrant:v1.19.2-unprivileged@sha256:" + "0" * 64),
            "1.19.2")

    def test_a_reference_without_a_version_is_refused(self):
        for ref in (QDRANT_REF_WITHOUT_VERSION, "qdrant/qdrant:latest"):
            with self.subTest(ref=ref):
                with self.assertRaises(StackError) as ctx:
                    catalog.qdrant_version(ref)
                self.assertEqual(ctx.exception.step, "catalog")

    def test_a_refused_version_names_the_tagged_image_fix(self):
        # R5 (fix-round-r2-rulings m5, part 2): the refusal has to tell the user what to
        # do, not only that the reference is unusable. The string is the ruling's, verbatim.
        with self.assertRaises(StackError) as ctx:
            catalog.qdrant_version(QDRANT_REF_WITHOUT_VERSION)
        self.assertEqual(ctx.exception.fix, "use a Qdrant image tagged vX.Y.Z")


class TestTheImageOverride(unittest.TestCase):
    def test_image_flags_parse_and_refuse(self):
        self.assertEqual(catalog.parse_image_flags(["qdrant=q:1"]), {"qdrant": "q:1"})
        self.assertEqual(
            catalog.parse_image_flags(["llama=a:1", "qdrant=b:2"]),
            {"llama": "a:1", "qdrant": "b:2"})
        # the dzn role is a catalogue entry (Task 4): an override may name it
        self.assertEqual(catalog.parse_image_flags(["llama-dzn=r:1"]), {"llama-dzn": "r:1"})
        for bad in (["qdrant"], ["dzn=x"], ["llama="]):
            with self.subTest(bad=bad):
                with self.assertRaises(StackError) as ctx:
                    catalog.parse_image_flags(bad)
                self.assertEqual(ctx.exception.step, "catalog")

    def test_an_unknown_role_points_at_the_override_form(self):
        with self.assertRaises(StackError) as ctx:
            catalog.parse_image_flags(["nope=x"])
        self.assertEqual(ctx.exception.fix, "--image ROLE=REF, ROLE is llama, qdrant or llama-dzn")

    def test_flags_beat_env_beat_catalog(self):
        got = catalog.resolve_images({"llama": "f"},
                                     {"QCTX_STACK_IMAGE_LLAMA": "e",
                                      "QCTX_STACK_IMAGE_QDRANT": "  "})
        self.assertEqual(got, {"llama": "f", "qdrant": catalog.QDRANT_IMAGE,
                               "llama-dzn": catalog.LLAMA_DZN_IMAGE})

    def test_env_beats_catalog_and_blank_env_is_no_value(self):
        got = catalog.resolve_images({},
                                     {"QCTX_STACK_IMAGE_LLAMA": "e",
                                      "QCTX_STACK_IMAGE_QDRANT": "   "})
        self.assertEqual(got, {"llama": "e", "qdrant": catalog.QDRANT_IMAGE,
                               "llama-dzn": catalog.LLAMA_DZN_IMAGE})
        self.assertEqual(catalog.resolve_images({}, {}), catalog.IMAGES)

    def test_the_dzn_role_resolves_by_flag_and_env(self):
        # flag > env > catalogue, the same precedence as every role
        self.assertEqual(catalog.resolve_images({"llama-dzn": "f"},
                                                {"QCTX_STACK_IMAGE_LLAMA_DZN": "e"})["llama-dzn"],
                         "f")
        self.assertEqual(catalog.resolve_images({},
                                                {"QCTX_STACK_IMAGE_LLAMA_DZN": "e"})["llama-dzn"],
                         "e")
        self.assertEqual(catalog.resolve_images({}, {})["llama-dzn"], catalog.LLAMA_DZN_IMAGE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
