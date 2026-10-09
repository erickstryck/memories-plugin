"""What the stack pins: the three images, the two GGUFs, the server flags and the ports.

This is DATA, not behaviour: every other `stack` module renders the compose, fetches the
models and talks to a runtime from the constants here, so a version bump is an edit of
this file (and of the pinning tables in `tests/test_stack_catalog.py`), reviewed against
the digest literals.

BUMP PROCEDURE (spec, "Bump e override"), once per release that touches this file:

1. llama.cpp: choose the build and read the digest of the `server-vulkan-bNNNNN` INDEX
   (a GET on the manifest with the OCI index Accept header; the digest comes back in the
   `docker-content-digest` header). Pin `tag@digest`, never the tag alone.
2. Qdrant: the NEXT MINOR ONLY, never skip. Qdrant guarantees storage compatibility only
   between consecutive minors, and `qdrant_version` feeds the guard that enforces it on
   `up --upgrade`.
3. llama-dzn: read the digest of the published image from the build workflow's run
   (`.github/workflows/llama-dzn.yml`, which the install builds locally when the pin
   cannot be pulled) and pin `tag@digest` in `LLAMA_DZN_IMAGE`, the same way as llama.
   Until that first publish the pin is the tag alone.
4. Run the opt-in integration (`QCTX_STACK_IT=1`) against the new pins, then
   regenerate the golden fixtures: `python3 tests/test_stack_compose.py --regen`.

OVERRIDES: `--image ROLE=REF` (parsed by `parse_image_flags`) beats the
`QCTX_STACK_IMAGE_LLAMA` / `QCTX_STACK_IMAGE_QDRANT` / `QCTX_STACK_IMAGE_LLAMA_DZN`
environment, which beats this catalogue (`resolve_images`). What was used is written to
`stack.json`; `qctx stack up` without `--upgrade` repeats exactly that. Nothing here
updates on its own.
"""
import re
from dataclasses import dataclass
from typing import Mapping

from . import StackError

LLAMA_IMAGE = ("ghcr.io/ggml-org/llama.cpp:server-vulkan-b11382"
               "@sha256:431561ee79ee67b3980a02ff47ed9dc19496127643b75671d9789ef693ca57f9")
QDRANT_IMAGE = ("docker.io/qdrant/qdrant:v1.19.2-unprivileged"
                "@sha256:efb96a9425a90d2d5a1a0a474156280df1892dcdf1af3e8515bd2589b1bfd88b")
#: The OWN build the Windows GPU profile runs: the official image plus Mesa's dzn driver
#: (Vulkan over D3D12), built by `.github/workflows/llama-dzn.yml`. The pin is the TAG
#: until that workflow's first publish; the digest is copied in from the run by the bump
#: (step 3 of the BUMP PROCEDURE), the same way as llama's.
LLAMA_DZN_IMAGE = "ghcr.io/erickstryck/llama-dzn:b11382-mesa26.0.3"

#: The roles an override may name. `llama-dzn` is the own build (Windows, the dzn
#: profile), a tag until the publish workflow's first digest arrives.
IMAGES = {"llama": LLAMA_IMAGE, "qdrant": QDRANT_IMAGE, "llama-dzn": LLAMA_DZN_IMAGE}
IMAGE_ENV = {"llama": "QCTX_STACK_IMAGE_LLAMA", "qdrant": "QCTX_STACK_IMAGE_QDRANT",
             "llama-dzn": "QCTX_STACK_IMAGE_LLAMA_DZN"}


@dataclass(frozen=True)
class Model:
    role: str
    repo: str
    revision: str
    filename: str
    size: int
    sha256: str
    license: str

    def url(self) -> str:
        """The revision-pinned download: a commit, not a branch, so the bytes never move
        under a running `stack up`."""
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"


EMBED_MODEL = Model("embed", "gpustack/bge-m3-GGUF",
                    "2d48f1737679ad900d5c26c5aad5410e9c70fdca",
                    "bge-m3-Q4_K_M.gguf", 437778496,
                    "6d39681b26c61279ac1f82db35a04a05009e94c415b51c858ff571489a82fc06",
                    "MIT")
RERANK_MODEL = Model("rerank", "gpustack/bge-reranker-v2-m3-GGUF",
                     "3093af03b1a635e67b084b1d8c03c5f5e020fd05",
                     "bge-reranker-v2-m3-Q4_K_M.gguf", 438376864,
                     "e186a244ed455b4ab66ec64339ce7427a6ae13f5c0b5e544de96e50f0f8b3673",
                     "Apache-2.0")
MODELS = (EMBED_MODEL, RERANK_MODEL)
MODELS_BYTES = sum(m.size for m in MODELS)

#: bge-m3 embeds to 1024 dimensions; the Qdrant collection is created from this.
EMBED_DIM = 1024
#: The context, batch and ubatch of both servers, pinned at 8192 (spec, measured on
#: b11382). A chunk over the ubatch is refused outright, so all three move together.
CONTEXT = 8192

#: Host ports, published on 127.0.0.1 only; a busy one moves to `port + PORT_FALLBACK_OFFSET`.
PORTS = {"qdrant": 6333, "embed": 8003, "rerank": 8004}
#: What the container itself listens on. Both llama-servers share 8080: they never run on
#: the same machine port at once because the host mapping keeps them apart.
CONTAINER_PORTS = {"qdrant": 6333, "embed": 8080, "rerank": 8080}
PORT_FALLBACK_OFFSET = 10000

PROJECT = "mnemosine"
SERVICES = ("qdrant", "embed", "rerank")
VOLUME = "mnemosine-qdrant"

_QDRANT_VERSION = re.compile(r"^v(\d+\.\d+\.\d+)")


def server_command(role: str, device: str | None) -> list[str]:
    """The llama-server arguments for one model.

    The flags are the ones measured on b11382: `--no-ui` because `--no-webui` is
    deprecated there; `-c/-b/-ub 8192` together because with embeddings on the server
    forces `n_batch = n_ubatch`, and `--reranking` turns the same mode on. `--host
    0.0.0.0` is required inside the container for the port mapping; what keeps the
    server off the network is publishing on 127.0.0.1 only.

    `device` is the `-dev` value: a `Vulkan<n>` name on a GPU profile, or None on the
    cpu profile, which then passes `-dev none`. The `none` is load-bearing: an engine
    can be configured to inject a GPU into EVERY container (the `devices` line in
    containers.conf), and a llama-server with no `-dev` then takes it.
    """
    if role == "embed":
        model, switch = EMBED_MODEL, "--embedding"
    elif role == "rerank":
        model, switch = RERANK_MODEL, "--reranking"
    else:
        raise StackError(
            f"unknown role: {role}", step="catalog",
            fix='role is "embed" or "rerank"')
    return ["-m", f"/models/{model.filename}",
            "--host", "0.0.0.0",
            "--port", str(CONTAINER_PORTS[role]),
            switch,
            "-c", str(CONTEXT), "-b", str(CONTEXT), "-ub", str(CONTEXT),
            "--no-ui",
            "-dev", device or "none"]


def outdated_pins(images: Mapping[str, str]) -> list[str]:
    """The catalogue roles whose pin moved past the recorded one, in the
    catalogue's order. The single owner of the comparison, so the status, the
    install step and the install report cannot drift apart (the three copies
    that used to live in `lifecycle.status`, `cli._step_managed` and
    `cli.check_section`). An empty recording counts every role as moved: a
    state written before the catalogue carried images is out of date."""
    return [role for role in IMAGES if images.get(role) != IMAGES[role]]


def qdrant_version(ref: str) -> str:
    """The `x.y.z` off a Qdrant reference's tag, for the minor guard on `up --upgrade`.

    A digest-only reference carries no version, and `latest` is not a version either:
    both refuse, because a guard that reads a guess is worse than no guard.
    """
    # the tag of the LAST path segment: a registry port (`host:5000/...`) never reaches it
    tag = ref.partition("@")[0].rsplit("/", 1)[-1].partition(":")[2]
    match = _QDRANT_VERSION.match(tag)
    if match is None:
        raise StackError(f"no Qdrant version in: {ref}", step="catalog",
                         fix="use a Qdrant image tagged vX.Y.Z")
    return match.group(1)


def parse_image_flags(values: list[str]) -> dict[str, str]:
    """The `--image ROLE=REF` pairs, checked against the roles this catalogue names."""
    out: dict[str, str] = {}
    for value in values:
        role, sep, ref = value.partition("=")
        if not sep:
            raise StackError(
                f"--image wants ROLE=REF, got: {value!r}", step="catalog",
                fix="--image ROLE=REF, ROLE is llama, qdrant or llama-dzn")
        if not ref:
            raise StackError(f"--image has an empty reference: {value!r}",
                             step="catalog",
                             fix="--image ROLE=REF, ROLE is llama, qdrant or llama-dzn")
        if role not in IMAGES:
            raise StackError(
                f"unknown --image role: {role}", step="catalog",
                fix="--image ROLE=REF, ROLE is llama, qdrant or llama-dzn")
        out[role] = ref
    return out


def resolve_images(flags: dict[str, str], env: Mapping[str, str],
                   base: dict[str, str] | None = None) -> dict[str, str]:
    """Which image each role runs on: the `--image` flag beats the environment, the
    environment beats this catalogue. A blank environment value is no value at all.
    """
    base = IMAGES if base is None else base
    out: dict[str, str] = {}
    for role, default in base.items():
        if role in flags:
            out[role] = flags[role]
        else:
            env_ref = env.get(IMAGE_ENV.get(role, ""), "").strip()
            out[role] = env_ref or default
    return out
