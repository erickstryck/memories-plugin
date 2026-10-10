# The llama-dzn image: the official llama.cpp server-vulkan build PLUS Mesa's
# dzn Vulkan driver (Vulkan-over-D3D12), the only GPU path inside WSL2
# (spec-mestra, "Imagem propria e GPU no Windows (fase 3)" -> "A imagem").
#
# AMDS64-ONLY: the official server-vulkan base and this build are amd64, and
# the dzn targets the WSL2 dxcore runtime. The Mesa version is pinned to
# 26.0.3 to match the mesa-vulkan-drivers of the official image, so the dzn
# does not diverge from the other drivers.
#
# Stage 1 (build): Ubuntu 26.04 — the same base the official image uses —
# compiles ONLY the dzn driver from the Mesa 26.0.3 tarball. The sha256 below
# is the MEASURED value of https://mesa3d.org/archive/mesa-26.0.3.tar.xz
# (43,855,948 bytes, measured 2026-10-09; the root URL without /archive/ is a
# 404). A mismatch fails the build: continuing on a wrong hash would be a
# supply-chain defect.
#
# Stage 2 (final): FROM the official image by its MULTI-ARCH INDEX digest
# (catalog.LLAMA_IMAGE) and ADD ONLY the dzn: libvulkan_dzn.so and its ICD
# manifest, at the location the Vulkan loader searches. The llama.cpp binary
# stays the official one — this image is the official image plus the driver.

###############################################################################
# Stage 1 — build the dzn driver from Mesa 26.0.3
###############################################################################
FROM ubuntu:26.04 AS dzn-build

# The /archive/ path is the live one; the root mesa3d.org URL is a 404.
ARG MESA_URL=https://mesa3d.org/archive/mesa-26.0.3.tar.xz
# The measured sha256 of the tarball at MESA_URL (never an invented value).
ARG MESA_SHA256=ddb7443d328e89aa45b4b6b80f077bf937f099daeca8ba48cabe32aab769e134

# Build toolchain: meson, ninja and python3-mako are the spec's list; the
# compiler and headers are the plain ones a Meson build needs. directx-headers
# is the package the dzn compiles against — directx-headers-dev 1.619.1-1
# exists in Ubuntu 26.04 (spec-mestra's fact table, measured).
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl g++ meson ninja python3 python3-mako \
        directx-headers-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Verify the tarball's sha256 BEFORE using it: the build fails on a mismatch.
RUN curl -fsSL "$MESA_URL" -o mesa.tar.xz \
    && echo "$MESA_SHA256  mesa.tar.xz" | sha256sum -c -

RUN tar -xJf mesa.tar.xz \
    && cd mesa-26.0.3 \
        # The dzn-only compile (recorded for the Task-12 spike, which runs this
        # on a real WSL2 build and may adjust it): meson setup with
        # -Dvulkan-drivers=microsoft-experimental and every other driver,
        # platform, GL and LLVM off. Verified against the 26.0.3 tree:
        #   * `platforms` and `gallium-drivers` are ARRAY options, so they
        #     take the empty list, never `disabled`;
        #   * the dzn pulls in the dxil compiler and the SPIRV-to-DXIL path
        #     from within src/microsoft/ (gated only on the driver being
        #     selected), and links the system `DirectX-Headers` >= 1.618.1
        #     via the directx-headers-dev package (required when dzn is on);
        #   * with no windowing platform the wsi compiles headless
        #     (wsi_common_headless.c) — the dzn loads WSL2's dxcore library
        #     at RUNTIME, so nothing WSI needs to be built in;
        #   * the ICD manifest's library path is generated as
        #     <prefix>/<libdir> = /usr/lib/x86_64-linux-gnu, which is where
        #     stage 2 copies libvulkan_dzn.so (Ubuntu's multiarch libdir).
        # The spec's own list (meson, ninja, python3-mako,
        # directx-headers-dev) is the apt install above; the compiler and
        # headers are the plain prerequisites of any Meson build.
        env -u MESA_CROSS_FILE \
        meson setup build \
            --prefix=/usr \
            -Dvulkan-drivers=microsoft-experimental \
            -Dgallium-drivers= \
            -Dplatforms= \
            -Degl=disabled \
            -Dgles1=disabled \
            -Dgles2=disabled \
            -Dopengl=false \
            -Dglx=disabled \
            -Dllvm=disabled \
        && meson compile -C build \
    # libvulkan_dzn.so in the build tree is the compiled driver itself:
    # a regular file, because the dzn's shared_library sets no
    # version/soversion, so meson emits no versioned form of it. cp -L
    # is used anyway, so the copy still works if a future Mesa bump
    # does make it a versioned library. The ICD manifest is the dzn's
    # only one in the build tree (dzn_devenv_icd is the dev-environment
    # variant and is not installed by default).
    && cp -L build/src/microsoft/vulkan/libvulkan_dzn.so /src/ \
    && find build -name 'dzn_icd*.json' -exec cp {} /src/dzn_icd.json \;

###############################################################################
# Stage 2 — the official image (digest-pinned) plus the dzn
###############################################################################
# The multi-arch INDEX digest of the official server-vulkan-b11382 image, the
# same pin as catalog.LLAMA_IMAGE. The base is overridden by ARG from the
# publish workflow (.github/workflows/llama-dzn.yml, bump input "base digest").
ARG LLAMA_BASE=ghcr.io/ggml-org/llama.cpp:server-vulkan-b11382@sha256:431561ee79ee67b3980a02ff47ed9dc19496127643b75671d9789ef693ca57f9
FROM ${LLAMA_BASE}

# Only the dzn is added: the driver library and its ICD manifest, at the
# location the Vulkan loader searches for ICDs. The manifest's library path
# was generated for /usr (stage 1's --prefix), so it already points at the
# destination. The official image's mesa-vulkan-drivers keeps its own ICDs
# (RADV, ANV, hasvk, lavapipe, NVK, asahi, virtio); the loader enumerates
# every ICD it finds, so the dzn simply appears beside them.
COPY --from=dzn-build /src/libvulkan_dzn.so /usr/lib/x86_64-linux-gnu/
COPY --from=dzn-build /src/dzn_icd.json /usr/share/vulkan/icd.d/
