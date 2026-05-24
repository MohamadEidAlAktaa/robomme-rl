"""Modal image definition for robomme-rl.

We build on top of RoboMME's upstream Dockerfile
(`third_party/robomme_policy_learning/Dockerfile`) so we stay in sync with their
canonical recipe (CUDA 12.8, uv, openpi env, robomme micromamba env). Our own
RFT/GRPO dependencies and source are layered on top.

Note: the upstream Dockerfile expects its submodule `robomme_benchmark` to be
checked out locally — Modal ships the local context tarball to its builder, so
`git submodule update --init` must have been run inside
`third_party/robomme_policy_learning/` before invoking Modal.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOMME_DIR = REPO_ROOT / "third_party" / "robomme_policy_learning"

# Upstream image: built from RoboMME's Dockerfile with their directory as context.
# First build is slow (~20-30 min); subsequent builds use Modal's layer cache.
robomme_base = modal.Image.from_dockerfile(
    path=str(ROBOMME_DIR / "Dockerfile"),
    context_dir=str(ROBOMME_DIR),
)

# Layer our own code and any extra deps on top. Keep extra deps minimal — RoboMME
# already pins torch/jax/transformers/wandb. Add RFT/GRPO-specific libs here as
# the loop grows.
#
# Why the symlink + PATH dance:
#   RoboMME's Dockerfile installs Python 3.11 via `uv` into /app/.venv/, with
#   `UV_PYTHON_DOWNLOADS=automatic`. The result is that no `python3` exists on
#   the default PATH, which makes Modal's image introspection fail with
#   "unable to determine the version of Python installed in the Image".
#   We expose the uv-managed Python so Modal can detect it AND so our function
#   runs in the env that has jax/torch/openpi installed.
image = (
    robomme_base
    .run_commands(
        # Fail fast in build if the uv venv isn't where we expect.
        "test -x /app/.venv/bin/python "
        "|| (echo 'expected uv venv at /app/.venv/bin/python — '"
        "'did the upstream Dockerfile change?' && exit 1)",
        # Make the uv Python discoverable for Modal's introspection and tools
        # that look in /usr/local/bin.
        "ln -sf /app/.venv/bin/python /usr/local/bin/python",
        "ln -sf /app/.venv/bin/python /usr/local/bin/python3",
        # Upstream openpi has a top-level `import pytest` in
        # src/openpi/models_pytorch/gemma_pytorch.py (likely a leftover from a
        # refactored test file). It's pulled in transitively by
        # serve_policy.py → mme_vla_suite.policies.policy → ...
        # The upstream Dockerfile installs deps with `uv sync --no-dev`, which
        # excludes pytest (it lives in the [dependency-groups].dev section of
        # pyproject.toml), so the import fails at serve start.
        # We add pytest into the uv venv to satisfy the import without
        # touching upstream source. Pin matches their dev pin.
        # Note: uv venvs don't include `pip` by default — use `uv pip install`
        # targeting the venv's interpreter instead.
        "uv pip install --python /app/.venv/bin/python 'pytest>=8.3.4'",
        # ManiSkill / Sapien need a Vulkan ICD manifest so the Vulkan loader
        # (`libvulkan1`, installed upstream) can find the NVIDIA driver lib
        # (`libGLX_nvidia.so.0`, injected at runtime by nvidia-container-toolkit).
        # Without this file, sim env construction fails with:
        #   RuntimeError: vk::PhysicalDevice::createDeviceUnique:
        #                 ErrorInitializationFailed
        # The base image (`nvidia/cuda:...-runtime-...`) doesn't ship the
        # manifest the way `nvidia/cudagl` does, so we lay it down ourselves.
        "mkdir -p /usr/share/vulkan/icd.d",
        # Use the *absolute* path so dlopen bypasses the dynamic linker search
        # (`LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64` is
        # preset by Modal's runtime; if it contains a stub libGLX_nvidia.so.0
        # that lacks the full Vulkan ICD symbols, the loader picks the stub
        # first and fails). The real file lives at
        # /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 (verified to export
        # vk_icdGetInstanceProcAddr via our find_vulkan_icd probe).
        "printf '%s\\n' "
        "'{' "
        "'  \"file_format_version\": \"1.0.0\",' "
        "'  \"ICD\": {' "
        "'    \"library_path\": \"/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0\",' "
        "'    \"api_version\": \"1.3.0\"' "
        "'  }' "
        "'}' "
        "> /usr/share/vulkan/icd.d/nvidia_icd.json",
        # LLVMpipe / lavapipe — software Vulkan via Mesa, CPU-rasterized.
        # Doesn't need any NVIDIA graphics support; works in any container.
        # Modal's NVIDIA datacenter driver in this image accepts vkCreateInstance
        # but fails on vkCreateDevice for graphics use — datacenter drivers
        # often gate graphics-Vulkan features. LLVMpipe provides a guaranteed
        # working Vulkan path so Sapien can render at all. It's slow
        # (~50-200ms/step instead of <50ms), but unblocks development.
        # Both ICD manifests live in /usr/share/vulkan/icd.d/ so the Vulkan
        # loader auto-discovers them; whichever can vkCreateDevice wins.
        "apt-get update && apt-get install -y --no-install-recommends "
        "mesa-vulkan-drivers && rm -rf /var/lib/apt/lists/*",
        # Upstream `examples/robomme/subgoal_prediction/qwenvl/api.py` calls
        # `PtEngine(..., attn_impl='flash_attention_2')`, but flash-attn isn't
        # installed in the robomme micromamba env (it would need a 15-30 min
        # build from source for torch 2.9.1+cu128 with no prebuilt wheel).
        # We've already verified attn_impl='sdpa' works in test_load_planner
        # (Qwen3-VL-4B loads + runs cleanly with PyTorch's native SDPA path).
        # In-place patch the relevant lines during image build. Both api.py
        # (regular Qwen3-VL planner) and api_memer.py (MemER variant) hit it.
        # We can revert to flash_attention_2 once we're ready to wait through
        # the flash-attn build for the eventual RL training throughput win.
        "sed -i \"s/attn_impl='flash_attention_2'/attn_impl='sdpa'/g\" "
        "/app/examples/robomme/subgoal_prediction/qwenvl/api.py "
        "/app/examples/robomme/subgoal_prediction/qwenvl/api_memer.py "
        "|| true",
    )
    .env({
        # Put the uv venv first so `python`, `pip`, and installed entry points
        # resolve to the env with RoboMME's deps.
        "PATH": "/app/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": "/app/src:/workspace/src",
        # Point openpi + HuggingFace caches at the Modal Volume mount (see
        # CACHE_VOLUME in modal/app.py, mounted at /cache). This is how
        # `openpi.shared.download.maybe_download` and `huggingface_hub` know
        # where to put the multi-GB checkpoints so they persist across runs.
        # Upstream default is ~/.cache/openpi, which would be inside the
        # ephemeral container fs and lost on shutdown.
        "OPENPI_DATA_HOME": "/cache/openpi",
        "HF_HOME": "/cache/hf",
        # JAX persists compiled kernels here. Without this, every VLA server
        # boot recompiles from scratch (~50-90 s). With it, second-and-later
        # boots are ~5-10 s. Big win across the dozens of containers in a
        # parallel run.
        "JAX_COMPILATION_CACHE_DIR": "/cache/jax_cache",
        # Force the Vulkan loader to use ONLY the LLVMpipe (lavapipe) ICD.
        # With both ICDs available, Sapien picks NVIDIA first and fails at
        # vkCreateDevice — datacenter NVIDIA drivers gate graphics-Vulkan
        # features even though vkCreateInstance succeeds. Pinning to lavapipe
        # gives us a guaranteed-working CPU Vulkan path. Slow but functional;
        # we can revisit a real GPU path (multi-GPU container, H100 host with
        # full graphics drivers, or `nvidia/cudagl` base) once the algorithmic
        # loop is shaped.
        # mesa-vulkan-drivers on Ubuntu 24.04 installs the lavapipe ICD as
        # `lvp_icd.json` (no `x86_64` suffix — that's an older Mesa convention).
        # Confirmed by `ls /usr/share/vulkan/icd.d/` post-install via vk_sapien.
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json",
    })
    .add_local_dir(
        str(REPO_ROOT / "src"),
        remote_path="/workspace/src",
        copy=True,
    )
)
