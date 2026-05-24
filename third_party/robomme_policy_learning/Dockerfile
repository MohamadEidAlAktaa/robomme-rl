FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    zip \
    unzip \
    tar \
    libegl1 \
    tmux \
    libgl1 \
    libglib2.0-0 \
    libvulkan1 \
    vulkan-tools \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=automatic \
    PYTHONPATH=/app/src \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    SAPIEN_RENDER_DEVICE=cuda \
    XDG_RUNTIME_DIR=/tmp/runtime-root

RUN mkdir -p "${XDG_RUNTIME_DIR}" /runs

# Copy project metadata and code.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY docs ./docs
COPY scripts ./scripts
COPY packages ./packages
COPY third_party ./third_party
COPY examples ./examples

# Install the project (includes workspace deps like openpi-client) from the lockfile.
RUN uv sync --frozen --no-dev --no-install-project --python 3.11

# Install micromamba and set up a dedicated robomme env for evaluation.
# This is only for evaluation, not for training.
# You can comment out this section if you don't need to evaluate the policies.
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba -O > /usr/local/bin/micromamba \
    && chmod +x /usr/local/bin/micromamba \
    && micromamba create -y -n robomme python=3.11 \
    && micromamba run -n robomme pip install -r examples/robomme/requirements.txt \
    && micromamba run -n robomme pip install -e third_party/robomme_benchmark \
    && micromamba run -n robomme pip install -e packages/openpi-client


CMD ["micromamba", "run", "-n", "robomme", "bash"]

