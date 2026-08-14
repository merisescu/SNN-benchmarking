FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN curl -fsSL https://opencode.ai/install | bash

RUN curl -sS https://starship.rs/install.sh | sh -s -- --yes \
    && echo 'eval "$(starship init bash)"' >> /root/.bashrc \
    && echo '[[ -f ~/.bashrc ]] && source ~/.bashrc' >> /root/.bash_profile \
    && echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> /root/.bashrc

RUN uv sync

WORKDIR /Workspace
