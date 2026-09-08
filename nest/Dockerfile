FROM python:3.14

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    parallel \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN curl -fsSL https://opencode.ai/install | bash

RUN curl -sS https://starship.rs/install.sh | sh -s -- --yes \
    && echo 'eval "$(starship init bash)"' >> /root/.bashrc \
    && echo '[[ -f ~/.bashrc ]] && source ~/.bashrc' >> /root/.bash_profile \
    && echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> /root/.bashrc

ENV NEST_INSTALL_DIR=/usr/local
ENV PYTHONPATH=${NEST_INSTALL_DIR}/lib/python3.14/site-packages

WORKDIR /workspace