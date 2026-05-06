FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Drop the yarn apt source shipped by the base image — its GPG key has rotated
# and breaks `apt-get update`. This repo isn't needed by anything we install.
RUN rm -f /etc/apt/sources.list.d/yarn.list

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        # Shell QoL
        bat \
        fd-find \
        locales \
        ripgrep \
        tmux \
        vim \
        zsh \
        # Build deps for constellaration's scientific extensions (booz-xform
        # needs NetCDF headers; cmake-built extensions use pkg-config; some
        # Fortran-extension wheels fall back to source builds).
        cmake \
        gfortran \
        libnetcdf-dev \
        ninja-build \
        pkg-config \
        # Build deps for REGCOIL (Fortran NetCDF bindings + BLAS/LAPACK).
        libnetcdff-dev \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/fdfind /usr/local/bin/fd \
    && ln -s /usr/bin/batcat /usr/local/bin/bat

# Build REGCOIL from source (https://github.com/landreman/regcoil).
# Upstream's makefile has no Debian/Ubuntu host config, so we override
# FC/EXTRA_*_FLAGS on the make command line. -fallow-argument-mismatch is
# required for gfortran >= 10 to compile the legacy mini_libstell sources.
ARG REGCOIL_REF=6f5e3f73027954e6be7bb7de8c63bb44ea2741f1
RUN set -eux; \
    git clone https://github.com/landreman/regcoil.git /tmp/regcoil; \
    cd /tmp/regcoil && git checkout "${REGCOIL_REF}"; \
    NF_CFLAGS="$(nf-config --fflags)"; \
    NF_LIBS="$(nf-config --flibs)"; \
    make -j"$(nproc)" \
        FC=gfortran \
        EXTRA_COMPILE_FLAGS="-O2 -ffree-line-length-none -fexternal-blas -fopenmp -fallow-argument-mismatch ${NF_CFLAGS}" \
        EXTRA_LINK_FLAGS="-fopenmp -lopenblas ${NF_LIBS}"; \
    install -m 0755 regcoil /usr/local/bin/regcoil; \
    rm -rf /tmp/regcoil

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV EDITOR=vim

# Install uv. Used to provide additional Python versions on demand:
# .pre-commit-config.yaml pins `language_version: python3.10` for black,
# but the base image only ships 3.12.
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh

# Install Claude
RUN curl -fsSL https://claude.ai/install.sh | bash

# Install bun (for Claude)
RUN curl -fsSL https://bun.sh/install | bash

# Install Python 3.12 system-wide and expose python3.12 in PATH.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv python install 3.12 \
    && ln -sf "$(uv python find 3.12)" /usr/local/bin/python3.12
