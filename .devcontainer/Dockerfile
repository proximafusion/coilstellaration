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
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/fdfind /usr/local/bin/fd \
    && ln -s /usr/bin/batcat /usr/local/bin/bat

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV EDITOR=vim

# Install uv. Used to provide additional Python versions on demand:
# .pre-commit-config.yaml pins `language_version: python3.10` for black,
# but the base image only ships 3.12.
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh

# Install Python 3.10 system-wide and expose python3.10 in PATH.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv python install 3.10 \
    && ln -sf "$(uv python find 3.10)" /usr/local/bin/python3.10
