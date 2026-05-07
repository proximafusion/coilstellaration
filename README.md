# coilstellaration

Code for analyzing and evaluating stellarator boundaries and coilsets, built on top of [constellaration](https://github.com/proximafusion/constellaration).

## Develop

```bash
hatch env create test
hatch run test:pytest
hatch run lint:pre-commit run --all-files
```

## Run examples without the devcontainer

The `.devcontainer/Dockerfile` is self-contained — it installs the system build
dependencies for `constellaration` (NetCDF, BLAS/LAPACK, gfortran, CMake) and
builds the `regcoil` binary into `/usr/local/bin`. You can use it directly with
`docker` if you'd rather not open the devcontainer:

```bash
# Build the image
docker build --platform=linux/amd64 -f .devcontainer/Dockerfile -t coilstellaration .

# Run an example. PYTHONPATH=src is normally set by devcontainer.json, so we
# pass it explicitly here. Hatch envs aren't pre-built outside the
# devcontainer lifecycle, so create the default env on first run.
docker run --rm -it --platform=linux/amd64 --shm-size=16g \
    -v "$PWD":/workspaces/constellaration_update \
    -w /workspaces/constellaration_update \
    -e PYTHONPATH=/workspaces/constellaration_update/src \
    coilstellaration \
    bash -c "pipx install hatch && hatch env create && hatch run python examples/run_regcoil.py"
```

Caveats:

- The bind mounts declared in `devcontainer.json` (`~/tmp/outputs`,
  `~/repo`, `~/constellaration`) are devcontainer-only conveniences. If an
  example writes to `/home/vscode/tmp/outputs`, create it in the container or
  redirect the output path.
- For repeated runs, mount a persistent volume at `./venv` (or run an editable
  install with `pip install -e .`) so you don't recreate the hatch env each
  time.
