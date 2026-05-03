# constellaration_update

Code for analyzing and evaluating stellarator boundaries and coilsets, built on top of [constellaration](https://github.com/proximafusion/constellaration).

## Install

```bash
pip install -e .
```

The build pulls in `constellaration==0.2.6` and its scientific-Python dependencies.

## Layout

- `coilset/` — DESC coilset utilities and optimization
- `metrics/` — evaluation metrics (`metrics`, `metrics_v2`)
- `checkpoint/` — Flax NNX model checkpoint helpers
- `data_generation/` — data-generation entry points
- `machine_learning/` — coil-predictor model, training, and example driver
- `held_out_set.py`, `sampling.py`, `source_configurations.py`, `types.py`, `visualization.py` — top-level helpers

## Develop

```bash
hatch env create test
hatch run test:pytest
hatch run lint:pre-commit run --all-files
```
