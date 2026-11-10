# Local model checkpoints

This directory holds one trained Flax NNX checkpoint,
`mlp_DHmCARHZ4ay9rsMDPdjVCu3.json`: an `mlp` model for the `fixed_shape` track
(NFP 3, 5 coils per half period). `examples/run_model_prediction.py`, the unit
tests and the fixed-shape golden file use it. Files here are bundled into
the wheel by hatchling's `include = ["src/coilstellaration/**", ...]` rule
(see `pyproject.toml`), so they ship with `pip install coilstellaration`
and are reachable at runtime via `importlib.resources` regardless of the
current working directory.

Files are named `<model_kind>_<id>.json`, where `<model_kind>` is the
checkpoint's `config.model_kind` and `<id>` identifies the training run it came
from. `paths.model_path("<id>")` resolves a file from its ID alone, and also
accepts a plain `<id>.json`, the name `examples/run_model_training.py` writes.

The on-disk format is JSON produced by pydantic's `model_dump_json()`. The
archive payload is `coilstellaration.types.Blob`; its JSON shape is a
superset of `dapper.Blob`'s, so a JSON dump of a dapper-era
`FlaxNnxCheckpoint` deserializes directly via `model_validate_json` (the
`dapper_is_blob` marker is silently dropped by pydantic's default
`extra="ignore"`).

## Loading from `coilstellaration`

```python
from coilstellaration.machine_learning import model_definition
from coilstellaration.machine_learning import utils as ml_utils

ckpt = ml_utils.read_checkpoint_from_id("<model_kind>", "<id>")
model = model_definition.read_model_from_checkpoint(ckpt)
```

`read_checkpoint_from_id` validates the JSON against the checkpoint type of
`<model_kind>`, and `read_model_from_checkpoint` rebuilds the matching module
in eval mode. To load a checkpoint you trained yourself, pass its path as
`read_checkpoint_from_id(..., model_data_path=...)`.

## Sizing note

Hatchling will happily bundle anything you drop here, but checkpoints
above ~10 MB make the wheel awkward to publish and slow to install. For
larger artifacts, prefer hosting them out-of-band (Git LFS in the repo,
HuggingFace Hub release assets, or a download-on-first-use helper) and
keep this directory for small example/golden-file checkpoints.
