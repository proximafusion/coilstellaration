# Local model checkpoints

This directory holds Flax NNX checkpoints copied out of the dapper-based
pipeline. Files here are bundled into
the wheel by hatchling's `include = ["src/coilstellaration/**", ...]` rule
(see `pyproject.toml`), so they ship with `pip install coilstellaration`
and are reachable at runtime via `importlib.resources` regardless of the
current working directory.

The on-disk format is JSON produced by pydantic's `model_dump_json()`. The
archive payload is `coilstellaration.types.Blob`; its JSON shape is a
superset of `dapper.Blob`'s, so a JSON dump of a dapper-era
`FlaxNnxCheckpoint` deserializes directly via `model_validate_json` (the
`dapper_is_blob` marker is silently dropped by pydantic's default
`extra="ignore"`).

## One-shot export

```python
from pathlib import Path

import dapper
from beta.coilstellaration.machine_learning import types as legacy_types

storage_id = "<paste_dapper_storage_id_here>"
out_path = Path(
    "/workspaces/constellaration_update/src/coilstellaration/data/models"
    "/<friendly_name>.json"
)

ckpt = dapper.read(legacy_types.CoilPredictorCheckpoint, storage_id)
out_path.write_text(ckpt.model_dump_json())
```

## Loading from `coilstellaration`

```python
from coilstellaration import flax_nnx_checkpoint_util, paths, types
from coilstellaration.machine_learning import model_definition

ckpt = types.CoilPredictorCheckpoint.model_validate_json(
    paths.model_path("<friendly_name>").read_text()
)
model = flax_nnx_checkpoint_util.from_checkpoint(
    ckpt, model_definition.CoilPredictor
)
```

## Sizing note

Hatchling will happily bundle anything you drop here, but checkpoints
above ~10 MB make the wheel awkward to publish and slow to install. For
larger artifacts, prefer hosting them out-of-band (Git LFS in the repo,
HuggingFace Hub release assets, or a download-on-first-use helper) and
keep this directory for small example/golden-file checkpoints.
