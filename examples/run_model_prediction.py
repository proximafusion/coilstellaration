from coilstellaration import flax_nnx_checkpoint_util, paths, types
from coilstellaration.machine_learning import model_definition

model = paths.model_path("D2HbzeYjo57Aif48z5T6axt")
model_checkpoint = types.CoilPredictorCheckpoint.model_validate_json(model.read_text())
coil_predictor = flax_nnx_checkpoint_util.from_checkpoint(
    model_checkpoint,
    module_cls=model_definition.CoilPredictor,
)

print(coil_predictor)
