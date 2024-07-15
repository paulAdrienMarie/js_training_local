import platform
import onnx
from onnxruntime.training import artifacts

assert platform.python_version_tuple() == ("3", "9", "6")

# Load the ONNX model
onnx_model = onnx.load("./public/inference_artifacts/initial_model.onnx")

# Check if the model has been loaded successfully
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print("The model is invalid: {}".format(e))
else:
    print("The exported model is valid!")

requires_grad = ["classifier.weight", "classifier.bias"]
frozen_params = [
    param.name
    for param in onnx_model.graph.initializer
    if param.name not in requires_grad
]

path_to_output_artifacts = "./public/training_artifacts"

artifacts.generate_artifacts(
    onnx_model,
    optimizer=artifacts.OptimType.AdamW,
    loss=artifacts.LossType.L1Loss,
    artifact_directory=path_to_output_artifacts,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
)
