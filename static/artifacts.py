import platform
import onnx
from onnxruntime.training import artifacts
import os

assert list(platform.python_version_tuple())[:-1] == ["3", "9"]

# Load the ONNX model
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

onnx_model = onnx.load(os.path.join(ARTIFACTS_DIR, "model.onnx"))

# Check if the model has been loaded successfully
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print("The model is invalid: {}".format(e))
else:
    print("The exported model is valid!")

requires_grad = ["classifier.weight"]
frozen_params = [
    param.name
    for param in onnx_model.graph.initializer
    if param.name not in requires_grad
]

artifacts.generate_artifacts(
    onnx_model,
    optimizer=artifacts.OptimType.AdamW,
    loss=artifacts.LossType.L1Loss,
    artifact_directory=ARTIFACTS_DIR,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
)
