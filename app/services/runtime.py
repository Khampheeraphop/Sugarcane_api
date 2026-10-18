"""Select an inference implementation without importing training dependencies in production."""

from pathlib import Path


def create_runtime(model_path: Path, policy_path: Path | None):
    if model_path.suffix == ".onnx":
        from ml.onnx_inference import OnnxModelRuntime

        return OnnxModelRuntime(model_path, policy_path)
    from ml.inference import ModelRuntime

    return ModelRuntime(model_path, policy_path)
