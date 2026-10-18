"""Low-memory CPU inference with the same frozen features, heads and decision evidence."""

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.domain.evidence import ModelScores


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class OnnxModelRuntime:
    def __init__(self, model_path: Path, policy_path: Path | None):
        directory = model_path.parent
        manifest = json.loads((directory / "runtime.json").read_text(encoding="utf-8"))
        for filename, expected in manifest["files"].items():
            if checksum(directory / filename) != expected:
                raise ValueError(f"Inference artifact checksum mismatch: {filename}")
        self.metadata = manifest["metadata"]
        self.preprocessing = manifest["preprocessing"]
        self.policy = None
        if policy_path and policy_path.exists():
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            if policy["model_sha256"] == manifest["source_model_sha256"]:
                self.policy = policy
        self.labels = np.asarray(self.policy["reference_labels"]) if self.policy else None
        with np.load(directory / "heads.npz", allow_pickle=False) as arrays:
            self.arrays = {name: arrays[name] for name in arrays.files}
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.lock = threading.Lock()

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        width, height = image.size
        short = self.preprocessing["resize_size"]
        size = (
            (short, int(short * height / width))
            if width <= height
            else (int(short * width / height), short)
        )
        image = image.resize(size, Image.Resampling.BILINEAR)
        crop = self.preprocessing["crop_size"]
        left, top = int(round((image.width - crop) / 2)), int(round((image.height - crop) / 2))
        image = image.crop((left, top, left + crop, top + crop))
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
        mean = np.asarray(self.preprocessing["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(self.preprocessing["std"], dtype=np.float32)[:, None, None]
        return ((pixels - mean) / std)[None]

    @staticmethod
    def softmax(logits):
        values = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return values / values.sum(axis=1, keepdims=True)

    def measure(self, image: Image.Image) -> ModelScores:
        with self.lock:
            features = self.session.run(["features"], {"image": self.preprocess(image)})[0]
            a = self.arrays
            normalized = (features - a["mean"]) / a["std"]
            probabilities = self.softmax(
                (normalized @ a["disease_weight"].T + a["disease_bias"])
                / self.metadata["disease_temperature"]
            )[0]
            gate = self.softmax(
                (normalized @ a["gate_weight"].T + a["gate_bias"])
                / self.metadata["gate_temperature"]
            )[0, 1]
            unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            similarities = unit @ a["reference"].T
            support = (
                [
                    float(similarities[:, self.labels == index].max())
                    for index in range(len(probabilities))
                ]
                if self.labels is not None
                else None
            )
            return ModelScores(
                probabilities.tolist(), float(gate), float(similarities.max()), support
            )
