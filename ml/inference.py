"""Torch-only runtime: returns measurements, never agricultural advice or HTTP errors."""

import hashlib
import json
import threading
from pathlib import Path

import torch
from PIL import Image

from app.domain.evidence import ModelScores as ModelScores
from ml.common import CLASSES, backbone, transform


class ModelRuntime:
    def __init__(self, model_path: Path, policy_path: Path | None = None):
        torch.set_num_threads(4)
        bundle = torch.load(model_path, map_location="cpu", weights_only=True)
        self.metadata = bundle["metadata"]
        self.net = backbone()
        self.net.load_state_dict(bundle["backbone"])
        self.net.eval()
        self.disease = torch.nn.Linear(576, len(CLASSES))
        self.disease.load_state_dict(bundle["disease"])
        self.disease.eval()
        self.gate = torch.nn.Linear(576, 2)
        self.gate.load_state_dict(bundle["gate"])
        self.gate.eval()
        self.mean, self.std, self.reference = bundle["mean"], bundle["std"], bundle["reference"]
        self.transform = transform()
        self.lock = threading.Lock()
        self.policy = self._load_policy(model_path, policy_path)
        self.reference_labels = (
            torch.tensor(self.policy["reference_labels"]) if self.policy else None
        )

    @staticmethod
    def _load_policy(model_path: Path, policy_path: Path | None) -> dict | None:
        if policy_path is None or not policy_path.exists():
            return None
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        with model_path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        return policy if policy.get("model_sha256") == checksum else None

    def measure(self, image: Image.Image) -> ModelScores:
        with self.lock, torch.inference_mode():
            features = self.net(self.transform(image).unsqueeze(0))
            normalized = (features - self.mean) / self.std
            gate = float(
                (self.gate(normalized) / self.metadata["gate_temperature"]).softmax(1)[0, 1]
            )
            similarities = torch.nn.functional.normalize(features, dim=1) @ self.reference.T
            probabilities = (
                self.disease(normalized) / self.metadata["disease_temperature"]
            ).softmax(1)[0]
            support = None
            if self.reference_labels is not None:
                support = [
                    float(similarities[:, self.reference_labels == index].max())
                    for index in range(len(CLASSES))
                ]
            return ModelScores(probabilities.tolist(), gate, float(similarities.max()), support)
