"""Set class-support cutoffs only from calibration data; never inspect feedback or test here."""

import hashlib
import json

import numpy as np
import torch

from ml.common import CLASSES, MODELS, ROOT, backbone, write_json
from ml.train import features


def main():
    torch.set_num_threads(4)
    bundle = torch.load(MODELS / "model.pt", map_location="cpu", weights_only=True)
    model = backbone()
    model.load_state_dict(bundle["backbone"])
    model.eval()
    rows = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    train = [row for row in rows if row["split"] == "train" and row["label"] in CLASSES]
    calibration = [row for row in rows if row["split"] == "calibration" and row["label"] in CLASSES]
    labels = torch.tensor([CLASSES.index(row["label"]) for row in train])
    embeddings = torch.nn.functional.normalize(features(model, calibration, "cpu"), dim=1)
    thresholds = {}
    counts = {}
    for index, name in enumerate(CLASSES):
        mask = torch.tensor([row["label"] == name for row in calibration])
        support = (embeddings[mask] @ bundle["reference"][labels == index].T).max(1).values.numpy()
        thresholds[name] = float(np.quantile(support, 0.05))
        counts[name] = int(mask.sum())
    with (MODELS / "model.pt").open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    policy = {
        "version": "class-support-v2",
        "model_sha256": checksum,
        "reference_labels": labels.tolist(),
        "class_support_thresholds": thresholds,
        "calibration_counts": counts,
        "quantile": 0.05,
        "requires_confirmation": ["RedRot"],
        "confidence_display_enabled": False,
        "limitations": [
            "Small calibration sets; support cutoffs cannot guarantee unseen-disease rejection.",
            "No independent Thai farm calibration; no diagnostic confidence percentage is shown.",
        ],
    }
    write_json(MODELS / "policy.json", policy)
    print(json.dumps({"thresholds": thresholds, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
