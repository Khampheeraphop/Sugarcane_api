"""Diagnostic only: inspect class-specific embedding support; never train on feedback."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml.common import CLASSES, MODELS, ROOT, backbone, transform
from ml.train import features

parser = argparse.ArgumentParser()
parser.add_argument("image", type=Path, help="Local diagnostic image; never used for training")
image_path = parser.parse_args().image
torch.set_num_threads(4)
bundle = torch.load(MODELS / "model.pt", map_location="cpu", weights_only=True)
net = backbone()
net.load_state_dict(bundle["backbone"])
net.eval()
rows = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
train = [r for r in rows if r["split"] == "train" and r["label"] in CLASSES]
labels = torch.tensor([CLASSES.index(r["label"]) for r in train])
cal = [r for r in rows if r["split"] == "calibration" and r["label"] in CLASSES]
x = torch.nn.functional.normalize(features(net, cal, "cpu"), dim=1)
reference = bundle["reference"]
thresholds = {}
for i, name in enumerate(CLASSES):
    mask = torch.tensor([r["label"] == name for r in cal])
    support = (x[mask] @ reference[labels == i].T).max(1).values.numpy()
    thresholds[name] = float(np.quantile(support, 0.05))
    print(
        name,
        "n",
        len(support),
        "min",
        round(float(support.min()), 3),
        "q05",
        round(thresholds[name], 3),
        "median",
        round(float(np.median(support)), 3),
        flush=True,
    )
path = image_path
with torch.inference_mode():
    with Image.open(path) as im:
        f = net(transform()(im.convert("RGB")).unsqueeze(0))
    x = (f - bundle["mean"]) / bundle["std"]
    probabilities = torch.nn.functional.linear(
        x, bundle["disease"]["weight"], bundle["disease"]["bias"]
    )
    probabilities = (probabilities / bundle["metadata"]["disease_temperature"]).softmax(1)[0]
    similarities = torch.nn.functional.normalize(f, dim=1) @ reference.T
    print(
        "User-reported feedback (not training data):",
        {
            name: {
                "classifier": round(float(probabilities[i]), 4),
                "support": round(float(similarities[:, labels == i].max()), 4),
                "calibration_q05": round(thresholds[name], 4),
            }
            for i, name in enumerate(CLASSES)
        },
        flush=True,
    )
