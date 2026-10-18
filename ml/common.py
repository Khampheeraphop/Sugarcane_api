from pathlib import Path

import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CLASSES = ["EyeSpot", "Healthy", "RedLeafSpot", "RedRot", "RingSpot"]
LABELS_TH = [
    "อาการคล้ายโรคใบจุดตา",
    "ไม่พบอาการผิดปกติที่โมเดลรู้จัก",
    "อาการคล้ายโรคใบจุดแดง",
    "อาการคล้ายโรคเน่าแดง",
    "อาการคล้ายโรคใบจุดวงแหวน",
]


def backbone(pretrained=False):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    model.classifier = torch.nn.Identity()
    return model.eval()


def transform():
    return MobileNet_V3_Small_Weights.DEFAULT.transforms()


def write_json(path, value):
    import json

    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
