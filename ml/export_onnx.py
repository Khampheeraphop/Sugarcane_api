"""Export the existing weights without retraining; verify checksums before publishing."""

import hashlib
import json
import sys

import numpy as np
import torch

from ml.common import MODELS, backbone, transform, write_json


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    source = MODELS / "model.pt"
    bundle = torch.load(source, map_location="cpu", weights_only=True)
    policy = json.loads((MODELS / "policy.json").read_text(encoding="utf-8"))
    if policy["model_sha256"] != digest(source):
        raise ValueError("Recalibrate the support policy for these weights before exporting")
    model = backbone()
    model.load_state_dict(bundle["backbone"])
    torch.onnx.export(
        model.eval(),
        (torch.zeros(1, 3, 224, 224),),
        str(MODELS / "backbone.onnx"),
        input_names=["image"],
        output_names=["features"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    np.savez_compressed(
        MODELS / "heads.npz",
        **{
            **{key: bundle[key].numpy() for key in ["mean", "std", "reference"]},
            **{
                f"{head}_{key}": bundle[head][key].numpy()
                for head in ["disease", "gate"]
                for key in ["weight", "bias"]
            },
        },
    )
    tf = transform()
    write_json(
        MODELS / "runtime.json",
        {
            "source_model_sha256": digest(source),
            "metadata": bundle["metadata"],
            "preprocessing": {
                "resize_size": tf.resize_size[0],
                "crop_size": tf.crop_size[0],
                "mean": tf.mean,
                "std": tf.std,
                "interpolation": "bilinear",
            },
            "files": {name: digest(MODELS / name) for name in ["backbone.onnx", "heads.npz"]},
        },
    )
    print("ONNX inference artifacts exported; original weights and policy unchanged.")


if __name__ == "__main__":
    main()
