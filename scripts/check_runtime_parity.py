"""Compare both runtimes on held-out images, without changing weights or thresholds."""

import json

import numpy as np
from PIL import Image

from app.domain.diagnosis_policy import decide_screening
from ml.common import MODELS, ROOT, write_json
from ml.inference import ModelRuntime
from ml.onnx_inference import OnnxModelRuntime


def main():
    original = ModelRuntime(MODELS / "model.pt", MODELS / "policy.json")
    exported = OnnxModelRuntime(MODELS / "backbone.onnx", MODELS / "policy.json")
    rows = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in rows if row["split"] == "test"]
    maximum_error = 0.0
    for row in rows:
        with Image.open(ROOT / row["path"]) as source:
            image = source.convert("RGB")
        a, b = original.measure(image), exported.measure(image)
        for left, right in [
            (a.probabilities, b.probabilities),
            (a.class_support, b.class_support),
            ([a.gate_score, a.similarity], [b.gate_score, b.similarity]),
        ]:
            maximum_error = max(maximum_error, float(np.max(np.abs(np.asarray(left) - right))))
            np.testing.assert_allclose(left, right, rtol=2e-4, atol=2e-5)
        assert decide_screening(a, original.metadata, original.policy) == decide_screening(
            b, exported.metadata, exported.policy
        ), row["path"]
    report = {
        "test_images": len(rows),
        "decision_matches": len(rows),
        "maximum_absolute_error": maximum_error,
        "note": "Runtime conversion parity only; does not add disease classes or establish field accuracy.",
    }
    write_json(MODELS / "runtime_parity.json", report)
    print(report)


if __name__ == "__main__":
    main()
