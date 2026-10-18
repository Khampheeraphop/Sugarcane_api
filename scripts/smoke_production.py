"""Exercise the production artifact and HTTP paths without importing training libraries."""

import io
import sys

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def main():
    pixels = np.random.default_rng(42).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    with TestClient(app) as client:
        assert client.get("/api/ready").status_code == 200
        catalog = client.get("/api/diseases")
        assert catalog.status_code == 200 and len(catalog.json()["classes"]) == 5
        scores = app.state.predictor.runtime.measure(image)
        assert len(scores.probabilities) == 5 and np.isfinite(scores.probabilities).all()
        buffer = io.BytesIO()
        Image.new("RGB", (224, 224), "green").save(buffer, format="PNG")
        response = client.post(
            "/api/analyze", files={"file": ("blank.png", buffer.getvalue(), "image/png")}
        )
        assert response.status_code == 200 and response.json()["status"] == "poor_quality"
        assert "torch" not in sys.modules, "Production unexpectedly imported PyTorch"
    print(
        "Production smoke passed: ONNX inference, ready, catalog and image validation; no PyTorch imported."
    )


if __name__ == "__main__":
    main()
