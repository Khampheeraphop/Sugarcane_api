import io

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.domain.knowledge import KNOWLEDGE, advice
from app.main import app
from app.services.images import assess_quality as quality
from app.services.images import decode_image


def image_bytes(size=(224, 224), color="green"):
    f = io.BytesIO()
    Image.new("RGB", size, color).save(f, format="PNG")
    return f.getvalue()


def test_decode_rejects_fake_image():
    with pytest.raises(HTTPException) as e:
        decode_image(b"not an image")
    assert e.value.status_code == 415


def test_uniform_and_small_images_need_retake():
    assert not quality(Image.new("RGB", (224, 224), "green"))["passed"]
    assert not quality(Image.new("RGB", (80, 80), "green"))["passed"]


def test_actual_image_format_is_checked():
    assert decode_image(image_bytes()).size == (224, 224)
    f = io.BytesIO()
    Image.new("RGB", (224, 224)).save(f, format="GIF")
    with pytest.raises(HTTPException) as e:
        decode_image(f.getvalue())
    assert e.value.status_code == 415


def test_general_advice_has_provenance():
    for name in KNOWLEDGE:
        a = advice(name)
        assert a["scope"] and a["care"]
        if name != "Healthy":
            assert a["sources"]


def test_api_input_errors_and_auth(monkeypatch):
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert (
            client.post("/api/analyze", files={"file": ("x.jpg", b"", "image/jpeg")}).status_code
            == 400
        )
        assert (
            client.post(
                "/api/analyze", files={"file": ("x.jpg", b"fake", "image/jpeg")}
            ).status_code
            == 415
        )
        assert (
            client.post(
                "/api/analyze",
                files={"file": ("x.jpg", b"x" * (8 * 1024 * 1024 + 1), "image/jpeg")},
            ).status_code
            == 413
        )
        monkeypatch.setenv("INFERENCE_API_SECRET", "test-secret")
        assert (
            client.post(
                "/api/analyze", files={"file": ("x.png", image_bytes(), "image/png")}
            ).status_code
            == 401
        )


def test_unknown_and_blank_do_not_return_disease():
    with TestClient(app) as client:
        if not client.get("/api/health").json()["ready"]:
            pytest.skip("No trained model yet")
        response = client.post(
            "/api/analyze", files={"file": ("blank.png", image_bytes(), "image/png")}
        )
        assert response.json()["status"] == "poor_quality"
        assert response.json()["confidence"] is None
        assert "advice" not in response.json()


def test_near_duplicate_groups_do_not_cross_splits():
    import json

    from ml.common import ROOT

    path = ROOT / "data/manifest.json"
    if not path.exists():
        pytest.skip("No data yet")
    rows = json.loads(path.read_text())
    splits = {}
    for row in rows:
        assert row["group"] not in splits or splits[row["group"]] == row["split"]
        splits[row["group"]] = row["split"]


def test_trained_leaf_prediction_includes_advice_only_when_accepted():
    import json

    from ml.common import MODELS, ROOT

    if not (MODELS / "test_predictions.json").exists():
        pytest.skip("No trained model yet")
    rows = json.loads((MODELS / "test_predictions.json").read_text(encoding="utf-8"))
    accepted = next(
        r
        for r in rows
        if r["accepted"]
        and r["label"] not in ("Healthy", "not_sugarcane")
        and r["prediction"] == r["label"]
    )

    def usable_negative(row):
        if (
            row["accepted"]
            or row["label"] != "not_sugarcane"
            or not row["source"].startswith("flowers")
        ):
            return False
        with Image.open(ROOT / row["path"]) as image:
            return quality(image.convert("RGB"))["passed"]

    negative = next(r for r in rows if usable_negative(r))
    with TestClient(app) as client:
        for row, expected in [(accepted, "suspected"), (negative, "unrecognized")]:
            path = ROOT / row["path"]
            response = client.post(
                "/api/analyze", files={"file": (path.name, path.read_bytes(), "image/jpeg")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == expected
            if expected == "suspected":
                assert data["advice"]["sources"] and data["advice"]["care"]
            else:
                assert "advice" not in data and data["confidence"] is None


def test_disease_catalog_explains_supported_and_withheld_classes():
    with TestClient(app) as client:
        response = client.get("/api/diseases")
        assert response.status_code == 200
        catalog = response.json()
        assert len(catalog["classes"]) == 5
        red_rot = next(item for item in catalog["classes"] if item["id"] == "RedRot")
        assert red_rot["requires_confirmation"] and not red_rot["leaf_output_enabled"]
        assert any("Rust" in item for item in catalog["unsupported"])
        for item in catalog["classes"]:
            assert item["advice"]["cause"] and item["advice"]["scope"]
