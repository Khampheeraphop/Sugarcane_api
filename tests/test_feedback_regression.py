import pytest
from PIL import Image

from app.services.screening import ScreeningService
from ml.common import MODELS, ROOT


@pytest.mark.integration
def test_reported_out_of_scope_leaf_is_not_named_red_rot():
    feedback = ROOT / "data/feedback/user_reported_leaf_pale.png"
    if not feedback.exists() or not (MODELS / "model.pt").exists():
        pytest.skip("Private feedback and local model are intentionally not published")
    service = ScreeningService(MODELS / "model.pt", MODELS / "policy.json")
    with Image.open(feedback) as image:
        result = service.predict(image.convert("RGB"))
    assert result["status"] == "uncertain"
    assert result["confidence"] is None
    assert "advice" not in result
    assert "class_id" not in result
