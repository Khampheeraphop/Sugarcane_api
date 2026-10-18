import pytest
from PIL import Image, ImageDraw

from app.services.field_analysis import compare_field


def test_inconsistent_reference_is_rejected_instead_of_ranking_the_whole_field():
    image = Image.new("RGB", (768, 768), "#367232")
    ImageDraw.Draw(image).rectangle((512, 512, 639, 639), fill="#aa6834")
    with pytest.raises(ValueError, match="ช่องอ้างอิง"):
        compare_field(image, [0, 1, 104])
