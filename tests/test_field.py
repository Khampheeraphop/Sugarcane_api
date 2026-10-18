import io

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from app.services.field_analysis import compare_field


def test_reference_comparison_finds_color_difference_not_a_disease():
    image = Image.new("RGB", (768, 768), "#367232")
    ImageDraw.Draw(image).rectangle((512, 512, 639, 639), fill="#aa6834")
    result = compare_field(image, [0, 1, 12])
    assert result["disease_prediction"] is None
    assert result["georeferenced"] is False
    assert any(p["row"] in (8, 9) and p["col"] in (8, 9) for p in result["points"])


def test_uniform_field_is_not_claimed_healthy():
    result = compare_field(Image.new("RGB", (768, 768), "green"), [0, 1, 12])
    assert result["points"] == []
    assert result["disease_prediction"] is None


def test_field_validation_and_does_not_replace_leaf_model():
    with TestClient(app) as client:
        predictor = app.state.predictor
        f = io.BytesIO()
        Image.new("RGB", (768, 768), "green").save(f, format="PNG")
        response = client.post(
            "/api/field/analyze",
            files={"file": ("field.png", f.getvalue(), "image/png")},
            data={"reference_tiles": "[0,1,12]"},
        )
        assert response.status_code == 200
        assert app.state.predictor is predictor
        response = client.post(
            "/api/field/analyze",
            files={"file": ("field.png", f.getvalue(), "image/png")},
            data={"reference_tiles": "[0,1]"},
        )
        assert response.status_code == 422
        assert client.get("/api/field/weather?latitude=100&longitude=0").status_code == 422


def test_weather_summary_uses_real_response_values(monkeypatch):
    import asyncio

    import app.services.weather as weather

    data = {
        "latitude": 16.4,
        "longitude": 102.8,
        "timezone": "Asia/Bangkok",
        "daily": {
            "time": ["2026-10-18"],
            "temperature_2m_min": [24.2],
            "temperature_2m_max": [32.1],
            "precipitation_sum": [5.4],
        },
        "hourly": {
            "time": ["2026-10-18T00:00", "2026-10-18T01:00", "2026-10-18T02:00"],
            "relative_humidity_2m": [90, 80, None],
        },
    }

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, params):
            assert url == "https://api.open-meteo.com/v1/forecast"
            assert params["latitude"] == 16.44
            return Response()

    monkeypatch.setattr(weather.httpx, "AsyncClient", Client)
    result = asyncio.run(weather.forecast(16.44, 102.83))
    assert result["days"][0]["humidity_mean"] == 85
    assert result["days"][0]["hours_humidity_ge_85"] == 1
    assert result["days"][0]["precipitation_mm"] == 5.4
    assert result["disease_risk"] is None
