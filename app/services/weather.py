"""Actual weather forecast summaries only; no fabricated disease risk model."""

from datetime import datetime, timezone

import httpx


async def forecast(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
        "forecast_days": 3,
        "hourly": "relative_humidity_2m",
        "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        data = response.json()
    days = []
    for i, date in enumerate(data["daily"]["time"]):
        humidities = [
            v
            for t, v in zip(data["hourly"]["time"], data["hourly"]["relative_humidity_2m"])
            if t.startswith(date) and v is not None
        ]
        days.append(
            {
                "date": date,
                "temperature_min": data["daily"]["temperature_2m_min"][i],
                "temperature_max": data["daily"]["temperature_2m_max"][i],
                "precipitation_mm": data["daily"]["precipitation_sum"][i],
                "humidity_mean": round(sum(humidities) / len(humidities), 1)
                if humidities
                else None,
                "hours_humidity_ge_85": sum(v >= 85 for v in humidities),
                "humidity_hours_available": len(humidities),
            }
        )
    return {
        "source": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "timezone": data["timezone"],
        "requested_coordinates": {"latitude": latitude, "longitude": longitude},
        "forecast_grid_coordinates": {"latitude": data["latitude"], "longitude": data["longitude"]},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "disease_risk": None,
        "notice": "พยากรณ์อากาศจากแบบจำลอง ไม่ใช่ค่าจากเซ็นเซอร์ในแปลง และไม่ใช่การพยากรณ์โรค",
        "humidity_note": "จำนวนชั่วโมง RH ≥ 85% เป็นเพียงสรุปข้อมูลอากาศ เกณฑ์นี้ยังไม่ได้ตรวจสอบเพื่อทำนายโรคอ้อย",
    }
