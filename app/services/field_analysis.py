"""Independent field-image comparison, NOT a trained disease detector.
Compares tile HSV histograms to user-selected reference tiles. No leaf-model imports.
"""

import math

import numpy as np

GRID = 12


def tile_features(image):
    image = image.copy()
    image.thumbnail((1536, 1536))
    arr = np.asarray(image.convert("HSV"), dtype=np.float32) / 255
    height, width = arr.shape[:2]
    features = []
    for row in range(GRID):
        for col in range(GRID):
            tile = arr[
                round(row * height / GRID) : round((row + 1) * height / GRID),
                round(col * width / GRID) : round((col + 1) * width / GRID),
            ]
            # H/S distribution with saturation as weight: reduce unstable hue of gray pixels.
            h = np.histogram(tile[:, :, 0], bins=16, range=(0, 1), weights=tile[:, :, 1])[0] + 1e-6
            s = np.histogram(tile[:, :, 1], bins=8, range=(0, 1))[0] + 1e-6
            v = np.histogram(tile[:, :, 2], bins=8, range=(0, 1))[0] + 1e-6
            # Value contributes less to avoid ranking only exposure, but lighting still matters.
            features.append(
                np.concatenate([0.55 * h / h.sum(), 0.3 * s / s.sum(), 0.15 * v / v.sum()])
            )
    return np.asarray(features)


def compare_field(image, reference_tiles):
    if min(image.size) < 384:
        raise ValueError("ภาพแปลงต้องมีด้านสั้นอย่างน้อย 384 พิกเซล")
    if (
        len(reference_tiles) < 3
        or len(reference_tiles) > 12
        or len(set(reference_tiles)) != len(reference_tiles)
    ):
        raise ValueError("เลือกช่องอ้างอิงที่ไม่ซ้ำกัน 3–12 ช่อง")
    if any(type(i) is not int or i < 0 or i >= GRID * GRID for i in reference_tiles):
        raise ValueError("ช่องอ้างอิงไม่ถูกต้อง")
    features = tile_features(image)
    reference = features[reference_tiles].mean(axis=0)
    reference_spread = np.sqrt(
        ((np.sqrt(features[reference_tiles]) - np.sqrt(reference)) ** 2).sum(axis=1)
    ) / math.sqrt(2)
    if reference_spread.max() > 0.25:
        raise ValueError("ช่องอ้างอิงมีสีต่างกันมาก กรุณาเลือกบริเวณที่สีใกล้เคียงกัน ไม่รวมดิน เงา หรือรอยผิดปกติ")
    # Hellinger distance: descriptive image distance, not a calibrated anomaly probability.
    distances = np.sqrt(((np.sqrt(features) - np.sqrt(reference)) ** 2).sum(axis=1)) / math.sqrt(2)
    points = []
    for i in np.argsort(-distances).tolist():
        if i in reference_tiles or distances[i] < 0.03:
            continue
        row, col = divmod(i, GRID)
        if any(abs(row - p["row"]) <= 1 and abs(col - p["col"]) <= 1 for p in points):
            continue
        points.append(
            {
                "tile": i,
                "row": row,
                "col": col,
                "rank": len(points) + 1,
                "difference": round(float(distances[i]), 4),
                "x": (col + 0.5) / GRID,
                "y": (row + 0.5) / GRID,
            }
        )
        if len(points) == 5:
            break
    return {
        "method": "reference-hsv-comparison-v1",
        "grid": GRID,
        "reference_tiles": reference_tiles,
        "points": points,
        "disease_prediction": None,
        "georeferenced": False,
        "message": "จัดอันดับจุดที่สีต่างจากช่องอ้างอิง ไม่ใช่ตำแหน่งยืนยันโรคหรือสัดส่วนพื้นที่เสียหาย",
        "limitations": [
            "ดิน เงา น้ำ วัชพืช และแสงอาจถูกจัดอันดับด้วย",
            "แม้แปลงปกติก็มีจุดที่ต่างจากอ้างอิงได้ ต้องเดินตรวจจริง",
            "พิกัดเป็นตำแหน่งในภาพ ไม่ใช่ GPS และยังไม่รองรับ GeoTIFF/ภาพหลายช่วงคลื่น",
            "ไม่ได้ใช้หรือปรับแก้โมเดลตรวจใบอ้อยระยะใกล้",
        ],
    }
