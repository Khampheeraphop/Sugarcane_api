"""Group near duplicates before splitting; no farm/plant IDs exist in source."""

import random
from collections import Counter, defaultdict

import imagehash
import numpy as np
from PIL import Image, ImageDraw

from ml.common import CLASSES, ROOT, write_json


def main():
    rng = random.Random(42)
    rows = []
    for label in CLASSES:
        paths = sorted((ROOT / "data/raw/sugarcane").rglob(f"{label}/*.jpg"))
        if not paths:
            raise RuntimeError(f"Missing data for {label}; run scripts/download_data.py")
        rows += [
            dict(path=p.relative_to(ROOT).as_posix(), label=label, source="sugarcaneld-bd")
            for p in paths
        ]
    for folder in ["daisy", "dandelion", "roses", "sunflowers", "tulips"]:
        paths = sorted((ROOT / "data/raw/flowers").rglob(f"{folder}/*.jpg"))
        rng.shuffle(paths)
        rows += [
            dict(
                path=p.relative_to(ROOT).as_posix(),
                label="not_sugarcane",
                source=f"flowers/{folder}",
            )
            for p in paths[:200]
        ]
    if not any(r["label"] == "not_sugarcane" for r in rows):
        raise RuntimeError("Missing negative images")
    # Generated controls help reject blank, noise and diagram inputs; not a substitute for real OOD photos.
    controls = ROOT / "data/controls"
    controls.mkdir(exist_ok=True)
    nrng = np.random.default_rng(42)
    for i in range(120):
        path = controls / f"{i:03}.png"
        if i % 3 == 0:
            image = Image.fromarray(nrng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
        else:
            image = Image.new("RGB", (224, 224), tuple(rng.randrange(256) for _ in range(3)))
            draw = ImageDraw.Draw(image)
            if i % 3 == 1:
                for j in range(6):
                    x, y = rng.randrange(170), rng.randrange(170)
                    draw.rectangle(
                        (x, y, x + 35, y + 35), fill=tuple(rng.randrange(256) for _ in range(3))
                    )
        image.save(path)
        rows.append(
            dict(
                path=path.relative_to(ROOT).as_posix(),
                label="not_sugarcane",
                source="synthetic-controls",
            )
        )
    hashes = []
    for row in rows:
        with Image.open(ROOT / row["path"]) as im:
            row["phash"] = str(imagehash.phash(im))
            hashes.append(int(row["phash"], 16))
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i in range(len(rows)):
        for j in range(i):
            if (hashes[i] ^ hashes[j]).bit_count() <= 4:
                parents[find(i)] = find(j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    buckets = defaultdict(list)
    conflicts = []
    for group in groups.values():
        if len({r["label"] for r in group}) > 1:
            conflicts += group
        else:
            buckets[group[0]["label"]].append(group)
    output = []
    for label, parts in buckets.items():
        rng.shuffle(parts)
        n = len(parts)
        if n < 15:
            raise RuntimeError(f"Insufficient independent image groups for {label}: {n}")
        for i, group in enumerate(parts):
            split = (
                "train"
                if i < int(n * 0.60)
                else "val"
                if i < int(n * 0.75)
                else "calibration"
                if i < int(n * 0.85)
                else "test"
            )
            for row in group:
                row.update(split=split, group=f"g{find(rows.index(row))}")
                output.append(row)
    (ROOT / "models").mkdir(exist_ok=True)
    write_json(ROOT / "data/manifest.json", output)
    report = {
        "seed": 42,
        "counts": dict(Counter(f"{r['split']}/{r['label']}" for r in output)),
        "near_duplicate_groups": len(groups),
        "excluded_conflicting_images": len(conflicts),
        "split_limitation": "pHash groups (distance <=4), not independent farms/plants. Metadata unavailable; field performance unknown.",
    }
    write_json(ROOT / "models/data_report.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
