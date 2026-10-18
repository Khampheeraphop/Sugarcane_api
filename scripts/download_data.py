"""Download public research data, retain provenance, extract without path traversal."""

import hashlib
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCES = [
    dict(
        name="sugarcane",
        file="sugarcane.zip",
        url="https://www.kaggle.com/api/v1/datasets/download/shifatearman/sugarcaneld-bd-dataset?datasetVersionNumber=6",
        citation="Arman et al. SugarcaneLD-BD, DOI:10.17632/n8mpzb7p4k.1",
        license="CC BY-SA 4.0 (Kaggle mirror); Mendeley record states CC BY 4.0. Follow stricter mirror terms.",
        source="https://www.kaggle.com/datasets/shifatearman/sugarcaneld-bd-dataset",
    ),
    dict(
        name="flowers",
        file="flower_photos.tgz",
        url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        citation="TensorFlow flower_photos; individual image authors in LICENSE.txt in archive",
        license="CC BY 2.0, retain per-image attribution in flower_photos/LICENSE.txt",
        source="https://www.tensorflow.org/datasets/catalog/tf_flowers",
    ),
]


def download(url, dest):
    if dest.exists():
        return
    temp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "SugarcaneCare research prototype"})
    with urllib.request.urlopen(req, timeout=120) as response, temp.open("wb") as f:
        total = 0
        while chunk := response.read(1024 * 1024):
            f.write(chunk)
            total += len(chunk)
            if total % (25 * 1024 * 1024) < 1024 * 1024:
                print(f"{dest.name}: {total // (1024 * 1024)} MB", flush=True)
    temp.replace(dest)


def main():
    DATA.mkdir(exist_ok=True)
    records = []
    for source in SOURCES:
        dest = DATA / source["file"]
        print("Downloading", source["name"], flush=True)
        download(source["url"], dest)
        target = DATA / "raw" / source["name"]
        if not (target / ".complete").exists():
            target.mkdir(parents=True, exist_ok=True)
            if dest.suffix == ".zip":
                with zipfile.ZipFile(dest) as z:
                    for item in z.infolist():
                        if not (target / item.filename).resolve().is_relative_to(target.resolve()):
                            raise ValueError("Unsafe archive path")
                    z.extractall(target)
            else:
                with tarfile.open(dest) as t:
                    t.extractall(target, filter="data")
            (target / ".complete").touch()
        records.append(
            {**source, "sha256": hashlib.file_digest(dest.open("rb"), "sha256").hexdigest()}
        )
        (DATA / "sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        images = [p for p in target.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        from collections import Counter

        print(source["name"], len(images), dict(Counter(p.parent.name for p in images)), flush=True)


if __name__ == "__main__":
    main()
