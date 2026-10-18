"""Create isolated QA fixtures. Never automatically load them in the user's browser."""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ROOT / "output")
out = parser.parse_args().output.resolve()
out.mkdir(parents=True, exist_ok=True)
rows = json.loads((ROOT / "models/test_predictions.json").read_text(encoding="utf-8"))
row = next(
    r for r in rows if r["accepted"] and r["label"] == "EyeSpot" and r["prediction"] == r["label"]
)
shutil.copy2(ROOT / row["path"], out / "qa_leaf.jpg")
image = Image.new("RGB", (768, 768), "#367232")
ImageDraw.Draw(image).rectangle((512, 512, 639, 639), fill="#aa6834")
image.save(out / "qa_field_synthetic.png")
Image.new("RGB", (768, 768), "#367232").save(out / "qa_uniform_field.png")
(out / "qa_fake.txt").write_text("not an image", encoding="utf-8")
(out / "qa_large.png").write_bytes(bytes(9 * 1024 * 1024))
(out / "QA_FIXTURES.md").write_text(
    f"QA ONLY. Never show these as real farm data.\n\nLeaf source: {row['path']}\n"
    "Source: Arman et al., SugarcaneLD-BD, Kaggle mirror CC BY-SA 4.0; see data/sources.json.\n"
    "Field images are synthetic flat-color test fixtures. They do not validate agricultural performance.\n"
    "qa_fake.txt and qa_large.png deliberately test invalid input.\n",
    encoding="utf-8",
)
print("QA fixtures ready; no browser state was changed.")
