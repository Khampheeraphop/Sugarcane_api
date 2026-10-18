"""Evaluate the application policy separately from the original classifier metrics."""

import json
from collections import Counter

from PIL import Image

from app.services.screening import ScreeningService
from ml.common import MODELS, ROOT, write_json


def main():
    service = ScreeningService(MODELS / "model.pt", MODELS / "policy.json")
    rows = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in rows if row["split"] == "test"]
    counts = Counter()
    accepted = correct = negative_outputs = 0
    for row in rows:
        with Image.open(ROOT / row["path"]) as image:
            result = service.predict(image.convert("RGB"))
        sugarcane = row["label"] != "not_sugarcane"
        counts[f"{'sugarcane' if sugarcane else 'negative'}/{result['status']}"] += 1
        if result["status"] in ("healthy", "suspected"):
            if sugarcane:
                accepted += 1
                correct += result["class_id"] == row["label"]
            else:
                negative_outputs += 1
    n = sum(row["label"] != "not_sugarcane" for row in rows)
    report = {
        "version": service.meta["version"],
        "test_sugarcane_count": n,
        "test_negative_count": len(rows) - n,
        "accepted_disease_count": accepted,
        "disease_coverage": accepted / n,
        "accepted_disease_accuracy": correct / accepted if accepted else None,
        "negative_disease_output_count": negative_outputs,
        "status_counts": dict(counts),
        "requires_confirmation": ["RedRot"],
        "note": "Application policy metrics, including quality checks and class support. Raw classifier metrics are separate. Test results never select cutoffs.",
    }
    write_json(MODELS / "policy_metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
