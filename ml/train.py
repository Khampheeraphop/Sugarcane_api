"""Frozen MobileNet embeddings + two small heads. Train/val/calibration/test are separate.
Only train embeddings are augmented. Test data never selects hyperparameters or thresholds.
"""

import argparse
import copy
import json
import random
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from ml.common import CLASSES, MODELS, ROOT, backbone, transform, write_json


class Images(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.tf = (
            T.Compose(
                [
                    T.RandomResizedCrop(224, scale=(0.78, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(12),
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            if augment
            else transform()
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(ROOT / self.rows[i]["path"]) as im:
            return self.tf(im.convert("RGB"))


def features(model, rows, device, augment=False):
    results = []
    with torch.inference_mode():
        for batch in DataLoader(Images(rows, augment), batch_size=32, num_workers=0):
            results.append(model(batch.to(device)).cpu())
    return torch.cat(results)


def head_fit(x, y, vx, vy, classes, seed=42):
    torch.manual_seed(seed)
    head = torch.nn.Linear(x.shape[1], classes)
    opt = torch.optim.AdamW(head.parameters(), lr=0.005, weight_decay=0.05)
    counts = torch.bincount(y, minlength=classes).float()
    weights = counts.sum() / (classes * counts.clamp_min(1))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    best = float("inf")
    wait = 0
    history = []
    state = None
    for epoch in range(250):
        head.train()
        opt.zero_grad()
        loss = loss_fn(head(x), y)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            val = torch.nn.functional.cross_entropy(head(vx), vy).item()
        history.append({"epoch": epoch + 1, "train_loss": loss.item(), "val_loss": val})
        if val < best - 0.0001:
            best = val
            state = copy.deepcopy(head.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= 20:
            break
    head.load_state_dict(state)
    head.eval()
    return head, history


def temperature(logits, labels):
    # Scalar post-hoc calibration, fitted only on calibration split.
    options = np.geomspace(0.5, 5, 100)
    losses = [torch.nn.functional.cross_entropy(logits / float(t), labels).item() for t in options]
    return float(options[int(np.argmin(losses))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(MODELS / "cache"))
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    print("Device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    started = time.time()
    rows = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    net = backbone(True).to(device)
    partitions = {
        s: [r for r in rows if r["split"] == s] for s in ["train", "val", "calibration", "test"]
    }
    xs = {}
    for s, part in partitions.items():
        print("Embedding", s, len(part), flush=True)
        xs[s] = features(net, part, device)
    # Two additional views of each training image; frozen backbone keeps 4 GB GPU use modest.
    train_x = torch.cat(
        [xs["train"]] + [features(net, partitions["train"], device, True) for _ in range(2)]
    )
    train_rows = partitions["train"] * 3
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(0.05)

    def norm(x):
        return (x - mean) / std

    def label(rr):
        return torch.tensor(
            [CLASSES.index(r["label"]) if r["label"] in CLASSES else -1 for r in rr]
        )

    ys = {s: label(rr) for s, rr in partitions.items()}
    ty = label(train_rows)
    mask = ty >= 0
    vm = ys["val"] >= 0
    print("Training disease head", flush=True)
    disease, dh = head_fit(
        norm(train_x[mask]), ty[mask], norm(xs["val"][vm]), ys["val"][vm], len(CLASSES)
    )
    print("Training sugarcane gate", flush=True)
    gate, gh = head_fit(
        norm(train_x), (ty >= 0).long(), norm(xs["val"]), (ys["val"] >= 0).long(), 2
    )
    with torch.no_grad():
        dc = disease(norm(xs["calibration"]))
        gc = gate(norm(xs["calibration"]))
        cm = ys["calibration"] >= 0
        dt = temperature(dc[cm], ys["calibration"][cm])
        gt = temperature(gc, cm.long())
        dp = (dc / dt).softmax(1)
        gp = (gc / gt).softmax(1)[:, 1]
        reference = torch.nn.functional.normalize(xs["train"][ys["train"] >= 0], dim=1)
        similarities = torch.nn.functional.normalize(xs["calibration"], dim=1) @ reference.T
        sim = similarities.max(1).values
        best = None
        # Tune joint OOD gate against held-out known negatives, not against disease softmax alone.
        for g in np.linspace(0.5, 0.995, 60):
            for o in np.linspace(0.35, 0.95, 60):
                accepted = (gp >= g) & (sim >= o)
                fpr = accepted[~cm].float().mean().item()
                tpr = accepted[cm].float().mean().item()
                if fpr <= 0.02 and (best is None or tpr > best[0]):
                    best = (tpr, float(g), float(o), fpr)
        if best is None:
            best = (0.0, 1.01, 1.01, 0.0)
        _, gate_threshold, similarity_threshold, _ = best
        gate_accepted = (gp >= gate_threshold) & (sim >= similarity_threshold) & cm
        confidence, pred = dp.max(1)
        disease_threshold = 1.01
        for threshold in np.linspace(0.5, 0.99, 50):
            accepted = gate_accepted & (confidence >= threshold)
            if (
                accepted.sum() >= 10
                and (pred[accepted] == ys["calibration"][accepted]).float().mean() >= 0.9
            ):
                disease_threshold = float(threshold)
                break
        # Held-out test: report raw classification, rejection, and selective performance separately.
        tm = ys["test"] >= 0
        tp = (disease(norm(xs["test"])) / dt).softmax(1)
        tg = (gate(norm(xs["test"])) / gt).softmax(1)[:, 1]
        ts = (torch.nn.functional.normalize(xs["test"], dim=1) @ reference.T).max(1).values
        conf, pred = tp.max(1)
        accepted = (tg >= gate_threshold) & (ts >= similarity_threshold)
        decided = accepted & (conf >= disease_threshold)
        correct = (pred == ys["test"]) & tm
        selected = decided & tm
        metrics = {
            "raw_disease_accuracy": accuracy_score(ys["test"][tm], pred[tm]),
            "raw_disease_macro_f1": f1_score(ys["test"][tm], pred[tm], average="macro"),
            "disease_report": classification_report(
                ys["test"][tm],
                pred[tm],
                labels=list(range(len(CLASSES))),
                target_names=CLASSES,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                ys["test"][tm], pred[tm], labels=list(range(len(CLASSES)))
            ).tolist(),
            "test_sugarcane_count": int(tm.sum()),
            "test_negative_count": int((~tm).sum()),
            "sugarcane_gate_recall": accepted[tm].float().mean().item(),
            "negative_false_accept_rate": accepted[~tm].float().mean().item(),
            "negative_disease_output_rate": decided[~tm].float().mean().item(),
            "disease_coverage": selected.sum().item() / max(1, tm.sum().item()),
            "accepted_disease_count": int(selected.sum()),
            "accepted_disease_accuracy": correct[selected].float().mean().item()
            if selected.any()
            else None,
        }
        test_predictions = []
        for i, r in enumerate(partitions["test"]):
            test_predictions.append(
                {
                    **r,
                    "prediction": CLASSES[int(pred[i])],
                    "confidence": float(conf[i]),
                    "gate_score": float(tg[i]),
                    "similarity": float(ts[i]),
                    "accepted": bool(decided[i]),
                }
            )
    metadata = {
        "version": "mobilenet-v3-small-frozen-v1",
        "classes": CLASSES,
        "seed": 42,
        "disease_temperature": dt,
        "gate_temperature": gt,
        "gate_threshold": gate_threshold,
        "similarity_threshold": similarity_threshold,
        "disease_threshold": disease_threshold,
        "device": device,
        "training_seconds": round(time.time() - started, 1),
        "status": "experimental",
        "weather_model_trained": False,
        "limitations": [
            "No independent Thai farm evaluation.",
            "No plant/farm IDs; pHash grouping only.",
            "Negatives are flowers and synthetic controls; grass, rice, maize, objects and unseen diseases are not validated.",
            "Labels describe image patterns; not laboratory-confirmed diagnosis. Healthy means no recognized visible symptom.",
            "Calibration sample is small; confidence is not a guarantee.",
        ],
    }
    net.cpu()
    torch.save(
        {
            "backbone": net.state_dict(),
            "disease": disease.state_dict(),
            "gate": gate.state_dict(),
            "mean": mean,
            "std": std,
            "reference": reference,
            "metadata": metadata,
        },
        MODELS / "model.pt.tmp",
    )
    write_json(MODELS / "metadata.json", metadata)
    write_json(MODELS / "metrics.json", metrics)
    write_json(MODELS / "history.json", {"disease": dh, "gate": gh})
    write_json(MODELS / "test_predictions.json", test_predictions)
    (MODELS / "model.pt.tmp").replace(MODELS / "model.pt")
    print(json.dumps({"metadata": metadata, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
