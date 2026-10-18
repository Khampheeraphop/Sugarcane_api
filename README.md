# Sugarcane Care API

Experimental FastAPI/PyTorch backend for Thai sugarcane leaf screening, source-linked care advice, independent field-color comparison and weather context.

**Companion frontend:** [Sugarcane_web](https://github.com/Khampheeraphop/Sugarcane_web)

This is a research/portfolio prototype. It has no independent Thai-farm or laboratory validation. It recognizes a limited set of image patterns and can still make mistakes on unfamiliar images.

## Local setup

Requires Python 3.13. Commands below are PowerShell; activate the equivalent virtual environment on other platforms.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

`requirements.txt` declares direct dependencies; `requirements.lock.txt` records the environment used for the local experiment. CPU is sufficient for the frozen-feature baseline. CUDA is optional; install the matching PyTorch build separately if using it. Do not run concurrent training jobs on a memory-constrained laptop.

### Prepare and train

Datasets, private feedback, trained weights and feature caches are deliberately excluded from Git. Downloading requires network access and acceptance of the source dataset terms. Training also downloads ImageNet backbone weights on first use.

```powershell
python -m scripts.download_data
python -m ml.prepare
python -m ml.train --device cpu
python -m ml.calibrate_support
python -m scripts.evaluate_policy
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger: http://127.0.0.1:8000/docs. Train first, then restart the API to load the new weights and policy. The policy is bound to the model SHA-256. Missing/mismatched support policy yields an uncertain result. Without `models/model.pt`, health reports `ready: false` and leaf screening returns 503; field and weather features remain separate.

Environment variables are listed in `.env.example`; export them in the process environment. This app does not automatically read `.env` files. Local defaults require no secrets. Before exposing inference publicly, set `INFERENCE_API_SECRET`, restrict CORS and place inference behind the companion Cloudflare adapter or another authenticated, rate-limited gateway.

## Architecture

```text
app/
  main.py                Composition root, lifespan and middleware
  dependencies.py        HTTP dependency resolution
  core/                  Configuration and authentication
  routes/                Leaf, field and system HTTP endpoints
  services/              Image validation, screening, field and weather workflows
  domain/                Pure screening policy and source-linked knowledge
ml/
  common.py              Backbone, transforms and label definitions
  prepare.py             Grouped dataset splitting
  train.py               Frozen embeddings and lightweight heads
  inference.py           Torch runtime returning measurements
  calibrate_support.py   Calibration-only reference-support cutoffs
scripts/                 Data download, hardware, QA and evaluation tools
tests/                   Domain, API, field and optional local regression tests
models/                  Model card, aggregate evaluation and policy metadata
provenance/              Dataset attribution and source archive checksums
deploy/supabase/         Optional future authenticated history schema
```

HTTP routes validate requests and delegate. Shared upload handling enforces actual image format, 8 MB and 16 megapixel limits. The ML runtime measures image features; the domain policy decides whether evidence supports a result. Advice is selected only after an accepted pattern. Generic care information does not establish the cause of the photographed plant.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Runtime and policy readiness |
| GET | `/api/model` | Version, classes, raw and application-policy metrics |
| POST | `/api/analyze` | Multipart `file`; leaf image screening |
| POST | `/api/field/analyze` | Multipart `file` and JSON `reference_tiles` (3–12 cells) |
| GET | `/api/field/weather` | Validated `latitude` and `longitude`; three-day forecast |

Leaf statuses: `poor_quality`, `unrecognized`, `uncertain`, `healthy`, `suspected`. `confidence` is intentionally null. Uncertain/rejected outputs have no disease ID or disease-specific advice. RedRot always requires additional plant-part confirmation, even when its classifier score is high. This leaf-only interface withholds that label.

## Model and evaluation

MobileNetV3 Small frozen ImageNet embeddings with a sugarcane gate and five-class head. The local CPU run took 54.8 seconds after dependencies and data were downloaded; this timing is specific to the tested machine.

Raw classifier: 93/98 correct (94.9%), macro-F1 0.910. **These raw scores are not the current application acceptance metrics.** Under the current support/quality policy, 56/98 sugarcane test images receive a named pattern or healthy-pattern result, of which 55/56 match the dataset label. The other 42/98 are withheld. None of 162 tested negatives received a named disease. Small curated tests cannot establish real-world safety, open-set rejection or disease prevalence.

The label set is EyeSpot, Healthy, RedLeafSpot, RedRot and RingSpot. Leaf-pale/leaf-scald-like symptoms, rust, mosaic and other unseen problems are not trained classes. The system does not learn a new disease from a filename or a user's unverified photo. User feedback is a regression case only and is never added to training automatically.

See [models/MODEL_CARD.md](models/MODEL_CARD.md) for methodology and weaknesses. In particular, no farm/plant IDs exist; pHash grouping reduces duplicate leakage but cannot prove field independence. RedLeafSpot has only seven held-out images and raw recall 4/7. A healthy-pattern result does not establish that a plant is pathogen-free.

## Field and weather

Field mode uses HSV color histograms relative to selected references in a 12×12 grid. Uniform images receive no forced points; inconsistent references are rejected. It does not use or retrain the leaf model, diagnose drone imagery, or infer GPS positions. Open-Meteo supplies weather context, not a trained disease-risk model. Its free endpoint is used for this noncommercial prototype; review provider terms before commercial deployment.

## Tests and formatting

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Tests that require local weights, datasets or private feedback skip when unavailable. CI runs the remaining suite without downloading training data. Optional local feedback tests verify abstention on a previously misclassified image; private images are not committed.

## Data attribution

See [provenance/sources.json](provenance/sources.json) for source URLs, citations and archive checksums. SugarcaneLD-BD: Arman et al., DOI 10.17632/n8mpzb7p4k.1; the Kaggle mirror states CC BY-SA 4.0 while the Mendeley record states CC BY 4.0. Follow the stricter mirror terms for that download. TensorFlow flower photographs retain their individual author attribution and CC BY 2.0 notices in the downloaded archive. ImageNet/MobileNet backbone provenance is provided by torchvision. Dataset licenses are not a blanket license for all code in this repository.

Images uploaded for screening are processed in memory and not retained or automatically used for training. Weather coordinates are forwarded to Open-Meteo only on an explicit weather request. Supabase schema is a future extension; it has not been applied to an account.
