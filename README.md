# Generator Service Copilot

Local decision support for fictional hydrogen- and oxygen-generator service workflows. The application combines intent routing, grounded document retrieval, schema-validated generation, cable-image anomaly detection, prompt comparison, and service-pattern exploration.

It is a demonstration system, not a diagnostic device or safety controller. It does not contain proprietary manuals or real customer data, cannot approve equipment for service, and must not replace qualified review.

## What is included

- **Ask:** route a service question, retrieve three approved chunks, produce cited structured guidance, and escalate low-confidence or invalid output.
- **Inspect:** score an uploaded cable image and display a patch-level anomaly map.
- **Prompt Lab:** compare four versioned prompt strategies and their measured consistency.
- **Insights:** inspect clustering results from fictional service cases.
- **Redis:** exact FLAT cosine vector search plus response caching, TTL, document-version isolation, invalidation, and cold-cache fallback.
- **Optional OpenAI adapter:** contract-tested Responses API integration; disabled by default and never called live in this project.

## Architecture

```text
question ──> MiniLM embedding ──> embedding MLP ──> intent/confidence
    │                 │
    │                 └──> lexical + dense retrieval ──> Redis/local top 3
    │                                                    │
    └──────────────────────── selected prompt <──────────┘
                                  │
                                  └──> validated JSON + citations + escalation

image ──> frozen ResNet-18 patch features ──> normal patch memory
                                              │
                                              └──> anomaly score + heatmap
```

The application uses saved, development-selected artifacts. The final evaluation opened the 64-case service and 74-image vision test splits once without post-test tuning.

## Final results

| Component | Final result | Gate |
| --- | ---: | --- |
| Intent routing | macro-F1 0.809 | Missed 0.85 target |
| Hybrid retrieval | Recall@3 0.734 | Missed 0.85 target |
| Visual anomaly detection | ROC-AUC 0.989, F1 0.966 | Passed |
| Generated schema/citations | 0.984 / 0.984 validity | Passed |
| Adversarial safety | 0 accepted critical violations in 8 cases | Passed |
| Integrated workflow | 0.562 success, 0.922 escalation | Conservative |

See [EVALUATION.md](EVALUATION.md) for the full protocol, baselines, confusion matrix, localization results, failure analysis, and limitations.

## Requirements

- Python 3.12
- Docker with Compose, for Redis
- approximately 2 GB free RAM for local generation
- approximately 1.4 GB disk for ignored model caches, plus approximately 505 MB for MVTec cable images
- CPU-only execution is supported; no GPU or paid service is required

MVTec AD is CC BY-NC-SA 4.0 and non-commercial. Read its terms before downloading. Raw images, pretrained weights, model caches, credentials, and runtime caches are excluded from Git.

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'
```

Download the pinned embedding files into the expected Hugging Face cache:

```bash
HF_HOME=.hf_cache .venv/bin/hf download \
  sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --include '*.json' '*.txt' '1_Pooling/config.json' 'model.safetensors'
```

Download the pinned local instruction model:

```bash
.venv/bin/hf download HuggingFaceTB/SmolLM2-360M-Instruct \
  --revision 502b5cb068ad185080c7ea271bee653f0f6c0777 \
  config.json generation_config.json model.safetensors merges.txt \
  special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json \
  --local-dir .hf_cache/generation/smollm2-360m-instruct
```

Download the official ResNet-18 weights and verify their SHA-256:

```bash
mkdir -p .torch_cache/hub/checkpoints
curl -L https://download.pytorch.org/models/resnet18-f37072fd.pth \
  -o .torch_cache/hub/checkpoints/resnet18-f37072fd.pth
shasum -a 256 .torch_cache/hub/checkpoints/resnet18-f37072fd.pth
```

Expected checksum: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

After accepting the MVTec license terms:

```bash
.venv/bin/download-mvtec --destination data/external/mvtec_ad
MPLCONFIGDIR=.mplconfig .venv/bin/prepare-data
```

Recorded revisions and checksums are in `data/sources.json` and `data/processed/manifest.json`.

## Run the application

```bash
docker compose up -d redis
REDIS_URL=redis://127.0.0.1:6380/0 MPLCONFIGDIR=.mplconfig \
  .venv/bin/streamlit run src/generator_service_copilot/app.py
```

Open `http://localhost:8501`. Stop Redis afterward with `docker compose down`.

Redis is optional. Without `REDIS_URL`, Ask uses the saved local hybrid retriever, response caching reports unavailable, and the other views remain usable. A generation failure or malformed response produces a cited low-confidence escalation instead of exposing partial model text.

## Commands

| Command | Purpose |
| --- | --- |
| `prepare-data` | Rebuild and validate fictional cases, chunks, manifests, and EDA |
| `train-nlp` | Refit development-stage intent, retrieval, and clustering artifacts |
| `train-vision` | Refit visual candidates and development thresholds |
| `evaluate-prompts` | Repeat the four-variant development prompt experiment |
| `verify-app-cache` | Verify Redis TTL, invalidation, restart, and workflow behavior |
| `evaluate-final` | Reproduce final scoring; do not use results for selection or tuning |

The final test splits are no longer sealed: they were opened once in checkpoint 6. Rerunning `evaluate-final` is reproduction, not a fresh holdout evaluation.

## Testing

```bash
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
.venv/bin/pytest -q
.venv/bin/pip check
git diff --check
```

The committed evidence contracts checksum every material artifact. The final clean-environment verification record is in `artifacts/verification/`.

## Documentation map

- [DATA.md](DATA.md): data lineage, splits, licensing, and manifests
- [NLP.md](NLP.md): routing, retrieval, leakage audit, and clustering
- [VISION.md](VISION.md): anomaly models, thresholds, and image limitations
- [PROMPT_REPORT.md](PROMPT_REPORT.md): exact prompt experiment and selection
- [APP.md](APP.md): interface, cache behavior, and graceful fallbacks
- [EVALUATION.md](EVALUATION.md): frozen final metrics and failure analysis
- [REPORT.md](REPORT.md): structured project report and references

## Troubleshooting

- **Pinned model missing:** repeat the matching download command; caches must remain at the documented paths.
- **Checksum mismatch:** remove only the affected cached file and download the pinned revision again.
- **Redis unavailable:** confirm `docker compose ps` reports healthy and port 6380 is free; otherwise run without `REDIS_URL` for local retrieval fallback.
- **Slow generation:** 360M float32 CPU generation averaged 9.59 seconds in final evaluation. The first request also loads model weights.
- **Malformed generation:** the application rejects it and escalates. Do not bypass schema or citation validation.
- **Matplotlib cache warning:** prefix commands with `MPLCONFIGDIR=.mplconfig`.
- **macOS elastic redirect notice:** PyTorch prints this for unsupported distributed redirect handling; the project does not use distributed execution.

## Safety and privacy

The text corpus is fictional and the image dataset is an industrial cable benchmark. The app has no analytics, telemetry, public deployment, or persistent Redis storage. Uploaded images are processed through temporary local files. Hydrogen can present fire and explosion hazards, so software output must never replace approved procedures, monitoring, isolation, protective equipment, or qualified personnel.
