# Visual anomaly detection

This checkpoint detects unusual cable or connector appearance from an uploaded image. It uses the MVTec AD cable category as a non-commercial industrial-inspection proxy; it does not establish performance on hydrogen or oxygen generators.

## Method

The 179 normal training images fit three development candidates. A grayscale-and-edge PCA reconstruction model provides a feature baseline. A 65,363-parameter convolutional autoencoder learns normal-image reconstruction from scratch. The selected detector instead extracts frozen ImageNet ResNet-18 `layer2` patch features, retains a deterministic 5,000-patch normal memory bank, and scores each patch by its nearest normal neighbor. No anomaly image or final-test image is used to fit either model.

Model and threshold selection used only 121 development images (74 normal and 47 anomalous). The image score averages the highest 10% of patch distances. Its decision threshold maximizes development F1, while the localization threshold independently maximizes pixel F1 on development anomaly masks. The 74-image final test split was later opened once with these thresholds frozen; see `EVALUATION.md`.

## Development results

| Candidate | ROC-AUC | Average precision | F1 |
| --- | ---: | ---: | ---: |
| PCA reconstruction | 0.7090 | 0.6742 | 0.6080 |
| Autoencoder, best aggregation | 0.5296 | 0.4633 | 0.5595 |
| ResNet patch memory, top 10% | 0.9865 | 0.9810 | 0.9213 |

At the selected threshold, the patch model records 73 true negatives, one false positive, six false negatives, and 41 true positives. Its development localization result is 0.9452 pixel ROC-AUC, 0.5463 pixel average precision, and 0.4126 micro-IoU on anomalous images. The weak autoencoder result is retained rather than hidden: pixel reconstruction responds strongly to normal cable-position and illumination variation, whereas pretrained patch features provide a much stronger representation for this small dataset.

## Reproduction and artifacts

Place the MVTec data using the commands in `DATA.md`, and cache the official torchvision ResNet-18 weights at `.torch_cache/hub/checkpoints/resnet18-f37072fd.pth`. The expected SHA-256 is `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. Then run:

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/train-vision
```

`artifacts/vision/report.json` contains metrics and provenance, `development_predictions.csv` contains per-image scores, and `artifact_contract.json` checksums all committed inference and evidence files. `models/patch_memory.npy` is the selected detector's fitted state. The official pretrained weights and raw images remain local and uncommitted because they are reproducible external dependencies.

## Limits

MVTec cable images are a visual proxy, not field data from the target equipment. The development set was used repeatedly for candidate and threshold selection, so its metrics are optimistic. The patch detector can identify visual novelty but cannot infer electrical safety, root cause, or repair suitability. Its output should support inspection triage only; a qualified person must make service and safety decisions.
