# Data notes

## Fictional service corpus

The service cases and manual-style documents are project-authored fictional fixtures. They do not describe Generator Pasifik operations, real customers, or manufacturer-approved procedures.

`prepare-data` deterministically creates 320 cases from 80 semantic scenario groups across eight intents. All four paraphrases of a scenario remain in one split. Each intent contributes 24 training, eight development, and eight final test cases. The test cases were not used for model selection and were opened once during the frozen final evaluation.

The eight fictional source documents produce 24 chunks. Every chunk records its source file, document identifier, version, equipment family, safety topic, and synthetic-data flag.

## MVTec AD cable subset

The local image data comes from the cable category of [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), mirrored in the `foersben/mvtec-ad` dataset repository at revision `c75b39616f84db43677bcc8228caaafaf5096d7f`.

The snapshot contains 224 normal training images, 58 normal test images, 92 anomalous test images, and a pixel mask for every anomaly. The manifest verifies 374 unique image hashes and 504,738,427 image bytes. A deterministic project split assigns 179 normal images to training, 121 images to development, and 74 images to a sealed test split while preserving every defect category in development and test.

MVTec AD is licensed under CC BY-NC-SA 4.0. Raw images and masks remain ignored by Git and may be used here only for non-commercial work. `data/sources.json` records the source revision, retrieval date, license, and license-file checksum. Reproduction requires explicit acceptance of the dataset terms before running:

```bash
download-mvtec --destination data/external/mvtec_ad
prepare-data
```

## Generated evidence

- `data/processed/manifest.json` records checksums for tracked source and processed data.
- `data/processed/mvtec_cable_manifest.csv` records image paths, source/project splits, defect classes, mask paths, hashes, and byte sizes without redistributing image content.
- `artifacts/eda/summary.json` records data counts and licensing status.
- `artifacts/eda/case_balance.png` and `document_chunk_lengths.png` summarize the fictional corpus.
