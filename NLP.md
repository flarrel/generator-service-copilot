# NLP and retrieval notes

## Evaluation boundary

Checkpoint 2 uses only the 192 training and 64 development service cases. The 64 service-case test rows remain sealed. All reported values are development results and must not be presented as final generalization estimates.

The embedding model is `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Only the approximately 92 MB required PyTorch files were downloaded; duplicate ONNX, OpenVINO, TensorFlow, and Rust exports were excluded. Inference is forced to CPU for consistent execution on machines without compatible acceleration.

## Leakage audit

A character 3–5 gram TF-IDF audit compares every development case with its nearest training case. The maximum cross-split similarity is 0.693, the 95th percentile is 0.677, and no pair reaches the 0.90 review threshold. Scenario-group and normalized-duplicate isolation from checkpoint 1 also remain valid.

## Intent routing

Three candidates were evaluated:

- a majority-class baseline: 0.028 macro-F1;
- word/character TF-IDF with logistic regression: 0.373 macro-F1;
- a two-layer MLP over 384-dimensional sentence embeddings: 0.854 macro-F1.

The MLP is trained with the training service cases plus threefold weighting of the 24 approved manual chunks as labeled class prototypes. This uses no development labels for fitting. At the development-selected 0.65 confidence threshold, the model accepts 84.4% of cases at 90.7% accuracy. Low-confidence cases are intended for human review.

The MLP is the checkpoint candidate. Per-case predictions, confidence, correctness, per-class metrics, and all weaker baselines remain in `artifacts/nlp/` for inspection. No final model claim is made before sealed-test evaluation.

## Retrieval

Lexical TF-IDF, dense cosine retrieval, and a fixed 60% lexical/40% dense hybrid were compared against the expected manual for each development query:

| Method | Recall@1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: |
| Lexical | 0.406 | 0.672 | 0.581 |
| Dense | 0.500 | 0.812 | 0.677 |
| Hybrid | 0.562 | 0.906 | 0.719 |

The hybrid is selected for later application integration. Redis still stores and serves the dense vectors; application-level lexical scores can be fused with Redis distances.

## Redis verification

Redis 8.2.3 ran in a disposable local container. A namespaced FLAT cosine index stored all 24 document chunks with 384-dimensional vectors and metadata fields. Three live KNN checks each returned the expected document within the top three. The container was stopped and auto-removed after verification.

## Data mining

K-means with eight clusters was applied to training-case embeddings. The silhouette score is 0.141, adjusted Rand index is 0.132, and normalized mutual information is 0.321. These weak-to-moderate values show that issue language does not naturally form eight clean intent clusters. The result is retained as honest exploratory evidence rather than represented as a successful taxonomy discovery.

Cluster composition, representative cases, and the two-dimensional PCA visualization are available in `artifacts/nlp/`.
