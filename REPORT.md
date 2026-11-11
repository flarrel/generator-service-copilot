# Generator Service Copilot

## Background

Service staff working around hydrogen and oxygen generation equipment need fast access to approved guidance, but a language model that invents settings or repair actions would create unacceptable risk. Hydrogen is highly flammable and can create fire and explosion hazards when mishandled, reinforcing the need for approved controls and qualified review rather than autonomous advice ([OSHA](https://www.osha.gov/green-jobs/hydrogen/fire-explosion)).

This project investigates a bounded local assistant for a fictional service workflow. It combines text classification, document retrieval, structured generation, visual anomaly detection, caching, and explicit escalation. The approach follows the NIST AI RMF principle that risk considerations should be incorporated across AI design, development, use, and evaluation ([NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)).

## Problem definition

The tool must route a service question, retrieve relevant approved text, generate a cited structured answer, flag unusual cable appearance, and expose uncertainty. It must run locally without a paid API, avoid proprietary data, and degrade safely when retrieval, generation, Redis, or schema validation fails.

The project tests four questions:

1. Can compact sentence embeddings and an MLP route eight fictional service intents?
2. Can lexical/dense retrieval provide the expected manual within the top three results?
3. Can a local instruction model produce consistent, cited, schema-valid output without unsafe accepted actions?
4. Can nominal patch features distinguish anomalous industrial cable images and localize defects?

## Data and engineering

The project-authored text corpus contains 320 fictional cases from 80 semantic groups, split group-wise into 192 training, 64 development, and 64 final test cases. Eight fictional versioned manuals yield 24 chunks with lineage and safety metadata. The data pipeline validates schemas, domains, unique identifiers, versions, missing values, and group isolation, then records checksums for tracked artifacts.

The visual data is the cable category of MVTec AD. The benchmark supplies defect-free training images, defective and normal test images, and pixel-level anomaly masks ([Bergmann et al., 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html)). Its CC BY-NC-SA license makes this a non-commercial research proxy; raw images are excluded from Git.

## Methods

### NLP, retrieval, and data mining

Three intent candidates were compared: majority class, word/character TF-IDF logistic regression, and a two-layer MLP over 384-dimensional MiniLM sentence embeddings. Sentence-BERT’s bi-encoder design supports independently computed semantic vectors and cosine comparison, making it suitable for retrieval and clustering ([Reimers and Gurevych, 2019](https://aclanthology.org/D19-1410/)).

Retrieval compared lexical TF-IDF, dense cosine similarity, and a development-selected 60/40 hybrid. Redis stored all chunk embeddings in an exact FLAT cosine index. FLAT search compares against every indexed vector and suits small collections where exact nearest neighbors matter ([Redis vector search documentation](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)). K-means over training embeddings provided exploratory issue-pattern analysis; weak silhouette and label-alignment scores were retained rather than overstated.

### Computer vision

Visual baselines used PCA reconstruction and a three-layer convolutional autoencoder trained only on normal images. The selected detector extracts frozen mid-level ResNet-18 patch features and compares them with a 5,000-patch nominal memory. This design is a simplified PatchCore-like approach; PatchCore established nominal patch-feature memory banks as an effective industrial anomaly-detection strategy ([Roth et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)).

Image-level thresholds and the pixel threshold were selected using development data only. Final metrics include ROC-AUC, average precision, fixed-threshold precision/recall/F1, normal false-positive rate, pixel ROC-AUC, pixel average precision, and micro-IoU.

### Prompt engineering and generation

Four versioned prompts tested direct instruction, explicit schema and grounding, a grounded few-shot example, and additional safety/escalation rules. Eight development cases ran three times per variant under fixed sampling settings. A predeclared scorecard weighted schema validity, citation validity, grounded-action coverage, field completeness, safety, and repeat consistency.

The selected local provider is SmolLM2-360M-Instruct in float32 CPU mode. A provider-level opening-brace prefill steers raw JSON generation; output is still parsed and validated rather than repaired. The selected few-shot prompt outperformed the more verbose safety prompt, showing that additional instructions can overload a small model.

An optional OpenAI Responses API adapter is implemented but not used live. It sends strict JSON Schema through `text.format`, reads `output_text`, and sets `store=false`, consistent with [official OpenAI Responses API guidance](https://developers.openai.com/api/docs/guides/migrate-to-responses?api-mode=responses) and [Structured Outputs guidance](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses). Mocked tests validate the contract without claiming live API experience.

### Application and safety controls

The four-view Streamlit application integrates routing, retrieval, generation, visual inspection, prompt results, and cluster insights. Response keys include a document-corpus checksum and expire after 15 minutes. Redis failures fall back to local hybrid retrieval; generation or schema failures return a cited low-confidence escalation. Uploaded images use temporary local files.

The final test protocol verified every prior artifact hash before opening test data. Models, thresholds, retrieval weights, prompt selection, and application rules were frozen. Test results were never used for retuning.

## Results

| Component | Main final result |
| --- | ---: |
| Intent MLP | accuracy 0.828, macro-F1 0.809 |
| Fixed abstention | 0.797 coverage, 0.961 accepted accuracy |
| Hybrid retrieval | Recall@3 0.734, MRR 0.636 |
| ResNet patch detector | ROC-AUC 0.989, AP 0.993, F1 0.966 |
| Patch localization | pixel ROC-AUC 0.943, micro-IoU 0.399 |
| Selected generation prompt | schema/citation validity 0.984/0.984 |
| Integrated success | 0.5625 |
| Operational escalation | 0.922 |

The visual detector passed its 0.85 AUROC gate with one false positive and two false negatives among 74 test images. Prompt schema, citation, and adversarial-safety gates also passed: eight adversarial requests caused no accepted critical-safety violation and all escalated operationally.

Intent macro-F1 and hybrid Recall@3 missed their 0.85 gates. Installation recall was only 0.25, while 17 hybrid retrieval misses were concentrated in electrical inspection and pressure/flow. Dense retrieval outperformed the frozen hybrid on the final split, but selection was not changed after test access.

Generation achieved strong formal compliance but only 0.578 grounded-action coverage. One output truncated at the token ceiling. The integrated success rate of 0.5625 and 0.922 escalation rate show that valid JSON and citations do not by themselves make an efficient support system.

## Discussion

The strongest result is visual anomaly discrimination, where pretrained patch features substantially outperform both reconstruction approaches. The result aligns with the industrial anomaly-detection literature, but it cannot be transferred directly to hydrogen or oxygen generators because the benchmark contains cable imagery under controlled conditions.

For text, the compact system is credible as a triage demonstrator but not ready for operational use. Synthetic language does not cover the ambiguity and vocabulary of real service records. Retrieval evaluation also shows that development gains did not generalize to the final split. A future study should collect approved, de-identified domain examples; add hard retrieval negatives; separate installation from maintenance language; and perform independent safety review.

The conservative escalation policy prevents accepted critical failures in this evaluation, but creates substantial human workload. Risk tolerance, escalation cost, and reviewer capacity must be defined with domain owners; they cannot be optimized from a small fictional benchmark alone.

## Ethics, safety, and limitations

- No real customers, proprietary manuals, or employer operational claims are represented.
- MVTec AD is non-commercial and used only as a proxy; actual equipment images require separate approval and validation.
- The system cannot detect all hazards, infer root cause, prescribe settings, or approve return to service.
- Automated grounded-action coverage is lexical and cannot replace expert factual review.
- Eight adversarial cases are useful evidence, not exhaustive red-team coverage.
- The OpenAI adapter was not called live; cost, credentials, account access, and live response behavior remain unverified.

## Conclusion

Generator Service Copilot demonstrates the requested technical workflow end to end: validated data engineering, NLP, neural models, retrieval, Redis, data mining, computer vision, prompt testing, structured generation, a local application, and frozen evaluation. It also exposes material shortcomings rather than claiming universal success. Vision and prompt-safety gates pass; intent, retrieval, grounded actions, integrated success, and escalation efficiency require further work before any operational trial.

## References

1. Bergmann, P. et al. (2019). [MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html). CVPR.
2. Reimers, N. and Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/). EMNLP-IJCNLP.
3. Roth, K. et al. (2022). [Towards Total Recall in Industrial Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html). CVPR.
4. Redis. [Vector search concepts](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/).
5. NIST. (2023). [Artificial Intelligence Risk Management Framework 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10).
6. OSHA. [Hydrogen Fuel Cells: Fire and Explosion](https://www.osha.gov/green-jobs/hydrogen/fire-explosion).
7. OpenAI. [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses?api-mode=responses) and [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).
