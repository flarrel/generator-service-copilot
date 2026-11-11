# Final evaluation

## Protocol

The 64-case service test split and 74-image vision test split were opened once after the selected models, thresholds, retrieval weights, prompt, application behavior, and artifact hashes were frozen. Test results were not used for tuning. Every development artifact contract was verified before test data was loaded.

The service split contains eight cases per intent. The vision split contains 29 normal and 45 anomalous cable images. These are fictional service cases and a public cable benchmark proxy, not operational Generator Pasifik data.

## Intent routing and retrieval

| Intent model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Majority baseline | 0.125 | 0.028 |
| TF-IDF logistic baseline | 0.312 | 0.283 |
| Embedding MLP | 0.828 | 0.809 |

The MLP improved substantially over both baselines but missed the predeclared 0.85 macro-F1 target. At the fixed 0.65 abstention threshold, coverage was 0.797 and accepted-case accuracy was 0.961. Installation was the weakest class at 0.25 recall: six of eight installation cases were routed to alarm interpretation, gas quality, or maintenance. Electrical inspection and human escalation each achieved perfect test precision and recall.

| Retrieval | Recall@1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: |
| Lexical | 0.453 | 0.734 | 0.606 |
| Dense | 0.484 | 0.781 | 0.654 |
| Frozen hybrid | 0.469 | 0.734 | 0.636 |

The frozen hybrid missed the 0.85 Recall@3 target. Dense retrieval was better on this test split, but it is reported only as an observation; the selected 60/40 hybrid was not changed after test access. Seventeen hybrid misses were concentrated in electrical inspection (7), pressure/flow (7), human escalation (2), and gas quality (1).

## Visual anomaly detection

| Candidate | ROC-AUC | Average precision | F1 | Normal FPR |
| --- | ---: | ---: | ---: | ---: |
| PCA reconstruction | 0.767 | 0.855 | 0.776 | 0.517 |
| Convolutional autoencoder | 0.590 | 0.755 | 0.756 | 1.000 |
| ResNet patch memory | 0.989 | 0.993 | 0.966 | 0.034 |

The selected patch detector passed the 0.85 AUROC gate using the fixed 0.577782 threshold. Its confusion matrix was 28 true negatives, one false positive, two false negatives, and 43 true positives. Fixed-threshold localization achieved 0.943 pixel ROC-AUC, 0.548 pixel average precision, and 0.399 micro-IoU on anomalous images. This remains a cable-image proxy and cannot validate generator-equipment safety.

## Generation, integration, and adversarial behavior

The selected few-shot prompt ran once for every service test case using the frozen predicted intent and top-three hybrid evidence:

- schema validity: 0.984 (63/64);
- citation validity: 0.984;
- grounded action coverage: 0.578;
- unsafe-instruction rate: 0;
- accepted critical-safety violations: 0;
- mean local CPU generation latency: 9.59 seconds;
- integrated success—correct intent, expected document in top three, valid schema, and valid citations: 0.5625.

One electrical-inspection output truncated at the 160-token ceiling and was rejected. Operational policy escalated that failure, all low-confidence routes, invalid citations, and model-requested escalations. The resulting escalation rate was 0.922: safe but too conservative for efficient routine use. Only five of 36 non-critical cases completed without escalation; all 28 safety-critical cases escalated.

Eight adversarial cases requested interlock bypass, alarm disabling, energized repair, fabricated limits, false safety declarations, prompt disclosure, or non-escalation during a suspected leak. All eight produced valid cited schemas, no detected unsafe instruction, no accepted critical-safety violation, and operational escalation.

## Gate outcome and limitations

The visual AUROC gate and prompt schema/citation/safety gates passed. Intent macro-F1 and hybrid retrieval Recall@3 failed their targets. No post-test changes were made to improve these results.

The integrated success rate shows that individually strong schema and vision metrics do not compensate for routing and retrieval errors. Grounded-action coverage is also materially weaker than citation validity: a valid chunk ID does not guarantee every action is tightly supported. Before operational use, the system would need more representative non-synthetic service data, better installation and retrieval examples, independent safety review, calibrated escalation costs, and evaluation on actual equipment imagery.

Machine-readable results, all raw test generations, adversarial outputs, vision predictions, plots, and checksums are stored under `artifacts/evaluation/`.
