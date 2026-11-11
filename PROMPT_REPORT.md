# Prompt experiment report

## Purpose and boundary

This experiment compares four versioned prompts for a bounded service-response generator. It uses eight fixed development cases—one per service intent—and three sampled runs per case. The 64-case final service test split remains sealed. The corpus is fictional, so results describe prompt behavior in this demo and not real Generator Pasifik operations.

The local provider is `HuggingFaceTB/SmolLM2-360M-Instruct` at revision `502b5cb068ad185080c7ea271bee653f0f6c0777`. Its downloaded files total 727,051,867 bytes, the 724 MB weight file has SHA-256 `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`, and inference uses float32 CPU. The provider pre-fills the opening `{` token but does not constrain, repair, or replace the remainder of the generated output.

## Prompt variants

The exact version 1.0 prompt bodies are stored in `prompts/variants.json`:

1. `v1_direct` asks directly for an evidence-only JSON response.
2. `v2_grounded_schema` defines the fields and grounding behavior explicitly.
3. `v3_few_shot` adds one valid grounded example while prohibiting reuse of its citation.
4. `v4_safety_rules` adds explicit unsafe-action prohibitions and escalation rules.

Each output must contain a summary, one to five actions, one to three exact evidence chunk IDs, an escalation flag, and low/medium/high confidence. The Pydantic schema rejects missing, malformed, or extra fields.

## Predeclared evaluation

Every variant receives the same question, predicted intent, safety flag, and three approved chunks. Each case runs at seeds fixed in code with temperature 0.2, top-p 0.9, and a 160-token ceiling. Metrics are schema validity, citation validity, grounded action coverage, required-field completeness, unsafe-instruction rate, accepted critical-safety violations, action consistency across repeats, and latency.

The score weights are 20% schema validity, 20% citation validity, 15% grounded action coverage, 10% completeness, 25% safety, and 10% consistency. A candidate is ineligible if it has any accepted critical-safety violation or falls below 95% schema or citation validity.

## Development results

| Variant | Schema | Citations | Grounded actions | Consistency | Mean latency | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Direct | 0.000 | 0.000 | 0.000 | 0.000 | 8.13 s | Fail |
| Grounded schema | 0.458 | 0.458 | 0.194 | 0.417 | 9.70 s | Fail |
| Few-shot | 0.958 | 0.958 | 0.771 | 0.819 | 9.30 s | Pass |
| Safety rules | 0.625 | 0.625 | 0.125 | 0.500 | 11.15 s | Fail |

`v3_few_shot` is selected with score 0.9267. It produced no detected unsafe instruction and no accepted critical-safety violation. One of its 24 outputs hit the token limit before citations and required fields, so the result is not perfect and the application must treat schema failure as an escalation rather than expose partial text.

## Human review and failure analysis

The direct prompt frequently copied source text into invented object shapes, embedded full evidence strings instead of chunk IDs, or omitted fields. The explicit schema improved compliance but sometimes used numeric placeholders, truncated before completion, or returned `escalate=false` for safety-critical cases. The safety-heavy prompt was counterproductive for this small model: it repeated safety sentences, emitted a literal confidence union, and exceeded the action-count limit. More rules did not mean better control.

The few-shot example gave the model a compact structural target without overwhelming it. Its valid actions generally paraphrased the supplied evidence and its citations were exact. Grounded action coverage of 0.771 still leaves room for unsupported or too-generic action wording, so the selected prompt remains bounded by schema rejection, citation validation, safety escalation, and human review.

An initial run without the JSON opening-token prefill produced zero schema-valid outputs across all variants. That provider failure was diagnosed from raw output and corrected before the authoritative experiment; it is not included as a favorable prompt result.

## Optional OpenAI provider

The optional adapter uses the Responses API, strict JSON Schema output under `text.format`, the configured model name, and `store=false`. This follows [official OpenAI Responses API guidance](https://developers.openai.com/api/docs/guides/migrate-to-responses?api-mode=responses). Mocked contract tests verify the request and response handling without a key or spend. No live OpenAI call occurred.

## Reproduction and limits

Run `.venv/bin/evaluate-prompts` after placing the pinned local model in `.hf_cache/generation/smollm2-360m-instruct`. Detailed raw generations, metrics, provenance, and checksums are in `artifacts/prompts/`. Development reuse makes these estimates optimistic; final claims wait for the sealed evaluation checkpoint. A 360M model is also not evidence that the same ranking transfers to a larger local or hosted model.
