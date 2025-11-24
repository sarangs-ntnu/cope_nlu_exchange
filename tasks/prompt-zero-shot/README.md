# Prompt-based zero/low-shot sentiment by aspect

## Objective
Evaluate instruction-tuned language models for aspect-aware sentiment without full supervised training, using zero-shot and few-shot prompting.

## Dataset assumptions
- Small labeled set available for few-shot examples; same columns as prior tasks.
- Sentiment label space known (e.g., positive/negative/neutral).

## Experiment steps
1. **Model selection**: Choose instruction models accessible in the environment (e.g., local Mistral/LLama variants or API-based analogs).
2. **Prompt design**:
   - Include aspect explicitly: "You are grading sentiment about the **teacher**."
   - Constrain the output label set and format (e.g., JSON with `sentiment`).
   - Prepare k-shot prompts with balanced examples per aspect.
3. **Decoding settings**: Use deterministic decoding (temperature=0–0.2) for consistency; limit max tokens.
4. **Evaluation**: Run zero-shot and few-shot settings; compute accuracy/macro F1 overall and per aspect. Record qualitative failures (hallucinated labels, ignoring aspect).
5. **Cost/runtime tracking**: Log latency and, if applicable, token costs.

## Reporting
Compare zero/low-shot performance to supervised baselines, noting cases where prompts succeed or fail (e.g., implicit sentiment, aspect confusion).
