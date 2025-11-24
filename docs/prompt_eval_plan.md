# Prompt-based Aspect Sentiment Evaluation Plan

This document outlines how to evaluate small instruction-tuned language models on aspect-based sentiment classification using zero-shot and few-shot prompts. It also covers metrics and qualitative comparisons with supervised baselines.

## Model Selection
Choose small, cost-efficient instruction models that support chat-style prompting. Example candidates:
- GPT-like analogs (e.g., `gpt-3.5-turbo` scale or smaller, including hosted light variants).
- Open-source instruction models: `mistral-instruct`, `llama-3-8b-instruct`, `phi-3-mini-instruct`, or comparable 7–8B chat checkpoints.

## Label and Aspect Schema
- **Aspects:** Define a fixed list per domain (e.g., `food`, `service`, `ambience`, `price`).
- **Sentiment labels:** Limit to `positive`, `neutral`, `negative`.
- Include a fallback label `none` when no opinion is expressed about an aspect.

## Prompt Templates
### Zero-shot prompt (chat style)
```
System: You are an assistant that extracts aspect-level sentiment from customer reviews.
User: Review: "{review_text}"
Identify the sentiment for each aspect in the list: {aspects_csv}.
Return JSON: [{"aspect": "<aspect>", "sentiment": "positive|neutral|negative|none", "evidence": "<short quote>"}]. Only include aspects in the list.
```

### Few-shot (k-shot) prompt
Provide k labeled examples before the target review:
```
System: You are an assistant that extracts aspect-level sentiment from customer reviews.
User: Example 1 Review: "<text>"\nAspects: food, service, ambience, price\nAnswer: [{"aspect": "food", "sentiment": "positive", "evidence": "fresh pasta"}, ...]
...
User: Example k Review: "<text>"\nAspects: <aspects_csv>\nAnswer: [...]
User: Target Review: "{review_text}"\nAspects: {aspects_csv}\nAnswer:
```
- Include varied sentiment coverage across examples (positive/neutral/negative/none) and multiple aspect mentions.
- Keep k small (2–5) to control latency.

### Output Validation
- Enforce JSON schema via lightweight regex or JSON parsing; if invalid, reprompt with the same context and an explicit schema reminder.
- Normalize aspect names and sentiments to lowercase before scoring.

## Evaluation Protocol
1. **Data split:** Use a labeled aspect-sentiment dataset with train/dev/test; keep the test split unseen.
2. **Baselines:** Train or reuse a supervised classifier per aspect (e.g., fine-tuned small transformer) for comparison.
3. **Inference settings:**
   - Temperature: 0–0.2 for determinism.
   - Max tokens: Sized to cover all aspects; usually 128–256.
   - For k-shot, cache the exemplar block to reduce latency.
4. **Metrics (per aspect and macro averages):**
   - Accuracy and F1 for `positive`, `neutral`, `negative`, `none`.
   - Report support counts per aspect and label.
5. **Error analysis:**
   - Manually sample mismatches vs. gold labels and compare against supervised model errors.
   - Note confusion patterns (e.g., neutral vs. none, aspect leakage, hallucinated aspects).
   - Record representative error cases with review text, predicted labels, gold labels, and which model produced the error.

## Experimental Matrix
| Condition | Model | Prompt | Shots | Notes |
|-----------|-------|--------|-------|-------|
| Zero-shot | Small instruct (e.g., mistral-instruct) | Zero-shot | 0 | Baseline prompt above |
| Few-shot | Same | k-shot | 2–5 | Exemplar diversity emphasized |
| Supervised | Aspect classifiers | N/A | N/A | Fine-tuned baseline |

## Reporting
- For each aspect, list Accuracy and F1 for zero-shot, few-shot, and supervised baselines.
- Summarize overall macro Accuracy/F1 across aspects.
- Include qualitative error table contrasting model types.
- Document prompt text, model version, and decoding parameters for reproducibility.
