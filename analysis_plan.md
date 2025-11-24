# Model Interpretation Plan

## Linear Models: N-gram Weights
- Extract top positive and negative n-grams per intent/class using model coefficients.
- Break down weights by aspect (e.g., sentiment, domain attribute) if mult-task or multi-label.
- Present results as tables with n-gram, weight, and example utterances where each appears.
- Filter out stopwords and extremely rare n-grams to avoid noise.
- Script outline:
  - Load vectorizer vocabulary and coefficient matrix.
  - Map feature indices back to n-grams.
  - For each class/aspect: sort coefficients, select top-k high/low weights.
  - Save to CSV/Markdown for review.

## Transformer Models: Attribution on Examples
- Apply Integrated Gradients to representative samples per class/aspect.
- Use at least one correctly classified and one misclassified example per category.
- Aggregate token attributions to subwords/words for readability.
- Visualize highlights with heatmaps or color-coded text and capture screenshots for reports.
- Cross-check attributions against predicted probabilities to ensure alignment.

## Error Pattern Summary
- Audit validation/test predictions to tag common failure modes:
  - Negation handling (e.g., "not good" vs. "good").
  - Sarcasm/irony cues.
  - Aspect confusion (wrong target entity/attribute).
- For each pattern, include frequency counts and 2–3 concrete examples.
- Recommend dataset or prompt refinements (extra negation examples, sarcasm cues, aspect-specific prompts).

## Reporting Checklist
- Collate findings into a concise memo:
  - Top n-grams by class/aspect with takeaways.
  - Attribution visualizations and interpretations.
  - Error pattern counts and recommended fixes.
- Store artifacts under `reports/` with date-stamped filenames.
