# Explainability and error analysis

## Objective
Provide interpretable insights into model decisions and diagnose common failure modes across aspects.

## Dataset assumptions
- Access to trained models from prior experiments (linear, embedding-based, transformer).

## Experiment steps
1. **Linear models**: Extract top weighted n-grams per sentiment class and aspect from Logistic Regression/SVM to surface indicative phrases.
2. **Transformer attributions**: Apply techniques like Integrated Gradients or attention rollout on representative teacher and course examples.
3. **Counterfactuals**: Use minimal edits (e.g., swap aspect tokens or key words) to probe model sensitivity.
4. **Error logging**: Collect misclassified examples with predicted vs. gold labels and confidence; categorize by phenomenon (negation, sarcasm, aspect confusion).
5. **Visualization**: Generate small tables/plots summarizing influential tokens and frequent error categories.

## Reporting
Summarize interpretability findings and actionable data/model improvements (e.g., add negation examples, strengthen aspect prompts).
