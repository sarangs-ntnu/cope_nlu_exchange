# Cross-Aspect Transfer Evaluation Plan

This document outlines how to measure transfer between teacher- and course-focused models.

## 1. Aspect-Specific Training and Cross-Aspect Testing
- Train separate models for each aspect (teacher-only and course-only data).
- Evaluate each model on its own aspect to confirm baseline quality.
- Cross-test: evaluate the teacher-trained model on course data and the course-trained model on teacher data to quantify zero-shot transfer.
- Report precision, recall, and F1 per aspect, and compare against baselines to spot asymmetries in transfer.

## 2. Joint Training with Aspect Tokens
- Combine teacher and course data into one corpus.
- Insert explicit aspect tokens (e.g., `<teacher>` or `<course>`) into inputs or labels to condition the model.
- Train a single model on the combined corpus.
- Evaluate per-aspect F1 on held-out teacher and course sets to understand how well a shared model performs relative to specialized models.
- Compare joint-training results to the cross-aspect tests above to assess whether shared representations help or hurt transfer.

## 3. Failure Analysis
- Inspect misclassified examples from both cross-testing and joint-training evaluations.
- Categorize errors (e.g., aspect confusion, missing context, domain-specific terminology, or label imbalance).
- Identify signals that should be shared vs. specialized (e.g., sentiment cues may transfer, but entity-specific phrasing may not).
- Use findings to decide whether to maintain specialized models, adopt a shared model with aspect tokens, or pursue hybrid approaches (e.g., multi-task heads or adapters).

## 4. Reporting
- Summarize results in a table with per-aspect precision/recall/F1 for:
  - Specialized models on in-aspect tests.
  - Cross-aspect tests (teacher→course, course→teacher).
  - Joint model with aspect tokens.
- Highlight failure categories and recommended next steps for model design.
