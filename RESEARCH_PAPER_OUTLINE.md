# Aspect-Aware Sentiment Modeling for Educational Feedback

## Title
Aspect-Aware Sentiment Modeling for Educational Feedback: Joint Analysis of Teacher and Course Narratives

## Rationale
Student feedback often interweaves perceptions of teaching quality with course design, yet many sentiment systems treat comments as single-topic signals. By explicitly modeling aspect references (teacher vs. course), we can disentangle pedagogical strengths from curricular issues, leading to clearer actionability for instructors and program directors.

## Objectives
- Develop and compare baseline through state-of-the-art models (n-gram, sentence embeddings, transformer fine-tuning) for aspect-conditioned sentiment classification.
- Quantify the value of aspect prompts and multi-task supervision (aspect + sentiment) in improving label fidelity and robustness to noisy aspect tags.
- Evaluate cross-aspect generalization (train-on-teacher, test-on-course) to understand transferability and specialization trade-offs.
- Provide reproducible command-line workflows that support augmentation, prompting, and explainability analyses for research exchange.

## Contributions
- A unified experimental suite covering classical, embedding-based, and transformer models tailored to teacher/course aspects with runnable CLI recipes.
- Multi-task and aspect-prompted configurations that demonstrate gains in macro-F1 and stability under aspect-label noise.
- Robustness and cross-domain evaluations highlighting how aspect-aware conditioning influences transfer between teacher and course feedback.
- Practical guidance for deployment, including augmentation strategies, zero/low-shot prompting baselines, and interpretability reports for stakeholders.

## Description (Methods Overview)
We structure the study around incremental modeling complexity. First, we establish n-gram TF-IDF baselines with linear classifiers to anchor performance. Next, we encode comments (with aspect prompts) using modern sentence embeddings and train shallow classifiers. We then fine-tune lightweight transformers with aspect-aware inputs, extending to dual-head multi-task setups that jointly predict aspect and sentiment. Zero/low-shot prompting baselines with instruction-tuned LLMs provide a complementary paradigm. Finally, we probe robustness via data augmentation, assess cross-aspect transfer, and generate explanation artifacts (feature weights, attribution maps) to contextualize model decisions. All experiments are exposed through consistent command-line entry points to facilitate reproducibility and comparison.
